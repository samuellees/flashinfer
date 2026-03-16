# TRT-LLM Fused MoE: Support float32 Routing Logits (Renormalize)

## Background

`csrc/trtllm_fused_moe_kernel_launcher.cu:1803` enforces `routing_logits` to be `bfloat16` for
non-DeepSeekV3 routing methods (including Renormalize). This document investigates the origin of
this restriction and what changes are needed to support `float32` routing logits.

## Current Restriction Chain

### Layer 1: FlashInfer Wrapper Type Check

Multiple entry functions enforce `dl_bfloat16` for `routing_logits`:

| Entry Function | Line | Allowed dtypes |
|----------------|------|----------------|
| `trtllm_bf16_moe` | 1618-1620 | float32, bfloat16 |
| `trtllm_fp8_moe` (use_routing_scales_on_input) | 1698 | bfloat16 only |
| `trtllm_fp8_moe` (DeepSeekV3) | 1700 | float32 only |
| `trtllm_fp8_moe` (other) | 1702 | bfloat16 only |
| `trtllm_fp8_block_scale_moe` (DeepSeekV3) | 1800 | float32 only |
| **`trtllm_fp8_block_scale_moe` (other)** | **1803** | **bfloat16 only** |
| `trtllm_fp8_block_scale_moe_v2` | 1934-1936 | float32, bfloat16 |
| `trtllm_moe_routing_only` | 2062-2063 | float32, bfloat16 |

**Observation**: `trtllm_bf16_moe`, `trtllm_fp8_block_scale_moe_v2`, and
`trtllm_moe_routing_only` already accept float32. The restriction is only in `trtllm_fp8_moe`
and `trtllm_fp8_block_scale_moe`.

### Layer 2: Runner Dispatch (`trtllm_fused_moe_runner.cu:145`)

```cpp
// In Routing::Runner::run(), for Renormalize/RenormalizeNaive/TopK:
routingData.mDtypeExpW = btg::Dtype::Bfloat16;  // hardcoded
routingData.mPtrScores = routingLogits;           // void*, no cast
```

`mDtypeExpW` is hardcoded to `Bfloat16`. This field simultaneously controls:
- **`InputT`**: the type used to read `mPtrScores` (routing_logits)
- **`OutputT`**: the type used to write `mPtrTopKWeights` (expert_weights)

### Layer 3: Kernel Dispatch Macro (`DevKernel.h:235-251`)

```cpp
#define LAUNCH_ROUTING_WITH_NUM_EXPERTS(data, ..., numExperts)
  if (data.mDtypeExpW == tg::Dtype::Fp32) {
    // Instantiates kernel with InputT=float, OutputT=float
    LAUNCH_TILEN(data, ..., LAUNCH_ESC(float, float, numExperts, ...), ...);
  } else if (data.mDtypeExpW == tg::Dtype::Bfloat16) {
    // Instantiates kernel with InputT=__nv_bfloat16, OutputT=__nv_bfloat16
    LAUNCH_TILEN(data, ..., LAUNCH_ESC(__nv_bfloat16, __nv_bfloat16, numExperts, ...), ...);
  }
```

**The Fp32 branch already exists.** The kernel template is already compiled for `InputT=float`.

### Layer 4: Kernel Template (`RoutingKernel.h:107-126`)

```cpp
template <typename InputT_, typename OutputT_, int MaxNumExperts_, bool isPow2_, bool UsePdl_>
struct KernelParamsBase {
  InputT const* mPtrScores = nullptr;  // routing_logits cast to InputT*
  OutputT* mPtrTopKWeights = nullptr;  // expert_weights cast to OutputT*
};
```

### Layer 5: Renormalize Data Struct (`RoutingKernel.h:271-278`)

```cpp
namespace routingRenormalize {
struct Data : public DataBase {
  tg::Dtype mDtypeExpW{tg::Dtype::Fp32};   // default is Fp32!
  tg::Dtype mDtypeElt{tg::Dtype::Bfloat16};
  bool mDoSoftmaxBeforeTopK{false};
  bool mNormTopkProb{true};
  bool mApplySoftmaxAfterTopK{false};
};
}
```

Note: The default for `routingRenormalize::Data::mDtypeExpW` is **Fp32**, but the runner
overrides it to Bfloat16.

## Key Coupling Issue

**`mDtypeExpW` controls both `InputT` (scores) and `OutputT` (expert_weights) simultaneously.**

If we change `mDtypeExpW` to `Fp32` to support float32 routing_logits, expert_weights will also
become float32, which affects:

1. **`expert_weights` allocation** in `prepare_routing()` — currently hardcoded to `dl_bfloat16`
2. **Finalize kernel** — the weighted sum kernel that uses `expert_weights` to combine expert
   outputs. Its dispatch macro (`LAUNCH_EXPW` in `DevKernel.h:119-140`) already supports
   `mDtypeExpW == Fp32` for all element types (fp16, bf16, e4m3).

## Required Changes for float32 Routing Logits (Renormalize)

### Change 1: Relax dtype checks in wrapper functions

**Files**: `csrc/trtllm_fused_moe_kernel_launcher.cu`

Lines 1698, 1702, 1803: Change from `dl_bfloat16` only to accept both `dl_float32` and
`dl_bfloat16`.

```cpp
// Before (line 1803):
TVM_FFI_ICHECK_EQ(routing_logits.value().dtype(), dl_bfloat16)
    << "routing_logits must be bfloat16.";

// After:
TVM_FFI_ICHECK(routing_logits.value().dtype() == dl_float32 ||
               routing_logits.value().dtype() == dl_bfloat16)
    << "routing_logits must be float or bfloat16.";
```

### Change 2: Pass routing_logits dtype to runner

**File**: `csrc/trtllm_fused_moe_runner.cu`

The `Routing::Runner::run()` signature currently receives `dtypeElt` (hidden_states dtype) but
not the routing_logits dtype. Two options:

**Option A** (minimal): Infer from `dtypeElt` parameter already passed to `run()`.

```cpp
// Line 145, before:
routingData.mDtypeExpW = btg::Dtype::Bfloat16;

// After: propagate dtypeElt (which already carries the right info for some callers)
// But this conflates hidden_states dtype with routing_logits dtype — not ideal.
```

**Option B** (cleaner): Add a `dtypeRouting` parameter to `Routing::Runner::run()`.

```cpp
void run(void* routingLogits, ..., btg::Dtype dtypeElt, btg::Dtype dtypeBias,
         btg::Dtype dtypeRouting,  // NEW
         bool useRoutingScalesOnInput, ...);
```

Then in the Renormalize branch:
```cpp
routingData.mDtypeExpW = dtypeRouting;  // was: btg::Dtype::Bfloat16
```

### Change 3: Update `expert_weights` allocation

**File**: `csrc/trtllm_fused_moe_kernel_launcher.cu`

In each launcher's `prepare_routing()`, `expert_weights` is allocated with hardcoded
`dl_bfloat16`. This needs to match `mDtypeExpW`:

```cpp
// Before (multiple locations, e.g., line 502):
FusedMoeLauncher::expert_weights =
    alloc_tensor({args->num_tokens, args->top_k}, dl_bfloat16, hidden_states.device());

// After:
auto ew_dtype = (mRoutingExpWDtype == btg::Dtype::Fp32) ? dl_float32 : dl_bfloat16;
FusedMoeLauncher::expert_weights =
    alloc_tensor({args->num_tokens, args->top_k}, ew_dtype, hidden_states.device());
```

### Change 4: Verify finalize kernel compatibility

**File**: `include/flashinfer/trtllm/fused_moe/DevKernel.h`

The `LAUNCH_EXPW` macro (lines 119-140) already handles `mDtypeExpW == Fp32` for all element
types (fp16, bf16, e4m3). **No change needed** for finalize kernels.

### Change 5: Update Python API (if applicable)

**File**: `flashinfer/fused_moe/*.py`

If the Python API enforces `routing_logits.dtype == torch.bfloat16`, relax it to also accept
`torch.float32`.

## Locations of `prepare_routing()` Overrides

| Launcher Class | Line | `expert_weights` dtype |
|----------------|------|------------------------|
| `Bf16MoeLauncher` | 479 | dl_bfloat16 (line 502) |
| `Fp8MoeLauncher` | 620 | dl_bfloat16 (line 652) |
| `Fp8BlockScaleMoeLauncher` | 884 | dl_bfloat16 (line 930) |
| `Fp8BlockScaleMoeLauncherV2` | 1181 | dl_bfloat16 (line 1206) |
| `MxFp8MoeLauncher` | 1366 | dl_bfloat16 (line 1419) |

All 5 locations need the same update.

## Risk Assessment

- **Low risk**: The TRT-LLM routing kernel already has float32 template instantiations compiled
  (`LAUNCH_ROUTING_WITH_NUM_EXPERTS` Fp32 branch).
- **Low risk**: The finalize kernel's `LAUNCH_EXPW` macro already supports `mDtypeExpW == Fp32`.
- **Medium risk**: Changing `expert_weights` from bf16 to fp32 doubles its memory footprint
  (`num_tokens * top_k * 4` bytes vs 2 bytes). For large batch sizes this could matter.
- **Medium risk**: The `mDtypeExpW` coupling means we cannot have float32 scores with bfloat16
  expert_weights without decoupling the field into `mDtypeScore` + `mDtypeExpW` (like
  DeepSeek's `mDtypeScore` / `mDtypeBias` / `mDtypeExpW` separation).

## Summary

The bfloat16 restriction on routing_logits for Renormalize is a **wrapper-level limitation**, not
a kernel-level one. The underlying TRT-LLM routing kernels are already templated and compiled for
float32 input. The main complexity is that `mDtypeExpW` couples score dtype and expert_weights
dtype together — changing one changes both, with downstream implications for memory allocation
and finalize kernels (though these are already supported).
