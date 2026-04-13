"""Test FP8 block scale MoE with near-zero scales (issue #2595).

Models like Qwen3.5-397B-A17B-FP8 have inactive experts with near-zero
block scales (~1e-23). The CUTLASS FP8 kernel must produce finite (near-zero)
output, not NaN.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import get_compute_capability


def _is_fp8_block_scale_supported():
    if not torch.cuda.is_available():
        return False
    cc = get_compute_capability(torch.device("cuda"))
    return cc[0] >= 9  # SM90 (Hopper) for cutlass_fused_moe FP8 block scale


def ceil_div(a, b):
    return (a + b - 1) // b


def per_block_cast_to_fp8(x, block_size_n=128):
    """2D block-wise FP8 quantization matching DeepSeek block scale format."""
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, block_size_n) * block_size_n),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // block_size_n, block_size_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


@pytest.mark.skipif(
    not _is_fp8_block_scale_supported(),
    reason="FP8 block scale MoE requires SM90+",
)
@pytest.mark.parametrize(
    "dead_expert_scale",
    [1e-10, 1e-15, 1e-20, 1e-23, 1e-30],
    ids=["1e-10", "1e-15", "1e-20", "1e-23(issue)", "1e-30"],
)
def test_fp8_block_scale_moe_near_zero_scales(dead_expert_scale):
    """Verify no NaN output when expert block scales are near-zero.

    Regression test for https://github.com/flashinfer-ai/flashinfer/issues/2595.
    Uses cutlass_fused_moe with use_deepseek_fp8_block_scale=True (SM90 path).
    Simulates dead experts by injecting near-zero block scales after quantization.
    """
    import flashinfer.fused_moe as fused_moe

    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("cutlass_fused_moe FP8 block scale only supported on SM90")

    torch.manual_seed(42)
    device = torch.device("cuda")
    otype = torch.bfloat16

    num_tokens = 4
    hidden_size = 256
    intermediate_size = 128
    num_experts = 8
    top_k = 2

    # Create and quantize weights
    w1 = (
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, device=device, dtype=otype
        )
        / 10
    )
    w2 = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, device=device, dtype=otype
        )
        / 10
    )

    w1_fp8 = torch.empty_like(w1, dtype=torch.float8_e4m3fn)
    w1_scale = torch.zeros(
        num_experts,
        ceil_div(2 * intermediate_size, 128),
        ceil_div(hidden_size, 128),
        dtype=torch.float32,
        device=device,
    )
    w2_fp8 = torch.empty_like(w2, dtype=torch.float8_e4m3fn)
    w2_scale = torch.zeros(
        num_experts,
        ceil_div(hidden_size, 128),
        ceil_div(intermediate_size, 128),
        dtype=torch.float32,
        device=device,
    )

    for e in range(num_experts):
        fp8, scale = per_block_cast_to_fp8(w1[e].float())
        w1_fp8[e] = fp8
        w1_scale[e] = scale
        fp8, scale = per_block_cast_to_fp8(w2[e].float())
        w2_fp8[e] = fp8
        w2_scale[e] = scale

    # Inject near-zero scales for dead expert 0 (simulating Qwen3.5 dead experts)
    w1_scale[0] = dead_expert_scale
    w2_scale[0] = dead_expert_scale

    # Input
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=otype)

    # Routing: token 0 → dead expert 0 + normal expert 1
    selected_experts = torch.zeros(num_tokens, top_k, dtype=torch.int32, device=device)
    selected_experts[0, 0] = 0  # dead expert
    selected_experts[0, 1] = 1
    for i in range(1, num_tokens):
        selected_experts[i, 0] = 1
        selected_experts[i, 1] = 2

    routing_weights = torch.ones(num_tokens, top_k, device=device, dtype=torch.float32)
    routing_weights = F.softmax(routing_weights, dim=1)

    result = fused_moe.cutlass_fused_moe(
        x.contiguous(),
        selected_experts,
        routing_weights,
        w1_fp8.contiguous(),
        w2_fp8.contiguous(),
        otype,
        use_deepseek_fp8_block_scale=True,
        quant_scales=[w1_scale.contiguous(), w2_scale.contiguous()],
    )
    output = result[0] if isinstance(result, list) else result

    assert not torch.isnan(output).any(), (
        f"NaN detected with dead_expert_scale={dead_expert_scale}. "
        f"NaN count: {torch.isnan(output).sum().item()}/{output.numel()}"
    )
    assert torch.isfinite(output).all(), (
        f"Inf detected with dead_expert_scale={dead_expert_scale}"
    )
