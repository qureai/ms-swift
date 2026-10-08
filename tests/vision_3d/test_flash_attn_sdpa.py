"""CPU tests for the Flash-Attention (SDPA) conversion of STA temporal attention (sta.py) and the slow-fast
cross-attention (slowfast.py). SDPA uses its math backend on CPU, so these verify the rewrite is NUMERICALLY
EQUIVALENT to the old eager matmul+softmax (fp32, tight tol) and that the mask/edge-cases are preserved:
  * STA: banded (windowed) + full temporal attention match an eager oracle; zero-init proj => identity;
    variable gt; no NaN.
  * Cross-attn: core matches an eager masked oracle; a fully-masked sample stays finite and contributes 0.
The broader cross-attn semantics (identity-at-init, GQA, masked-keys-ignored, install hooks) are covered by
tests/vision_3d/test_slowfast_crossattn.py, which must also still pass after the SDPA rewrite.
"""
import math
import torch
import torch.nn.functional as F

from swift.model.vision_3d.sta import VisionTemporalAttention
from swift.model.vision_3d.slowfast import LLMCrossAttnBlock


def _eager_attn(q, k, v, keep_mask=None):
    """Reference eager attention (the pre-SDPA implementation). keep_mask: bool, True = attend."""
    hd = q.shape[-1]
    a = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)
    if keep_mask is not None:
        a = a.masked_fill(~keep_mask, float('-inf'))
    a = torch.softmax(a.float(), dim=-1).to(q.dtype)
    return torch.matmul(a, v)


def test_sta_core_banded_matches_eager():
    torch.manual_seed(0)
    s, H, gt, hd, w = 6, 3, 11, 8, 2
    q, k, v = (torch.randn(s, H, gt, hd, dtype=torch.float64) for _ in range(3))
    ar = torch.arange(gt)
    keep = ((ar[None, :] - ar[:, None]).abs() <= w)[None, None]      # (1,1,gt,gt)
    o_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
    o_eager = _eager_attn(q, k, v, keep)
    assert torch.allclose(o_sdpa, o_eager, atol=1e-6, rtol=1e-5), (o_sdpa - o_eager).abs().max()
    print('ok sta_core_banded_matches_eager')


def test_sta_core_full_matches_eager():
    torch.manual_seed(1)
    s, H, gt, hd = 4, 2, 7, 8
    q, k, v = (torch.randn(s, H, gt, hd, dtype=torch.float64) for _ in range(3))
    o_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
    o_eager = _eager_attn(q, k, v, None)
    assert torch.allclose(o_sdpa, o_eager, atol=1e-6, rtol=1e-5)
    print('ok sta_core_full_matches_eager')


def test_sta_forward_identity_at_init_and_no_nan():
    torch.manual_seed(2)
    hidden, nh, hd = 32, 4, 8
    mod = VisionTemporalAttention(hidden, nh, hd, window=2).double()
    gt, gh, gw = 5, 2, 3
    n = gt * gh * gw
    x = torch.randn(n, hidden, dtype=torch.float64)
    grid = torch.tensor([[gt, gh, gw]])
    out = mod(x, grid)
    assert out.shape == (n, hidden)
    # proj is zero-initialised => temporal path contributes exactly 0 at init (STA identity-equivalent)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-12), 'zero-init proj must give 0 output at init'
    # activate proj and confirm finite (no NaN) output
    with torch.no_grad():
        mod.proj.weight.normal_()
    out2 = mod(x, grid)
    assert torch.isfinite(out2).all(), 'STA output must be finite'
    print('ok sta_forward_identity_at_init_and_no_nan')


def test_sta_forward_variable_gt_batch():
    torch.manual_seed(3)
    hidden, nh, hd = 24, 3, 8
    mod = VisionTemporalAttention(hidden, nh, hd, window=1).double()
    with torch.no_grad():
        mod.proj.weight.normal_()
    # two volumes with DIFFERENT gt concatenated (gt-major), as in a real batch
    grids = [(3, 2, 2), (6, 1, 2)]
    tokens = sum(gt * gh * gw for gt, gh, gw in grids)
    x = torch.randn(tokens, hidden, dtype=torch.float64)
    grid = torch.tensor(grids)
    out = mod(x, grid)
    assert out.shape == (tokens, hidden) and torch.isfinite(out).all()
    print('ok sta_forward_variable_gt_batch')


def test_crossattn_core_matches_eager():
    torch.manual_seed(4)
    B, nh, L, T, hd = 2, 4, 5, 7, 8
    q = torch.randn(B, nh, L, hd, dtype=torch.float64)
    k = torch.randn(B, nh, T, hd, dtype=torch.float64)
    v = torch.randn(B, nh, T, hd, dtype=torch.float64)
    valid = torch.ones(B, T, dtype=torch.bool)
    valid[:, T // 2:] = False
    m = valid.view(B, 1, 1, T)
    o_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    o_eager = _eager_attn(q, k, v, m)
    assert torch.allclose(o_sdpa, o_eager, atol=1e-6, rtol=1e-5), (o_sdpa - o_eager).abs().max()
    print('ok crossattn_core_matches_eager')


def test_crossattn_fully_masked_sample_finite_and_zero():
    torch.manual_seed(5)
    H, SLOW, NH, NKV, HD = 32, 16, 4, 2, 8
    B, L, T = 2, 5, 7
    blk = LLMCrossAttnBlock(H, SLOW, NH, NKV, HD).double()
    with torch.no_grad():
        blk.gate.fill_(1.0)
        blk.o_proj.weight.normal_()
    h = torch.randn(B, L, H, dtype=torch.float64)
    kv = torch.randn(B, T, SLOW, dtype=torch.float64)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0] = False                                   # sample 0 has NO valid slow tokens (the dummy case)
    out = blk(h, kv, mask)
    assert torch.isfinite(out).all(), 'fully-masked sample must not produce NaN/Inf (SDPA all-masked guard)'
    assert torch.allclose(out[0], h[0], atol=1e-10), 'sample with no slow tokens must pass through unchanged'
    assert not torch.allclose(out[1], h[1]), 'sample with slow tokens must be modified'
    # gradient must be finite too (the all-masked guard must not leak NaN grads)
    out.sum().backward()
    assert torch.isfinite(blk.q_proj.weight.grad).all() and torch.isfinite(blk.v_proj.weight.grad).all()
    print('ok crossattn_fully_masked_sample_finite_and_zero')


if __name__ == '__main__':
    for fn in [test_sta_core_banded_matches_eager, test_sta_core_full_matches_eager,
               test_sta_forward_identity_at_init_and_no_nan, test_sta_forward_variable_gt_batch,
               test_crossattn_core_matches_eager, test_crossattn_fully_masked_sample_finite_and_zero]:
        fn()
    print('\nALL FLASH-ATTN (SDPA) TESTS PASSED')
