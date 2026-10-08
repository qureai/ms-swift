"""CPU tests for the FAST-channel anti-collapse additions (post-proj VICReg + orthogonality + proj_fast init).
Covers: ortho_reg behaviour (0 on orthonormal, large on rank-1, scale-free), proj_fast orthogonal init is
full-rank (vs the rank-1 the zero-init seeds), and rankreg_loss raises rank on a collapsed fast-token batch.
"""
import math
import torch

from swift.model.vision_3d.slowfast import ortho_reg, build_proj_fast, rankreg_loss


def test_ortho_reg_zero_on_orthonormal():
    torch.manual_seed(0)
    W = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))[0]   # orthonormal -> WWᵀ = I
    assert ortho_reg(W) < 1e-10, f'orthonormal weight should give ~0, got {ortho_reg(W)}'
    print('ok ortho_reg_zero_on_orthonormal')


def test_ortho_reg_large_on_rank1():
    v = torch.randn(64, 1, dtype=torch.float64)
    W = (v @ torch.randn(1, 64, dtype=torch.float64))                 # rank-1 -> far from orthonormal
    assert ortho_reg(W) > 0.1, f'rank-1 weight should be heavily penalised, got {ortho_reg(W)}'
    assert ortho_reg(W) > ortho_reg(torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))[0])
    print('ok ortho_reg_large_on_rank1')


def test_ortho_reg_scale_free_in_dim():
    # /d^2 normalisation -> an orthonormal matrix stays ~0 regardless of dimension (so a fixed lambda transfers)
    for d in (16, 64, 256):
        W = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]
        assert ortho_reg(W) < 1e-9
    print('ok ortho_reg_scale_free_in_dim')


def test_ortho_reg_rectangular():
    torch.manual_seed(1)
    W = torch.randn(32, 128, dtype=torch.float64)      # out<=in -> uses WWᵀ (32x32)
    r = ortho_reg(W)
    assert torch.isfinite(r) and r >= 0
    print('ok ortho_reg_rectangular')


def test_proj_fast_orthogonal_init_is_full_rank():
    pf = build_proj_fast(in_dim=2 * 1024, llm_hidden_size=2560)
    last = pf[-1]
    # zero-init (the old default) -> rank 0; orthogonal-init -> (near) full rank on the smaller dim
    torch.nn.init.zeros_(last.weight)
    assert torch.linalg.matrix_rank(last.weight.double()).item() == 0
    torch.nn.init.orthogonal_(last.weight)
    rk = torch.linalg.matrix_rank(last.weight.double()).item()
    assert rk == min(last.weight.shape), f'orthogonal init should be full-rank, got rank {rk}'
    print('ok proj_fast_orthogonal_init_is_full_rank')


def test_rankreg_penalises_collapsed_fast_tokens():
    # a near rank-1 fast-token batch (what we observed) must incur a HIGH cov term vs a diverse batch
    torch.manual_seed(2)
    N, D = 1024, 256
    direction = torch.randn(1, D, dtype=torch.float64)
    collapsed = direction.repeat(N, 1) + 0.01 * torch.randn(N, D, dtype=torch.float64)  # ~rank 1
    diverse = torch.randn(N, D, dtype=torch.float64)                                     # full rank
    lc = rankreg_loss(collapsed, max_tokens=1024)
    ld = rankreg_loss(diverse, max_tokens=1024)
    assert lc > ld, f'collapsed fast tokens must be penalised more than diverse ones ({lc} vs {ld})'
    print(f'ok rankreg_penalises_collapsed_fast_tokens (collapsed={lc:.3f} > diverse={ld:.3f})')


if __name__ == '__main__':
    for fn in [test_ortho_reg_zero_on_orthonormal, test_ortho_reg_large_on_rank1,
               test_ortho_reg_scale_free_in_dim, test_ortho_reg_rectangular,
               test_proj_fast_orthogonal_init_is_full_rank, test_rankreg_penalises_collapsed_fast_tokens]:
        fn()
    print('\nALL FAST-COLLAPSE-FIX TESTS PASSED')
