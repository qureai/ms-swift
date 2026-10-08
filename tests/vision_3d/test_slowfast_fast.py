"""CPU unit tests for the slow-fast FAST branch (no model load). Covers S1 shape seams."""
import torch
from swift.model.vision_3d.slowfast import fast_pool_3d, fast_out_dim, build_proj_fast

C = 8
TARGET = (2, 3, 3)          # Tf,Hf,Wf -> fast_count = 18
FC = TARGET[0] * TARGET[1] * TARGET[2]


def _tokens(gt, gh, gw, c=C):
    return torch.randn(gt * gh * gw, c)


def test_shapes_all_modes():
    t = _tokens(6, 8, 8)
    assert fast_pool_3d(t, (6, 8, 8), TARGET, 'avg').shape == (FC, C)
    assert fast_pool_3d(t, (6, 8, 8), TARGET, 'max').shape == (FC, C)
    assert fast_pool_3d(t, (6, 8, 8), TARGET, 'avgmax').shape == (FC, 2 * C)
    print('ok shapes_all_modes')


def test_fixed_count_variable_resolution():
    # two DIFFERENT input grids must produce the SAME fixed fast_count (S1: variable-resolution).
    a = fast_pool_3d(_tokens(4, 6, 10), (4, 6, 10), TARGET, 'avgmax')
    b = fast_pool_3d(_tokens(12, 16, 5), (12, 16, 5), TARGET, 'avgmax')
    assert a.shape == b.shape == (FC, 2 * C)
    print('ok fixed_count_variable_resolution')


def test_upsample_when_target_bigger():
    # target larger than source in an axis (small volume) still yields exactly fast_count, no crash (S1).
    t = _tokens(1, 2, 2)
    out = fast_pool_3d(t, (1, 2, 2), TARGET, 'avgmax')
    assert out.shape == (FC, 2 * C)
    print('ok upsample_when_target_bigger')


def test_avgmax_is_concat_of_avg_and_max():
    t = _tokens(6, 6, 6)
    avg = fast_pool_3d(t, (6, 6, 6), TARGET, 'avg')
    mx = fast_pool_3d(t, (6, 6, 6), TARGET, 'max')
    am = fast_pool_3d(t, (6, 6, 6), TARGET, 'avgmax')
    assert torch.allclose(am[:, :C], avg) and torch.allclose(am[:, C:], mx)
    print('ok avgmax_is_concat')


def test_gt_major_order_and_values():
    # identity-target (target == source grid) must reproduce the tokens exactly for avg AND max.
    gt, gh, gw = 2, 3, 3
    t = torch.arange(gt * gh * gw * C, dtype=torch.float32).reshape(gt * gh * gw, C)
    outa = fast_pool_3d(t, (gt, gh, gw), (gt, gh, gw), 'avg')
    outm = fast_pool_3d(t, (gt, gh, gw), (gt, gh, gw), 'max')
    assert torch.allclose(outa, t), 'avg identity-pool must return tokens in gt-major order'
    assert torch.allclose(outm, t), 'max identity-pool must return tokens in gt-major order'
    print('ok gt_major_order_and_values')


def test_z_is_pooled_not_dropped():
    # a signal on ONE slice must survive under max (peak preserved), unlike a naive z-subsample.
    gt, gh, gw = 8, 3, 3
    t = torch.zeros(gt * gh * gw, C)
    t[3 * gh * gw:4 * gh * gw] = 5.0                  # slice index 3 is "hot"
    out_max = fast_pool_3d(t, (gt, gh, gw), (2, gh, gw), 'max')
    assert out_max.max().item() == 5.0, 'max pool over z must carry the single-slice peak'
    print('ok z_is_pooled_not_dropped')


def test_grid_mismatch_raises():
    try:
        fast_pool_3d(_tokens(4, 4, 4), (4, 4, 5), TARGET, 'avg')
    except ValueError:
        print('ok grid_mismatch_raises'); return
    raise AssertionError('expected ValueError on grid/M mismatch')


def test_out_dim_and_proj():
    assert fast_out_dim(C, 'avgmax') == 2 * C and fast_out_dim(C, 'avg') == C
    proj = build_proj_fast(2 * C, 16)
    y = proj(torch.randn(FC, 2 * C))
    assert y.shape == (FC, 16)
    print('ok out_dim_and_proj')


if __name__ == '__main__':
    for fn in [test_shapes_all_modes, test_fixed_count_variable_resolution, test_upsample_when_target_bigger,
               test_avgmax_is_concat_of_avg_and_max, test_gt_major_order_and_values, test_z_is_pooled_not_dropped,
               test_grid_mismatch_raises, test_out_dim_and_proj]:
        fn()
    print('\nALL FAST-BRANCH TESTS PASSED')
