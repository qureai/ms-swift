"""Backbone slow-fast forward integration test with a MOCK adapter (no model load). Covers S1 seams:
per-volume merged-count split, variable-resolution, fast_count token accounting, slow stash, and that
the DEFAULT (slowfast off) path is untouched."""
import torch
import torch.nn as nn
from swift.model.vision_3d.backbone import Vision3DBackbone

ENC, LLM = 16, 32
TPS, PS, M = 2, 14, 2


class MockQwenAdapter:
    temporal_patch_size, patch_size, merge_size = TPS, PS, M

    def __init__(self, enc_dim):
        self.enc_dim = enc_dim
        self.last_total = None

    def _merged(self, v):
        gt, gh, gw = v.shape[0] // TPS, v.shape[2] // PS, v.shape[3] // PS
        return gt * (gh // M) * (gw // M)

    def __call__(self, encoder, vols, max_tokens, grid=None, variable=False, pool=True):
        assert variable, 'slowfast path must call adapter with variable=True'
        total = sum(self._merged(v) for v in vols)
        self.last_total = total
        return torch.randn(total, self.enc_dim)          # concatenated merged tokens, volume order


def _bb(slowfast, fast_grid=(2, 2, 2), pool='avgmax'):
    enc = nn.Linear(1, 1)                                # dummy: gives .parameters()/dtype
    return Vision3DBackbone(enc, encoder_dim=ENC, llm_hidden_size=LLM, max_tokens=100,
                            adapter=MockQwenAdapter(ENC), variable_tokens=True,
                            slowfast=slowfast, fast_grid=fast_grid, fast_pool=pool)


def _vols():
    # two DIFFERENT-sized volumes -> different merged counts (8 and 24)
    v1 = torch.randn(4, 11, 56, 56)     # gt2 gh4 gw4 -> 2*2*2 = 8
    v2 = torch.randn(6, 11, 112, 56)    # gt3 gh8 gw4 -> 3*4*2 = 24
    return [v1, v2], [8, 24]


def test_fast_output_and_slow_stash():
    bb = _bb(True)
    vols, merged = _vols()
    out = bb(vols)
    fc = bb.num_fast_tokens()
    assert fc == 8, f'fast_count should be 2*2*2=8, got {fc}'
    assert out.shape == (2 * fc, LLM), f'fast output {tuple(out.shape)} != {(2*fc, LLM)}'
    assert bb._slow_tokens is not None and len(bb._slow_tokens) == 2
    assert [t.shape[0] for t in bb._slow_tokens] == merged, 'slow tokens must keep per-volume merged counts'
    assert all(t.shape[1] == ENC for t in bb._slow_tokens), 'slow tokens are pre-proj (enc_dim)'
    print('ok fast_output_and_slow_stash')


def test_proj_fast_input_dim_matches_pool():
    for pool, mult in [('avg', 1), ('max', 1), ('avgmax', 2)]:
        bb = _bb(True, pool=pool)
        assert bb.proj_fast[0].in_features == ENC * mult, f'{pool}: proj_fast in {bb.proj_fast[0].in_features}'
        out = bb(_vols()[0])
        assert out.shape == (2 * bb.num_fast_tokens(), LLM)
    print('ok proj_fast_input_dim_matches_pool')


def test_num_tokens_switches_with_slowfast():
    assert _bb(True).num_tokens() == 8                   # fast_count
    off = _bb(False)
    assert off.num_tokens() == 100 and off.proj_fast is None and off.slowfast is False
    print('ok num_tokens_switches_with_slowfast')


def test_default_path_untouched_when_off():
    # slowfast off -> _slowfast_forward is never entered; the dispatch is a no-op.
    bb = _bb(False)
    assert bb.slowfast is False and bb.proj_fast is None
    print('ok default_path_untouched_when_off')


if __name__ == '__main__':
    for fn in [test_fast_output_and_slow_stash, test_proj_fast_input_dim_matches_pool,
               test_num_tokens_switches_with_slowfast, test_default_path_untouched_when_off]:
        fn()
    print('\nALL BACKBONE SLOW-FAST TESTS PASSED')
