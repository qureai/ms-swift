"""CPU tests documenting that sequence PACKING (`--packing true`) is INCOMPATIBLE with the slow-fast branch.

Finding (see PACKING_ANALYSIS.md): padding-free packing collapses a pack of N samples into ONE sequence
(B=1). `_stash_slow_tokens` keys the slow K/V to batch rows and bails to an all-masked dummy whenever
`len(_slow_tokens) != B` -- which is ALWAYS true once a pack holds >1 volume -> the slow branch is SILENTLY
disabled (no error, degraded model). And `LLMCrossAttnBlock` has no per-packed-sample block mask, so if a
pack ever held exactly one volume the cross-attn would leak it to every packed sample's text (leakage).

These tests assert the CURRENT behavior so the incompatibility is explicit + regression-guarded. If someone
later adds segment-aware packing support (per-sample slow K/V + block-diagonal cross-attn mask), the first
two tests are the ones that must be updated.
"""
import types
import torch

from swift.template.templates.volume_mixin import Volume3DTemplateMixin
from swift.model.vision_3d.slowfast import LLMCrossAttnBlock

H, SLOW, NH, NKV, HD, T = 32, 16, 4, 2, 8, 7


class _BB:
    """Minimal stand-in for Vision3DBackbone, exposing just what _stash_slow_tokens reads."""
    slowfast = True
    encoder_dim = SLOW
    dtype = torch.float32
    slow_max_tokens = 4096

    def __init__(self):
        self._slow_tokens = None
        self._slow_kv_padded = None
        self._slow_mask_padded = None


def _call_stash(slow_tokens, B):
    bb = _BB()
    bb._slow_tokens = slow_tokens
    model = types.SimpleNamespace(visual_3d=bb)
    stub_self = types.SimpleNamespace(is_training=True, _ct_debug=False)
    input_ids = torch.zeros(B, 50, dtype=torch.long)
    Volume3DTemplateMixin._stash_slow_tokens(stub_self, model, input_ids)
    return bb


def test_packing_silently_disables_slowfast():
    # Packed: padding-free forces B=1, but the pack holds N=3 volumes -> len(slow)=3 != B=1 -> guard trips.
    bb = _call_stash([torch.randn(10, SLOW), torch.randn(8, SLOW), torch.randn(12, SLOW)], B=1)
    assert bb._slow_kv_padded.shape == (1, 1, SLOW), 'guard must install the (B,1) all-masked dummy'
    assert bb._slow_mask_padded is not None and not bb._slow_mask_padded.any(), \
        'packing (B=1, N>1 volumes) must leave the slow K/V ALL-MASKED -> slow branch contributes 0 (disabled)'
    print('ok packing_silently_disables_slowfast')


def test_unpacked_1to1_builds_real_slow_kv():
    # Contrast: no packing, B==N (1 volume per row) -> real slow K/V is built (slow-fast actually runs).
    bb = _call_stash([torch.randn(10, SLOW), torch.randn(8, SLOW)], B=2)
    assert bb._slow_kv_padded.shape == (2, 10, SLOW), 'T_max=10 padded slow K/V for the 1:1 case'
    assert bb._slow_mask_padded.any(), 'real (non-dummy) slow tokens must be present when B==N'
    assert bool(bb._slow_mask_padded[0].sum() == 10) and bool(bb._slow_mask_padded[1].sum() == 8)
    print('ok unpacked_1to1_builds_real_slow_kv')


def test_crossattn_has_no_packed_sample_isolation():
    # If cross-attn DID run on a packed row (the 1-volume edge case), it would leak: there is no per-sample
    # block mask, so query positions belonging to "sample B" attend the only volume's slow tokens too.
    blk = LLMCrossAttnBlock(H, SLOW, NH, NKV, HD).double()
    with torch.no_grad():
        blk.gate.fill_(1.0)
        blk.o_proj.weight.normal_()
    La, Lb = 3, 4                                   # a packed sequence: sample A (0:3) + sample B (3:7)
    h = torch.randn(1, La + Lb, H, dtype=torch.float64)
    kv = torch.randn(1, T, SLOW, dtype=torch.float64)   # ONE volume's slow tokens (survived the guard)
    mask = torch.ones(1, T, dtype=torch.bool)
    out = blk(h, kv, mask)
    # sample B's positions were modified by a volume that (under correct packing) isn't theirs -> leakage
    assert not torch.allclose(out[:, La:], h[:, La:], atol=1e-9), \
        'cross-attn modifies sample-B positions from another sample\'s volume -> no packing isolation (leakage)'
    print('ok crossattn_has_no_packed_sample_isolation')


if __name__ == '__main__':
    for fn in [test_packing_silently_disables_slowfast, test_unpacked_1to1_builds_real_slow_kv,
               test_crossattn_has_no_packed_sample_isolation]:
        fn()
    print('\nALL PACKING-COMPAT TESTS PASSED (packing is confirmed INCOMPATIBLE with slow-fast)')
