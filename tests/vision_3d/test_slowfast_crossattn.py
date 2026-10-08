"""CPU tests for the slow-fast SLOW branch (cross-attention). Covers S2: identity-at-init (backwards-compat),
GQA, masking / no-slow-token guard, and install_slowfast HOOKING the correct layers (no layer renaming)."""
import torch
import torch.nn as nn
from swift.model.vision_3d.slowfast import LLMCrossAttnBlock, install_slowfast

H, SLOW, NH, NKV, HD = 32, 16, 4, 2, 8
B, L, T = 2, 5, 7


def test_identity_at_init():
    blk = LLMCrossAttnBlock(H, SLOW, NH, NKV, HD)
    h = torch.randn(B, L, H)
    kv = torch.randn(B, T, SLOW)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = blk(h, kv, mask)
    assert torch.allclose(out, h, atol=1e-6), 'zero-gate must make cross-attn an identity at init'
    print('ok identity_at_init')


def test_shape_and_gqa_and_active():
    blk = LLMCrossAttnBlock(H, SLOW, NH, NKV, HD)
    with torch.no_grad():
        blk.gate.fill_(1.0)                      # activate
        blk.o_proj.weight.normal_()
    h = torch.randn(B, L, H)
    out = blk(h, torch.randn(B, T, SLOW), torch.ones(B, T, dtype=torch.bool))
    assert out.shape == (B, L, H)
    assert not torch.allclose(out, h), 'active gate must change the output (GQA nkv<nh path)'
    print('ok shape_and_gqa_and_active')


def test_masking_ignores_invalid_keys():
    blk = LLMCrossAttnBlock(H, SLOW, NH, NKV, HD)
    with torch.no_grad():
        blk.gate.fill_(1.0); blk.o_proj.weight.normal_()
    h = torch.randn(B, L, H)
    kv = torch.randn(B, T, SLOW)
    m_full = torch.ones(B, T, dtype=torch.bool)
    m_half = m_full.clone(); m_half[:, T // 2:] = False
    # changing the MASKED-OUT keys must not change the output
    kv2 = kv.clone(); kv2[:, T // 2:] = torch.randn_like(kv2[:, T // 2:])
    o1 = blk(h, kv, m_half)
    o2 = blk(h, kv2, m_half)
    assert torch.allclose(o1, o2, atol=1e-6), 'masked keys must not affect the output'
    print('ok masking_ignores_invalid_keys')


def test_no_slow_tokens_row_is_identity():
    blk = LLMCrossAttnBlock(H, SLOW, NH, NKV, HD)
    with torch.no_grad():
        blk.gate.fill_(1.0); blk.o_proj.weight.normal_()
    h = torch.randn(B, L, H)
    kv = torch.randn(B, T, SLOW)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0] = False                              # sample 0 has NO valid slow tokens
    out = blk(h, kv, mask)
    assert torch.allclose(out[0], h[0], atol=1e-6), 'a sample with no slow tokens must pass through unchanged'
    assert not torch.allclose(out[1], h[1]), 'sample 1 (with slow tokens) should be modified'
    print('ok no_slow_tokens_row_is_identity')


class _MockDecoderLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.lin = nn.Linear(h, h)
    def forward(self, hidden_states, *args, **kwargs):
        return self.lin(hidden_states)           # returns a plain tensor, like Qwen3.5


class _Cfg:
    hidden_size = H
    num_attention_heads = NH
    num_key_value_heads = NKV
    head_dim = HD
    layer_types = ['linear_attention', 'full_attention', 'linear_attention', 'full_attention',
                   'linear_attention', 'full_attention']


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_MockDecoderLayer(H) for _ in range(6)])


class _LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _Inner()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _LM()
        self.config = _Cfg()


class _MockBackbone(nn.Module):
    slowfast = True
    def __init__(self):
        super().__init__()
        self._slow_kv_padded = None
        self._slow_mask_padded = None


def test_install_hooks_full_attention_layers_no_rename():
    m = _Model(); bb = _MockBackbone()
    idxs = install_slowfast(m, bb, slow_dim=SLOW, every_n=1)     # every full-attn layer
    assert idxs == [1, 3, 5], f'should hook full_attention layers, got {idxs}'
    layers = m.model.language_model.layers
    # layers are NOT wrapped/renamed (resume-safety): still the original class
    assert all(type(layers[i]).__name__ == '_MockDecoderLayer' for i in range(6)), 'layers must not be wrapped'
    # cross blocks live on the backbone -> saved under visual_3d.cross_blocks.* and overlaid on resume
    assert hasattr(bb, 'cross_blocks') and len(bb.cross_blocks) == 3
    assert any('cross_blocks' in n for n, _ in bb.named_parameters()), 'cross params must be under the backbone'
    h = torch.randn(1, 3, H)
    expected = layers[1].lin(h)                                 # the raw layer output (bypassing the hook)
    # identity when no slow kv set
    bb._slow_kv_padded = None
    assert torch.allclose(layers[1](h), expected, atol=1e-6), 'hook is a no-op when no slow kv (identity)'
    # identity at init EVEN WITH slow kv (zero gate)
    bb._slow_kv_padded = torch.randn(1, 4, SLOW); bb._slow_mask_padded = torch.ones(1, 4, dtype=torch.bool)
    assert torch.allclose(layers[1](h), expected, atol=1e-6), 'zero-gate => identity even with slow kv (backwards-compat)'
    # active gate -> output changes
    with torch.no_grad():
        bb.cross_blocks[0].gate.fill_(1.0); bb.cross_blocks[0].o_proj.weight.normal_()
    assert not torch.allclose(layers[1](h), expected), 'active cross-attn hook changes the layer output'
    print('ok install_hooks_full_attention_layers_no_rename')


def test_install_stride():
    m = _Model(); bb = _MockBackbone()
    idxs = install_slowfast(m, bb, slow_dim=SLOW, every_n=2)     # full-attn [1,3,5] strided by 2 -> [1,5]
    assert idxs == [1, 5], f'stride-2 over full-attn should give [1,5], got {idxs}'
    assert len(bb.cross_blocks) == 2
    print('ok install_stride')


if __name__ == '__main__':
    for fn in [test_identity_at_init, test_shape_and_gqa_and_active, test_masking_ignores_invalid_keys,
               test_no_slow_tokens_row_is_identity, test_install_hooks_full_attention_layers_no_rename,
               test_install_stride]:
        fn()
    print('\nALL CROSS-ATTN TESTS PASSED')
