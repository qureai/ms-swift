# Copyright (c) ModelScope Contributors. All rights reserved.
"""Slow-Fast token production for the CT 3D VLM (opt-in; nothing here runs unless --vision_3d_slowfast).

FAST branch: 3D adaptive avg+max pool of a volume's post-encoder tokens down to a FIXED small grid
(Tf,Hf,Wf) -> a bounded `fast_count = Tf*Hf*Wf` tokens regardless of volume size. These are spliced into
the <video> positions (short sequence -> the LLM-sequence-length win). Pooling spans z (slices) too --
CT slices are not redundant, so z is POOLED not subsampled; the max-over-window carries a single-slice
finding's peak up into the overview that avg alone would dilute.

SLOW branch: keep the full (optionally capped) encoder tokens for the cross-attention K/V (built later).

This file holds pure-tensor helpers (unit-testable on CPU with no model) + the projector factory. The
cross-attention hybrid decoder layer is added in a later step.
"""
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def fast_pool_3d(tokens: torch.Tensor,
                 grid_thw: Sequence[int],
                 target: Tuple[int, int, int],
                 mode: str = 'avgmax') -> torch.Tensor:
    """3D adaptive pool a single volume's tokens to a fixed (Tf,Hf,Wf) grid.

    Args:
        tokens:   (M, C) post-encoder tokens for ONE volume, in gt-major order (gt outer, then gh, then gw),
                  where M == gt*gh*gw.
        grid_thw: (gt, gh, gw) merged-token grid for this volume (M == gt*gh*gw).
        target:   (Tf, Hf, Wf) fixed output grid. Output token count is ALWAYS Tf*Hf*Wf, independent of the
                  input size (adaptive pooling handles both down- and up-sampling), so the fast token count
                  is bounded and constant across variable-resolution volumes.
        mode:     'avg' | 'max' | 'avgmax'. 'avgmax' concatenates avg and max along the feature dim -> 2C.

    Returns:
        (Tf*Hf*Wf, C)   for 'avg'/'max'
        (Tf*Hf*Wf, 2C)  for 'avgmax'  (avg features first, then max features)
    """
    gt, gh, gw = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    M, C = tokens.shape
    if gt * gh * gw != M:
        raise ValueError(f'fast_pool_3d: grid_thw {(gt, gh, gw)} (prod={gt*gh*gw}) != tokens M={M}')
    Tf, Hf, Wf = int(target[0]), int(target[1]), int(target[2])
    if min(Tf, Hf, Wf) < 1:
        raise ValueError(f'fast_pool_3d: target grid must be >=1 in every axis, got {target}')
    # (M, C) gt-major -> (gt, gh, gw, C) -> (1, C, gt, gh, gw) for pool3d
    x = tokens.reshape(gt, gh, gw, C).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, gt, gh, gw)
    out = []
    if mode in ('avg', 'avgmax'):
        out.append(F.adaptive_avg_pool3d(x, (Tf, Hf, Wf)))
    if mode in ('max', 'avgmax'):
        out.append(F.adaptive_max_pool3d(x, (Tf, Hf, Wf)))
    if not out:
        raise ValueError(f"fast_pool_3d: mode must be 'avg'|'max'|'avgmax', got {mode!r}")
    pooled = torch.cat(out, dim=1)                     # (1, C or 2C, Tf, Hf, Wf)
    Cout = pooled.shape[1]
    # (1, Cout, Tf, Hf, Wf) -> (Tf, Hf, Wf, Cout) -> (Tf*Hf*Wf, Cout), same gt-major token order
    return pooled.squeeze(0).permute(1, 2, 3, 0).reshape(Tf * Hf * Wf, Cout)


def fast_out_dim(encoder_dim: int, mode: str) -> int:
    """Input feature dim the fast projector must accept for a given pool mode."""
    return encoder_dim * 2 if mode == 'avgmax' else encoder_dim


def ortho_reg(W: torch.Tensor) -> torch.Tensor:
    """Soft orthogonality penalty `||G - I||_F^2 / d^2` on a weight `W (out, in)`, where G = W Wᵀ (if out<=in)
    or WᵀW (if out>in) -- keeps the projection FULL-RANK in weight space (so it can't collapse the rich
    encoder features to a low-rank output) and bounds its singular values (curbs the norm blow-up). Scale-free
    (divided by d^2) so a fixed lambda is meaningful across dims."""
    W = W.float()
    o, i = W.shape[0], W.shape[1]
    G = (W @ W.t()) if o <= i else (W.t() @ W)
    d = G.shape[0]
    eye = torch.eye(d, device=W.device, dtype=W.dtype)
    return ((G - eye) ** 2).sum() / (d * d)


def build_proj_fast(in_dim: int, llm_hidden_size: int) -> nn.Sequential:
    """Fast-token projector (mirrors the base `proj` MLP shape) mapping pooled features -> LLM space."""
    return nn.Sequential(
        nn.Linear(in_dim, llm_hidden_size),
        nn.GELU(),
        nn.Linear(llm_hidden_size, llm_hidden_size),
    )


# ============================ SLOW branch: LLM cross-attention ============================
import math  # noqa: E402


def _find_decoder_layers(model):
    """Return the LLM decoder-layer ModuleList on a (possibly VLM) model, trying the known paths."""
    for path in ('model.language_model.layers', 'language_model.layers', 'model.model.layers', 'model.layers'):
        obj = model
        try:
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    if hasattr(model, 'get_decoder'):
        dec = model.get_decoder()
        if hasattr(dec, 'layers'):
            return dec.layers
    raise ValueError('slowfast: could not locate the LLM decoder layers ModuleList on the model.')


class LLMCrossAttnBlock(nn.Module):
    """Gated cross-attention: text hidden states (queries) attend to the volume SLOW tokens (K/V).

    Identity at init (the gate is zero-initialised -> tanh(0)=0), so inserting it is backwards-compatible:
    the augmented decoder layer reproduces the original layer bit-for-bit until trained. GQA-aware
    (num_kv_heads may be < num_heads). Fully-masked query rows (a sample with no slow tokens) contribute 0.
    """

    def __init__(self, hidden_size: int, slow_dim: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 eps: float = 1e-6):
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(slow_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(slow_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))          # zero-init => identity at start

    def forward(self, hidden_states: torch.Tensor, slow_kv: torch.Tensor,
                slow_mask: torch.Tensor = None) -> torch.Tensor:
        """hidden_states (B,L,H); slow_kv (B,T,slow_dim); slow_mask (B,T) bool (True=valid). -> (B,L,H)."""
        B, L, H = hidden_states.shape
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        x = self.norm(hidden_states)
        q = self.q_proj(x).view(B, L, nh, hd).transpose(1, 2)                 # (B, nh, L, hd)
        k = self.k_proj(slow_kv).view(B, -1, nkv, hd).transpose(1, 2)         # (B, nkv, T, hd)
        v = self.v_proj(slow_kv).view(B, -1, nkv, hd).transpose(1, 2)
        if nkv != nh:                                                        # GQA: expand kv heads
            rep = nh // nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # Flash/mem-efficient attention via SDPA (dispatches to the FlashAttention-2 / mem-efficient kernel on
        # CUDA, math fallback on CPU) -- replaces the eager matmul+softmax, which materialised the full
        # (B,nh,L,T) score matrix. SDPA's default scale is 1/sqrt(head_dim), matching the old `/sqrt(hd)`.
        attn_mask = None
        valid_scale = None
        if slow_mask is not None:
            m = slow_mask.bool().view(B, 1, 1, -1)                           # (B,1,1,T) True = valid key
            any_valid = m.any(dim=-1, keepdim=True)                          # (B,1,1,1)
            # A sample with NO valid slow token would be an all -inf row -> SDPA returns NaN (and NaN grads).
            # Let those rows attend to ALL keys (well-defined softmax), then zero that sample's output below,
            # reproducing the old `nan_to_num -> 0` "no slow tokens contribute nothing" semantics -- and keeping
            # the cross-attn params in the graph with finite (0) grad for the distributed all-masked dummy.
            attn_mask = m | (~any_valid)                                     # (B,1,1,T) bool
            valid_scale = any_valid.to(q.dtype)                              # (B,1,1,1): 1 if sample has slow tokens else 0
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, nh, L, hd)
        if valid_scale is not None:
            o = o * valid_scale                                              # samples with no slow tokens -> 0 contribution
        o = o.transpose(1, 2).reshape(B, L, nh * hd)                         # (B, L, nh*hd)
        return hidden_states + torch.tanh(self.gate) * self.o_proj(o)


def install_slowfast(model, backbone, *, slow_dim: int, every_n: int = 4, layer_idxs=None, num_heads=None,
                     num_kv_heads=None, head_dim=None):
    """Install gated cross-attention over the volume SLOW tokens onto selected decoder layers via forward
    HOOKS (NOT by wrapping the layer). The `LLMCrossAttnBlock` modules are stored on the backbone
    (`visual_3d.cross_blocks.*`), so:
      * the decoder layers' own params are NEVER renamed  -> resume loads them normally;
      * the cross-attn weights save under `visual_3d.*`   -> the existing `_load_trained_vision_3d_weights`
        overlay restores them on resume, and swift's `--vit_lr`/vision_tower group already targets them.
    The hook reads the padded slow K/V + mask that the template stashes on the backbone per forward
    (`backbone._slow_kv_padded` / `_slow_mask_padded`); when unset it's a pure no-op (identity).
    Default: insert on the FULL-attention layers at stride `every_n`. Returns the chosen layer indices."""
    layers = _find_decoder_layers(model)
    cfg = getattr(model, 'config', None)
    tcfg = getattr(cfg, 'text_config', cfg)                                  # VLM: text sub-config
    hidden = int(getattr(tcfg, 'hidden_size'))
    nh = int(num_heads or getattr(tcfg, 'num_attention_heads'))
    nkv = int(num_kv_heads or getattr(tcfg, 'num_key_value_heads', nh))
    hd = int(head_dim or getattr(tcfg, 'head_dim', hidden // nh))
    if layer_idxs is None:
        layer_types = list(getattr(tcfg, 'layer_types', ['full_attention'] * len(layers)))
        full = [i for i, t in enumerate(layer_types) if t == 'full_attention']
        layer_idxs = full[::max(1, int(every_n))] if full else list(range(0, len(layers), max(1, int(every_n))))
    p = next(model.parameters())
    cross_blocks = nn.ModuleList()
    handles = []
    backbone._slow_kv_padded = None
    backbone._slow_mask_padded = None

    def _make_hook(block):
        def hook(module, args, output):
            kv = getattr(backbone, '_slow_kv_padded', None)
            if kv is None:                                                   # no volumes / off -> identity
                return output
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            h = block(h, kv, getattr(backbone, '_slow_mask_padded', None))
            return (h, *output[1:]) if is_tuple else h
        return hook

    for i in layer_idxs:
        cb = LLMCrossAttnBlock(hidden, slow_dim, nh, nkv, hd).to(device=p.device, dtype=p.dtype)
        cross_blocks.append(cb)
        handles.append(layers[i].register_forward_hook(_make_hook(cb)))
    backbone.cross_blocks = cross_blocks                                     # saved under visual_3d.cross_blocks.*
    backbone._slowfast_layer_idxs = list(layer_idxs)
    backbone._slowfast_handles = handles
    return list(layer_idxs)


# ============================ VICReg rank-reg (anti-collapse) ============================
def rankreg_loss(Z: torch.Tensor, gamma: float = 1.0, var_w: float = 1.0, cov_w: float = 1.0,
                 max_tokens: int = 2048, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style variance+covariance loss on pre-proj tokens `Z (N, D)`, reduced over the TOKEN axis
    (our failure is token uniformity). Variance term floors each dim's std at `gamma` (a constant/collapsed
    representation is maximally penalised); covariance term decorrelates dims -> raises effective rank
    (the operative term per the diagnostic). Covariance is computed on a token subsample for cost."""
    Z = Z.float()
    if Z.dim() != 2 or Z.shape[0] < 2:
        return Z.new_zeros(())
    std = torch.sqrt(Z.var(dim=0) + eps)                       # (D,)
    var_term = torch.relu(gamma - std).mean()
    n = Z.shape[0]
    if max_tokens and n > max_tokens:                          # subsample tokens for the O(n*D^2) covariance
        idx = torch.randperm(n, device=Z.device)[:max_tokens]
        Zc = Z[idx]
    else:
        Zc = Z
    m = Zc.shape[0]
    Zc = Zc - Zc.mean(dim=0, keepdim=True)
    # Standardise per-dim so the off-diagonals are CORRELATIONS (scale-free, in [-1,1]); otherwise the cov
    # term's magnitude tracks the (uncontrolled bf16) feature scale and lambda becomes meaningless. The
    # variance/scale is handled separately by var_term above.
    Zc = Zc / Zc.std(dim=0, keepdim=True).clamp(min=1e-4)
    corr = (Zc.T @ Zc) / (m - 1)                               # (D, D), ~correlation matrix (diag ~1)
    D = corr.shape[0]
    # MEAN squared off-diagonal correlation in [0,1] (divide by the off-diagonal COUNT, not D) -> the term
    # does not scale with D, so a fixed lambda is meaningful across models: 0 = decorrelated, ->1 = collapsed.
    cov_term = (corr.pow(2).sum() - corr.diagonal().pow(2).sum()) / (D * (D - 1))
    return var_w * var_term + cov_w * cov_term


@torch.no_grad()
def collapse_metrics(Z: torch.Tensor, max_tokens: int = 2048):
    """Zero-extra-forward collapse detector: centered effective rank (participation ratio) + top
    singular-value fraction of pre-proj tokens `Z (N, D)` the forward already produced. Returns
    (eff_rank, sv0) or None. Tracks the 12->1 rank collapse Phase-0 measured."""
    Z = Z.detach().float()
    if Z.dim() != 2 or Z.shape[0] < 2:
        return None
    n = Z.shape[0]
    if max_tokens and n > max_tokens:
        Z = Z[torch.randperm(n, device=Z.device)[:max_tokens]]
    Zc = Z - Z.mean(0)
    try:
        sv = torch.linalg.svdvals(Zc)
    except Exception:
        return None
    en = (sv ** 2)
    en = en / en.sum().clamp(min=1e-9)
    return float(1.0 / (en ** 2).sum()), float(en[0])
