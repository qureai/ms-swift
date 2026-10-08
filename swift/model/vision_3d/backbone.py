# Copyright (c) ModelScope Contributors. All rights reserved.
"""`Vision3DBackbone`: wraps an extracted/converted vision tower + a projector into the LLM subspace.

The wrapper exposes a uniform contract regardless of the underlying encoder:
  * `forward(pixel_values_volumes, grid=None) -> (n_vol * max_tokens, llm_hidden_size)` token features;
  * `num_tokens(grid)` so the template can expand exactly the right number of <video> placeholders.

An optional `adapter` (callable) handles encoders whose `forward` is non-standard (e.g. Atlas/Pillar0,
which takes a `{modality: tensor}` dict and pools to a global vector — the adapter taps the pre-pool
token features instead). When `adapter is None`, a best-effort generic path is used.
"""
from typing import Callable, Optional

import torch
import torch.nn as nn


def pool_tokens_to(feats: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Adaptive-average-pool a (M, dim) token sequence to exactly `n_tokens` tokens (no-op if already M)."""
    if feats.shape[0] == n_tokens:
        return feats
    pooled = torch.nn.functional.adaptive_avg_pool1d(
        feats.transpose(0, 1).unsqueeze(0).float(), n_tokens).squeeze(0).transpose(0, 1)
    return pooled.to(feats.dtype)


class Vision3DBackbone(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 *,
                 encoder_dim: int,
                 llm_hidden_size: int,
                 max_tokens: Optional[int] = None,
                 adapter: Optional[Callable] = None,
                 resampler: Optional[nn.Module] = None,
                 variable_tokens: bool = False,
                 slowfast: bool = False,
                 fast_grid: tuple = (4, 8, 8),
                 fast_pool: str = 'avgmax',
                 slow_max_tokens: int = 4096):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.llm_hidden_size = llm_hidden_size
        self.max_tokens = max_tokens
        # variable_tokens: emit each volume's NATIVE (variable) token count (<= max_tokens) instead of a
        # fixed budget -> no pooling, no resampler; the projector maps each token to llm space.
        self.variable_tokens = variable_tokens
        self.adapter = adapter
        # Token aggregator + projector into LLM space. Either:
        #  * `resampler` (Perceiver): learned latent queries cross-attend to the encoder tokens and map
        #    encoder_dim -> llm_hidden_size, emitting exactly max_tokens tokens (replaces proj + mean-pool); or
        #  * `proj` (default): a 2-layer MLP applied after a blind mean-pool to max_tokens.
        self.resampler = resampler
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        # --- Slow-Fast (opt-in). FAST tokens: 3D avg+max pool -> proj_fast into <video>. SLOW tokens: full
        # encoder tokens stashed per-forward for the LLM cross-attention (wired in a later step). Everything
        # here is None/off unless `slowfast` is True, so the default forward path is byte-identical. ---
        self.slowfast = bool(slowfast)
        self.fast_grid = tuple(int(x) for x in fast_grid)
        self.fast_pool = str(fast_pool)
        self.slow_max_tokens = int(slow_max_tokens)
        self.proj_fast = None
        if self.slowfast:
            from .slowfast import build_proj_fast, fast_out_dim
            self.proj_fast = build_proj_fast(fast_out_dim(encoder_dim, self.fast_pool), llm_hidden_size)
        self._slow_tokens = None  # list[(M_i, enc_dim)] stashed per forward when slowfast + training/infer
        # Pre-projector tokens stashed per training forward for the VICReg rank-reg loss (the trainer reads
        # this iff --vision_3d_rankreg_lambda>0). They are already in the autograd graph, so this is free.
        self._rankreg_feats = None
        # Auxiliary ML-Decoder grounding head (set by attach_vision_3d_encoder when lambda>0). During
        # training the variable-path forward stashes the PRE-projector per-volume tokens in `_aux_feats`
        # for the template to run the head + ASL loss on. Defaults keep the head fully off/no-op.
        self.aux_head = None
        self.aux_lambda = 0.0
        self.aux_table = None
        self._aux_feats = None

    @property
    def dtype(self) -> torch.dtype:
        return self.proj[0].weight.dtype

    def _to_tokens(self, feats: torch.Tensor) -> torch.Tensor:
        """Coerce an encoder output into a (tokens, encoder_dim) sequence."""
        if feats.dim() == 5:  # (B, C, D, H, W) feature map -> (B*D*H*W, C)
            feats = feats.permute(0, 2, 3, 4, 1).reshape(-1, feats.shape[1])
        elif feats.dim() == 3:  # (B, N, C) -> (B*N, C)
            feats = feats.reshape(-1, feats.shape[-1])
        elif feats.dim() != 2:  # 2 -> already (N, C)
            raise ValueError(f'vision_3d: unexpected encoder output rank {feats.dim()} (shape {tuple(feats.shape)}).')
        return feats

    def _encode_one(self, volume: torch.Tensor) -> torch.Tensor:
        """Run the encoder on a single (T, C, H, W) volume -> (M, encoder_dim) tokens (generic path)."""
        x = volume.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, D, H, W)
        feats = self.encoder(x.to(next(self.encoder.parameters()).dtype))
        if hasattr(feats, 'last_hidden_state'):
            feats = feats.last_hidden_state
        return self._to_tokens(feats)

    def forward(self, pixel_values_volumes: torch.Tensor, grid=None) -> torch.Tensor:
        """Map a batch of volumes (n_vol, T, C, H, W) to (n_vol * max_tokens, llm_hidden_size)."""
        if self.max_tokens is None:
            raise ValueError('Vision3DBackbone requires max_tokens (== vision_3d_max_tokens) to be set for volumes.')
        vols = pixel_values_volumes
        if torch.is_tensor(vols) and vols.dim() == 4:  # single volume tensor -> add volume-batch axis
            vols = vols.unsqueeze(0)
        # (variable mode passes a LIST of (T,C,H,W) tensors with differing sizes -> iterate as-is)

        if self.slowfast:
            return self._slowfast_forward(vols, grid)

        if self.variable_tokens:
            # Variable-length: each volume keeps its native (variable) token count. No pooling, no
            # resampler -- just project each encoder token into LLM space and concat across volumes.
            if self.resampler is not None:
                raise ValueError('Vision3DBackbone: variable_tokens is incompatible with a resampler '
                                 '(a resampler emits a fixed number of tokens).')
            if self.adapter is not None:
                feats = self.adapter(self.encoder, vols, self.max_tokens, grid, variable=True)  # (sum M_i, enc_dim)
            else:
                feats = torch.cat([self._encode_one(v) for v in vols], dim=0)
            # Stash pre-projector per-volume tokens for the aux ML-Decoder head (training only). Split the
            # concatenated (sum M_i, enc_dim) by each volume's merged token count (derived from its shape
            # via the adapter's patch geometry -- matches the adapter's own per-volume split).
            if self.aux_head is not None and self.training and self.adapter is not None:
                ad = self.adapter
                counts = [(v.shape[0] // ad.temporal_patch_size) * (v.shape[2] // ad.patch_size)
                          * (v.shape[3] // ad.patch_size) // (ad.merge_size ** 2) for v in vols]
                self._aux_feats = list(torch.split(feats, counts, dim=0))
            else:
                self._aux_feats = None
            if self.training:
                self._rankreg_feats = feats             # pre-proj tokens for the VICReg rank-reg loss
            return self.proj(feats.to(self.dtype))  # (sum M_i, llm_hidden_size)

        if self.resampler is not None:
            # Learned path: get the encoder's FULL (unpooled) tokens per volume, then let the resampler
            # cross-attend max_tokens latent queries over them and map to llm_hidden_size (no mean-pool, no proj).
            if self.adapter is not None:
                feats = self.adapter(self.encoder, vols, self.max_tokens, grid, pool=False)  # (n_vol, M, enc_dim)
            else:
                feats = torch.stack([self._encode_one(v) for v in vols], dim=0)  # (n_vol, M, enc_dim)
            latents = self.resampler(feats)  # (n_vol, max_tokens, llm_hidden_size)
            return latents.reshape(-1, latents.shape[-1])  # (n_vol * max_tokens, llm_hidden_size)

        if self.adapter is not None:
            # adapter handles the encoder's native input/forward and returns (n_vol * max_tokens, encoder_dim)
            feats = self.adapter(self.encoder, vols, self.max_tokens, grid)
        else:
            feats = torch.cat([pool_tokens_to(self._encode_one(v), self.max_tokens) for v in vols], dim=0)
        return self.proj(feats.to(self.dtype))

    def _slowfast_forward(self, vols, grid) -> torch.Tensor:
        """Slow-Fast path. Produce FAST tokens (3D avg+max pool of each volume's merged tokens -> proj_fast,
        a fixed `fast_count` per volume, spliced into <video>) and stash the full per-volume SLOW tokens for
        the LLM cross-attention (consumed in a later step). Qwen-VL adapter only (needs its patch geometry)."""
        from .slowfast import fast_pool_3d
        ad = self.adapter
        if ad is None or not all(hasattr(ad, a) for a in ('temporal_patch_size', 'patch_size', 'merge_size')):
            raise ValueError('vision_3d slowfast requires a Qwen-VL adapter (patch geometry); got '
                             f'{type(ad).__name__ if ad else None}.')
        tps, ps, m = ad.temporal_patch_size, ad.patch_size, ad.merge_size
        # full (unpooled) merged tokens per volume, concatenated in volume order, then split per volume.
        feats = ad(self.encoder, vols, self.max_tokens, grid, variable=True)  # (sum_i M_i, enc_dim)
        merged, grids = [], []
        for v in vols:
            gt, gh, gw = v.shape[0] // tps, v.shape[2] // ps, v.shape[3] // ps
            gh_m, gw_m = gh // m, gw // m
            merged.append(gt * gh_m * gw_m)
            grids.append((gt, gh_m, gw_m))              # gt-major merged grid (matches qwen_vl merger layout)
        if sum(merged) != feats.shape[0]:
            raise ValueError(f'vision_3d slowfast: merged token counts {sum(merged)} != adapter output '
                             f'{feats.shape[0]} (grid/merge mismatch).')
        per_vol = list(torch.split(feats, merged, dim=0))
        fast_out, slow_out = [], []
        for tok, g in zip(per_vol, grids):
            fast = fast_pool_3d(tok, g, self.fast_grid, self.fast_pool)          # (fast_count, C or 2C)
            fast_out.append(self.proj_fast(fast.to(self.dtype)))                  # (fast_count, llm_hidden)
            slow_out.append(tok)                                                 # full tokens (cap at conditioning)
        # Stash slow tokens (pre-proj, enc_dim) for the cross-attention conditioning step; kept in-graph.
        self._slow_tokens = slow_out
        self._rankreg_feats = torch.cat(slow_out, dim=0) if self.training else None   # PRE-proj (encoder) tokens
        fast = torch.cat(fast_out, dim=0)               # (n_vol * fast_count, llm_hidden)
        # POST-proj FAST tokens, for the fast-channel anti-collapse rank-reg (the fast <video> tokens were
        # observed to collapse to rank ~1 while the pre-proj encoder stayed high-rank; see TRAINING/SLOWFAST md).
        self._rankreg_fast_feats = fast if self.training else None
        return fast

    def num_fast_tokens(self) -> int:
        """Fixed fast-token count per volume when slow-fast is on (== Tf*Hf*Wf)."""
        return int(self.fast_grid[0] * self.fast_grid[1] * self.fast_grid[2])

    def num_tokens(self, grid=None) -> Optional[int]:
        """Tokens emitted per volume (== the pooling budget; fast_count when slow-fast is on)."""
        if self.slowfast:
            return self.num_fast_tokens()
        return self.max_tokens
