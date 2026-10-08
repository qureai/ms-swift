"""ML-Decoder multi-label head on the 3D vision-encoder tokens (auxiliary grounding objective).

The generative LM loss alone lets the model predict the report marginal, so the encoder collapses to a
constant, input-independent feature (verified: reldiff -> ~0). A per-volume multi-label classification
head trained with Asymmetric Loss (ASL) cannot be satisfied by a constant feature -> it forces the encoder
to encode each volume's findings.

MLDecoder (query-based transformer-decoder head) is vendored/trimmed from
vision_foundation_2d/.../nnmodule/models/ml_decoder.py (non-ZSL path only). It consumes the encoder tokens
directly (initial_num_features == encoder out dim), so no shape adapter is needed.
"""
from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_activation_fn


@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            w_i = duplicate_pooling[i, :, :]
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class TransformerDecoderLayerOptimal(nn.Module):
    """ML-Decoder's decoder layer (query self-attn dropped for speed). Fixed to honor
    `memory_key_padding_mask` so padded tokens from variable-length volumes are ignored."""

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt: Tensor, memory: Tensor, memory_key_padding_mask: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        tgt = self.norm1(tgt + self.dropout1(tgt))
        tgt2 = self.multihead_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class MLDecoder(nn.Module):
    def __init__(self, num_classes, initial_num_features, num_of_groups=-1, decoder_embedding=768):
        super().__init__()
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        embed_len_decoder = min(embed_len_decoder, num_classes)
        self.embed_standart = nn.Linear(initial_num_features, decoder_embedding)
        query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
        query_embed.requires_grad_(False)  # non-learnable queries (as in the original)
        self.query_embed = query_embed
        self.decoder = TransformerDecoderLayerOptimal(d_model=decoder_embedding, dim_feedforward=2048)
        self.num_classes = num_classes
        self.duplicate_factor = (num_classes + embed_len_decoder - 1) // embed_len_decoder
        self.duplicate_pooling = nn.Parameter(
            torch.empty(embed_len_decoder, decoder_embedding, self.duplicate_factor))
        self.duplicate_pooling_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_normal_(self.duplicate_pooling)
        self.group_fc = GroupFC(embed_len_decoder)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # x: (B, N, initial_num_features); key_padding_mask: (B, N) bool, True == pad (ignored)
        emb = torch.nn.functional.relu(self.embed_standart(x))            # (B, N, D)
        tgt = self.query_embed.weight.unsqueeze(1).expand(-1, emb.shape[0], -1)  # (Q, B, D)
        h = self.decoder(tgt, emb.transpose(0, 1), memory_key_padding_mask=key_padding_mask)  # (Q, B, D)
        h = h.transpose(0, 1)                                             # (B, Q, D)
        out = h.new_zeros(h.shape[0], h.shape[1], self.duplicate_factor)
        self.group_fc(h, self.duplicate_pooling, out)
        logits = out.flatten(1)[:, :self.num_classes] + self.duplicate_pooling_bias
        return logits                                                    # (B, num_classes)


class AsymmetricLoss(nn.Module):
    """Canonical multi-label ASL (Ben-Baruch et al.), self-contained. logits/targets: (B, C) float."""

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        logits, targets = logits.float(), targets.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss = loss * torch.pow(1 - pt, gamma)
        return -loss.sum(dim=-1).mean()


class Vision3DMLDecoderHead(nn.Module):
    """Auxiliary multi-label head over the 3D encoder tokens. `enc_dim` MUST match the encoder's output
    feature dim (out_hidden_size). Consumes a LIST of per-volume token tensors (variable token counts)."""

    def __init__(self, enc_dim: int, num_classes: int, num_of_groups: int = -1, decoder_embedding: int = 768,
                 gamma_neg: float = 4, gamma_pos: float = 1, clip: float = 0.05):
        super().__init__()
        self.decoder = MLDecoder(num_classes, initial_num_features=enc_dim,
                                 num_of_groups=num_of_groups, decoder_embedding=decoder_embedding)
        self.loss_fn = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
        self.num_classes = num_classes

    def forward(self, feats: List[Tensor]) -> Tensor:
        """feats: list of (M_i, enc_dim) per-volume encoder tokens -> logits (n_vol, num_classes)."""
        dtype, device = self.decoder.embed_standart.weight.dtype, self.decoder.embed_standart.weight.device
        lengths = [f.shape[0] for f in feats]
        max_m = max(lengths)
        x = feats[0].new_zeros(len(feats), max_m, feats[0].shape[-1])
        mask = torch.ones(len(feats), max_m, dtype=torch.bool, device=feats[0].device)
        for i, f in enumerate(feats):
            x[i, :lengths[i]] = f
            mask[i, :lengths[i]] = False  # False == real token
        return self.decoder(x.to(device=device, dtype=dtype), key_padding_mask=mask.to(device))

    def loss(self, feats: List[Tensor], targets: Tensor) -> Tensor:
        """targets: (n_vol, num_classes) multi-hot -> scalar ASL loss."""
        logits = self.forward(feats)
        return self.loss_fn(logits, targets.to(logits.device))
