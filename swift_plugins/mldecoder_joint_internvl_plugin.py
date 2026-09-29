#!/usr/bin/env python
"""Joint training plugin for InternVL3.5 + ML-Decoder head (ASL + AR loss).

Port of mldecoder_joint_plugin.py (Qwen3.5) to InternVL3.5. Same objective:
    total = AR_WEIGHT*autoregressive_CE + ASL_WEIGHT*ASL(cls_logits, 132 tags)
Load with:
    --external_plugins swift_plugins/mldecoder_joint_internvl_plugin.py
    --packing false --padding_free false          # REQUIRED (image->row correspondence)
    --use_liger_kernel true

ARCHITECTURE DIFFERENCE vs Qwen. InternVL has no `.visual` tower with a pre/post-merger pair.
Its vision path is:  pixel_values (total_tiles,3,448,448) -> vision_model (InternViT) ->
drop CLS -> pixel_shuffle(0.5) -> mlp1 -> vit_embeds (total_tiles, 256, llm_hidden=2048).
Each IMAGE is split into a variable number of 448px TILES; `Internvl2Template._encode`
computes `num_patches = [tiles per image]`. The head therefore:
  * taps `model.mlp1`'s output (the post-projector embeds the LLM actually consumes),
  * regroups the flat tile stream into images using num_patches,
  * runs MLDecoder over each image's (n_tiles*256) tokens -> one 132-logit vector per image.
The per-row 132-dim label vector (sidecar) is matched per image, repeated for multi-image rows
exactly as in the Qwen plugin.

Everything else (ASL, label sidecar, head param-group, ASL-weight schedule, eval AUC dump,
JSON-logprob AUC) is identical to the Qwen plugin.
"""
import json
import os
import sys

import torch
import torch.nn as nn

from swift.utils import get_logger

logger = get_logger()

FM_REPO = os.environ.get(
    'FM_ACCELERATE_REPO',
    '/localstorage/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate')
SIDECAR = os.environ.get(
    'CLS_LABEL_SIDECAR',
    '/localstorage/sahil_work/VLM_qure/data_preparation/training_data/cls_label_sidecar')

AR_WEIGHT = float(os.environ.get('AR_WEIGHT', '1.0'))
ASL_WEIGHT_START = float(os.environ.get('ASL_WEIGHT_START', '1.0'))
ASL_WEIGHT_END = float(os.environ.get('ASL_WEIGHT_END', '0.03'))
ASL_DECAY_STEPS = int(os.environ.get('ASL_DECAY_STEPS', '1000'))
ASL_DECAY_SHAPE = os.environ.get('ASL_DECAY_SHAPE', 'cosine')
_ASL_CONST = os.environ.get('ASL_WEIGHT')
ASL_WEIGHT = float(_ASL_CONST) if _ASL_CONST is not None else None

_SCHED = {'step': 0, 'total': None}
_EVAL = {'logits': [], 'labels': [], 'keys': []}
EVAL_PRED_DIR = os.environ.get('MLD_EVAL_PRED_DIR')
_LM = {'score': [], 'label': [], 'tag': [],
       '_saved_score': None, '_saved_label': None, '_saved_tag': None}
_TOK = {'tok': None, 'ontology': set()}
_TF = {'true': None, 'false': None}
_RUN = {'ar': 0.0, 'asl': 0.0, 'n': 0}

DECODER_EMBEDDING = int(os.environ.get('MLD_DECODER_EMBEDDING', '768'))
NUM_OF_GROUPS = int(os.environ.get('MLD_NUM_OF_GROUPS', '-1'))
HEAD_DTYPE = getattr(torch, os.environ.get('MLD_HEAD_DTYPE', 'bfloat16'))
HEAD_LR = float(os.environ.get('MLD_HEAD_LR', '5e-4'))
# cap the tokens fed to the head per image: n_tiles*256 can be large (12 tiles -> 3072). The
# ML-Decoder attends over all of them; a cap keeps memory bounded on many-tile studies.
HEAD_MAX_TOKENS = int(os.environ.get('MLD_HEAD_MAX_TOKENS', '0'))  # 0 = no cap

if FM_REPO not in sys.path:
    sys.path.insert(0, FM_REPO)
try:
    from nnmodule.models.ml_decoder import MLDecoder
except Exception as e:  # pragma: no cover
    raise ImportError(
        f'Could not import MLDecoder from {FM_REPO}. Set FM_ACCELERATE_REPO. Original error: {e}') from e


def current_asl_weight():
    if ASL_WEIGHT is not None:
        return ASL_WEIGHT
    if ASL_DECAY_STEPS <= 0 or ASL_WEIGHT_START == ASL_WEIGHT_END:
        return ASL_WEIGHT_END
    step = _SCHED['step']
    if step >= ASL_DECAY_STEPS:
        return ASL_WEIGHT_END
    t = step / ASL_DECAY_STEPS
    if ASL_DECAY_SHAPE == 'linear':
        f = 1.0 - t
    elif ASL_DECAY_SHAPE == 'exp':
        return ASL_WEIGHT_START * (ASL_WEIGHT_END / ASL_WEIGHT_START) ** t
    else:
        import math as _m
        f = 0.5 * (1.0 + _m.cos(_m.pi * t))
    return ASL_WEIGHT_END + (ASL_WEIGHT_START - ASL_WEIGHT_END) * f


# =========================================================================================
# label sidecar (identical to Qwen plugin)
# =========================================================================================
class LabelTable:

    def __init__(self, sidecar_dir):
        import pandas as pd
        meta = json.load(open(f'{sidecar_dir}/tags.json'))
        self.tags = meta['tags']
        self.num_classes = meta['num_classes']
        self.class_instance_nums = torch.tensor(meta['class_instance_nums'], dtype=torch.float)
        self.total_instance_num = float(meta['total_instance_num'])
        df = pd.read_parquet(f'{sidecar_dir}/labels.parquet')
        self.table = {k: v for k, v in zip(df['key'].tolist(), df['labels'].tolist())}
        logger.info(f'[mldecoder-internvl] label sidecar: {len(self.table):,} keys · '
                    f'{self.num_classes} classes')

    def get(self, path):
        if not path:
            return None
        parts = str(path).strip('/').split('/')
        v = self.table.get(os.path.basename(str(path)))
        if v is None:
            v = self.table.get('/'.join(parts[-3:]))
        return None if v is None else torch.tensor(v, dtype=torch.float)


MLD_DEBUG = os.environ.get('MLD_DEBUG', '0') == '1'
_DBG_SEEN = {}


def _dbg(tag):
    if not MLD_DEBUG:
        return
    n = _DBG_SEEN.get(tag, 0)
    if n < 5:
        _DBG_SEEN[tag] = n + 1
        logger.info(f'[mldecoder-internvl][dbg] {tag}')


_LABELS = None


def get_label_table():
    global _LABELS
    if _LABELS is None:
        _LABELS = LabelTable(SIDECAR)
    return _LABELS


# =========================================================================================
# the head, over InternVL post-projector tile features grouped per image
# =========================================================================================
class VisionMLDecoderHeadInternVL(nn.Module):
    """MLDecoder over each image's post-projector vit_embeds (mlp1 output).

    vit_embeds: (total_tiles, tokens_per_tile, in_features).  num_patches: tiles per image.
    """

    def __init__(self, num_classes, in_features, decoder_embedding=768, num_of_groups=-1):
        super().__init__()
        self.in_features = in_features
        self.ml_decoder = MLDecoder(
            num_classes=num_classes,
            num_of_groups=num_of_groups,
            decoder_embedding=decoder_embedding,
            initial_num_features=in_features,
        )
        logger.info(f'[mldecoder-internvl] head: in_features={in_features} -> {num_classes} classes')

    def forward(self, vit_embeds, num_patches):
        """vit_embeds (total_tiles, T, D) · num_patches list[int] summing to total_tiles."""
        vit_embeds = vit_embeds.to(HEAD_DTYPE)
        T, D = vit_embeds.shape[1], vit_embeds.shape[2]
        seqs = []
        off = 0
        for n in num_patches:
            n = int(n)
            img = vit_embeds[off:off + n].reshape(n * T, D)      # (n_tiles*T, D)
            if HEAD_MAX_TOKENS and img.shape[0] > HEAD_MAX_TOKENS:
                img = img[:HEAD_MAX_TOKENS]
            seqs.append(img)
            off += n
        n_tok = max(s.shape[0] for s in seqs)
        bs = len(seqs)
        x = vit_embeds.new_zeros(bs, n_tok, D)
        mask = torch.ones(bs, n_tok, dtype=torch.bool, device=x.device)  # True == ignore
        for i, s in enumerate(seqs):
            x[i, :s.shape[0]] = s
            mask[i, :s.shape[0]] = False
        return self.ml_decoder(x.to(HEAD_DTYPE), mask=mask, honor_mask=True)


def _unwrap(model):
    # strip DDP (.module) and PEFT/LoRA (PeftModel.base_model(LoraModel).model) to reach the
    # InternVLChatModel that actually owns `.mlp1`.
    m = model.module if hasattr(model, 'module') else model
    if not hasattr(m, 'mlp1') and hasattr(m, 'base_model'):
        bm = m.base_model
        m = getattr(bm, 'model', bm)
    return m


def attach_head(model):
    """Add the head, hook mlp1 for features, fold ASL into the model's loss. Idempotent."""
    if getattr(model, '_mldecoder_attached', False):
        return model
    mlp1 = getattr(model, 'mlp1', None)
    if mlp1 is None:
        raise RuntimeError('model has no `.mlp1` projector; expected InternVLChatModel')
    # llm hidden = mlp1 output dim; robustly read from config
    cfg = model.config
    llm_hidden = getattr(cfg, 'hidden_size', None)
    if llm_hidden is None and hasattr(cfg, 'llm_config'):
        llm_hidden = cfg.llm_config.hidden_size
    tbl = get_label_table()
    device = next(model.parameters()).device
    head = VisionMLDecoderHeadInternVL(
        num_classes=tbl.num_classes,
        in_features=int(llm_hidden),
        decoder_embedding=DECODER_EMBEDDING,
        num_of_groups=NUM_OF_GROUPS,
    ).to(device=device, dtype=HEAD_DTYPE)
    model.ml_decoder_head = head
    model.asl_loss = ASLwithClassWeight(
        tbl.class_instance_nums, tbl.total_instance_num).to(device=device, dtype=HEAD_DTYPE)

    def mlp1_hook(module, args, output):
        # keep the graph so head grads flow into mlp1 + vision_model
        _unwrap(model).__dict__['_cls_vit_embeds'] = output
        return output

    mlp1.register_forward_hook(mlp1_hook)

    state = {'warned': False, 'lm_warned': False}

    def loss_hook(module, args, kwargs, out):
        base = _unwrap(module)
        vit_embeds = base.__dict__.pop('_cls_vit_embeds', None)
        num_patches = base.__dict__.pop('_cls_num_patches', None)
        labels = base.__dict__.pop('_cls_labels', None)
        keys = base.__dict__.pop('_cls_keys', None)
        loss = getattr(out, 'loss', None)
        _dbg(f'losshook vit={vit_embeds is not None} np={num_patches is not None} '
             f'labels={labels is not None} loss={loss is not None}')
        if vit_embeds is None or num_patches is None or labels is None or loss is None:
            if vit_embeds is not None and labels is None and not state['warned']:
                logger.warning('[mldecoder-internvl] vision features but no cls_labels — ASL skipped.')
                state['warned'] = True
            return out
        total_tiles = int(sum(int(n) for n in num_patches))
        if total_tiles != vit_embeds.shape[0]:
            if not state['warned']:
                logger.warning(f'[mldecoder-internvl] tile mismatch: sum(num_patches)={total_tiles} '
                               f'vs vit_embeds rows={vit_embeds.shape[0]} — ASL skipped (packing on?)')
                state['warned'] = True
            return out
        logits = base.ml_decoder_head(vit_embeds, num_patches)
        labels = labels.to(logits.device)
        if labels.shape[0] != logits.shape[0]:
            if not state['warned']:
                logger.warning(f'[mldecoder-internvl] label/logit row mismatch {tuple(labels.shape)} '
                               f'vs {tuple(logits.shape)} — ASL skipped')
                state['warned'] = True
            return out
        if not module.training:
            _EVAL['logits'].append(logits.detach().float().cpu())
            _EVAL['labels'].append(labels.detach().float().cpu())
            n = logits.shape[0]
            _EVAL['keys'].extend(keys if (keys is not None and len(keys) == n) else [None] * n)
            lm_logits = getattr(out, 'logits', None)
            lm_labels = kwargs.get('labels')
            tok = _TOK['tok']
            if lm_logits is not None and lm_labels is not None and tok is not None:
                try:
                    _collect_lm_presence(lm_logits.detach().float(), lm_labels.detach(), tok, _TOK['ontology'])
                except Exception as e:
                    if not state.get('lm_warned'):
                        logger.warning(f'[mldecoder-internvl] LM presence collection failed: {e}')
                        state['lm_warned'] = True
        asl = base.asl_loss(logits, labels)
        w = current_asl_weight()
        ar_t = AR_WEIGHT * loss
        asl_t = w * asl
        if module.training:
            _RUN['ar'] += float(loss.detach()); _RUN['asl'] += float(asl.detach()); _RUN['n'] += 1
        out.loss = ar_t + asl_t
        return out

    model.register_forward_hook(loss_hook, with_kwargs=True)
    model._mldecoder_attached = True
    _w = f'{ASL_WEIGHT}' if ASL_WEIGHT is not None else \
         f'[{ASL_WEIGHT_START}->{ASL_WEIGHT_END} over {ASL_DECAY_STEPS} steps]'
    logger.info(f'[mldecoder-internvl] head attached · hooks registered · loss = {AR_WEIGHT}*ar + {_w}*asl')
    return model


# =========================================================================================
# ASL + AR loss (identical to Qwen plugin)
# =========================================================================================
class ASLwithClassWeight(nn.Module):

    def __init__(self, class_instance_nums, total_instance_num, gamma_neg=8, gamma_pos=1,
                 clip=0.05, eps=1e-8):
        super().__init__()
        p = class_instance_nums / max(total_instance_num, 1.0)
        self.register_buffer('pos_weights', torch.exp(1 - p))
        self.register_buffer('neg_weights', torch.exp(p))
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, pred, label):
        pred = pred.to(HEAD_DTYPE)
        label = label.to(HEAD_DTYPE)
        weight = label * self.pos_weights + (1 - label) * self.neg_weights
        xs_pos = torch.sigmoid(pred)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = label * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - label) * torch.log(xs_neg.clamp(min=self.eps))
        loss = (los_pos + los_neg) * weight
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * label + xs_neg * (1 - label)
            one_sided_gamma = self.gamma_pos * label + self.gamma_neg * (1 - label)
            loss = loss * torch.pow(1 - pt, one_sided_gamma)
        return -loss.sum(dim=-1).mean()


# =========================================================================================
# template: carry cls_labels + num_patches from the sidecar/encode through collate
# =========================================================================================
def patch_template():
    from swift.template.templates.internvl import Internvl2Template

    orig_encode_public = Internvl2Template.encode
    orig_encode = Internvl2Template._encode
    orig_collator = Internvl2Template._data_collator
    orig_post_encode = Internvl2Template._post_encode

    def _path_of(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get('path')
        return getattr(x, 'filename', None)

    def encode(self, inputs, *args, **kwargs):
        """Look labels up from the RAW ROW DICT (paths still strings here)."""
        paths = []
        if isinstance(inputs, dict):
            v = inputs.get('images') or inputs.get('image')
            cand = [v] if not isinstance(v, (list, tuple)) else list(v)
            paths = [q for q in (_path_of(x) for x in cand) if isinstance(q, str)]
        tbl = get_label_table()
        pairs = [(os.path.basename(str(p)), tbl.get(p)) for p in paths]
        vecs = [v for _, v in pairs if v is not None]
        self._pending_cls_labels = torch.stack(vecs) if vecs else None
        self._pending_cls_keys = [k for k, v in pairs if v is not None] if vecs else None
        _dbg(f'encode_row_paths={len(paths)} labels={len(vecs)}')
        return orig_encode_public(self, inputs, *args, **kwargs)

    def _encode(self, inputs):
        encoded = orig_encode(self, inputs)
        # num_patches per image for THIS row, recovered from pixel_values tile count.
        pv = encoded.get('pixel_values')
        pending = getattr(self, '_pending_cls_labels', None)
        if pending is not None:
            encoded['cls_labels'] = pending
            encoded['cls_key'] = getattr(self, '_pending_cls_keys', None)
            # per-image tile counts: recompute exactly as Internvl2Template._encode did
            n_imgs = pending.shape[0]
            if pv is not None and self.num_image_token:
                # tiles-per-image = (#IMG_CONTEXT positions for that image) / num_image_token;
                # simpler: transform_image split counts. We reconstruct from pixel_values total
                # and the row's image count by re-running the tile split is costly; instead trust
                # that pixel_values rows are the SUM of tiles and store per-row total + n_imgs so
                # the collator can only split evenly. To get exact per-image counts we stash the
                # per-image tile list computed here:
                encoded['cls_num_patches'] = getattr(self, '_pending_num_patches', None)
            _dbg('encode_attached')
        self._pending_cls_labels = None
        self._pending_cls_keys = None
        self._pending_num_patches = None
        return encoded

    # capture num_patches at the exact point Internvl2Template._encode computes it
    _orig_inner = Internvl2Template._encode

    def _encode_capture(self, inputs):
        # wrap: call the ORIGINAL _encode but intercept num_patches by monkeypatching torch.cat?
        # Simpler: replicate the tile-count computation from inputs.images here.
        from swift.template.vision_utils import transform_image
        imgs = getattr(inputs, 'images', None)
        if imgs:
            if self.num_image_token is None:
                self.num_image_token = int((self.input_size // 14) ** 2 * (0.5 ** 2))
            try:
                self._pending_num_patches = [transform_image(im, self.input_size, self.max_num).shape[0]
                                             for im in imgs]
            except Exception:
                self._pending_num_patches = None
        else:
            self._pending_num_patches = None
        return _encode(self, inputs)

    def _data_collator(self, batch, *, padding_to=None):
        res = orig_collator(self, batch, padding_to=padding_to)
        vecs = [b['cls_labels'] for b in batch if b.get('cls_labels') is not None]
        if vecs:
            vecs = [v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float) for v in vecs]
            vecs = [v.unsqueeze(0) if v.dim() == 1 else v for v in vecs]
            res['cls_labels'] = torch.cat(vecs, dim=0)
            keys, nps = [], []
            for b in batch:
                if b.get('cls_labels') is not None:
                    k = b.get('cls_key')
                    keys.extend(k if isinstance(k, (list, tuple)) else [k])
                    np_ = b.get('cls_num_patches')
                    if np_ is not None:
                        nps.extend(np_ if isinstance(np_, (list, tuple)) else [np_])
            res['cls_key'] = keys
            res['cls_num_patches'] = nps if nps else None
        _dbg(f'collator cls_rows={len(vecs)} np={res.get("cls_num_patches") is not None}')
        return res

    def _post_encode(self, model, inputs):
        cls_labels = inputs.pop('cls_labels', None)
        cls_key = inputs.pop('cls_key', None)
        cls_np = inputs.pop('cls_num_patches', None)
        base = _unwrap(model)
        if cls_labels is not None:
            base._cls_labels = cls_labels
            base._cls_keys = cls_key
            base._cls_num_patches = cls_np
            _dbg('post_encode_stashed')
        else:
            _dbg('post_encode_missing')
        return orig_post_encode(self, model, inputs)

    Internvl2Template.encode = encode
    Internvl2Template._encode = _encode_capture
    Internvl2Template._data_collator = _data_collator
    Internvl2Template._post_encode = _post_encode
    logger.info('[mldecoder-internvl] Internvl2Template patched (cls_labels + num_patches)')


def _resolve_tf_ids(tok):
    if _TF['true'] is not None:
        return _TF['true'], _TF['false']
    t, f = set(), set()
    for s_, bucket in ((' true', t), ('true', t), (' True', t), ('True', t),
                       (' false', f), ('false', f), (' False', f), ('False', f)):
        ids = tok.encode(s_, add_special_tokens=False)
        if len(ids) == 1:
            bucket.add(ids[0])
    _TF['true'], _TF['false'] = sorted(t), sorted(f)
    logger.info(f'[mldecoder-internvl] presence token ids · true={_TF["true"]} false={_TF["false"]}')
    return _TF['true'], _TF['false']


_NAME_RE = None


def _collect_lm_presence(logits, lm_labels, tok, ontology):
    global _NAME_RE
    if _NAME_RE is None:
        import re as _re
        _NAME_RE = _re.compile(r'"([A-Za-z0-9_]+)"\s*:\s*\{')
    true_ids, false_ids = _resolve_tf_ids(tok)
    tf = set(true_ids) | set(false_ids)
    B, T, _ = logits.shape
    for b in range(B):
        lab = lm_labels[b]
        pos = (lab.unsqueeze(-1) == lab.new_tensor(sorted(tf))).any(-1).nonzero().flatten()
        for t in pos.tolist():
            if t == 0:
                continue
            row = logits[b, t - 1]
            lt = max(float(row[i]) for i in true_ids)
            lf = max(float(row[i]) for i in false_ids)
            win = lab[max(0, t - 48):t]
            win = win[win >= 0]
            if win.numel() == 0:
                continue
            names = _NAME_RE.findall(tok.decode(win.tolist()))
            if not names:
                continue
            tag = names[-1]
            if tag not in ontology:
                continue
            _LM['score'].append(lt - lf)
            _LM['label'].append(1 if int(lab[t]) in set(true_ids) else 0)
            _LM['tag'].append(tag)


def _per_tag_auc(logits, labels, tags):
    from sklearn.metrics import average_precision_score, roc_auc_score
    out, aps, skipped = {}, {}, 0
    for i, tag in enumerate(tags):
        y = labels[:, i]
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == len(y):
            skipped += 1
            continue
        out[tag] = float(roc_auc_score(y, logits[:, i]))
        aps[tag] = float(average_precision_score(y, logits[:, i]))
    return out, aps, skipped


def patch_head_param_group():
    from swift.trainers.mixin import SwiftMixin
    orig = SwiftMixin.create_optimizer

    def create_optimizer(self, model=None):
        opt = orig(self, model=model)
        m = self.model
        base = _unwrap(m)
        head = getattr(base, 'ml_decoder_head', None)
        if opt is None or head is None:
            return opt
        in_opt = {id(p) for g in opt.param_groups for p in g['params']}
        head_params = [p for p in head.parameters() if p.requires_grad and id(p) not in in_opt]
        if head_params:
            n = sum(p.numel() for p in head_params)
            opt.add_param_group({'params': head_params, 'lr': HEAD_LR,
                                 'initial_lr': HEAD_LR, 'weight_decay': 0.0})
            logger.info(f'[mldecoder-internvl] head param group ADDED: {len(head_params)} tensors '
                        f'({n:,} params) at lr={HEAD_LR}. The head now trains.')
        else:
            logger.warning('[mldecoder-internvl] no new head params to add — head LR may be wrong.')
        return opt

    SwiftMixin.create_optimizer = create_optimizer
    logger.info(f'[mldecoder-internvl] create_optimizer patched for head lr={HEAD_LR}')


def patch_step_probe():
    from transformers import TrainerCallback
    from swift.trainers.mixin import SwiftMixin

    class _StepProbe(TrainerCallback):

        def on_train_begin(self, args, state, control, **kwargs):
            _SCHED['total'] = state.max_steps
            tr = getattr(self, '_trainer', None)
            tok = getattr(tr, 'processing_class', None) or getattr(tr, 'tokenizer', None)
            tok = getattr(tok, 'tokenizer', tok)
            _TOK['tok'] = tok
            try:
                _TOK['ontology'] = set(json.load(open(f'{SIDECAR}/ontology_tags.json')))
                logger.info(f'[mldecoder-internvl] JSON-logprob AUC over {len(_TOK["ontology"])} ontology tags')
            except Exception as e:
                _TOK['ontology'] = set()
                logger.warning(f'[mldecoder-internvl] no ontology_tags.json ({e}); JSON-logprob AUC off')

        def on_step_begin(self, args, state, control, **kwargs):
            _SCHED['step'] = state.global_step

        def on_step_end(self, args, state, control, **kwargs):
            every = max(int(getattr(args, 'logging_steps', 1) or 1), 1)
            if _RUN['n'] == 0 or state.global_step % every != 0:
                return
            n = _RUN['n']
            trainer = getattr(self, '_trainer', None)
            payload = {
                'train/ar_loss': _RUN['ar'] / n,
                'train/asl_loss': _RUN['asl'] / n,
                'train/asl_weight': current_asl_weight(),
                'train/asl_weighted': current_asl_weight() * _RUN['asl'] / n,
            }
            _RUN['ar'] = _RUN['asl'] = 0.0
            _RUN['n'] = 0
            if trainer is not None:
                trainer.log(payload)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            import torch as _t
            if not _EVAL['logits']:
                logger.warning('[mldecoder-internvl] eval produced no cls logits — no val AUC')
                return
            logits_t = _t.cat(_EVAL['logits']); labels_t = _t.cat(_EVAL['labels'])
            keys_l = list(_EVAL['keys'])
            _EVAL['logits'].clear(); _EVAL['labels'].clear(); _EVAL['keys'].clear()
            is_rank0 = True
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    is_rank0 = dist.get_rank() == 0
                    parts = [None] * dist.get_world_size()
                    dist.all_gather_object(parts, (logits_t.numpy(), labels_t.numpy(), keys_l))
                    import numpy as _np
                    logits_t = _t.from_numpy(_np.concatenate([p[0] for p in parts]))
                    labels_t = _t.from_numpy(_np.concatenate([p[1] for p in parts]))
                    keys_l = [k for p in parts for k in p[2]]
            except Exception as e:
                logger.warning(f'[mldecoder-internvl] val AUC gather failed ({e}); this rank only')
            logits = logits_t.numpy(); labels = labels_t.numpy()
            tags = get_label_table().tags
            aucs, aps, skipped = _per_tag_auc(logits, labels, tags)
            if not aucs:
                logger.warning('[mldecoder-internvl] no tag had both classes in val — no AUC')
                return
            payload = {f'val_auc/{k}': v for k, v in aucs.items()}
            payload.update({f'val_ap/{k}': v for k, v in aps.items()})
            payload['val/macro_auc'] = sum(aucs.values()) / len(aucs)
            payload['val/macro_ap'] = sum(aps.values()) / len(aps)
            payload['val/n_tags_scored'] = len(aucs)
            payload['val/n_studies'] = int(labels.shape[0])
            if _LM['score']:
                import numpy as _np
                from collections import defaultdict as _dd
                from sklearn.metrics import roc_auc_score as _auc
                by = _dd(lambda: ([], []))
                for sc_, lb_, tg_ in zip(_LM['score'], _LM['label'], _LM['tag']):
                    by[tg_][0].append(sc_); by[tg_][1].append(lb_)
                lm_aucs = {}
                for tg_, (sv, lv) in by.items():
                    lv = _np.asarray(lv)
                    if lv.sum() == 0 or lv.sum() == len(lv):
                        continue
                    lm_aucs[tg_] = float(_auc(lv, _np.asarray(sv)))
                _LM['score'].clear(); _LM['label'].clear(); _LM['tag'].clear()
                if lm_aucs:
                    payload.update({f'val_json_auc/{k}': v for k, v in lm_aucs.items()})
                    payload['val_json/macro_auc'] = sum(lm_aucs.values()) / len(lm_aucs)
            if is_rank0:
                try:
                    import numpy as _np
                    step = int(getattr(state, 'global_step', 0))
                    out_dir = EVAL_PRED_DIR or os.path.join(getattr(args, 'output_dir', '.'), 'eval_predictions')
                    os.makedirs(out_dir, exist_ok=True)
                    _np.savez_compressed(
                        os.path.join(out_dir, f'head_step{step:06d}.npz'),
                        logits=logits.astype('float32'),
                        probs=(1.0 / (1.0 + _np.exp(-logits))).astype('float32'),
                        labels=labels.astype('int8'),
                        keys=_np.array([k if k is not None else '' for k in keys_l], dtype=object),
                        tags=_np.array(tags, dtype=object), step=step)
                    logger.info(f'[mldecoder-internvl] saved eval predictions -> head_step{step:06d}.npz')
                except Exception as e:
                    logger.warning(f'[mldecoder-internvl] eval-prediction dump failed ({e}); continuing')
            trainer = getattr(self, '_trainer', None)
            if trainer is not None:
                trainer.log(payload)
            top = sorted(aucs.items(), key=lambda kv: -kv[1])[:5]
            logger.info(f'[mldecoder-internvl] VAL macro AUC {payload["val/macro_auc"]:.4f} · '
                        f'{len(aucs)} tags · n={labels.shape[0]}')
            logger.info('[mldecoder-internvl]   best: ' + ', '.join(f'{k}={v:.3f}' for k, v in top))

    orig_init = SwiftMixin.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        cb = _StepProbe(); cb._trainer = self
        self.add_callback(cb)

    SwiftMixin.__init__ = __init__
    logger.info('[mldecoder-internvl] step probe callback registered')


def patch_model_prepare():
    from swift.pipelines.train.tuner import TunerMixin
    orig = TunerMixin.prepare_model.__func__

    @classmethod
    def prepare_model(cls, args, model, **kwargs):
        model = orig(cls, args, model, **kwargs)
        attach_head(_unwrap(model))
        return model

    TunerMixin.prepare_model = prepare_model
    logger.info('[mldecoder-internvl] hooked TunerMixin.prepare_model (head lands before optimizer)')


from swift_plugins.qure_plugin_mode import is_inference  # noqa: E402

if is_inference():
    logger.info('[mldecoder-internvl] inference mode — train-only patches NOT installed.')
else:
    patch_template()
    patch_step_probe()
    patch_model_prepare()
    patch_head_param_group()
    logger.info('[mldecoder-internvl] plugin loaded — run with --packing false')
