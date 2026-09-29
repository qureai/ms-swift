#!/usr/bin/env python
"""Attach the trained ML-Decoder head for INFERENCE and dump its per-image logits.

    swift infer --external_plugins swift_plugins/mldecoder_infer_plugin.py \
                --infer_backend pt ...

WHY A SEPARATE PLUGIN FROM THE TRAINING ONE
mldecoder_joint_plugin.py attaches the head from `TunerMixin.prepare_model`, which only runs
on the TRAINING path. Loading a checkpoint with just that plugin under `swift infer` leaves the
head unattached, so the 25 `ml_decoder_head.*` tensors in the checkpoint are treated as
unexpected keys and dropped SILENTLY -- you would get a randomly-initialized head and no error.
This plugin hooks the inference engine instead.

BACKEND: pt only. vLLM and sglang reimplement the model, so a forward hook on `model.visual`
never fires there. Generation of the JSON text works on any backend; reading the HEAD requires
`--infer_backend pt`.

Writes `<result_path>.cls_logits.jsonl`, one row per image:
    {"image": "...", "logits": [132 floats], "probs": [132 floats], "top": [[tag, prob], ...]}
Set MLD_CLS_OUT to override the path.
"""
import json
import os
import sys

import torch

from swift.utils import get_logger

logger = get_logger()

FM_REPO = os.environ.get(
    'FM_ACCELERATE_REPO',
    '/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate')
SIDECAR = os.environ.get(
    'CLS_LABEL_SIDECAR',
    '/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/cls_label_sidecar')
OUT_PATH = os.environ.get('MLD_CLS_OUT')
TOP_K = int(os.environ.get('MLD_TOP_K', '15'))
HEAD_DTYPE = getattr(torch, os.environ.get('MLD_HEAD_DTYPE', 'bfloat16'))

if FM_REPO not in sys.path:
    sys.path.insert(0, FM_REPO)

# reuse the exact head definition used in training -- a re-implementation here could silently
# diverge in shape or token grouping and load the checkpoint weights into the wrong slots
from swift_plugins.mldecoder_joint_plugin import VisionMLDecoderHead  # noqa: E402

_TAGS = json.load(open(f'{SIDECAR}/tags.json'))['tags']
_STATE = {'fh': None, 'n': 0}


def _open_out():
    if _STATE['fh'] is None:
        path = OUT_PATH or 'cls_logits.jsonl'
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        _STATE['fh'] = open(path, 'w')
        logger.info(f'[mldecoder-infer] writing head logits -> {path}')
    return _STATE['fh']


def attach_for_inference(model):
    """Build the head, load its checkpoint weights, and hook the vision tower."""
    if getattr(model, '_mldecoder_infer_attached', False):
        return
    visual = getattr(model, 'visual', None)
    if visual is None:
        logger.warning('[mldecoder-infer] model has no .visual — head not attached')
        return
    vcfg = model.config.vision_config
    dev = next(model.parameters()).device
    head = VisionMLDecoderHead(
        num_classes=len(_TAGS), vis_hidden=vcfg.hidden_size, out_hidden=vcfg.out_hidden_size,
        spatial_merge_size=vcfg.spatial_merge_size).to(device=dev, dtype=HEAD_DTYPE)

    # The head must be attached BEFORE weights are loaded for from_pretrained to fill it. By the
    # time the engine hands us the model that has already happened, so pull the tensors out of
    # the checkpoint explicitly and report exactly how many matched -- a silent 0/23 here is the
    # failure mode this plugin exists to prevent.
    sd, src = {}, getattr(model, 'model_dir', None) or getattr(model, 'name_or_path', None)
    if src and os.path.isdir(src):
        from safetensors.torch import load_file
        import glob
        for f in sorted(glob.glob(os.path.join(src, '*.safetensors'))):
            for k, v in load_file(f).items():
                if k.startswith('ml_decoder_head.'):
                    sd[k[len('ml_decoder_head.'):]] = v
    if sd:
        missing, unexpected = head.load_state_dict(sd, strict=False)
        logger.info(f'[mldecoder-infer] loaded {len(sd)} head tensors from {src} '
                    f'(missing={len(missing)}, unexpected={len(unexpected)})')
        if missing:
            logger.warning(f'[mldecoder-infer] MISSING head tensors: {list(missing)[:5]} — '
                           'these stay randomly initialized')
    else:
        logger.error(f'[mldecoder-infer] NO ml_decoder_head.* tensors found in {src}. The head '
                     'is RANDOM and its outputs are meaningless. Check that the checkpoint was '
                     'trained with mldecoder_joint_plugin.py.')
    head.eval()
    model.ml_decoder_head = head

    def hook(module, args, kwargs, output):
        grid = kwargs.get('grid_thw')
        if grid is None and len(args) >= 2:
            grid = args[1]
        if grid is None or not hasattr(output, 'last_hidden_state'):
            return output
        with torch.no_grad():
            logits = head(output.last_hidden_state, output.pooler_output, grid)
        model._cls_logits = logits.detach().float().cpu()
        return output

    visual.register_forward_hook(hook, with_kwargs=True)
    model._mldecoder_infer_attached = True
    logger.info(f'[mldecoder-infer] head attached for inference ({len(_TAGS)} classes)')


def patch_engine():
    from swift.infer_engine.transformers_engine import TransformersEngine

    orig = TransformersEngine.__init__

    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        m = self.model
        attach_for_inference(m.module if hasattr(m, 'module') else m)

    TransformersEngine.__init__ = __init__
    logger.info('[mldecoder-infer] TransformersEngine.__init__ patched (use --infer_backend pt)')


def patch_template_dump():
    """After each forward, write the head logits for the images of that request."""
    from swift.template.templates.qwen import Qwen3_5Template

    orig_post = Qwen3_5Template._post_encode

    def _post_encode(self, model, inputs):
        return orig_post(self, model, inputs)

    Qwen3_5Template._post_encode = _post_encode

    # the engine calls template.decode per response; piggyback the dump there so each row is
    # emitted exactly once per generated sample
    orig_decode = Qwen3_5Template.decode

    def decode(self, generate_ids, **kwargs):
        text = orig_decode(self, generate_ids, **kwargs)
        ti = kwargs.get('template_inputs')
        model = self._get_model() if hasattr(self, '_get_model') else None
        base = getattr(model, 'module', model)
        logits = getattr(base, '_cls_logits', None) if base is not None else None
        if logits is not None:
            imgs = getattr(ti, 'images', None) or []
            probs = torch.sigmoid(logits)
            for i in range(logits.shape[0]):
                p = probs[i].tolist()
                order = sorted(range(len(p)), key=lambda j: -p[j])[:TOP_K]
                img = imgs[i] if i < len(imgs) else None
                img = img if isinstance(img, str) else getattr(img, 'filename', None)
                _open_out().write(json.dumps({
                    'image': img,
                    'logits': [round(v, 5) for v in logits[i].tolist()],
                    'probs': [round(v, 6) for v in p],
                    'top': [[_TAGS[j], round(p[j], 4)] for j in order],
                }) + '\n')
                _STATE['n'] += 1
            _open_out().flush()
            try:
                del base._cls_logits
            except Exception:
                pass
        return text

    Qwen3_5Template.decode = decode
    logger.info('[mldecoder-infer] Qwen3_5Template.decode patched for logit dumping')


patch_engine()
patch_template_dump()
logger.info('[mldecoder-infer] plugin loaded')
