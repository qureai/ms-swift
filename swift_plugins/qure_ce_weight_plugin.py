#!/usr/bin/env python
"""Token-weighted AR loss at ZERO cost, via Liger's fused-CE `ce_weight`.

    --external_plugins swift_plugins/qure_ce_weight_plugin.py
    --use_liger_kernel true            # required: this rides on the fused kernel

WHY THIS INSTEAD OF A GRADED loss_scale + CUSTOM CHUNKED CE
swift's non-binary `loss_scale` path routes through `per_token_loss_func`, which reads
`outputs.logits` and UPCASTS TO FP32 (swift/trainers/utils.py:101-103). At bs=28 with a 248,320
vocab that is ~180 GB of logits alone -- impossible, and it also disables the fused kernel the
throughput depends on. Hand-writing a chunked CE would avoid the memory but reimplement a tuned
Triton kernel in Python slices, i.e. it would be SLOWER than the fused path, not faster.

Liger's fused CE already takes `ce_weight`, a vocab-sized vector gathered by the TARGET token id
(liger_kernel/ops/fused_linear_cross_entropy.py:85-92) and normalized by the sum of the gathered
weights. So any weighting that depends on *which token is being predicted* is expressible for
free, inside the existing kernel, with no extra memory and no logits tensor.

WEIGHTS
    6.0   ` true` / `true` / ` True` / `True`      <- 6x false, as specified
    1.0   ` false` / `false` / ` False` / `False`
    0.1   JSON structure: braces, brackets, quotes, colons, commas, `null`, and the attribute
          keys presence / size / location / characteristics
    1.0   everything else

WHAT THIS CANNOT DO -- be aware of the limits, they are real:
  * Weighting is context-FREE. It applies wherever that token id is the target, so the JSON
    punctuation down-weight also applies inside the report turn's JSON. That is arguably
    desirable, but it is not "classification turn only".
  * Finding NAME tokens are NOT down-weighted here. Name subwords (`nod`, `ules`, ...) also
    occur in the ASP and graph turns and in ordinary prose, so a class weight would leak far
    beyond the classification JSON.
  * ASP / graph findings CANNOT be up-weighted this way at all -- "is this line a finding" is a
    property of position, not of token identity. That part genuinely needs either binary
    masking via loss_scale or a custom loss.
"""
import os

import torch

from swift.utils import get_logger

logger = get_logger()

W_TRUE = float(os.environ.get('QURE_W_TRUE', '6.0'))
W_FALSE = float(os.environ.get('QURE_W_FALSE', '1.0'))
W_FORMAT = float(os.environ.get('QURE_W_FORMAT', '0.10'))
W_DEFAULT = float(os.environ.get('QURE_W_DEFAULT', '1.0'))
# Coordinate-token weighting (bbox IoU lever). Up-weights the digit tokens that carry box
# coordinates. CRUDE and GLOBAL: ce_weight is keyed by token id, so this also up-weights digits
# elsewhere (sizes, counts) — acceptable since digits are overwhelmingly bbox coords here. It
# has no geometry (a 1px and a 100px error weigh the same); it only concentrates gradient on the
# coordinate tokens. Off by default (1.0); set QURE_W_COORD>1 to enable.
W_COORD = float(os.environ.get('QURE_W_COORD', '1.0'))

TRUE_STRS = (' true', 'true', ' True', 'True')
FALSE_STRS = (' false', 'false', ' False', 'False')
# single-digit tokens (Qwen tokenizes numbers digit-by-digit) — the atoms of box coordinates
DIGIT_STRS = tuple(str(d) for d in range(10)) + tuple(' ' + str(d) for d in range(10))
# JSON scaffolding: punctuation, the attribute keys, and null
FORMAT_STRS = (
    '{', '}', '[', ']', '"', ':', ',', '{"', '":', '": ', '", ', '"}', '"},', '}}', '}},',
    ' {', ' {"', ' [', ' ]', '\n', '\n  ', '\n    ', '\n      ',
    ' null', 'null', ' null,', 'null,',
    'presence', 'size', 'location', 'characteristics', 'characterization',
    '"presence"', '"size"', '"location"', '"characteristics"',
)

_STATE = {'vec': None, 'applied': False}


def build_ce_weight(tokenizer, vocab_size):
    """Vocab-sized weight vector, indexed by target token id."""
    w = torch.full((vocab_size,), W_DEFAULT, dtype=torch.float32)
    hits = {'true': 0, 'false': 0, 'format': 0, 'coord': 0}

    def mark(strings, value, key):
        for s in strings:
            ids = tokenizer.encode(s, add_special_tokens=False)
            # only single-token spellings can be weighted by class; a multi-token spelling
            # would need position-aware weighting, which ce_weight cannot express
            if len(ids) == 1 and 0 <= ids[0] < vocab_size:
                w[ids[0]] = value
                hits[key] += 1

    mark(TRUE_STRS, W_TRUE, 'true')
    mark(FALSE_STRS, W_FALSE, 'false')
    mark(FORMAT_STRS, W_FORMAT, 'format')
    if W_COORD != 1.0:                         # only touch digit weights when explicitly enabled
        mark(DIGIT_STRS, W_COORD, 'coord')
    logger.info(f'[qure-ce] ce_weight built over {vocab_size:,} classes · '
                f'true={W_TRUE} ({hits["true"]} ids) false={W_FALSE} ({hits["false"]} ids) '
                f'format={W_FORMAT} ({hits["format"]} ids) coord={W_COORD} ({hits["coord"]} ids) '
                f'default={W_DEFAULT}')
    return w


def patch_liger_ce_weight():
    """Inject ce_weight into liger's fused-CE FUNCTION.

    transformers' `use_liger_kernel` does not instantiate LigerFusedLinearCrossEntropyLoss as a
    module -- it sets `model.loss_function = LigerForCausalLMLoss`, which calls
    `loss_utils.fixed_fused_linear_cross_entropy(...)` and forwards **kwargs into the fused op.
    Patching the nn.Module's __init__ therefore never fires (the first attempt logged
    "no LigerFusedLinearCrossEntropyLoss instance found" and trained with NO weighting).
    Wrapping the function is the injection point that actually runs.
    """
    from liger_kernel.transformers.model import loss_utils

    orig_fn = loss_utils.fixed_fused_linear_cross_entropy

    def wrapped(*a, **kw):
        if kw.get('ce_weight') is None and _STATE['vec'] is not None:
            kw['ce_weight'] = _STATE['vec']
            if not _STATE['applied']:
                _STATE['applied'] = True
                logger.info('[qure-ce] token weighting ACTIVE on the fused CE '
                            f'(true={W_TRUE} false={W_FALSE} format={W_FORMAT})')
        elif _STATE['vec'] is None and not _STATE.get('warned'):
            _STATE['warned'] = True
            logger.warning('[qure-ce] fused CE called before ce_weight was built — this step is '
                           'UNWEIGHTED')
        return orig_fn(*a, **kw)

    loss_utils.fixed_fused_linear_cross_entropy = wrapped
    # LigerForCausalLMLoss resolves the name from this module at call time, so the rebind is
    # enough; also patch any already-imported alias.
    try:
        from liger_kernel.transformers.model import loss_utils as _lu
        _lu.fixed_fused_linear_cross_entropy = wrapped
    except Exception:
        pass
    logger.info('[qure-ce] liger fixed_fused_linear_cross_entropy wrapped')


def patch_step_probe():
    """Build the vector once the tokenizer exists, before the first forward."""
    from swift.pipelines.train.tuner import TunerMixin

    orig = TunerMixin.prepare_model.__func__

    @classmethod
    def prepare_model(cls, args, model, **kwargs):
        model = orig(cls, args, model, **kwargs)
        base = model.module if hasattr(model, 'module') else model
        tok = getattr(kwargs.get('template', None), 'tokenizer', None)
        if tok is None:
            from swift.model import get_processor
            tok = None
        if tok is not None:
            V = base.config.text_config.vocab_size
            dev = next(base.parameters()).device
            _STATE['vec'] = build_ce_weight(tok, V).to(dev)
            logger.info(f'[qure-ce] ce_weight ready on {dev}; it is injected on the first '
                        f'fused-CE call. Watch for "[qure-ce] token weighting ACTIVE".')
        else:
            logger.warning('[qure-ce] no tokenizer available; token weighting NOT active')
        return model

    TunerMixin.prepare_model = prepare_model


from swift_plugins.qure_plugin_mode import is_inference  # noqa: E402

if is_inference():
    # Nothing to weight when generating: there is no loss. Installing the wrapper anyway made
    # the fused CE log "called before ce_weight was built — this step is UNWEIGHTED" during
    # inference, which reads like a training bug and is not one.
    logger.info('[qure-ce] inference mode — token weighting NOT patched (no loss to weight)')
else:
    patch_liger_ce_weight()
    patch_step_probe()
    logger.info(f'[qure-ce] plugin loaded · true={W_TRUE} false={W_FALSE} format={W_FORMAT}')
