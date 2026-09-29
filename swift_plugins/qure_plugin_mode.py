#!/usr/bin/env python
"""Is this process training or serving? Used by the qure plugins to skip train-only patches.

WHY THIS EXISTS
swift restores `external_plugins` from the checkpoint's `args.json` when `load_args=True`
(the default). So every training plugin listed at `swift sft` time is ALSO imported by
`swift infer` / `swift deploy` / `swift eval` on that checkpoint, whether or not the infer
command asks for it. That is how a train-only `encode(self, inputs, **kwargs)` override killed
a 10k inference run with "takes 2 positional arguments but 3 were given": the inference path
calls `encode(inputs, return_template_inputs)` positionally.

Rather than make every plugin defensively support both call conventions, train-only patches
should simply not be installed when we are serving.

TWO INDEPENDENT SIGNALS, deliberately:
  * this module decides whether to INSTALL a patch (import time, before any model exists)
  * `Template.is_training` decides whether a patch ACTS (per batch)
`is_training` is `self.mode not in {'transformers','vllm','lmdeploy','sglang'}`
(swift/template/base.py:1590), so it is already False for every inference backend. Keeping
both means a wrong answer here cannot silently turn augmentation on during inference.

OVERRIDE
    QURE_PLUGIN_MODE=train   force train behaviour (install everything)
    QURE_PLUGIN_MODE=infer   force inference behaviour (skip train-only patches)
"""
import os
import sys

from swift.utils import get_logger

logger = get_logger()

# swift CLI verbs that serve a model rather than update weights
_INFER_VERBS = {'infer', 'deploy', 'eval', 'export', 'app', 'web-ui', 'rollout'}
_TRAIN_VERBS = {'sft', 'pt', 'rlhf', 'dpo', 'grpo', 'kto', 'orpo', 'ppo', 'sample', 'megatron'}


def _detect() -> str:
    forced = (os.environ.get('QURE_PLUGIN_MODE') or '').strip().lower()
    if forced in ('train', 'infer'):
        return forced
    for a in sys.argv[1:]:
        v = a.lstrip('-').lower()
        if v in _INFER_VERBS:
            return 'infer'
        if v in _TRAIN_VERBS:
            return 'train'
    # Unknown invocation (a bare python -c, a notebook, a test): assume train, because the
    # cost of being wrong that way is a patch that then no-ops on `is_training` anyway, while
    # the opposite would silently drop augmentation and token weighting from a real run.
    return 'train'


MODE = _detect()


def is_inference() -> bool:
    return MODE == 'infer'


def skip(plugin: str, what: str) -> bool:
    """True if `what` should be skipped; logs the reason exactly once per call site."""
    if not is_inference():
        return False
    logger.info(f'[{plugin}] inference mode (argv verb) — {what} NOT installed')
    return True
