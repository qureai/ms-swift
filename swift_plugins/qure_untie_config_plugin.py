#!/usr/bin/env python
"""Opt-in: make every saved checkpoint say what training actually does -- untied embeddings.

    --external_plugins ... "$SWIFT_ROOT/swift_plugins/qure_untie_config_plugin.py"

List it LAST (safest). It wraps TunerMixin.prepare_model with a PLAIN function (getattr-__func__
pattern), so it chains after whatever is already installed. Every prepare_model wrapper in
swift_plugins/ now uses the same getattr pattern (mldecoder_joint, mldecoder_joint_internvl and
qure_ce_weight used `TunerMixin.prepare_model.__func__` and crashed at import when listed after
this plugin), but a third-party plugin that still reads `.__func__` must be listed before it.

THE BUG (F3: train/serve lm_head mismatch)
Every Qwen3.5 checkpoint since stage 2 stores a SEPARATE, trained `lm_head.weight` next to
`model.language_model.embed_tokens.weight` (they have diverged), yet config.json still says
`tie_word_embeddings: true` at the top level AND in text_config:
  * HF / swift training and `--infer_backend pt` do NOT tie: both keys are in the checkpoint, so
    transformers refuses to tie and warns "You should update the config with
    tie_word_embeddings=False" (transformers/modeling_utils.py:2489-2503). The model trains and
    runs with the separate lm_head.
  * sglang 0.5.10 builds Qwen3.5's LM from `config.text_config` (sglang/srt/models/qwen3_vl.py:
    1094) and with text_config.tie_word_embeddings=true makes `lm_head = model.embed_tokens`
    (qwen3_vl.py:1105-1106) and copies the embedding into it on load (qwen3_5.py:1386-1397): it
    SERVES lm_head = embed_tokens, i.e. not the trained output head.
  * vLLM 0.20.2 does the same from `hf_text_config` (vllm/model_executor/models/qwen3_5.py:468,
    498-499); for VL models built through init_vllm_registered_model it additionally copies the
    TOP-LEVEL value into the text config (vllm/config/vllm.py:616-621).
  * HF itself reads the TOP-LEVEL field (modeling_utils.py:2421), which is also what a later
    `resize_token_embeddings()` -> `tie_weights()` consults: with `true` it RE-TIES and silently
    discards the trained lm_head.
  => the flag must be false at BOTH levels. examples/deploy/untie_lm_head.py fixes existing
     checkpoints (dry-run by default); this plugin fixes checkpoints written from now on.

WHAT THIS DOES
At TunerMixin.prepare_model (before the Trainer exists, so before any save), if the model's
output-embedding weight and input-embedding weight are DISTINCT tensors, set
tie_word_embeddings=False on the config and on its text/llm/language sub-config. No weight is
touched and nothing about the forward changes -- training is bit-identical (verified on CPU:
ckpt-608 loaded with the tied and the untied config gives identical lm_head / embed_tokens). A
genuinely tied model (lm_head IS embed_tokens) is left alone.

LIBRARY USE
`untie_tied_config(model) -> bool` can be imported by other plugins (sem_latent_plugin's
SEM_UNTIE_CONFIG) WITHOUT installing this plugin's patch: the patch is installed only when this
file is loaded by --external_plugins (top-level module name `qure_untie_config_plugin`), not when
it is imported as `swift_plugins.qure_untie_config_plugin`.

ENV
    QURE_UNTIE_CONFIG=0   load the plugin but install nothing (escape hatch)
Mode: train-only patch (skipped when qure_plugin_mode.is_inference()); prepare_model never runs
at inference anyway.
"""
# F1 -- must stay the first statements of the module (qure_plugin_mode.dedup_module).
# Only the --external_plugins copy (top-level name) registers both names: a library import
# (`swift_plugins.qure_untie_config_plugin`) must not claim the plugin name, or a later
# --external_plugins entry would be a cached no-op and never install the patch. (If a library
# copy was imported first, the plugin copy stays separate: the library copy has no side
# effects, and install() is once()-guarded.)
import sys  # noqa: E402
try:
    from swift_plugins.qure_plugin_mode import dedup_module as _dedup_module, once as _once
except ImportError:  # repo root not on sys.path; swift put this plugin's own dir there
    from qure_plugin_mode import dedup_module as _dedup_module, once as _once
_AS_PLUGIN = __name__ == 'qure_untie_config_plugin'
if _AS_PLUGIN and 'swift_plugins.qure_untie_config_plugin' not in sys.modules:
    _dedup_module(__name__, 'qure_untie_config_plugin')

import os  # noqa: E402

from swift.utils import get_logger  # noqa: E402

logger = get_logger()

ENABLED = (os.environ.get('QURE_UNTIE_CONFIG') or '1').strip().lower() not in ('0', 'false', 'no', 'off')
_SUB_KEYS = ('text_config', 'llm_config', 'language_config')


def _unwrap(model):
    """DDP/DeepSpeed `.module`, then PEFT/Swift base model, to the HF PreTrainedModel."""
    m = model
    for _ in range(4):
        nxt = getattr(m, 'module', None)
        if nxt is None or nxt is m:
            break
        m = nxt
    for attr in ('get_base_model',):
        fn = getattr(m, attr, None)
        if callable(fn):
            try:
                m = fn()
            except Exception:
                pass
    return m


def _configs(config):
    """The config and its LM sub-configs (the ones that carry tie_word_embeddings)."""
    out = [config] if config is not None else []
    for k in _SUB_KEYS:
        sub = getattr(config, k, None)
        if sub is not None and all(sub is not c for c in out):
            out.append(sub)
    return out


def _distinct(a, b) -> bool:
    if a is None or b is None or a is b:
        return False
    try:
        # ZeRO-3 partitions params to numel()==0 placeholders whose data_ptr is meaningless
        if a.numel() > 0 and b.numel() > 0 and a.data_ptr() == b.data_ptr() and a.shape == b.shape:
            return False
    except Exception:
        pass
    return True


def untie_tied_config(model) -> bool:
    """Set tie_word_embeddings=False on config + text_config when lm_head and embed_tokens are
    distinct tensors. Returns True iff a flag was changed. Never touches weights."""
    base = _unwrap(model)
    try:
        emb = base.get_input_embeddings()
        head = base.get_output_embeddings()
    except Exception:
        return False
    ew, hw = getattr(emb, 'weight', None), getattr(head, 'weight', None)
    if not _distinct(ew, hw):
        return False
    changed = []
    for c in _configs(getattr(base, 'config', None)):
        if getattr(c, 'tie_word_embeddings', False):
            c.tie_word_embeddings = False
            changed.append(type(c).__name__)
    if changed:
        logger.info(f'[qure-untie] lm_head and embed_tokens are distinct tensors -> '
                    f'tie_word_embeddings=False on {changed}; saved checkpoints now serve the '
                    'trained lm_head under sglang/vLLM and a resize cannot re-tie it')
    return bool(changed)


def install() -> None:
    """Wrap TunerMixin.prepare_model so untie_tied_config runs on every training model."""
    if not _once('qure_untie_config_plugin.install'):
        return
    from swift.pipelines.train.tuner import TunerMixin

    # plain function or classmethod depending on what earlier plugins installed
    orig = getattr(TunerMixin.prepare_model, '__func__', TunerMixin.prepare_model)

    def prepare_model(cls, args, model, **kwargs):
        model = orig(cls, args, model, **kwargs)
        if not untie_tied_config(model):
            logger.info('[qure-untie] config left unchanged (already untied, or lm_head is '
                        'embed_tokens)')
        return model

    TunerMixin.prepare_model = prepare_model
    logger.info('[qure-untie] prepare_model wrapped (tie_word_embeddings fixed before any save)')


if _AS_PLUGIN:
    from swift_plugins.qure_plugin_mode import is_inference  # noqa: E402
    if not ENABLED:
        logger.info('[qure-untie] QURE_UNTIE_CONFIG=0 -- nothing installed')
    elif is_inference():
        logger.info('[qure-untie] inference mode -- nothing installed (prepare_model is train-only)')
    else:
        install()
