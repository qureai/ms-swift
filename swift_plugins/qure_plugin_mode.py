#!/usr/bin/env python
"""Is this process training or serving? Used by the qure plugins to skip train-only patches.
Also: the one-copy-per-process guard every plugin calls first (F1, below).

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

HOW THE MODE IS DECIDED (first match wins)
  1. QURE_PLUGIN_MODE=train|infer                      explicit override
  2. basename(sys.argv[0]) when argv[0] is swift/cli/<verb>.py (or swift/cli/_megatron/<verb>.py)
  3. a verb anywhere in sys.argv[1:]                   (the original, pre-F2 signal)
  4. train                                             (unknown invocation: python -c, notebook)

  F2 BUG FIX (signal 2). The `swift` console script (swift/cli/main.py:85-94) STRIPS the verb
  and re-executes `python swift/cli/<verb>.py <args>` (or torchrun on that file), so signal 3
  never saw `infer`/`deploy` in the child that actually imports the plugins: every CLI
  inference run installed the train-only patches (augmentation, cls-label lookup that loads the
  1.1M-key sidecar parquet, ASL/optimizer wrappers). The child's argv[0] IS the verb file, so it
  is now read directly:
      infer.py deploy.py eval.py export.py app.py web_ui.py rollout.py merge_lora.py -> infer
      sft.py pt.py rlhf.py sample.py                                                  -> train
      _megatron/{sft,pt,rlhf}.py -> train          _megatron/export.py              -> infer
  EXCEPTION: `export` with `--to_cached_dataset true` ENCODES TRAINING DATA
  (examples/train/training_test_split/split_qwen3_5_2b_cls_json_mldecoder.sh runs it with
  mldecoder_joint_plugin to bake cls_labels into the cache), so it stays `train` -- without
  this the cache would silently carry no cls labels and the next run no ASL term. The flag is
  matched the way swift's parser matches it: `--to_cached_dataset` or `--to-cached-dataset`, any
  unambiguous abbreviation (`--to_cached`, `--to-c`), `=value` or a following value, last one wins,
  read from RAY_SWIFT_ARGS instead of argv when that is set.
  `sample` stays `train` exactly as in the original verb table.
  Nothing inference needs is gated: every plugin's train-only branch is patches that either
  never run at inference (prepare_model / create_optimizer / Trainer callbacks / fused CE) or
  are no-ops there (`is_training` guards; cls keys popped again before forward).
  Inference-only plugins (mldecoder_infer / force_presence / record_presence) are not gated.

OVERRIDE
    QURE_PLUGIN_MODE=train   force train behaviour (install everything) -- restores the legacy
                             (pre-F2) behaviour of the swift CLI, which always said train
    QURE_PLUGIN_MODE=infer   force inference behaviour (skip train-only patches)

F1 BUG FIX -- ONE MODULE OBJECT PER PLUGIN (dedup_module / once)
swift imports each --external_plugins file as a TOP-LEVEL module (swift/utils/utils.py:395:
the plugin's directory goes on sys.path, then import_module('<basename>')), while plugins import
each other as `swift_plugins.<name>` (the repo root is on PYTHONPATH). Python treats those as two
unrelated modules, so the second name EXECUTED THE PLUGIN AGAIN and re-installed its patches:
  * qure_stage3_plugin's `import swift_plugins.qure_augment_plugin` put a second augmentation
    wrapper in the chain: non-bbox images were RandAugment-ed TWICE, and the bbox guard zeroed
    AUG_PROB on the copy it had imported while the top-level copy still applied the GEOMETRIC
    pipeline to bbox rows (6c train log: "_preprocess_inputs patched" twice).
  * mldecoder_infer_plugin's `from swift_plugins.mldecoder_joint_plugin import ...` executed a
    second joint plugin whenever the joint plugin was also listed.
Every plugin now calls `dedup_module(__name__, '<name>')` as its FIRST statement: the module
being executed is registered under BOTH 'X' and 'swift_plugins.X' in sys.modules, so whichever
name is imported later returns the SAME object and nothing re-executes. `once(key)` is a
process-global sentinel each patch function checks before installing (belt and braces: even a
second copy that somehow executes cannot stack a second wrapper). This module dedups itself the
same way, and the once() registry lives in sys.modules (not in a module global), so it is one
registry even if this file were imported under two names.
A plugin whose body RAISES during import (qure_augment_plugin's deliberate load-time albumentations
check) would leave the alias pointing at a half-initialised module; `drop_failed_import(name)` pops
such leftovers (importers that catch and retry -- qure_stage3's augment resolution, mldecoder_infer's
head-class import -- call it first, so the retry raises the same error, as before F1), and
dedup_module() never keeps a leftover as "the earlier copy".

    QURE_PLUGIN_DEDUP=0      legacy behaviour: no aliasing, once() always True -> the double
                             import (and double augmentation) is reproduced exactly
"""
import os
import sys

_PKG = 'swift_plugins'
_REGISTRY = '_qure_plugin_registry'   # process-global state, kept in sys.modules
_FALSE = ('0', 'false', 'no', 'off')


def _dedup_enabled() -> bool:
    return (os.environ.get('QURE_PLUGIN_DEDUP') or '1').strip().lower() not in _FALSE


def _registry():
    """Process-global registry, independent of which copy of this module is asking."""
    reg = sys.modules.get(_REGISTRY)
    if reg is None:
        import types
        reg = types.ModuleType(_REGISTRY, 'qure plugin process-global state (qure_plugin_mode)')
        reg.once_keys = set()
        reg.superseded = []               # globals() of copies that lost to an earlier copy
        reg.dedup_log = []                # (module_name, canonical, action) for tests/debugging
        sys.modules[_REGISTRY] = reg
    return reg


def _log(level: str, msg: str) -> None:
    # lazy: dedup_module runs for this very module before `logger` exists
    try:
        from swift.utils import get_logger
        getattr(get_logger(), level)(msg)
    except Exception:  # pragma: no cover - logging must never break an import
        print(msg, file=sys.stderr)


def _is_stale(mod) -> bool:
    """True for a module object a FAILED import left behind under an alias name.

    importlib pops only the name it was importing when the module body raises (and clears
    `__spec__._initializing`), so the alias dedup_module() registered at the TOP of the body is
    the only reference left: its own spec name no longer maps to it. A module that is still
    executing (circular import) or that loaded fine is never stale.
    """
    spec = getattr(mod, '__spec__', None)
    name = getattr(spec, 'name', None)
    if not name or getattr(spec, '_initializing', False):
        return False
    return sys.modules.get(name) is not mod


def drop_failed_import(canonical: str) -> list:
    """Pop the aliases a FAILED import of `canonical` left in sys.modules; returns the names removed.

    dedup_module() aliases the executing module under the other name BEFORE the body runs, so a
    plugin that raises during import (e.g. qure_augment_plugin's deliberate load-time
    albumentations check) would leave `X` / `swift_plugins.X` pointing at a half-initialised
    module, and an importer that catches the error and retries under the other name (qure_stage3's
    augment resolution) would silently get it. Importers that retry call this first, so the retry
    re-executes the file and raises the same error -- exactly the pre-F1 behaviour.
    """
    removed = []
    for n in (canonical, f'{_PKG}.{canonical}'):
        m = sys.modules.get(n)
        if m is not None and _is_stale(m):
            del sys.modules[n]
            removed.append(n)
    pkg = sys.modules.get(_PKG)
    m = getattr(pkg, canonical, None) if pkg is not None else None
    if m is not None and getattr(m, '__spec__', None) is not None and _is_stale(m):
        try:
            delattr(pkg, canonical)
        except Exception:  # pragma: no cover
            pass
    if removed:
        _registry().dedup_log.append((removed[0], canonical, 'dropped-failed-import'))
        _log('info', f'[qure-plugin] {canonical}: dropped {removed} left behind by a failed import')
    return removed


def dedup_module(module_name: str, canonical: str) -> None:
    """Register the executing module under both `canonical` and `swift_plugins.<canonical>`.

    Call it as the first statement of a plugin: `dedup_module(__name__, 'qure_augment_plugin')`.
    No-op when QURE_PLUGIN_DEDUP=0, when `module_name` is neither of the two names (e.g.
    __main__, or a copy under another package), or when the module is not in sys.modules.
    If the OTHER name already holds a DIFFERENT module object (a copy imported before this
    helper existed, or from another file), that earlier object is kept for both names -- it is
    the one whose patches are installed -- and every once() call made from THIS execution's
    functions returns False, so the superseded copy installs nothing. An other-name object left
    behind by a FAILED import (_is_stale) is not a loaded copy: it is replaced by this execution.
    """
    if not _dedup_enabled():
        return
    names = (canonical, f'{_PKG}.{canonical}')
    if module_name not in names:
        return
    mod = sys.modules.get(module_name)
    if mod is None:
        return
    reg = _registry()
    other_name = names[1] if module_name == names[0] else names[0]
    other = sys.modules.get(other_name)
    if other is not None and other is not mod and _is_stale(other):
        reg.dedup_log.append((other_name, canonical, 'replaced-failed-import'))
        other = None
    if other is not None and other is not mod:
        # keep the first copy: import machinery returns sys.modules[name] after exec, so the
        # importer of `module_name` gets the earlier object too
        sys.modules[module_name] = other
        reg.superseded.append(mod.__dict__)
        reg.dedup_log.append((module_name, canonical, 'kept-earlier-copy'))
        _log('warning', f'[qure-plugin] {canonical}: a different copy is already loaded as '
                        f'{other_name!r}; keeping it for {module_name!r} too (this execution\'s '
                        'patches are refused by once())')
        mod = other
    else:
        sys.modules[other_name] = mod
        reg.dedup_log.append((module_name, canonical, 'aliased'))
    # `import swift_plugins.X` (no `as`) resolves X as an attribute of the package
    pkg = sys.modules.get(_PKG)
    if pkg is not None and getattr(pkg, canonical, None) is not mod:
        try:
            setattr(pkg, canonical, mod)
        except Exception:  # pragma: no cover
            pass


def once(key: str) -> bool:
    """Process-global idempotency sentinel: True the first time `key` is seen, False after.

    Convention: key = '<plugin module name>.<patch function>'. Also False when called from a
    copy that dedup_module() superseded. Always True when QURE_PLUGIN_DEDUP=0 (legacy: every
    execution installs its patches).
    """
    if not _dedup_enabled():
        return True
    reg = _registry()
    caller = sys._getframe(1).f_globals
    if any(caller is g for g in reg.superseded):
        _log('info', f'[qure-plugin] {key} refused: this copy of {caller.get("__name__")!r} was '
                     'superseded by an earlier copy (its patches are the installed ones)')
        return False
    keys = reg.once_keys
    if key in keys:
        _log('info', f'[qure-plugin] {key} already installed in this process -- '
                     'duplicate install skipped')
        return False
    keys.add(key)
    return True


# F1 applies to this module too: one object under 'qure_plugin_mode' and
# 'swift_plugins.qure_plugin_mode'. Must run before anything else below.
dedup_module(__name__, 'qure_plugin_mode')

from swift.utils import get_logger  # noqa: E402

logger = get_logger()

# swift CLI verbs that serve a model rather than update weights (signal 3, pre-F2 table)
_INFER_VERBS = {'infer', 'deploy', 'eval', 'export', 'app', 'web-ui', 'rollout'}
_TRAIN_VERBS = {'sft', 'pt', 'rlhf', 'dpo', 'grpo', 'kto', 'orpo', 'ppo', 'sample', 'megatron'}
# swift/cli/<file>.py stems (signal 2). `main`/`utils` decide nothing.
_ARGV0_INFER = {'infer', 'deploy', 'eval', 'export', 'app', 'web_ui', 'rollout', 'merge_lora'}
_ARGV0_TRAIN = {'sft', 'pt', 'rlhf', 'sample'}


def _cli_verb(argv0: str):
    """'infer' for .../swift/cli/infer.py (or .../swift/cli/_megatron/export.py -> 'export')."""
    if not argv0:
        return None
    p = os.path.abspath(argv0)
    stem, ext = os.path.splitext(os.path.basename(p))
    if ext != '.py':
        return None
    parent = os.path.dirname(p)
    up1, up2 = os.path.basename(parent), os.path.basename(os.path.dirname(parent))
    if (up1 == 'cli' and up2 == 'swift') or (up1 == '_megatron' and up2 == 'cli'):
        return stem
    return None


_TRUE = ('true', '1', 'yes', 'y', 't')   # transformers string_to_bool
# `--to_cached_dataset` as swift's HfArgumentParser(ExportArguments) resolves it: both spellings are
# registered and argparse accepts unambiguous abbreviations (allow_abbrev). 'to_c' is the shortest
# unambiguous prefix (to_hf / to_mcore / to_ollama / to_peft_format share 'to_'); tests pin this
# against the real parser.
_TO_CACHED_SPELLINGS = ('to_cached_dataset', 'to-cached-dataset')
_TO_CACHED_MIN_PREFIX = 4


def _flag_true(argv, flag: str, spellings=None, min_prefix: int = 0) -> bool:
    """HfArgumentParser bool flag: `--f`, `--f true`, `--f=true`; the LAST occurrence wins (argparse).

    `spellings` (option names without '--') also accepts every registered spelling of the flag and
    each prefix of one that is >= `min_prefix` chars (argparse abbreviations). Default: `flag` only.
    """
    names = tuple(spellings) if spellings else (flag[2:] if flag.startswith('--') else flag, )
    val = False
    for i, a in enumerate(argv):
        if not isinstance(a, str) or not a.startswith('--'):
            continue
        opt, eq, explicit = a[2:].partition('=')
        if not any(n == opt or (min_prefix and len(opt) >= min_prefix and n.startswith(opt)) for n in names):
            continue
        if eq:
            val = explicit.strip().lower() in _TRUE
        else:
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            val = nxt is None or not isinstance(nxt, str) or nxt.startswith('--') \
                or nxt.strip().lower() in _TRUE
    return val


def _swift_argv(argv):
    """The option list swift itself parses: RAY_SWIFT_ARGS (JSON) replaces sys.argv[1:] when set
    (swift/utils/utils.py parse_args)."""
    ray = os.environ.get('RAY_SWIFT_ARGS')
    if ray:
        try:
            import json
            v = json.loads(ray)
            if isinstance(v, list):
                return [str(x) for x in v]
        except Exception:  # noqa: BLE001 - swift fails on it itself; fall back to argv
            pass
    return argv[1:]


def _detect():
    """(mode, source) -- see the module docstring for the order."""
    forced = (os.environ.get('QURE_PLUGIN_MODE') or '').strip().lower()
    if forced in ('train', 'infer'):
        return forced, 'QURE_PLUGIN_MODE'
    argv = list(sys.argv or [])
    verb = _cli_verb(argv[0] if argv else '')
    if verb is not None:
        if verb == 'export' and _flag_true(_swift_argv(argv), '--to_cached_dataset', _TO_CACHED_SPELLINGS,
                                           _TO_CACHED_MIN_PREFIX):
            return 'train', f'argv[0]={verb}.py --to_cached_dataset (encodes training data)'
        if verb in _ARGV0_INFER:
            return 'infer', f'argv[0]={verb}.py'
        if verb in _ARGV0_TRAIN:
            return 'train', f'argv[0]={verb}.py'
    for a in argv[1:]:
        v = a.lstrip('-').lower()
        if v in _INFER_VERBS:
            return 'infer', f'argv verb {v!r}'
        if v in _TRAIN_VERBS:
            return 'train', f'argv verb {v!r}'
    # Unknown invocation (a bare python -c, a notebook, a test): assume train, because the
    # cost of being wrong that way is a patch that then no-ops on `is_training` anyway, while
    # the opposite would silently drop augmentation and token weighting from a real run.
    return 'train', 'default'


MODE, MODE_SOURCE = _detect()
logger.info(f'[qure-plugin] mode={MODE} (from {MODE_SOURCE}; override with QURE_PLUGIN_MODE, '
            f'plugin dedup={"on" if _dedup_enabled() else "OFF (QURE_PLUGIN_DEDUP=0)"})')


def is_inference() -> bool:
    return MODE == 'infer'


def skip(plugin: str, what: str) -> bool:
    """True if `what` should be skipped; logs the reason exactly once per call site."""
    if not is_inference():
        return False
    logger.info(f'[{plugin}] inference mode ({MODE_SOURCE}) — {what} NOT installed')
    return True
