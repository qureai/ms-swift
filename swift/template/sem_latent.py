# Copyright (c) ModelScope Contributors. All rights reserved.
"""SEM latent tokens: K reserved special tokens placed right after every assistant header.

WHAT THIS IS
A tokenizer that contains ``<|sem_0|> .. <|sem_{K-1}|>`` (added with
``--new_special_tokens examples/train/liger/token.txt examples/train/liger/token_sem.txt``)
gets those K tokens inserted IMMEDIATELY after every ``'<|im_start|>assistant\\n'`` of the
prompt, in every encode mode (train / transformers(pt) / sglang / vllm / lmdeploy, history turns,
tool-context headers, the inference generation prompt). The K final hidden states of the block
see only image + prompt (causal) and are what the SEM training plugin
(``swift_plugins/sem_latent_plugin.py``) regresses onto teacher report embeddings. The block is
ordinary prompt: labels -100, never generated, never part of the decoded response.

WHY RIGHT AFTER THE HEADER (and not after the empty think block)
The empty ``<think>\\n\\n</think>\\n\\n`` block is present/absent inconsistently between train
and inference (pre-existing): training adds it to every str-content assistant turn but never to
merged list-content (tool-calling) turns, while inference strips it from history and adds it to
the generation prompt only. ``header + SEM`` is the only position that is identical everywhere.
The last SEM token's hidden state is CE-supervised to predict whatever follows (``<think>``,
``<tool_call>``, content) -- expected and harmless.

WHERE IT HOOKS
``Template._encode_context_list`` (swift/template/base.py) is the one function every mode
reaches: in vllm/lmdeploy/sglang mode ``_encode_truncated`` calls ``Template._encode(self, ..)``
directly, bypassing any ``Qwen3_5Template._encode`` patch. The hook is 4 lines::

    sem_str = get_sem_str(self.tokenizer)
    if sem_str:
        context_list, loss_scale_list = insert_sem_into_contexts(context_list, loss_scale_list, sem_str)
        warn_if_sem_thinking_prefix(self, context_list, sem_str)
        warn_if_sem_header_trained(self, context_list, loss_scale_list)

``Template.decode`` also strips SEM token strings from the decoded response (``strip_sem``; same
``if sem_str`` gate). SEM ids are special tokens with no logit bias, so a model can in principle emit
one (measured max P ~1e-6 per position on ckpt-608 weights); swift decodes special tokens as text,
and a caller that fed such a response back as history would get a stray / extra block.

It lives in swift core (not a plugin) because most eval harnesses import no plugins at all
(in-process ``SwiftInfer(... --load_args false)``, direct ``TransformersEngine(ckpt)``, ``-lm``
exports without args.json). Activation is decided by the TOKENIZER alone, so a SEM checkpoint
behaves the same on every path and every pre-SEM checkpoint is untouched.

INSERTION RULE (string level, before tokenisation)
Only ``str`` contexts whose loss scale is 0 are touched (prompt pieces, tool-context headers,
the generation prompt, the masked empty-think block that ``_simplify_context_list`` merged onto
the prompt). Every occurrence of HEADER becomes ``HEADER + sem_str`` unless ``sem_str`` already
follows -- also across a context boundary (HEADER at the end of context i and context i+1
starting with ``sem_str``), so feeding back history that already carries SEM text never doubles
the block. Pre-tokenised id-list contexts and loss-scale>0 contexts are never touched. The list
lengths stay aligned. Because SEM tokens are special (added) tokens, tokenisation splits on them:
``header + SEM + '<think>'`` tokenises to exactly ``header ids + SEM ids + think ids`` (the only
case that re-tokenises differently is content starting with ``'\\n'`` -- ``'assistant\\n\\n{'``
is 271 without SEM, 198/SEM/198 with it -- which the Qwen3.5 template trims away anyway).

BACKWARD COMPATIBILITY (hard requirement)
Without ``<|sem_0|>`` in the tokenizer (every checkpoint before stage 7), ``get_sem_str`` returns
'' and the hook does nothing: the SAME list objects flow on and the encode output is
byte-identical (tests/qure/test_sem_insertion.py compares against the pre-change base.py).

K
Always derived from the tokenizer: count ``<|sem_i|>`` for i = 0, 1, ... until one is missing.
Never from an env var (env vars are not restored from args.json at inference). The Qwen3.5-2B
checkpoints have 248211 tokens and 248320 embedding rows, i.e. MAX_SEM = 109 spare rows; K > 109
forces a resize, and HF's ``tie_weights()`` then silently re-ties the trained lm_head to
embed_tokens. This module only WARNS on K > MAX_SEM (inference must keep working for whatever
the tokenizer says); the training plugin refuses to start.

CACHING / THREAD SAFETY
The result is cached on the tokenizer object (attribute ``_swift_sem_cache``); every call re-
validates the cache with two O(1) vocab lookups (``<|sem_0|>`` id unchanged, ``<|sem_K|>`` still
absent), so a deepcopy/pickled tokenizer or one extended later cannot serve a stale answer.
``len(tokenizer)`` is deliberately NOT used: on fast tokenizers it rebuilds the whole 248k vocab
(~60 ms). All other functions are pure (no shared mutable state except the warn-once set), so the
pt engine's ThreadPoolExecutor and sglang's run_in_executor are safe.

THINKING MODE
qwen3_5 has ``is_thinking=True``, so ``enable_thinking=None`` resolves the inference response
prefix to ``'<think>\\n'`` (a prompt the model was never trained on). With SEM active the block
still goes right after the header (``header + SEM + '<think>\\n'``) and a one-time WARNING is
logged; pass ``--enable_thinking false`` (every examples/deploy script already does).

CHAT TEMPLATE (native sglang / vLLM OpenAI servers)
Those servers render the checkpoint's ``chat_template.jinja`` instead of swift's template.
``make_sem_chat_template`` edits the Qwen3.5 jinja so it renders SEM right after the header in
history turns, in the last-turn-with-reasoning branch and in the generation prompt (both
enable_thinking branches), stripping a leading ``sem_str`` from assistant content first
(idempotent). ``install_sem_chat_template(processor)`` puts it on the processor and its
tokenizer so every checkpoint saved afterwards ships it. Token-id parity with swift's
sglang-mode encode is asserted by tests/qure/test_sem_insertion.py.

ENV
    SWIFT_SEM_LATENT=0   disable SEM insertion even when the tokenizer has SEM tokens (escape
                         hatch / A-B tests). Default: enabled iff ``<|sem_0|>`` is in the vocab.

KNOWN CAVEATS
  * A literal ``'<|im_start|>assistant\\n'`` inside user/tool text also gets a block; the plugin's
    block-count == assistant-message-count check marks such rows unsupervised.
  * ``--truncation_strategy left/right`` protects only image pads and can cut a block; the
    stage-6/7 runs use ``delete`` (row dropped). ``--cached_dataset`` bypasses encode: re-export.
  * ``template_backend='jinja'`` encodes the whole text at loss scale 1, so nothing is inserted
    by this hook there (the SEM-aware jinja does it) and training on that backend would label SEM
    (the SEM plugin refuses it for training; inference on the jinja backend with the SEM jinja is fine).
  * ``--loss_scale all`` / ``all+X`` puts the headers in loss-scale > 0 contexts: training inserts
    nothing while inference inserts after every header. ``warn_if_sem_header_trained`` warns once; the
    SEM plugin refuses the run.
  * THIS IS A FORK FEATURE. Upstream ms-swift (or any swift build without this module) serves a SEM
    checkpoint with ZERO SEM tokens on the swift template backend and logs nothing. There, use
    ``--template_backend jinja`` (the checkpoint ships the SEM-aware jinja) or a native sglang/vLLM
    server on a ``-lm`` export; see examples/train/liger/sem_latent/README.md for a canary.
  * Lookups are silent: the Rust ``token_to_id`` on fast tokenizers, ``_piece_to_id(token, False)`` on
    transformers' MistralCommonBackend (its ``convert_tokens_to_ids`` logs a WARNING per miss, i.e.
    per encode for every non-SEM Tekken tokenizer), else ``convert_tokens_to_ids``.
"""
import logging
import os
import re
from typing import Any, List, Optional, Sequence, Tuple

from swift.utils import get_logger

logger = get_logger()

SEM_TOKEN_FMT = '<|sem_{}|>'
HEADER = '<|im_start|>assistant\n'
# Spare embedding rows of the Qwen3.5-2B checkpoints (vocab_size 248320 - len(tokenizer) 248211).
MAX_SEM = 109
ENV_SWITCH = 'SWIFT_SEM_LATENT'

_MAX_SCAN = 4096  # hard stop for the <|sem_i|> scan
_CACHE_ATTR = '_swift_sem_cache'  # (sem0_id, k_scanned, ids, sem_str) on the tokenizer object
_WARNED = set()  # warn-once keys (module level; tests may .clear() it)

_JINJA_MARKER = '{#- [sem] SEM-aware chat template: swift/template/sem_latent.py make_sem_chat_template #}'
_JINJA_SET_RE = re.compile(r"\{%- set sem_tokens = '((?:[^'\\]|\\.)*)' %\}")
# Qwen3.5 / Qwen3.6 chat_template.jinja render statements (checkpoint-608 / -945 byte-identical).
_J_HIST = r"""{{- '<|im_start|>' + message.role + '\n' + content }}"""
_J_LAST = (r"""{{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' """
           r"""+ content }}""")
_J_GEN = r"""{{- '<|im_start|>assistant\n' }}"""
_J_STRIP = r"""{%- if content.startswith(sem_tokens) %}{%- set content = content[sem_tokens|length:] %}{%- endif %}"""
_J_HIST_NEW = r"""{{- '<|im_start|>' + message.role + '\n' + sem_tokens + content }}"""
_J_LAST_NEW = (r"""{{- '<|im_start|>' + message.role + '\n' + sem_tokens + '<think>\n' + reasoning_content """
               r"""+ '\n</think>\n\n' + content }}""")
_J_GEN_NEW = r"""{{- '<|im_start|>assistant\n' + sem_tokens }}"""


def _warn_once(key: str, msg: str, level: int = logging.WARNING) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.log(level, msg)


def sem_enabled() -> bool:
    """False when the escape hatch SWIFT_SEM_LATENT is 0/false/no/off."""
    return (os.environ.get(ENV_SWITCH) or '').strip().lower() not in {'0', 'false', 'no', 'off'}


def _as_tokenizer(obj: Any) -> Any:
    """Accept a tokenizer, a processor (``.tokenizer``) or a swift Template (``.tokenizer`` property)."""
    tok = getattr(obj, 'tokenizer', None)
    if tok is not None and tok is not obj and hasattr(tok, 'convert_tokens_to_ids'):
        return tok
    return obj


def _token_id(tokenizer: Any, token: str) -> Optional[int]:
    """Vocab id of ``token`` or None when absent. O(1): a Rust hash lookup on fast tokenizers."""
    backend = getattr(tokenizer, '_tokenizer', None)
    token_to_id = getattr(backend, 'token_to_id', None)
    if token_to_id is not None:
        try:
            return token_to_id(token)
        except Exception:  # noqa: BLE001 - fall back to the python API below
            pass
    tid = None
    # transformers' MistralCommonBackend (Tekken / SentencePiece via mistral-common) logs a WARNING on every
    # convert_tokens_to_ids miss; its own lookup with warn=False is silent (a non-SEM tokenizer misses
    # <|sem_0|> on EVERY encode, so the python API would print 2 warning lines per row).
    try:
        piece_to_id = None if hasattr(tokenizer, 'get_added_vocab') else getattr(tokenizer, '_piece_to_id', None)
        if callable(piece_to_id):
            tid = piece_to_id(token, False)
    except Exception:  # noqa: BLE001 - another signature / not a tokenizer: use the public API below
        tid = None
    if tid is None:
        try:
            tid = tokenizer.convert_tokens_to_ids(token)
        except Exception:  # noqa: BLE001 - not a tokenizer
            return None
    if not isinstance(tid, int) or tid < 0 or tid == getattr(tokenizer, 'unk_token_id', None):
        return None
    return tid


def _scan_sem(tokenizer: Any, sem0: int) -> Tuple[int, Tuple[int, ...], str]:
    ids = [sem0]
    while len(ids) < _MAX_SCAN:
        tid = _token_id(tokenizer, SEM_TOKEN_FMT.format(len(ids)))
        if tid is None:
            break
        ids.append(tid)
    k = len(ids)
    toks = [SEM_TOKEN_FMT.format(i) for i in range(k)]
    try:
        bad = [t for t, tid in zip(toks, ids) if tokenizer.encode(t, add_special_tokens=False) != [tid]]
    except Exception as e:  # noqa: BLE001
        bad = [f'<encode failed: {e!r}>']
    if bad or len(set(ids)) != k:
        _warn_once(f'bad:{sem0}:{k}', f'[sem] SEM tokens found in the tokenizer but they do not encode as single '
                   f'distinct tokens ({bad[:4]}); SEM insertion DISABLED for this tokenizer.')
        return k, (), ''
    if ids != list(range(ids[0], ids[0] + k)):
        _warn_once(f'noncontig:{sem0}:{k}', f'[sem] SEM token ids are not contiguous: {ids[:8]}...')
    if k > MAX_SEM:
        _warn_once(
            f'kmax:{sem0}:{k}', f'[sem] K={k} SEM tokens exceeds the {MAX_SEM} spare embedding rows of the '
            'Qwen3.5-2B checkpoints: adding them forced an embedding resize, which re-ties lm_head to '
            'embed_tokens. The SEM training plugin refuses this; inference insertion continues.')
    if k >= _MAX_SCAN:
        _warn_once(f'scan:{sem0}', f'[sem] stopped scanning SEM tokens at K={_MAX_SCAN}.')
    try:
        added = getattr(tokenizer, 'added_tokens_decoder', None) or {}
        not_special = [str(added[i]) for i in ids if i in added and not getattr(added[i], 'special', True)]
    except Exception:  # noqa: BLE001
        not_special = []
    if not_special:
        _warn_once(f'special:{sem0}', f'[sem] SEM tokens are not registered as special tokens: {not_special[:4]}')
    sem_str = ''.join(toks)
    _warn_once(f'active:{sem0}:{k}', f'[sem] SEM latent tokens active: K={k}, ids {ids[0]}..{ids[-1]} '
               f'(inserted right after every {HEADER!r}; {ENV_SWITCH}=0 disables).', logging.INFO)
    return k, tuple(ids), sem_str


def _sem_entry(tokenizer: Any) -> Tuple[Tuple[int, ...], str]:
    if tokenizer is None or not sem_enabled():
        return (), ''
    tokenizer = _as_tokenizer(tokenizer)
    sem0 = _token_id(tokenizer, SEM_TOKEN_FMT.format(0))
    if sem0 is None:
        return (), ''
    cache = getattr(tokenizer, _CACHE_ATTR, None)
    if (isinstance(cache, tuple) and len(cache) == 4 and cache[0] == sem0
            and _token_id(tokenizer, SEM_TOKEN_FMT.format(cache[1])) is None):
        return cache[2], cache[3]
    k, ids, sem_str = _scan_sem(tokenizer, sem0)
    try:
        setattr(tokenizer, _CACHE_ATTR, (sem0, k, ids, sem_str))
    except Exception:  # noqa: BLE001 - uncacheable object: recompute next time (warnings are once-only)
        pass
    return ids, sem_str


def get_sem_ids(tokenizer: Any) -> Tuple[int, ...]:
    """SEM token ids ``(<|sem_0|>, .., <|sem_{K-1}|>)``; ``()`` when absent or SWIFT_SEM_LATENT=0.

    Accepts a tokenizer, a processor or a swift Template. Cached on the tokenizer object.
    """
    return _sem_entry(tokenizer)[0]


def get_sem_str(tokenizer: Any) -> str:
    """Concatenated SEM token strings (``'<|sem_0|>...<|sem_{K-1}|>'``); ``''`` when inactive."""
    return _sem_entry(tokenizer)[1]


sem_string = get_sem_str  # spec alias


def sem_active(template: Any) -> bool:
    """True when the template's (or processor's / tokenizer's) tokenizer activates SEM insertion."""
    return bool(get_sem_ids(template))


def _insert_in_str(s: str, sem_str: str, next_ctx: Any) -> str:
    hl = len(HEADER)
    pos = s.find(HEADER)
    parts: List[str] = []
    start = 0
    changed = False
    while pos != -1:
        end = pos + hl
        if end < len(s):
            has_sem = s.startswith(sem_str, end)
        else:  # header ends the context: look at the next one (idempotent across context boundaries)
            has_sem = isinstance(next_ctx, str) and next_ctx.startswith(sem_str)
        parts.append(s[start:end])
        if not has_sem:
            parts.append(sem_str)
            changed = True
        start = end
        pos = s.find(HEADER, end)
    if not changed:
        return s
    parts.append(s[start:])
    return ''.join(parts)


def insert_sem_into_contexts(context_list: List[Any], loss_scale_list: Optional[List[float]],
                             sem_str: str) -> Tuple[List[Any], Optional[List[float]]]:
    """Put ``sem_str`` right after every HEADER in loss-scale-0 ``str`` contexts.

    ``loss_scale_list=None`` means all contexts are loss scale 0 (swift's convention). Returns the
    SAME ``(context_list, loss_scale_list)`` objects when nothing changes; otherwise a new context
    list of the same length (the loss-scale list is returned unchanged -- alignment is preserved).
    Idempotent. Pure: safe to call from any thread.
    """
    if not sem_str or not context_list:
        return context_list, loss_scale_list
    new_list = None
    n = len(context_list)
    for i, ctx in enumerate(context_list):
        if not isinstance(ctx, str) or HEADER not in ctx:
            continue
        if loss_scale_list is not None and loss_scale_list[i] > 0:
            continue
        new_ctx = _insert_in_str(ctx, sem_str, context_list[i + 1] if i + 1 < n else None)
        if new_ctx is not ctx:
            if new_list is None:
                new_list = list(context_list)
            new_list[i] = new_ctx
    if new_list is None:
        return context_list, loss_scale_list
    return new_list, loss_scale_list


def warn_if_sem_thinking_prefix(template: Any, context_list: List[Any], sem_str: str) -> bool:
    """One-time WARNING when the (SEM-inserted) inference generation prompt ends in the thinking prefix.

    Returns True when the prompt matched (whether or not the warning was already emitted).
    """
    if not sem_str or not context_list or getattr(template, 'is_training', False):
        return False
    last = context_list[-1]
    meta = getattr(template, 'template_meta', None)
    thinking_prefix = getattr(meta, 'thinking_prefix', None) or '<think>\n'
    if isinstance(last, str) and last.endswith(HEADER + sem_str + thinking_prefix):
        _warn_once(
            'thinking', f'[sem] SEM tokens are active but the inference response prefix is the thinking prefix '
            f'{thinking_prefix!r} (enable_thinking None/True): the prompt is header+SEM+{thinking_prefix!r}, '
            'which the SEM model was not trained on. Pass --enable_thinking false.')
        return True
    return False


def warn_if_sem_header_trained(template: Any, context_list: List[Any], loss_scale_list: Optional[List[float]]) -> bool:
    """One-time WARNING when a TRAINING encode leaves an assistant header in a loss-scale > 0 str context.

    ``--loss_scale all`` / ``all+X`` gives the prompt (headers included) a non-zero scale, so the hook
    inserts NOTHING there while every inference path does: a model trained that way on a SEM tokenizer
    is served K never-seen tokens after every header. The SEM plugin refuses such runs; this warns any
    other training run on a SEM tokenizer (e.g. a later non-SEM stage started from a SEM checkpoint).
    Returns True when such a context was found. Called only when SEM is active.
    """
    if not context_list or loss_scale_list is None or not getattr(template, 'is_training', False):
        return False
    for ctx, ls in zip(context_list, loss_scale_list):
        if isinstance(ctx, str) and ls and ls > 0 and HEADER in ctx:
            _warn_once(
                'header_trained', f'[sem] training encode: an assistant header {HEADER!r} is in a loss-scale > 0 '
                'context (--loss_scale all?), so NO SEM block is inserted there -- but every inference path inserts '
                'one after every header. Train SEM tokenizers with a loss_scale whose base strategy is default or '
                'last_round (e.g. ignore_empty_think).')
            return True
    return False


def find_sem_blocks(input_ids: Sequence[int], sem_ids: Sequence[int]) -> List[int]:
    """Start positions of complete, non-overlapping SEM blocks (the K ids in order) in a 1-D id sequence."""
    if not sem_ids:
        return []
    ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
    sem = list(sem_ids)
    k, first = len(sem), sem[0]
    starts: List[int] = []
    i = 0
    while True:
        try:
            i = ids.index(first, i)
        except ValueError:
            break
        if ids[i:i + k] == sem:
            starts.append(i)
            i += k
        else:
            i += 1
    return starts


def count_sem_blocks(input_ids: Sequence[int], sem_ids: Sequence[int]) -> int:
    """Number of complete SEM blocks in a 1-D id sequence (list or 1-D tensor)."""
    return len(find_sem_blocks(input_ids, sem_ids))


def sem_position_mask(input_ids: Any, sem_ids: Sequence[int]) -> Any:
    """Bool tensor (same shape as ``input_ids``, any rank) marking every SEM-id position. torch imported lazily.

    Marks ids, not blocks: callers that need whole blocks check ``count_sem_blocks`` per row.
    """
    import torch
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids)
    if not sem_ids:
        return torch.zeros_like(input_ids, dtype=torch.bool)
    sem = torch.as_tensor(list(sem_ids), dtype=input_ids.dtype, device=input_ids.device)
    return torch.isin(input_ids, sem)


sem_positions = sem_position_mask  # spec alias


_SEM_TOKEN_RE = re.compile(r'<\|sem_\d+\|>')


def strip_sem(text: str, sem_str: str) -> str:
    """Remove the SEM block ``sem_str`` AND every single SEM token string of it from a text.

    Used by ``Template.decode`` (a generated SEM id is never response text: fed back as history it would
    give a stray / extra block) and by callers cleaning history. Only the K token strings of ``sem_str``
    are removed (a literal ``<|sem_99|>`` beyond K is left alone). No-op without ``'<|sem_'`` in the text.
    """
    if not sem_str or not text or '<|sem_' not in text:
        return text
    text = text.replace(sem_str, '')
    if '<|sem_' in text:
        for tok in _SEM_TOKEN_RE.findall(sem_str):
            text = text.replace(tok, '')
    return text


def _jinja_quote(s: str) -> str:
    return s.replace('\\', '\\\\').replace("'", "\\'")


def make_sem_chat_template(jinja_src: str, sem_str: str) -> Optional[str]:
    """Qwen3.5 ``chat_template.jinja`` -> SEM-aware jinja (SEM right after every assistant header).

    Edits: history turns, the last-turn-with-reasoning branch (SEM before ``<think>``) and the
    generation prompt (SEM before either thinking branch); a leading ``sem_str`` is stripped from
    assistant content before re-adding it. Returns None when the expected statements are not all
    found exactly once (caller warns and keeps the original). Idempotent: an already SEM-aware
    template only gets its ``sem_tokens`` value updated.
    """
    if not isinstance(jinja_src, str) or not jinja_src or not sem_str:
        return None
    set_stmt = "{%- set sem_tokens = '" + _jinja_quote(sem_str) + "' %}"
    if _JINJA_MARKER in jinja_src:
        if len(_JINJA_SET_RE.findall(jinja_src)) != 1:
            return None
        return _JINJA_SET_RE.sub(lambda _: set_stmt, jinja_src, count=1)
    out = jinja_src
    for old, new, strip in ((_J_HIST, _J_HIST_NEW, True), (_J_LAST, _J_LAST_NEW, True), (_J_GEN, _J_GEN_NEW, False)):
        pat = re.compile(r'(?m)^([ \t]*)' + re.escape(old) + r'[ \t]*$')
        found = pat.findall(out)
        if len(found) != 1:
            return None
        indent = found[0]
        repl = (indent + _J_STRIP + '\n' if strip else '') + indent + new
        out = pat.sub(lambda _: repl, out, count=1)
    return _JINJA_MARKER + '\n' + set_stmt + '\n' + out


def install_sem_chat_template(processor: Any) -> bool:
    """Install the SEM-aware chat template on ``processor`` (and its tokenizer) in memory.

    Returns True when every string chat_template found was converted (or was already SEM-aware),
    False when SEM is inactive for this tokenizer or a template could not be converted (warned).
    ``save_pretrained`` then writes the SEM-aware ``chat_template.jinja`` into every checkpoint.
    """
    tok = _as_tokenizer(processor)
    sem_str = get_sem_str(tok)
    if not sem_str:
        return False
    targets = [processor] if tok is processor else [processor, tok]
    n_ok, n_seen = 0, 0
    for obj in targets:
        src = getattr(obj, 'chat_template', None)
        if not isinstance(src, str) or not src:
            if src is not None:
                logger.warning(f'[sem] {type(obj).__name__}.chat_template is {type(src).__name__}, not a str; '
                               'left unchanged.')
            continue
        n_seen += 1
        new = make_sem_chat_template(src, sem_str)
        if new is None:
            logger.warning(f'[sem] {type(obj).__name__}.chat_template does not match the Qwen3.5 layout; '
                           'left unchanged (native sglang/vLLM servers will NOT insert SEM tokens).')
            continue
        if new != src:
            obj.chat_template = new
        n_ok += 1
    if n_seen and n_ok == n_seen:
        logger.info(f'[sem] SEM-aware chat template installed on {n_ok} object(s) (K={len(get_sem_ids(tok))}).')
        return True
    return False
