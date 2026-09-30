#!/usr/bin/env python
"""SEM latent-token targets: THE single definition of "which assistant turn is a report, and
what text does the teacher embed for it". Shared, byte-for-byte, by
  * the offline teacher sidecar builder  examples/train/liger/sem_latent/build_sem_targets.py
  * the training plugin                  swift_plugins/sem_latent_plugin.py
If the two ever disagreed, every training lookup would miss the cache. So nothing in here
reads an env var that changes a target, and any change to a rule MUST bump RULE_VERSION (the
sidecar stores it in meta.json; the plugin refuses a sidecar built under another version).

PURE: stdlib + numpy only at import (no torch, no swift). Safe in dataloader workers.

WHICH TURNS ARE SUPERVISED (T1 only -- spec S2-DECISIONS, census in map_data.md)
A turn is supervised when BOTH signals agree:
  1. the preceding user prompt (after clean_user_prompt) starts with a T1 prefix, and
  2. the answer validates for that family.
      family          prompt prefix (cleaned)                                answer must be
      onto            "Generate a detailed ontology report"                  JSON dict with Findings
      fict            "Generate the Findings, Impression, and CT"            prose starting "Findings:"
      medrep          "Generate a medical report"                            JSON dict with Findings
      sec_findings    "Generate the Findings of the given"                   plain text (template G)
      sec_summary     "Generate the Summary of the given"                    plain text (template G)
      sec_impression  "Generate the Impression of the given"                 plain text (template G)
  EXCLUDED (no target): sec_reasoning, sec_prominence, sjson, graph ("hierarchical positive
  findings graph"), repcmp, followup, T3 coordinate JSON (describe / nod_char / nod_assess /
  lines), cls, bbox, region, asp, verify, audit, loccmp, and every tool-call / tool-response
  message. The answer-format check alone is ambiguous (RL's _kind labels graph/describe/lines
  "report"), hence the prompt gate; the prompt alone lets empty/garbled answers through
  (sec_summary has 17 empty answers in the mixture), hence the format gate.

PROMPT CLEANING (clean_user_prompt)
  * the head-prior block of HG / B3 rows: a prompt starting "Automated screening prior" loses
    everything up to and including its first blank line (head_context.serialize_context +
    build_headguided_s5.py: f'{prior}\\n\\n{prompt}')
  * the 5 B3 tool-use cue lines (build_component_b_v2.py CUES; always a full line)
  * <image> tags and leading "Image N:" labels, then .strip()

CANONICAL TARGET TEXT (canonical_target) -- one rule, per turn
      "Findings: ...\\nSummary: ...\\nImpression: ..."
  built from whichever of those three sections the turn itself contains, in that order:
      onto -> F+S+I     medrep -> F+I     fict -> F+I (parsed from the prose)
      sec_findings -> "Findings: <text>"  sec_summary -> "Summary: <text>"
      sec_impression -> "Impression: <text>"
  Complete Reasoning, Recommendations, Prominence_Score and the region sections are dropped.
  Findings/Impression strings are identical across families for the same film (150/150), so
  the rule gives a consistent per-film target. Section bodies: every line .strip()-ed, blank
  lines dropped, "* " bullets kept verbatim; list/dict values flattened to "key: value" lines.
  Answer cleaning before parsing, in order: leading <think>...</think> block (or everything up
  to a lone </think>), tool_call/tool_response blocks incl. unterminated tails (exactly
  output/rl_cxr/rl_rewards.py:strip_tools), VLM special tokens (|<ASP>| |</ASP>| <|ASP|>, the
  132 <TAG> tokens of examples/train/liger/token.txt, <|sem_i|>, chat control tokens), ```
  fences, .strip(). JSON is parsed with json raw_decode, then ast.literal_eval (python repr).
  Final .strip(), capped at MAX_CHARS (6000). A list-content (swift-merged) assistant message
  uses its LAST element: after tool rounds that is the final answer.
  With user_prompt=None (e.g. scoring generated text) detection is by FORMAT only: a JSON dict
  with Findings -> onto (has Summary) / medrep (no Summary); prose starting "Findings:" -> fict.

KEYS
  target_key(text): first 8 bytes (big-endian) of sha1(text.encode('utf-8')) as uint64.
  image_key(path) : same hash of the FULL normalised path (storage root /localstorage |
                    /fastdata3 | /fastdata4 stripped, swift's image_<W>x<H>_ basename prefix
                    removed, os.path.normpath). Full path, not basename: CheXpert basenames
                    collide (view1_frontal.png alone covers 9,012 turns). 0 = no identity.

LEAK GUARD (is_leaked, used by the plugin under SEM_SKIP_LEAKED=1)
  A turn is leaked when its Findings body (or, for a single-section turn, its whole body)
  appears verbatim in any EARLIER text of the row. "Verbatim" is compared after undoing JSON
  string escapes in the prior text (\\n \\t \\" \\/ \\\\) and collapsing whitespace on both sides,
  so a section turn after an onto turn of the same film is caught although the onto answer
  holds its Findings as an escaped JSON string.

RAW ROWS
  row_messages(row)     normalises messages/conversations, role/from, content/value;
                        drops system messages; keeps the per-message `loss` flag (None = unset).
  assistant_turn_groups(msgs) maximal runs of assistant/tool_call messages = the assistant
                        turns swift renders behind ONE header each = one SEM block each.
  iter_row_targets(row) (msg_index, family, canonical) for every RAW assistant message with a
                        target (a SUPERSET of what the plugin sees on merged messages, so every
                        training lookup hits the cache), plus explicit overrides:
                          sem_targets: list aligned with assistant_turn_groups (str or null)
                          sem_target : str -> the last assistant turn
                        family 'override'; msg_index = last message of that group (-1 when the
                        list is longer than the row has turns). Override text is only
                        .strip()-ed and capped (normalise_override); it is embedded as given.

SIDECAR (written by write_sidecar / build_sem_targets.py --merge, read by SidecarReader)
  keys.u64.npy        (N,)       uint64, strictly increasing
  emb.f16.npy         (N, 2560)  float16, L2-normalised, UNcentred teacher space, rows = keys
  mu.f32.npy          (2560,)    float32 mean of the N unique rows (NOT normalised, |mu|~0.7)
  in_train.u8.npy     (N,)       1 = key seen in a --train_datasets file
  families.u8.npy     (N,)       optional bitmask of FAMILY_BITS the key was produced by
  probe_logits.f16.npy(N, 131)   optional raw logits of lp23sep head_xray_bce.pt on the row
  ntok.i32.npy        (N,)       optional untruncated teacher token count (EOT included)
  texts.jsonl                    optional, one {"k","t","f"} per row in key order
  meta.json                      written LAST (its presence = sidecar complete): rule_version,
                                 teacher_path, teacher_fingerprint, max_length, dtype, n, dim,
                                 datasets, created, probe_head, truncation_rate,
                                 n_texts_per_family, ...
  SidecarReader memory-maps emb/probe (never loads the matrix); keys/mu/in_train are small.
  It re-opens its maps after unpickling, so it is safe to build in every dataloader worker.

ENV (paths only -- nothing here changes a target)
  SEM_TEACHER_PATH   default teacher dir   (DEFAULT_TEACHER, bb_23sep)
  SEM_PROBE_HEAD     default probe head    (DEFAULT_PROBE_HEAD, lp23sep/head_xray_bce.pt)
  QURE_PLUGIN_DEDUP  =0 disables the one-module-object alias below (same knob as the F1 fix)
  Both paths go through resolve_storage_path: tried as given, then under the other storage
  roots (/localstorage <-> /fastdata3 <-> /fastdata4), like swift/template/vision_utils.py.
"""
import ast
import hashlib
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---- one module object under both import names (F1 pattern, stdlib-only copy) ---------------
# swift imports plugin files as TOP-LEVEL modules (swift_plugins/ goes on sys.path) while other
# code imports `swift_plugins.sem_targets`. Pure module, so a second copy would only duplicate
# caches -- but aliasing keeps `SidecarReader` / constants identical objects everywhere.
if os.environ.get('QURE_PLUGIN_DEDUP', '1') != '0' and __name__ in ('sem_targets',
                                                                     'swift_plugins.sem_targets'):
    _other = 'swift_plugins.sem_targets' if __name__ == 'sem_targets' else 'sem_targets'
    _prev = sys.modules.get(_other)
    if _prev is not None and _prev is not sys.modules.get(__name__):
        sys.modules[__name__] = _prev          # keep the first copy for both names
    elif __name__ in sys.modules:
        sys.modules[_other] = sys.modules[__name__]

logger = logging.getLogger('sem_targets')

RULE_VERSION = 'v1'
SIDECAR_FORMAT = 'sem_sidecar_v1'
KEY_RULE = "uint64 = big-endian first 8 bytes of sha1(text.encode('utf-8'))"

T1_FAMILIES = ('onto', 'fict', 'medrep', 'sec_findings', 'sec_summary', 'sec_impression')
OVERRIDE_FAMILY = 'override'
FAMILY_BITS = {f: 1 << i for i, f in enumerate(T1_FAMILIES)}
FAMILY_BITS[OVERRIDE_FAMILY] = 1 << 6

MAX_CHARS = 6000
MIN_BODY_CHARS = 4          # a section body shorter than this is not a report ('' / '.' / 'N/A')
EMB_DIM = 2560
TEACHER_MAX_LENGTH = 1024
TEACHER_EOT_ID = 151643     # bb_23sep <|endoftext|>, appended by add_special_tokens=True

STORAGE_ROOTS = ('/localstorage', '/fastdata3', '/fastdata4')
_DEFAULT_TEACHER = '/fastdata4/sahil_work/tag_extractor/bb_23sep'
_DEFAULT_PROBE_HEAD = '/fastdata4/sahil_work/tag_extractor/lp23sep/head_xray_bce.pt'

SECTION_NAMES = ('Findings', 'Summary', 'Impression')

# ============================================================================================
# paths
# ============================================================================================


def resolve_storage_path(path: Optional[str]) -> Optional[str]:
    """`path` if it exists, else the same suffix under another storage root, else `path`
    unchanged (the caller decides whether a missing path is fatal). None/'' pass through."""
    if not path:
        return path
    if os.path.exists(path):
        return path
    for root in STORAGE_ROOTS:
        if path.startswith(root + '/'):
            suffix = path[len(root):]
            for alt in STORAGE_ROOTS:
                if alt != root and os.path.exists(alt + suffix):
                    return alt + suffix
            break
    return path


def default_teacher_path() -> str:
    return resolve_storage_path(os.environ.get('SEM_TEACHER_PATH') or _DEFAULT_TEACHER)


def default_probe_head_path() -> str:
    return resolve_storage_path(os.environ.get('SEM_PROBE_HEAD') or _DEFAULT_PROBE_HEAD)


DEFAULT_TEACHER = _DEFAULT_TEACHER
DEFAULT_PROBE_HEAD = _DEFAULT_PROBE_HEAD

# ============================================================================================
# user prompt -> family
# ============================================================================================

PRIOR_HEADER = 'Automated screening prior'
# build_component_b_v2.py CUES (the 5 tool-use cue lines B3 inserts as a full line)
TOOL_CUES = (
    'Use the available tools and show each step before the final answer.',
    'Work through this with the provided tools, then give the final answer.',
    'Call the relevant tools first; reply with the final answer after the tool steps.',
    'Solve this step by step using tool calls.',
    'Use tools where helpful and finish with the final answer.',
)
_CUE_LINE_RE = re.compile(r'(?m)^[ \t]*(?:' + '|'.join(re.escape(c) for c in TOOL_CUES) + r')[ \t]*(?:\n|$)')
_IMAGE_TAG_RE = re.compile(r'<image>')
_IMAGE_LABEL_RE = re.compile(r'(?m)^[ \t]*Image \d+:[ \t]*')

# ordered: the longer 'Generate the Findings, Impression, and CT' must win over sec_findings
_T1_PREFIXES = (
    ('fict', 'Generate the Findings, Impression, and CT'),
    ('onto', 'Generate a detailed ontology report'),
    ('medrep', 'Generate a medical report'),
    ('sec_findings', 'Generate the Findings of the given'),
    ('sec_summary', 'Generate the Summary of the given'),
    ('sec_impression', 'Generate the Impression of the given'),
)
_GRAPH_MARK = 'hierarchical positive findings graph'


def _content_text(content: Any, last: bool = False) -> str:
    """str content as is; list content -> joined text parts (or the LAST str part when
    `last`, i.e. the final answer of a swift-merged assistant turn); dict -> its text."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        v = content.get('text', content.get('content', ''))
        return v if isinstance(v, str) else ''
    if isinstance(content, (list, tuple)):
        parts = [_content_text(c) for c in content if c is not None]
        parts = [p for p in parts if isinstance(p, str)]
        if last:
            return parts[-1] if parts else ''
        return ''.join(parts)
    return str(content)


def clean_user_prompt(text: Any) -> str:
    """Strip the head-prior header block, the 5 tool-use cue lines, <image> tags and leading
    "Image N:" labels; .strip(). List content is joined."""
    t = _content_text(text)
    t = t.lstrip()
    if t.startswith(PRIOR_HEADER):
        parts = t.split('\n\n', 1)
        if len(parts) == 2:
            t = parts[1]
    t = _CUE_LINE_RE.sub('', t)
    t = _IMAGE_TAG_RE.sub('', t)
    t = _IMAGE_LABEL_RE.sub('', t)
    return t.strip()


def prompt_family(user_prompt: Any) -> Optional[str]:
    """T1 family named by the user prompt alone (no answer check), or None."""
    p = clean_user_prompt(user_prompt)
    if not p or _GRAPH_MARK in p.lower():
        return None
    for fam, pfx in _T1_PREFIXES:
        if p.startswith(pfx):
            return fam
    return None


# ============================================================================================
# assistant answer cleaning / parsing
# ============================================================================================

# the 132 finding tags of examples/train/liger/token.txt (frozen here so the rule does not
# depend on a file; tests assert it still equals token.txt)
_TAGS = """PNEUMONITIS BRONCHIECTASIS GROUND_GLASS_OPACITIES DIALYSIS_CATHETER CLAVICLE_FRACTURE ATELECTASIS
CHEST_WALL_DEFORMITY PICC_LINE CHEST_TUBE HYDROPNEUMOTHORAX RIB_FRACTURE AORTIC_ECTASIA PNEUMOTHORAX LYMPHADENOPATHY
SUPPORT_DEVICE AORTIC_TORTUOSITY MALIGNANCY SPINAL_HARDWARE LOOP_RECORDER MPA_DILATION PULMONARY_HYPERTENSION
OSTEOPENIA TENTED_DIAPHRAGM MASS ROTATION ARTIFACT SCARRING ENDOTRACHEAL_TUBE_REMOVED SURGICAL_HARDWARE ANEURYSM
AERATION CARDIAC_DEFIBRILLATOR INTUBATION HEALED_RIB_FRACTURE HYPERINFLATION EPIDURAL_CATHETER TRACHEAL_SHIFT
LORDOSIS_ABNORMALITY CALCIFIED_PULMONARY_NODULE HILAR_ENLARGEMENT GRANULOMA PULMONARY_VASCULAR_CONGESTION
AORTIC_ABNORMALITY HARDWARE_WIRE LUNG_SURGERY PULMONARY_ARTERY_CATHETER FISSURAL_THICKENING OPACITY HILAR_MASS
HEMOTHORAX PLEURAL_THICKENING INFILTRATE SOLITARY_PULMONARY_NODULE RETROCARDIAC_OPACITY PACEMAKER ARDS
PLEURAL_CALCIFICATION ASPIRATION POSTOPERATIVE_CHANGES BASILAR_OPACITY ATHEROSCLEROSIS ELEVATED_DIAPHRAGM
PERIHILAR_OPACITIES METASTASIS CALCIFICATION DEMINERALIZATION MEDIASTINAL_WIDENING ORTHOPEDIC_HARDWARE
PLEURAL_EFFUSION CATHETER MEDIASTINAL_DRAIN COMPRESSION_FRACTURE EMPYEMA SCOLIOSIS PULMONARY_NODULE MEDIASTINAL_SHIFT
PERICARDIAL_EFFUSION STERNOTOMY_WIRES INTERSTITIAL_LUNG_DISEASE CAVITY PNEUMOPERITONEUM ENTERIC_TUBE_REMOVED LESION
REACTIVE_AIRWAYS_DISEASE NORMAL PROSTHETIC_VALVE DEGENERATIVE_CHANGES OSSEOUS_STRUCTURES PULMONARY_EDEMA TRACHEOSTOMY
CARDIOMEGALY LVAD DIAPHRAGMATIC_HERNIA PERIBRONCHIAL_THICKENING AIRSPACE_OPACITY FEEDING_TUBE BONE_METASTASIS
GROUND_GLASS_NODULE COPD DIFFUSE_PULMONARY_NODULES FIBROSIS AORTIC_ANEURYSM CONSOLIDATION EMPHYSEMA PNEUMOMEDIASTINUM
FIBRO_LESION PROSTHETIC SOFT_TISSUE_SWELLING STERNOTOMY KERLEY_LINES CALCIFIED_GRANULOMA CENTRAL_VENOUS_CATHETER
CAVITARY_NODULE PLEURAL_SCARRING PORT_CATHETER HEART_FAILURE NIPPLE_SHADOWS END_OF_THE_VESSEL SUBCUTANEOUS_EMPHYSEMA
LOW_LUNG_VOLUMES CABG KYPHOSIS LUNG_TRANSPLANT HILAR_LYMPHADENOPATHY TRACHEOSTOMY_TUBE ENDOTRACHEAL_TUBE
RESPIRATORY_FAILURE BRONCHITIS PNEUMONIA MEDIASTINAL_MASS PULMONARY_VENOUS_HYPERTENSION PLEURAL_PLAQUE"""
TAG_TOKENS = tuple(f'<{t}>' for t in _TAGS.split())
VLM_SPECIAL_TOKENS = ('|<ASP>|', '|</ASP>|') + TAG_TOKENS

_SPECIAL_RE = re.compile(
    r'<\|sem_\d+\|>'
    r'|\|</?ASP>\||<\|/?ASP\|>'
    r'|<\|(?:im_start|im_end|endoftext|vision_start|vision_end|image_pad|video_pad)\|>'
    r'|<(?:' + '|'.join(sorted(_TAGS.split(), key=len, reverse=True)) + r')>')
_THINK_LEAD_RE = re.compile(r'^\s*<think>.*?</think>\s*', re.S)
_FENCE_HEAD_RE = re.compile(r'^```[A-Za-z0-9_-]*[ \t]*\n?')
_FENCE_TAIL_RE = re.compile(r'\n?[ \t]*```$')


def strip_tools(t: Optional[str]) -> str:
    """EXACTLY output/rl_cxr/rl_rewards.py:strip_tools (tool blocks, unterminated tails, a
    leading empty think block, strip)."""
    t = re.sub(r'<tool_call>.*?</tool_call>', '', t or '', flags=re.S)
    t = re.sub(r'<tool_response>.*?</tool_response>', '', t, flags=re.S)
    t = re.sub(r'<tool_call>.*$|<tool_response>.*$', '', t, flags=re.S)
    return re.sub(r'^<think>\s*</think>\s*', '', t.strip())


def clean_assistant_text(content: Any) -> str:
    """Answer text the rules parse: LAST element of list content; leading think block (or up
    to a lone </think>); strip_tools; VLM special tokens; ``` fences; .strip()."""
    t = _content_text(content, last=True)
    if not t:
        return ''
    if '</think>' in t:
        if '<think>' in t:
            t = _THINK_LEAD_RE.sub('', t, count=1)
        else:                                   # thinking-mode generation: prompt held <think>
            t = t.rsplit('</think>', 1)[1]
    t = strip_tools(t)
    t = _SPECIAL_RE.sub('', t)
    t = t.strip()
    if t.startswith('```'):
        t = _FENCE_HEAD_RE.sub('', t, count=1)
        t = _FENCE_TAIL_RE.sub('', t).strip()
    return t


def _parse_obj(t: str) -> Any:
    """JSON (tolerating trailing junk), then python-repr; None when neither parses."""
    if not t or t[0] not in '{[':
        return None
    try:
        return json.JSONDecoder().raw_decode(t)[0]
    except Exception:
        pass
    if len(t) <= 500_000:
        try:
            return ast.literal_eval(t)
        except Exception:
            pass
    return None


def _norm_key(k: Any) -> str:
    return re.sub(r'[\s_]+', ' ', str(k)).strip().lower()


def _flatten(v: Any) -> str:
    """list/dict section values -> readable 'key: value' lines."""
    if v is None:
        return ''
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, int, float)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return '\n'.join(s for s in (_flatten(x) for x in v) if s.strip())
    if isinstance(v, dict):
        out = []
        for k, x in v.items():
            s = _flatten(x)
            if not s.strip():
                continue
            out.append(f'{k}:\n{s}' if '\n' in s.strip() else f'{k}: {s.strip()}')
        return '\n'.join(out)
    return str(v)


def _norm_body(s: Any) -> str:
    s = _flatten(s).replace('\r\n', '\n').replace('\r', '\n')
    return '\n'.join(ln.strip() for ln in s.split('\n') if ln.strip())


def _dict_sections(d: dict) -> Dict[str, str]:
    """{'Findings'|'Summary'|'Impression': body} from a report dict (case/underscore
    insensitive keys; the first matching key wins)."""
    want = {n.lower(): n for n in SECTION_NAMES}
    out: Dict[str, str] = {}
    for k, v in d.items():
        name = want.get(_norm_key(k))
        if name and name not in out:
            out[name] = _norm_body(v)
    return out


_PROSE_HDR_RE = re.compile(
    r'(?im)^[ \t]*(findings|summary|impression|recommendations?|complete reasoning|prominence[ _]score)'
    r'[ \t]*:[ \t]*')


def _prose_sections(t: str) -> Dict[str, str]:
    """'Findings:\\n...\\n\\nImpression:\\n...[\\n\\nRecommendations:...]' -> sections."""
    want = {n.lower(): n for n in SECTION_NAMES}
    ms = list(_PROSE_HDR_RE.finditer(t))
    out: Dict[str, str] = {}
    for i, m in enumerate(ms):
        name = want.get(m.group(1).lower())
        end = ms[i + 1].start() if i + 1 < len(ms) else len(t)
        if name and name not in out:
            out[name] = _norm_body(t[m.end():end])
    return out


def _compose(sections: Dict[str, str]) -> Optional[str]:
    parts = [f'{n}: {sections[n]}' for n in SECTION_NAMES if sections.get(n)]
    if not parts:
        return None
    text = '\n'.join(parts).strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rstrip()
    return text or None


_SEC_FAMILY_SECTION = {'sec_findings': 'Findings', 'sec_summary': 'Summary', 'sec_impression': 'Impression'}


def _single_section(name: str, t: str) -> Optional[str]:
    if not t or t[0] in '{[<':                   # structured / markup answer, not a section
        return None
    m = re.match(r'(?i)^[ \t]*' + name + r'[ \t]*:[ \t]*', t)
    if m:                                        # answer repeats its own header
        t = t[m.end():]
    body = _norm_body(t)
    if len(body) < MIN_BODY_CHARS:
        return None
    return _compose({name: body})


def _analyse(user_prompt: Any, content: Any) -> Tuple[Optional[str], Optional[str]]:
    """-> (family, canonical) or (None, None). One code path for classify/canonical so that
    classify_turn(...) is not None  <=>  canonical_target(...) is not None."""
    t = clean_assistant_text(content)
    if not t:
        return None, None
    if user_prompt is None:                      # format-only detection (generated text)
        obj = _parse_obj(t)
        if isinstance(obj, dict):
            if not any(_norm_key(k) == 'findings' for k in obj):
                return None, None
            sec = _dict_sections(obj)
            can = _compose(sec)
            return ('onto' if 'Summary' in sec else 'medrep', can) if can else (None, None)
        if t.startswith('Findings:'):
            can = _compose(_prose_sections(t))
            return ('fict', can) if can else (None, None)
        return None, None
    fam = prompt_family(user_prompt)
    if fam is None:
        return None, None
    if fam in ('onto', 'medrep'):
        obj = _parse_obj(t)
        if not isinstance(obj, dict) or not any(_norm_key(k) == 'findings' for k in obj):
            return None, None
        can = _compose(_dict_sections(obj))
    elif fam == 'fict':
        if not t.startswith('Findings:'):
            return None, None
        can = _compose(_prose_sections(t))
    else:
        can = _single_section(_SEC_FAMILY_SECTION[fam], t)
    return (fam, can) if can else (None, None)


def classify_turn(user_prompt: Any, assistant_text: Any) -> Optional[str]:
    """T1 family name of this (prompt, answer) turn, or None. `assistant_text` may be swift
    list content (the last element is used). user_prompt=None -> format-only detection."""
    return _analyse(user_prompt, assistant_text)[0]


def canonical_target(assistant_text: Any, user_prompt: Any = None) -> Optional[str]:
    """'Findings: ...\\nSummary: ...\\nImpression: ...' (present sections) or None."""
    return _analyse(user_prompt, assistant_text)[1]


def turn_target(user_prompt: Any, assistant_text: Any) -> Optional[Tuple[str, str]]:
    """(family, canonical) in one pass, or None."""
    fam, can = _analyse(user_prompt, assistant_text)
    return (fam, can) if fam is not None else None


def normalise_override(text: Any) -> Optional[str]:
    """Explicit `sem_targets` / `sem_target` entry -> text embedded as given (.strip(), cap)."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    if len(t) > MAX_CHARS:
        t = t[:MAX_CHARS].rstrip()
    return t or None


# ============================================================================================
# keys
# ============================================================================================


def target_key(canonical_text: str) -> int:
    return int.from_bytes(hashlib.sha1(canonical_text.encode('utf-8')).digest()[:8], 'big')


def target_keys(texts: Iterable[str]) -> np.ndarray:
    return np.fromiter((target_key(t) for t in texts), dtype=np.uint64)


_WXH_PREFIX_RE = re.compile(r'^image_\d+x\d+_(.+)$')


def normalise_image_path(path: Any) -> Optional[str]:
    if isinstance(path, dict):
        path = path.get('path')
    if not isinstance(path, str) or not path.strip():
        return None
    p = path.strip()
    if p.startswith(('http://', 'https://')):
        return p
    p = os.path.normpath(p)
    d, b = os.path.split(p)
    m = _WXH_PREFIX_RE.match(b)
    if m:
        p = os.path.join(d, m.group(1))
    for root in STORAGE_ROOTS:
        if p.startswith(root + '/'):
            p = p[len(root):]
            break
    return p


def image_key(path: Any) -> int:
    """uint64 identity of an image (hard-negative masking). 0 when there is no usable path
    (PIL image, bytes, None): 0 never masks anything."""
    p = normalise_image_path(path)
    if p is None:
        return 0
    return int.from_bytes(hashlib.sha1(p.encode('utf-8')).digest()[:8], 'big')


# ============================================================================================
# leak guard
# ============================================================================================

_UNESCAPE = (('\\\\', '\x00'), ('\\n', '\n'), ('\\t', '\t'), ('\\"', '"'), ('\\/', '/'), ('\x00', '\\'))
_CANON_HDR_RE = re.compile(r'(?m)^(Findings|Summary|Impression): ')


def split_canonical(canonical_text: str) -> Dict[str, str]:
    """Inverse of the canonical composition: {'Findings': body, ...}."""
    ms = list(_CANON_HDR_RE.finditer(canonical_text or ''))
    out: Dict[str, str] = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(canonical_text)
        out.setdefault(m.group(1), canonical_text[m.end():end].strip())
    return out


def _leak_norm(s: str) -> str:
    for a, b in _UNESCAPE:
        s = s.replace(a, b)
    return ' '.join(s.split())


def leak_body(canonical_text: str) -> str:
    """The text the leak guard looks for: the Findings body, else the first section's body."""
    sec = split_canonical(canonical_text)
    if 'Findings' in sec:
        return sec['Findings']
    for n in SECTION_NAMES:
        if n in sec:
            return sec[n]
    return (canonical_text or '').strip()


def is_leaked(canonical_text: str, prior_texts: Sequence[Any]) -> bool:
    body = _leak_norm(leak_body(canonical_text))
    if not body:
        return False
    for p in prior_texts or ():
        s = _content_text(p)
        if s and body in _leak_norm(s):
            return True
    return False


# ============================================================================================
# raw rows
# ============================================================================================

_ROLE_MAP = {'human': 'user', 'user': 'user', 'gpt': 'assistant', 'assistant': 'assistant', 'bot': 'assistant',
             'system': 'system', 'function_call': 'tool_call', 'tool_call': 'tool_call',
             'observation': 'tool', 'tool': 'tool', 'tool_response': 'tool', 'function': 'tool'}


def row_messages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """[{'role','content','loss'}] of a raw dataset row; system messages dropped; content kept
    as-is (str or list); loss = the row's per-message flag or None."""
    msgs = row.get('messages')
    if msgs is None:
        msgs = row.get('conversations')
    out = []
    for m in msgs or ():
        if not isinstance(m, dict):
            continue
        r = m.get('role')
        if r is None:
            r = m.get('from')
        role = _ROLE_MAP.get(str(r).lower(), str(r)) if r is not None else None
        if role == 'system' or role is None:
            continue
        content = m['content'] if 'content' in m else m.get('value')
        out.append({'role': role, 'content': content, 'loss': m.get('loss')})
    return out


def assistant_turn_groups(msgs: Sequence[Dict[str, Any]]) -> List[List[int]]:
    """Maximal runs of assistant/tool_call messages: the turns swift renders behind one
    '<|im_start|>assistant\\n' header each (tool_call -> assistant, consecutive assistant
    messages merged; a role-tool or user message ends the run)."""
    groups: List[List[int]] = []
    cur: List[int] = []
    for i, m in enumerate(msgs):
        if m.get('role') in ('assistant', 'tool_call'):
            cur.append(i)
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def iter_row_targets(row: Dict[str, Any]) -> List[Tuple[int, str, str]]:
    """(index in row_messages(row), family, canonical) over RAW assistant messages (+ explicit
    overrides, family 'override'). See the module header."""
    msgs = row_messages(row)
    out: List[Tuple[int, str, str]] = []
    last_user = None
    for i, m in enumerate(msgs):
        role = m['role']
        if role == 'user':
            last_user = m['content']
            continue
        if role != 'assistant':
            continue
        fam, can = _analyse(last_user if last_user is not None else '', m['content'])
        if fam is not None:
            out.append((i, fam, can))
    groups = None
    ov = row.get('sem_targets')
    if isinstance(ov, (list, tuple)):
        groups = assistant_turn_groups(msgs)
        for j, t in enumerate(ov):
            t = normalise_override(t)
            if t is not None:
                out.append((groups[j][-1] if j < len(groups) else -1, OVERRIDE_FAMILY, t))
    one = normalise_override(row.get('sem_target'))
    if one is not None:
        groups = groups if groups is not None else assistant_turn_groups(msgs)
        out.append((groups[-1][-1] if groups else -1, OVERRIDE_FAMILY, one))
    return out


# ============================================================================================
# sidecar
# ============================================================================================

SIDECAR_FILES = {
    'keys': 'keys.u64.npy', 'emb': 'emb.f16.npy', 'mu': 'mu.f32.npy', 'in_train': 'in_train.u8.npy',
    'families': 'families.u8.npy', 'probe': 'probe_logits.f16.npy', 'ntok': 'ntok.i32.npy',
    'texts': 'texts.jsonl', 'meta': 'meta.json',
}


def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    tmp = path + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def write_sidecar(out_dir: str, keys: np.ndarray, emb: np.ndarray, in_train: np.ndarray,
                  meta: Dict[str, Any], probe_logits: Optional[np.ndarray] = None,
                  families: Optional[np.ndarray] = None, ntok: Optional[np.ndarray] = None,
                  texts: Optional[Sequence[str]] = None, mu: Optional[np.ndarray] = None,
                  overwrite: bool = False, norm_tol: float = 2e-3) -> Dict[str, Any]:
    """Sort by key, dedup (duplicate keys must carry the same row: cos > 0.999, their
    in_train/families are OR-ed), check unit norms, compute mu over the unique rows (from
    the given precision, before the fp16 cast), write every array atomically and meta.json
    LAST. Returns the final meta."""
    meta_path = os.path.join(out_dir, SIDECAR_FILES['meta'])
    if os.path.exists(meta_path) and not overwrite:
        raise FileExistsError(f'[sem-targets] sidecar already complete: {meta_path} (overwrite=True to replace)')
    os.makedirs(out_dir, exist_ok=True)
    keys = np.asarray(keys, dtype=np.uint64).ravel()
    n0 = keys.shape[0]
    emb = np.asarray(emb)
    if emb.ndim != 2 or emb.shape[0] != n0:
        raise ValueError(f'[sem-targets] emb shape {emb.shape} vs {n0} keys')
    in_train = np.asarray(in_train).astype(np.uint8).ravel()
    fams = np.asarray(families).astype(np.uint8).ravel() if families is not None else None
    order = np.argsort(keys, kind='stable')
    keys = keys[order]
    uniq, first = np.unique(keys, return_index=True)
    if uniq.shape[0] != n0:                      # dedup: rows must agree, flags OR-ed
        e = emb[order].astype(np.float32)
        inv = np.searchsorted(uniq, keys)
        rep = e[first][inv]
        cos = (e * rep).sum(1) / np.maximum(np.linalg.norm(e, axis=1) * np.linalg.norm(rep, axis=1), 1e-12)
        if cos.min() < 0.999:
            raise ValueError(f'[sem-targets] duplicate key with different embeddings (min cos {cos.min():.5f})')
        tr = np.zeros(uniq.shape[0], np.uint8)
        np.maximum.at(tr, inv, in_train[order])
        in_train_s = tr
        if fams is not None:
            f = np.zeros(uniq.shape[0], np.uint8)
            np.bitwise_or.at(f, inv, fams[order])
            fams = f
        sel = order[first]
    else:
        sel = order
        in_train_s = in_train[order]
        if fams is not None:
            fams = fams[order]
    keys = uniq
    emb_s = emb[sel]
    n, dim = emb_s.shape
    e32 = emb_s.astype(np.float32)
    norms = np.linalg.norm(e32, axis=1) if n else np.zeros(0, np.float32)
    if n and (not np.isfinite(e32).all() or np.abs(norms - 1).max() > norm_tol):
        raise ValueError(f'[sem-targets] rows must be finite and L2-normalised (norm range '
                         f'{norms.min():.5f}..{norms.max():.5f})')
    if mu is None:
        mu = e32.astype(np.float64).mean(0) if n else np.zeros(dim)
    mu = np.asarray(mu, dtype=np.float32).ravel()
    del e32
    _atomic_save_npy(os.path.join(out_dir, SIDECAR_FILES['keys']), keys)
    _atomic_save_npy(os.path.join(out_dir, SIDECAR_FILES['emb']), emb_s.astype(np.float16))
    _atomic_save_npy(os.path.join(out_dir, SIDECAR_FILES['mu']), mu)
    _atomic_save_npy(os.path.join(out_dir, SIDECAR_FILES['in_train']), in_train_s.astype(np.uint8))
    if fams is not None:
        _atomic_save_npy(os.path.join(out_dir, SIDECAR_FILES['families']), fams.astype(np.uint8))
    if probe_logits is not None:
        _atomic_save_npy(os.path.join(out_dir, SIDECAR_FILES['probe']),
                         np.asarray(probe_logits)[sel].astype(np.float16))
    if ntok is not None:
        _atomic_save_npy(os.path.join(out_dir, SIDECAR_FILES['ntok']), np.asarray(ntok)[sel].astype(np.int32))
    if texts is not None:
        tp = os.path.join(out_dir, SIDECAR_FILES['texts'])
        with open(tp + '.tmp', 'w', encoding='utf-8') as f:
            for j, i in enumerate(sel):
                rec = {'k': str(int(keys[j])), 't': texts[i]}
                if fams is not None:
                    rec['f'] = int(fams[j])
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        os.replace(tp + '.tmp', tp)
    meta = dict(meta)
    meta.update({
        'format': SIDECAR_FORMAT, 'rule_version': meta.get('rule_version', RULE_VERSION), 'key_rule': KEY_RULE,
        'n': int(n), 'dim': int(dim), 'storage_dtype': 'float16', 'mu_norm': float(np.linalg.norm(mu)),
        'emb_norm_min': float(norms.min()) if n else None, 'emb_norm_max': float(norms.max()) if n else None,
        'n_in_train': int(in_train_s.sum()), 'has_probe_logits': probe_logits is not None,
        'has_texts': texts is not None, 'has_families': fams is not None, 'has_ntok': ntok is not None,
        'family_bits': FAMILY_BITS,
    })
    if fams is not None:
        meta['n_texts_per_family'] = {f: int(((fams & b) != 0).sum()) for f, b in FAMILY_BITS.items()}
    meta.setdefault('dtype', 'float16')      # teacher COMPUTE dtype when the builder sets it (bf16|fp32)
    meta.setdefault('created', time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    with open(meta_path + '.tmp', 'w') as f:
        json.dump(meta, f, indent=1)
    os.replace(meta_path + '.tmp', meta_path)
    return meta


class SidecarReader:
    """Memory-mapped teacher-target sidecar. lookup() -> row index (-1 = miss); rows() /
    probe_logits() / in_train() give zeros / False for -1. The embedding and probe matrices
    are ALWAYS memory-mapped (pages are shared across dataloader workers through the page
    cache); `mmap=False` only loads the small key / flag arrays into RAM."""

    def __init__(self, dir: str, mmap: bool = True):
        self.dir = dir
        self._mmap = mmap
        self._open()

    def _p(self, k: str) -> str:
        return os.path.join(self.dir, SIDECAR_FILES[k])

    def _open(self) -> None:
        mp = self._p('meta')
        if not os.path.isfile(mp):
            raise FileNotFoundError(f'[sem-targets] no complete sidecar at {self.dir} (meta.json missing)')
        with open(mp) as f:
            self.meta = json.load(f)
        mode = 'r' if self._mmap else None
        self.keys = np.load(self._p('keys'), mmap_mode=mode)
        self._emb = np.load(self._p('emb'), mmap_mode='r')
        self.mu = np.asarray(np.load(self._p('mu')), dtype=np.float32)
        self._in_train = np.load(self._p('in_train'), mmap_mode=mode)
        self._probe = np.load(self._p('probe'), mmap_mode='r') if os.path.isfile(self._p('probe')) else None
        self._fams = np.load(self._p('families'), mmap_mode=mode) if os.path.isfile(self._p('families')) else None
        self._ntok = np.load(self._p('ntok'), mmap_mode=mode) if os.path.isfile(self._p('ntok')) else None
        self._text_off = None
        n = int(self.keys.shape[0])
        if self.keys.dtype != np.uint64 or self._emb.dtype != np.float16 or self._emb.ndim != 2:
            raise ValueError(f'[sem-targets] bad sidecar dtypes keys={self.keys.dtype} emb={self._emb.dtype}')
        if self._emb.shape[0] != n or self._in_train.shape[0] != n or int(self.meta.get('n', n)) != n:
            raise ValueError(f'[sem-targets] inconsistent sidecar sizes at {self.dir}')
        if self.mu.shape != (self._emb.shape[1], ):
            raise ValueError(f'[sem-targets] mu shape {self.mu.shape} vs dim {self._emb.shape[1]}')
        if self._probe is not None and self._probe.shape[0] != n:
            raise ValueError('[sem-targets] probe_logits rows != n')
        if n > 1 and not bool(np.all(self.keys[1:] > self.keys[:-1])):
            raise ValueError(f'[sem-targets] keys not strictly increasing at {self.dir}')

    # pickling (spawned workers / pickled datasets) re-opens the maps instead of copying them
    def __getstate__(self):
        return {'dir': self.dir, '_mmap': self._mmap}

    def __setstate__(self, st):
        self.dir, self._mmap = st['dir'], st['_mmap']
        self._open()

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return (f'SidecarReader({self.dir!r}, n={self.n}, dim={self.dim}, rule={self.meta.get("rule_version")}, '
                f'probe={self._probe is not None})')

    @property
    def n(self) -> int:
        return int(self.keys.shape[0])

    @property
    def dim(self) -> int:
        return int(self._emb.shape[1])

    @property
    def emb(self) -> np.ndarray:
        """the memory-mapped (n, dim) float16 matrix (read-only)."""
        return self._emb

    @property
    def rule_version(self) -> Optional[str]:
        return self.meta.get('rule_version')

    def check_rule_version(self, expected: str = RULE_VERSION) -> None:
        if self.rule_version != expected:
            raise ValueError(f'[sem-targets] sidecar {self.dir} was built with rule_version '
                             f'{self.rule_version!r}, this code is {expected!r}: rebuild it')

    def lookup(self, keys: Any) -> np.ndarray:
        q = np.asarray(keys, dtype=np.uint64)
        flat = q.ravel()
        if self.n == 0 or flat.size == 0:
            return np.full(q.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self.keys, flat)
        pc = np.minimum(pos, self.n - 1)
        hit = (pos < self.n) & (np.asarray(self.keys[pc]) == flat)
        return np.where(hit, pc, -1).astype(np.int64).reshape(q.shape)

    def lookup_texts(self, texts: Iterable[str]) -> np.ndarray:
        return self.lookup(target_keys(texts))

    def _gather(self, arr: np.ndarray, idx: Any, dtype, width: Optional[int]) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64).ravel()
        shape = (idx.shape[0], width) if width is not None else (idx.shape[0], )
        out = np.zeros(shape, dtype=dtype)
        ok = idx >= 0
        if ok.any():
            sel = idx[ok]
            if (sel >= self.n).any():
                raise IndexError(f'[sem-targets] row index out of range (n={self.n})')
            order = np.argsort(sel, kind='stable')     # sequential page access on the map
            vals = np.asarray(arr[sel[order]])
            tmp = np.empty_like(vals)
            tmp[order] = vals
            out[ok] = tmp
        return out

    def rows(self, idx: Any) -> np.ndarray:
        return self._gather(self._emb, idx, np.float16, self.dim)

    def probe_logits(self, idx: Any) -> Optional[np.ndarray]:
        if self._probe is None:
            return None
        return self._gather(self._probe, idx, np.float16, int(self._probe.shape[1]))

    def in_train(self, idx: Any) -> np.ndarray:
        return self._gather(self._in_train, idx, np.uint8, None).astype(bool)

    def families(self, idx: Any) -> Optional[np.ndarray]:
        if self._fams is None:
            return None
        return self._gather(self._fams, idx, np.uint8, None)

    def ntok(self, idx: Any) -> Optional[np.ndarray]:
        if self._ntok is None:
            return None
        return self._gather(self._ntok, idx, np.int32, None)

    def texts(self, idx: Any) -> Optional[List[Optional[str]]]:
        """canonical texts (needs texts.jsonl; the line offsets are indexed on first use)."""
        tp = self._p('texts')
        if not os.path.isfile(tp):
            return None
        if self._text_off is None:
            offs = []
            with open(tp, 'rb') as f:
                pos = 0
                for line in f:
                    offs.append(pos)
                    pos += len(line)
            self._text_off = np.asarray(offs, dtype=np.int64)
        out: List[Optional[str]] = []
        with open(tp, 'rb') as f:
            for i in np.asarray(idx, dtype=np.int64).ravel():
                if i < 0:
                    out.append(None)
                    continue
                f.seek(int(self._text_off[i]))
                out.append(json.loads(f.readline())['t'])
        return out
