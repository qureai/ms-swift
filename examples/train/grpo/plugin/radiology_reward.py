"""
Radiology structured-report + grounding reward for GRPO (ms-swift).
=====================================================================

A single composite ORM (`radiology_report`). The embedding model is the MAIN,
densest signal; the rule-based terms (cls / bbox / token) shape specifics.

Reward components (v2 design)
-----------------------------
  1. CLS      -> _cls_reward
                 EXACT-match TP/FP/FN over a fixed abnormality vocabulary
                 (NO hierarchical / soft-positive mapping -- resemble GT exactly).
                 reward = w_youden * Youden(J = sens + spec - 1)
                        - w_hamming * asymmetric_Hamming
                 asymmetry: lambda_fn > lambda_fp (miss hurts more, but FP is NOT free).
  2. BBOX     -> _bbox_reward
                 GT box present  -> DIoU localization (see DIoU-vs-Dice note below).
                 GT box absent   -> align box LABELS with GT cls via precision/recall.
                 + anti-degeneracy (tiny / whole-image / duplicate boxes).
  3. TOKEN    -> _token_reward
                 Weighted bag-of-tokens F1 vs GT report text; tokens of GT-present
                 findings are up-weighted (positive findings matter more).
  4. EMBEDDING (main) -> _embedding_sections + _retrieval_margins + group z-scoring
                 (a) Encode findings / impression / reasoning SEPARATELY against GT and
                     reward the group-relative rank (absolute cosine ignored).
                 (b) NEW retrieval: reward a rollout whose findings retrieve ITS OWN
                     study's GT above other studies' GTs in the batch (R@1 -- the ranker's
                     real strength), not an absolute score.
  5. CONSISTENCY (NEW) -> _consistency_reward
                 Impression must not assert an abnormality absent from the findings
                 (rule-based; catches "conclusion invents a finding"; hard to hack).
  6. FORMAT   -> _format_penalty   (fence-stripped + parseable; bonus if clean, floor if not)
  7. LENGTH   -> _length_penalty   (mild, one-sided, on finding count)

JSON robustness: ```json ... ``` fences are stripped before parsing, and the parser
falls back to the outermost {...} block, so a well-formed report wrapped in a code
fence is not spuriously floored.

DIoU vs Dice (boxes)
--------------------
DIoU is used, not Dice/IoU. Dice and IoU are monotonic reparametrizations and are
BOTH zero when boxes are disjoint -> no gradient for a far-off box (fatal for small
findings). DIoU adds a center-distance term that stays informative even with zero
overlap, giving a monotone pull toward the target everywhere. Dice is a segmentation
(mask) metric; only revisit soft-Dice if grounding moves from boxes to masks.

Assumed model / GT output schema (tolerant parser; adjust helpers if yours differs)
-----------------------------------------------------------------------------------
    {
      "reasoning": "....",                         # optional (or <think>...</think>)
      "findings":  "free text ...",                # optional; else rendered from tags
      "impression":"free text ...",                # optional (aka summary/conclusion)
      "general":   {"pneumothorax": {"presence": true, "location": "right apex",
                                     "bbox": [x1,y1,x2,y2]}, ...},   # bbox optional, [0,1]
      "edema": {...}, "copd": {...}, "ild": {...},
      "pulmonary_hypertension": {...}, "lung_cancer": {...}
      # "other_findings" is intentionally DROPPED from the reward
    }

Ground truth
------------
The GT source of truth is the pre-extracted `reference` column (normalized finding_set,
`prominence` {finding: 1..5}, absent_set, and section texts) built at data-creation time.
This is what activates the PROMINENCE-WEIGHTED cls match (missing a prominence-5 finding
costs far more than a prominence-2). If `reference` is absent, the reward falls back to
parsing the raw `solution` text (uniform weights). `solution` is otherwise kept only as
verbatim text for the token/embedding terms.

Normalization
-------------
EVERY sub-reward returns a value in [0, 1] (1 = perfect), the same scale as the
embedding cosine mapped via (cos+1)/2. The final reward is a convex combination of
the ACTIVE sub-rewards, so it is also in [0, 1]. Unparseable JSON scores 0.

Embedding model
---------------
Path comes from the swift arg `--reward_embedding_model <path>` (added to GRPOArguments)
or the env var RADREWARD_EMBED_MODEL. If neither is set, the embedding sub-reward is
disabled and the remaining terms are renormalized. Loaded with SentenceTransformers
(bfloat16), encoded with normalize_embeddings=True (cosine == dot product).

Usage
-----
    swift rlhf --rlhf_type grpo \
      --external_plugins examples/train/grpo/plugin/radiology_reward.py \
      --reward_funcs radiology_report \
      --reward_embedding_model /path/to/checkpoint-7000 ...

Environment overrides
---------------------
    RADREWARD_EMBED_MODEL   fallback for --reward_embedding_model
    RADREWARD_VOCAB         JSON list of canonical abnormality labels (CLS label space)
    RADREWARD_GT_KEY        dataset column holding the GT report JSON (default 'solution')
    RADREWARD_ENABLE_BBOX   '1' to enable the bbox head (default '1')
"""
import ast
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from swift.rewards import ORM, orms
from swift.utils import get_logger

logger = get_logger()


# ----------------------------------------------------------------------------- #
#  Configuration.
# ----------------------------------------------------------------------------- #
@dataclass
class RewardConfig:
    """Every sub-reward returns a value in [0, 1] (1 = perfect), same scale as the
    embedding cosine mapped to [0,1]. The composite is a convex combination of the
    ACTIVE sub-rewards, so the total is also in [0, 1]. Outer `w_*` are the mixing
    weights (need not sum to 1 -- they are renormalized over whatever is active)."""

    # --- outer mixing weights (embedding is the main signal) ---
    w_cls: float = 0.25
    w_bbox: float = 0.15
    w_token: float = 0.10
    w_emb: float = 0.35              # BASE embedding weight (see abnormality scaling below)
    w_consistency: float = 0.10
    w_format: float = 0.03
    w_length: float = 0.02

    # embedding weight scales UP with the number of GT abnormalities: richer reports ->
    # the embedding term (which captures graded content) matters more; near-normal reports
    # (few findings, compressed cosine band) -> it matters less.
    #   w_emb_eff = w_emb * (1 + emb_abn_gain * min(n_abn, emb_abn_cap) / emb_abn_cap)
    emb_abn_gain: float = 1.0        # 0 -> constant weight; 1.0 -> doubles at/above the cap
    emb_abn_cap: int = 6             # saturating number of abnormalities

    # --- CLS internals (all fold into a [0,1] convex combo) ---
    cls_w_sens: float = 0.40         # recall (penalizes FN) -- weighted highest => FN>FP asymmetry
    cls_w_spec: float = 0.20         # specificity (penalizes FP) -> Youden = sens+spec-1
    cls_w_overlap: float = 0.40      # asymmetric Tversky overlap (Hamming analog, discriminative)
    tversky_fn: float = 1.5          # FN weight in overlap (heavier)
    tversky_fp: float = 1.0          # FP weight in overlap (not free)
    prom_default: float = 0.6        # weight for findings with no GT prominence (cls tasks)

    # --- BBOX internals (convex -> [0,1]) ---
    enable_bbox: bool = True
    box_w_loc: float = 0.5           # DIoU localization vs GT box (where annotated)
    box_w_align: float = 0.4         # label precision/recall vs GT cls
    box_w_degen: float = 0.1         # (1 - degeneracy) reward
    box_align_beta: float = 2.0      # F-beta; >1 favors recall
    box_area_min: float = 5e-4
    box_area_max: float = 0.5
    box_dup_diou: float = 0.9

    # --- TOKEN (task-aware; see _token_reward) ---
    # cls_tokens: each abnormality token weighted by prominence; cls_json: the presence
    # BOOLEAN per finding weighted by prominence; other tasks: plain unweighted word F1.
    token_gain: float = 2.0          # prominence weight: 1 + token_gain*(prominence/5)
    token_default: float = 1.0       # weight for findings with no prominence

    # --- EMBEDDING (cosine -> (cos+1)/2 in [0,1]; convex over available sections) ---
    emb_w_findings: float = 0.45
    emb_w_impression: float = 0.35
    emb_w_reasoning: float = 0.20

    # --- FORMAT / LENGTH (as [0,1] goodness) ---
    lambda_fmt: float = 0.10         # per-violation drop from 1.0
    len_alpha: float = 0.25          # tolerance band
    len_slope: float = 0.5           # drop per unit of relative over-length

    # unparseable JSON -> whole reward = 0.0 (the floor, since rewards are in [0,1])
    reward_floor: float = 0.0

    # === constrained-RL additions (all OFF by default -> identical to legacy behaviour) ======= #
    # Presence as (fixed-lambda) constraint terms, added to the convex combo. Set >0 to activate.
    # Structured as fixed alpha now; a dual-ascent updater can later overwrite w_recall/w_precision.
    w_recall: float = 0.0            # lambda_R * prominence-weighted recall R_w (completeness)
    w_precision: float = 0.0         # lambda_P * precision P (anti-hallucination)
    # Token-probability calibration reward (cls_json only; needs response_token_ids+rollout_logprobs).
    cal_mode: str = 'none'           # 'none' | 'brier'
    w_cal: float = 0.0               # weight of the calibration term in the combo
    cal_true_weight: float = 6.0     # w: weight of TRUE (present) tokens vs 1.0 for FALSE in the Brier
    # Attribute agreement via the embedding model (per matched present tag, only where BOTH sides
    # have a non-null attribute value). Needs GT attributes in reference['attributes'].
    w_attr: float = 0.0
    # Embedding reliability weight beta(n)=n/(n+beta_k); beta_k<=0 disables it (beta=1) and keeps the
    # legacy abnormality-count scaling. When enabled, beta(n) REPLACES that scaling.
    beta_k: float = 0.0
    emb_baseline: float = 0.0        # S_used=max(S,emb_baseline) when P==R==1 (benign-dip cap); 0=off

    # --- dual-ascent (PPO-Lagrangian) for the recall/precision constraints -------------------------
    # off (default) => fixed-lambda scalarization (w_recall/w_precision are the prices). On => lambda_R,
    # lambda_P become dual variables updated each batch: lam += dual_lr*(target - measured), clipped.
    dual_ascent: bool = False
    recall_target: float = 0.85
    precision_target: float = 0.85
    dual_lr: float = 0.05
    lambda_min: float = 0.0
    lambda_max: float = 1.0

    # --- subtype partial credit -------------------------------------------------------------------
    # alpha in [0,1]: a predicted tag in the SAME clinical family as a GT tag (but not exact) scores
    # `subtype_credit`*weight on recall/Tversky instead of 0 (a miss) + is a reduced (alpha) FP instead
    # of a full one. 0 (default) = exact-match only (legacy). Family map from RADREWARD_FAMILIES json.
    subtype_credit: float = 0.0

    # --- group-relative embedding ------------------------------------------------------------------
    # When on, each rollout's embedding sub-reward is re-expressed as a WITHIN-GROUP z-score (rollouts
    # sharing a prompt = same reference+task). GSPO extracts gradient from within-group variance, so an
    # absolute embedding score that is ~flat across the group (e.g. all ~0.80) yields no gradient; the
    # z-score restores it by rewarding the group's best report. Off (default) = absolute (cos+1)/2.
    emb_group_relative: bool = False

    # --- grouping ---
    num_generations: int = 8

    def __post_init__(self):
        """Override the constrained-RL additions from env (RADREWARD_*), so they are CLI/config
        tunable without touching swift-core args. Legacy fields keep their dataclass defaults."""
        import os as _os

        def _f(name, cur):
            v = _os.getenv(name)
            return float(v) if v not in (None, '') else cur
        self.w_cls = _f('RADREWARD_W_CLS', self.w_cls)          # lower this when enabling w_recall/w_precision
        self.w_recall = _f('RADREWARD_W_RECALL', self.w_recall)
        self.w_precision = _f('RADREWARD_W_PRECISION', self.w_precision)
        self.w_emb = _f('RADREWARD_W_EMB', self.w_emb)         # base embedding weight (env-tunable)
        self.subtype_credit = _f('RADREWARD_SUBTYPE_CREDIT', self.subtype_credit)  # family partial credit alpha
        self.emb_group_relative = _os.getenv('RADREWARD_EMB_GROUPREL',
                                             '1' if self.emb_group_relative else '0') == '1'
        self.w_cal = _f('RADREWARD_W_CAL', self.w_cal)
        self.cal_true_weight = _f('RADREWARD_TRUE_WEIGHT', self.cal_true_weight)
        self.w_attr = _f('RADREWARD_W_ATTR', self.w_attr)
        self.beta_k = _f('RADREWARD_BETA_K', self.beta_k)
        self.emb_baseline = _f('RADREWARD_EMB_BASELINE', self.emb_baseline)
        self.cal_mode = _os.getenv('RADREWARD_CAL_MODE', self.cal_mode)
        # cls recall knobs (1_oct: raise to fix the cls_tokens recall collapse -- heavier FN penalty
        # and a stronger sensitivity weight so under-calling findings costs reward).
        self.tversky_fn = _f('RADREWARD_TVERSKY_FN', self.tversky_fn)
        self.tversky_fp = _f('RADREWARD_TVERSKY_FP', self.tversky_fp)
        self.cls_w_sens = _f('RADREWARD_CLS_W_SENS', self.cls_w_sens)
        self.cls_w_spec = _f('RADREWARD_CLS_W_SPEC', self.cls_w_spec)
        # dual-ascent (PPO-Lagrangian) for recall/precision: env-gated. When on, w_recall/w_precision
        # become the INITIAL dual variables and are then updated each batch toward the targets.
        self.dual_ascent = _os.getenv('RADREWARD_DUAL_ASCENT', '1' if self.dual_ascent else '0') == '1'
        self.recall_target = _f('RADREWARD_RECALL_TARGET', self.recall_target)
        self.precision_target = _f('RADREWARD_PRECISION_TARGET', self.precision_target)
        self.dual_lr = _f('RADREWARD_DUAL_LR', self.dual_lr)
        self.lambda_min = _f('RADREWARD_LAMBDA_MIN', self.lambda_min)
        self.lambda_max = _f('RADREWARD_LAMBDA_MAX', self.lambda_max)


# fixed CLS label space (needed for specificity / Hamming denominators). Override via
# RADREWARD_VOCAB. Keep this to true abnormalities (devices/normal-variants excluded)
# so specificity actually has teeth instead of saturating at ~1.
_DEFAULT_VOCAB: List[str] = [
    'nodules', 'solitary_nodules', 'tiny_nodules', 'diffused_nodules', 'peripheral_nodules',
    'cavitary_nodules', 'calcified_nodules', 'mass', 'mediastinal_mass',
    'consolidation', 'pneumonia', 'atelectasis', 'perihilar_opacities',
    'effusion', 'pleural_effusion', 'hemothorax', 'pleural_thickening', 'pneumothorax',
    'edema', 'pulmonary_edema', 'fibrosis', 'diffuse_fibrosis', 'reticular_opacities',
    'interstitial_thickening', 'emphysema', 'copd', 'hyperinflation', 'bronchitis',
    'cardiomegaly', 'pericardial_effusion', 'mediastinal_widening', 'hilar_enlargement',
    'hilar_lymphadenopathy', 'aortic_aneurysm', 'rib_fractures', 'healed_rib_fracture',
    'clavicle_fracture', 'scoliosis', 'kyphosis', 'calcification', 'granuloma',
    'diaphragm_elevation',
]

SCORED_BUCKETS = {'general', 'edema', 'copd', 'ild', 'pulmonary_hypertension', 'lung_cancer'}
DROPPED_BUCKETS = {'other_findings'}
ATTR_KEYS = {'presence', 'size', 'location', 'characteristics', 'bbox', 'prominence'}
FINDINGS_KEYS = ('findings',)
IMPRESSION_KEYS = ('impression', 'summary', 'conclusion')
REASONING_KEYS = ('reasoning', 'thinking', 'rationale', 'analysis')

_TOKEN_RE = re.compile(r'[a-z0-9]+')


def _norm_tag(name: str) -> str:
    """Canonicalize a raw tag string to a lookup key (case/space/punct only -- NO synonyms)."""
    return re.sub(r'[^a-z0-9]+', '_', str(name).strip().lower()).strip('_')


def _tokens(text: Optional[str]) -> Set[str]:
    return set(_TOKEN_RE.findall(text.lower())) if text else set()


def load_vocab() -> Set[str]:
    path = os.getenv('RADREWARD_VOCAB')
    if path and os.path.exists(path):
        with open(path) as f:
            return {_norm_tag(x) for x in json.load(f)}
    logger.warning('RADREWARD_VOCAB not set; using the built-in default abnormality vocabulary. '
                   'Set it to your real CLS label space so specificity/Hamming are correct.')
    return {_norm_tag(x) for x in _DEFAULT_VOCAB}


def load_families() -> Dict[str, frozenset]:
    """{normalized_tag: frozenset(families)} for subtype partial credit. A tag may be in MULTIPLE families
    (e.g. mediastinal_mass -> {mass, mediastinal}); two tags are related if their family sets intersect.
    Accepts json values that are a list of families or a single string. From RADREWARD_FAMILIES, else {}."""
    path = os.getenv('RADREWARD_FAMILIES')
    if path and os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        out = {}
        for k, v in raw.items():
            fams = v if isinstance(v, (list, tuple, set)) else [v]
            out[_norm_tag(k)] = frozenset(str(x) for x in fams)
        return out
    return {}


# ----------------------------------------------------------------------------- #
#  Embedding model wrapper (main contribution). Lazy, cached, inference-only.
# ----------------------------------------------------------------------------- #
class EmbeddingScorer:
    """Fine-tuned radiology encoder; L2-normalized encodes memoized by text.

    Loaded exactly as in the report_embedding repo:
        SentenceTransformer(path, device, model_kwargs={'torch_dtype': 'bfloat16'})
        .encode(..., normalize_embeddings=True)   # cosine == dot product
    """

    def __init__(self, path: str):
        self.path = path
        self._model = None
        self._cache: Dict[str, object] = {}
        # Fail LOUD (not silently) if the loader is missing: without sentence_transformers the
        # embedding sub-reward + A_attr are disabled (cosine_pairs -> []), which quietly changes the
        # objective. Warn once at construction so a fresh env can't hide it. See RL.md 1 / 3.
        try:
            import sentence_transformers  # noqa: F401
        except Exception as e:  # noqa
            logger.error('EmbeddingScorer: `sentence_transformers` is NOT importable (%s) -> the '
                         'embedding sub-reward AND attribute agreement (A_attr) will be DISABLED for '
                         '`%s`. Install it:  pip install --no-deps sentence-transformers', e, path)

    def _lazy(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._model = SentenceTransformer(
                self.path, device=device, model_kwargs={'torch_dtype': 'bfloat16'})
            self._model.eval()
        return self._model

    def encode(self, texts: List[str]):
        import numpy as np
        model = self._lazy()
        missing = [t for t in texts if t not in self._cache]
        if missing:
            import torch
            with torch.inference_mode():
                embs = model.encode(missing, normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False)
            for t, e in zip(missing, embs):
                self._cache[t] = e
        return np.stack([self._cache[t] for t in texts])

    def cosine_pairs(self, a: List[str], b: List[str]) -> List[float]:
        """cos(a[i], b[i]) for aligned lists; returns [] on failure (term disabled)."""
        try:
            import numpy as np
            uniq = list({t for t in (a + b)})
            emb = {t: e for t, e in zip(uniq, self.encode(uniq))}
            return [float(np.dot(emb[x], emb[y])) for x, y in zip(a, b)]
        except Exception as e:
            logger.warning(f'embedding term disabled: {e}')
            return []

    def matrix(self, queries: List[str], keys: List[str]) -> List[List[float]]:
        """Full cosine matrix sims[i][j] = cos(queries[i], keys[j]); [] on failure."""
        try:
            import numpy as np
            uniq = list({t for t in (queries + keys)})
            emb = {t: e for t, e in zip(uniq, self.encode(uniq))}
            q = np.stack([emb[t] for t in queries])
            k = np.stack([emb[t] for t in keys])
            return (q @ k.T).tolist()
        except Exception as e:
            logger.warning(f'retrieval term disabled: {e}')
            return []


# ----------------------------------------------------------------------------- #
#  Geometry — DIoU.
# ----------------------------------------------------------------------------- #
def diou(a: List[float], b: List[float]) -> float:
    """Distance-IoU for normalized xyxy boxes. Range (-1, 1]; informative even at 0 overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    iou = inter / union if union > 0 else 0.0
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    center_d2 = (acx - bcx) ** 2 + (acy - bcy) ** 2
    diag2 = (max(ax2, bx2) - min(ax1, bx1)) ** 2 + (max(ay2, by2) - min(ay1, by1)) ** 2
    return iou if diag2 <= 0 else iou - center_d2 / diag2


def _loc_reward(d: float) -> float:
    """DIoU (-1, 1] -> [0, 1] monotone localization credit (perfect=1)."""
    return max(0.0, min(1.0, (d + 1.0) / 2.0))


def _valid_box(b) -> bool:
    return (isinstance(b, (list, tuple)) and len(b) == 4
            and all(isinstance(v, (int, float)) for v in b)
            and all(-0.01 <= v <= 1.01 for v in b))


def _area(b: List[float]) -> float:
    return abs(b[2] - b[0]) * abs(b[3] - b[1])


# ----------------------------------------------------------------------------- #
#  Parsing.
# ----------------------------------------------------------------------------- #
@dataclass
class Finding:
    label: str                         # canonical (normalized) name
    bbox: Optional[List[float]] = None
    location: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)   # {size, location, characteristics} raw values


@dataclass
class ParsedReport:
    ok: bool
    present: Set[str] = field(default_factory=set)     # canonical present labels
    findings: List[Finding] = field(default_factory=list)
    findings_text: Optional[str] = None
    impression_text: Optional[str] = None
    reasoning_text: Optional[str] = None
    prominence: Dict[str, float] = field(default_factory=dict)   # {finding: 1..5} (GT only)
    absent: Set[str] = field(default_factory=set)               # confirmed-absent labels (GT only)
    presence_map: Dict[str, bool] = field(default_factory=dict)  # {finding: presence bool}
    attrs_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {finding: {size,location,characteristics}}
    n_violations: int = 0
    raw: str = ''


_CLS_TOKEN_RE = re.compile(r'<([A-Za-z_][A-Za-z0-9_]*)>')
# ASP block markers in every form seen in rollouts: special tokens |<ASP>| / |</ASP>|, the legacy
# string <|ASP|> (and </|ASP|>), and the malformed close <|/ASP>| that SFT sometimes writes. The
# special-token form contains '<ASP>', which _CLS_TOKEN_RE would otherwise read as a finding 'asp'.
_ASP_MARK_RE = re.compile(r'\|</?ASP>\||</?\|ASP\|>|<\|/ASP>\|')
_ASP_OPEN_ANY_RE = re.compile(r'\|<ASP>\||<\|ASP\|>')
_ASP_ANY_CLOSE_RE = re.compile(r'\|</ASP>\||</\|ASP\|>|<\|/ASP>\|')
_CLS_TAG_UPPER_RE = re.compile(r'<[A-Z][A-Z0-9_]*>')          # a real cls_tokens tag, e.g. <PLEURAL_EFFUSION>
_JSON_START_RE = re.compile(r'\n\s*\{\s*"')


def has_cls_tag_tokens(text: str) -> bool:
    """True if `text` contains at least one cls_tokens tag (<UPPER_SNAKE>), ignoring ASP markers.
    False for an empty answer, an ASP-only answer, or an ontology/JSON report written instead of tags."""
    return bool(_CLS_TAG_UPPER_RE.search(_ASP_MARK_RE.sub(' ', text or '')))


def asp_block_span(text: str) -> Optional[Tuple[int, int]]:
    """(start, end) character span of the ASP block, or None if the text has no ASP opening marker.
    Ends at the closing marker when present; otherwise (legacy <|ASP|> has no close, or the block was
    truncated) at the first cls_tokens tag or JSON object after the opening, else end of text."""
    t = text or ''
    mo = _ASP_OPEN_ANY_RE.search(t)
    if not mo:
        return None
    ends = [len(t)]
    for rx in (_ASP_ANY_CLOSE_RE, _CLS_TAG_UPPER_RE, _JSON_START_RE):
        m = rx.search(t, mo.end())
        if m:
            ends.append(m.end() if rx is _ASP_ANY_CLOSE_RE else m.start())
    return mo.start(), min(ends)


def parse_cls_tokens(text: str) -> ParsedReport:
    """Parse the cls_tokens output `<PLEURAL_EFFUSION> <CONSOLIDATION> ...` into a report.
    ASP markers are stripped first so the |<ASP>| special token is not counted as a finding."""
    labels = {_norm_tag(t) for t in _CLS_TOKEN_RE.findall(_ASP_MARK_RE.sub(' ', text or ''))}
    findings = [Finding(label=l) for l in sorted(labels)]
    return ParsedReport(ok=isinstance(text, str), present=set(labels), findings=findings,
                        findings_text=_render_findings(findings),
                        presence_map={l: True for l in labels})


def gt_from_reference(ref_json: str, vocab: Set[str]) -> Optional[ParsedReport]:
    """Build the GT ParsedReport from the pre-extracted `reference` column (the source
    of truth: normalized finding_set, prominence, absent_set, section texts). Returns
    None if the reference is empty/unusable so the caller can fall back to `solution`."""
    try:
        ref = json.loads(ref_json) if isinstance(ref_json, str) else (ref_json or {})
    except Exception:
        return None
    if not isinstance(ref, dict) or 'finding_set' not in ref:
        return None
    present = {_norm_tag(x) for x in ref.get('finding_set', [])}
    prom = {_norm_tag(k): float(v) for k, v in (ref.get('prominence') or {}).items()
            if isinstance(v, (int, float))}
    absent = {_norm_tag(x) for x in ref.get('absent_set', [])}
    findings = [Finding(label=f) for f in sorted(present)]
    findings_text = ref.get('findings_text') or _render_findings(findings)
    presence_map = {f: True for f in present}
    presence_map.update({a: False for a in absent})
    # GT per-tag attributes for the attribute-agreement term (A_attr). Populated by the consolidated
    # cls->report data build as reference['attributes'] = {tag: {size, location, characteristics}}.
    attrs_map = {}
    for k, v in (ref.get('attributes') or {}).items():
        if isinstance(v, dict):
            attrs_map[_norm_tag(k)] = {a: v.get(a) for a in ('size', 'location', 'characteristics')}
    return ParsedReport(ok=True, present=present, findings=findings,
                        findings_text=findings_text,
                        impression_text=ref.get('impression_text') or None,
                        reasoning_text=ref.get('reasoning_text') or None,
                        prominence=prom, absent=absent, presence_map=presence_map, attrs_map=attrs_map)


_FENCE_RE = re.compile(r'```(?:json)?\s*(.*?)```', re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Remove a ```json ... ``` (or bare ``` ... ```) wrapper if present."""
    if not text:
        return text
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # dangling fences with no closing marker
    return re.sub(r'```(?:json)?', '', text, flags=re.IGNORECASE).strip()


def _extract_json(text: str) -> Optional[dict]:
    """Parse the report/cls JSON, tolerating code fences, surrounding prose, and
    single-quote Python-dict repr (json.loads -> ast.literal_eval fallback)."""
    if not text:
        return None
    t = _strip_fences(text)
    s, e = t.find('{'), t.rfind('}')
    candidates = [t]
    if s != -1 and e > s:
        candidates.append(t[s:e + 1])
    for cand in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return None


def _get_text(obj: dict, keys) -> Optional[str]:
    for kk in obj:
        if _norm_tag(kk) in {_norm_tag(k) for k in keys}:
            v = obj[kk]
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _render_findings(findings: List[Finding]) -> str:
    if not findings:
        return 'no acute cardiopulmonary findings'
    parts = []
    for f in sorted(findings, key=lambda x: x.label):
        s = f.label.replace('_', ' ')
        if f.location:
            s += f' in the {f.location}'
        parts.append(s)
    return 'findings: ' + '; '.join(parts)


def parse_report(text: str, vocab: Set[str]) -> ParsedReport:
    """Parse a model/GT report into present labels, boxes, and section texts.

    Drops `other_findings`. NO synonym/hierarchy collapse -- labels are exact
    (normalized) strings. ok=False only when the JSON is unparseable (-> reward floor).
    """
    obj = _extract_json(text)
    # reasoning can live in a <think> block outside the JSON
    reasoning = None
    mth = re.search(r'<think>(.*?)</think>', text or '', re.DOTALL)
    if mth:
        reasoning = mth.group(1).strip()

    if obj is None or not isinstance(obj, dict):
        return ParsedReport(ok=False, reasoning_text=reasoning, raw=text or '')

    violations = 0
    by_label: Dict[str, Finding] = {}
    presence_map: Dict[str, bool] = {}
    for bucket, findings in obj.items():
        bkey = _norm_tag(bucket)
        if bkey in DROPPED_BUCKETS or bkey in {_norm_tag(k) for k in
                                               FINDINGS_KEYS + IMPRESSION_KEYS + REASONING_KEYS}:
            continue
        if not isinstance(findings, dict):
            continue                                    # free-text section, handled below
        if bkey not in SCORED_BUCKETS:
            violations += 1                             # unknown bucket, still permissive
        for name, attrs in findings.items():
            if not isinstance(attrs, dict):
                violations += 1
                continue
            presence_map[_norm_tag(name)] = attrs.get('presence', True) is not False
            if attrs.get('presence', True) is False:
                continue
            for k in attrs:
                if _norm_tag(k) not in {_norm_tag(x) for x in ATTR_KEYS}:
                    violations += 1
            label = _norm_tag(name)
            bbox = attrs.get('bbox')
            if bbox is not None and not _valid_box(bbox):
                violations += 1
                bbox = None
            f = Finding(label=label, bbox=[float(v) for v in bbox] if bbox else None,
                        location=attrs.get('location'),
                        attrs={k: attrs.get(k) for k in ('size', 'location', 'characteristics')})
            prev = by_label.get(label)
            if prev is None:
                by_label[label] = f
            elif prev.bbox is None and f.bbox is not None:
                prev.bbox = f.bbox

    findings = list(by_label.values())
    findings_text = _get_text(obj, FINDINGS_KEYS) or _render_findings(findings)
    impression_text = _get_text(obj, IMPRESSION_KEYS)
    # reasoning: primarily the ontology-report "reasoning" key (same as findings/impression);
    # fall back to a <think>...</think> block if present.
    reasoning_text = _get_text(obj, REASONING_KEYS) or reasoning
    attrs_map = {f.label: f.attrs for f in findings if f.attrs}
    return ParsedReport(ok=True, present={f.label for f in findings}, findings=findings,
                        findings_text=findings_text, impression_text=impression_text,
                        reasoning_text=reasoning_text, presence_map=presence_map,
                        attrs_map=attrs_map, n_violations=violations, raw=text or '')


# ----------------------------------------------------------------------------- #
#  Constrained-RL helpers: presence P/R, token-prob Brier, attribute agreement, beta(n).
# ----------------------------------------------------------------------------- #
def _beta(n: int, k: float) -> float:
    """Embedding reliability weight beta(n)=n/(n+k). k<=0 -> disabled (returns 1.0)."""
    if k <= 0:
        return 1.0
    denom = float(n) + k
    return (float(n) / denom) if denom > 0 else 0.0


def _precision(pred_present: Set[str], gt_present: Set[str]) -> float:
    """|TP|/|A|. Empty prediction is NOT free precision: it scores 1.0 only when GT is also empty (a
    truly normal study -- predicting nothing is correct), else 0.0. Returning 1.0 for an empty pred on
    an ABNORMAL study made "predict all-normal" a reward-hacking optimum (free P=1.0) that collapsed the
    policy to normal; this keeps precision from rewarding wholesale under-calling."""
    if not pred_present:
        return 1.0 if not gt_present else 0.0
    return len(pred_present & gt_present) / len(pred_present)


_BOOL_ID_CACHE: Dict[int, Tuple[frozenset, frozenset]] = {}


def _bool_token_id_sets(tokenizer) -> Tuple[frozenset, frozenset]:
    """(true_ids, false_ids): single-token ids whose text is a JSON boolean 'true'/'false' variant."""
    key = id(tokenizer)
    if key in _BOOL_ID_CACHE:
        return _BOOL_ID_CACHE[key]
    t_ids, f_ids = set(), set()
    for variants, bucket in ((('true', ' true', 'True', ' True'), t_ids),
                             (('false', ' false', 'False', ' False'), f_ids)):
        for s in variants:
            try:
                ids = tokenizer.encode(s, add_special_tokens=False)
            except Exception:
                continue
            if len(ids) == 1:
                bucket.add(ids[0])
    res = (frozenset(t_ids), frozenset(f_ids))
    _BOOL_ID_CACHE[key] = res
    return res


def _ordered_cls_tags(text: str) -> Optional[List[str]]:
    """Normalized tag labels of a cls-json completion in document order (one per presence slot)."""
    obj = _extract_json(text)
    if not isinstance(obj, dict):
        return None
    tags: List[str] = []
    for bucket, findings in obj.items():
        if _norm_tag(bucket) in DROPPED_BUCKETS or not isinstance(findings, dict):
            continue
        for name, attrs in findings.items():
            if isinstance(attrs, dict) and 'presence' in attrs:
                tags.append(_norm_tag(name))
    return tags


def _weighted_brier(pred_text: str, token_ids, logprobs, true_ids: frozenset, false_ids: frozenset,
                    gt_present: Set[str], true_weight: float) -> Optional[float]:
    """Weighted-Brier calibration reward in [0,1] from the presence-token probabilities.

    For each cls-json tag: p_true = P('true') = exp(lp) if a 'true' token was sampled at that slot,
    else 1-exp(lp) (the presence slot is ~always true/false, so the complement is tight); y = GT
    presence. reward = 1 - [ sum_t v_t (p_true - y)^2 / sum_t v_t ], with v_t = true_weight for
    present (y=1) tags and 1 for absent (y=0) tags -> present findings dominate the score. Returns
    None (term skipped, combo renormalizes) if token data is missing or misaligned with the tags."""
    if not token_ids or not logprobs or len(token_ids) != len(logprobs):
        return None
    tags = _ordered_cls_tags(pred_text)
    if not tags:
        return None
    seq = []
    for tid, lp in zip(token_ids, logprobs):
        if tid in true_ids:
            seq.append((True, math.exp(lp)))
        elif tid in false_ids:
            seq.append((False, math.exp(lp)))
    if len(seq) != len(tags):                 # stray true/false or malformed json -> skip
        return None
    num = den = 0.0
    for tag, (is_true, p_s) in zip(tags, seq):
        p_true = p_s if is_true else (1.0 - p_s)
        y = 1.0 if tag in gt_present else 0.0
        v = true_weight if y == 1.0 else 1.0
        num += v * (p_true - y) ** 2
        den += v
    return (1.0 - num / den) if den > 0 else None


def _attr_agreement(pred: ParsedReport, gt: ParsedReport, embedder) -> Optional[float]:
    """Embedding attribute agreement on matched present tags, ONLY where BOTH sides give a non-null
    value for that axis (size/location/characteristics). Mean of (cos+1)/2 over the comparable
    (tag, axis) pairs; None if there are none (term skipped). GT-silent/pred-only -> not compared."""
    if embedder is None:
        return None
    a_list, b_list = [], []
    for tag in (pred.present & gt.present):
        pa = pred.attrs_map.get(tag) or {}
        ga = gt.attrs_map.get(tag) or {}
        for axis in ('size', 'location', 'characteristics'):
            pv, gv = pa.get(axis), ga.get(axis)
            if pv not in (None, '', 'null') and gv not in (None, '', 'null'):
                a_list.append(str(pv)[:200])
                b_list.append(str(gv)[:200])
    if not a_list:
        return None
    cos = embedder.cosine_pairs(a_list, b_list)
    if not cos:                          # embedder failed (returns [] on error) -> skip the term
        return None
    vals = [(c + 1.0) / 2.0 for c in cos]
    return sum(vals) / len(vals)


# ----------------------------------------------------------------------------- #
#  Reward.
# ----------------------------------------------------------------------------- #
class RadiologyReportReward(ORM):
    """Composite CXR report reward. Returns one scalar per completion.

    Reshapes the flat batch into G rollouts per study so the embedding sections can be
    scored group-relatively (rank within the group; absolute cosine ignored).
    Per-component values are stashed on `self.last_components` for external logging.
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.cfg = RewardConfig()
        self.cfg.enable_bbox = os.getenv('RADREWARD_ENABLE_BBOX', '1') == '1'
        g = getattr(args, 'num_generations', None)
        if g:
            self.cfg.num_generations = int(g)
        # dual-ascent state: running dual variables, seeded from the configured fixed weights.
        # Persist across batches so the Lagrangian multipliers accumulate over training.
        self._lam_R = self.cfg.w_recall
        self._lam_P = self.cfg.w_precision
        self.gt_key = os.getenv('RADREWARD_GT_KEY', 'solution')
        self.vocab = load_vocab()
        self.families = load_families()                  # {tag: family} for subtype partial credit ({} = off)
        if self.families and self.cfg.subtype_credit > 0:
            logger.info('subtype partial credit ON: alpha=%.2f, %d tags across %d families',
                        self.cfg.subtype_credit, len(self.families), len(set(self.families.values())))
        # embedding model path: swift arg --reward_embedding_model, else env var.
        # If neither is given, the embedding sub-reward is disabled (no embedder).
        emb_path = getattr(args, 'reward_embedding_model', None) or os.getenv('RADREWARD_EMBED_MODEL')
        self.embedder = EmbeddingScorer(emb_path) if emb_path else None
        if self.embedder is None:
            logger.warning('No embedding model (reward_embedding_model / RADREWARD_EMBED_MODEL) '
                           '-> embedding sub-reward disabled.')
        # tokenizer (lazy) for the token-probability Brier: maps 'true'/'false' presence tokens to ids
        self._model_path = getattr(args, 'model', None) or os.getenv('RADREWARD_MODEL_PATH')
        self._tokenizer = None
        self._bool_ids = None
        self.last_components: Dict[str, List[float]] = {}

    def _get_bool_ids(self):
        """Lazily load the policy tokenizer and cache the true/false presence-token id sets."""
        if self._bool_ids is not None:
            return self._bool_ids
        if not self._model_path:
            self._bool_ids = (frozenset(), frozenset())
            return self._bool_ids
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
            self._bool_ids = _bool_token_id_sets(self._tokenizer)
        except Exception as e:  # noqa
            logger.warning(f'radiology_report: could not load tokenizer for Brier ({type(e).__name__}); '
                           'calibration term disabled.')
            self._bool_ids = (frozenset(), frozenset())
        return self._bool_ids

    def _match_weights(self, pred_set, gt_set, wfn):
        """Weighted TP/GT + FP counts between a predicted and a GT tag set, with family partial credit.
        Returns (W_tp, W_gt, fp_full, fp_partial). Exact matches score full weight; with subtype_credit
        alpha>0, a GT tag missed exactly but whose clinical family IS predicted scores alpha*weight (and
        such a predicted tag is an alpha-FP, counted in fp_partial, not fp_full). alpha=0 -> exact only."""
        a = self.cfg.subtype_credit
        W_gt = sum(wfn(g) for g in gt_set)
        W_tp = sum(wfn(f) for f in (pred_set & gt_set))
        fp = pred_set - gt_set
        if a > 0 and self.families:
            fam = lambda t: self.families.get(t) or frozenset()   # tag -> set of families ({} if none)
            pred_fams = set()
            for p in pred_set:
                pred_fams |= fam(p)
            gt_fams = set()
            for g in gt_set:
                gt_fams |= fam(g)
            # a missed GT tag gets partial credit if ANY of its families is among the predicted families
            W_tp += a * sum(wfn(g) for g in (gt_set - pred_set) if fam(g) & pred_fams)
            fp_part = sum(1 for p in fp if fam(p) & gt_fams)     # a near-miss pred (shares a GT family)
            fp_full = len(fp) - fp_part
        else:
            fp_full, fp_part = len(fp), 0
        return W_tp, W_gt, fp_full, fp_part

    # -- CLS: PROMINENCE-WEIGHTED sens/spec (Youden) + asymmetric Tversky, [0,1] --- #
    def _cls_reward(self, pred: ParsedReport, gt: ParsedReport) -> Tuple[float, float, float, float]:
        """Return (reward in [0,1], sensitivity, specificity, youden).

        TP/FN are weighted by GT prominence (missing a prominence-5 finding costs far
        more than a prominence-2). When no prominence is available (cls tasks), every
        finding weighs `prom_default` -> reduces to the plain count-based version.
          reward = cls_w_sens*sens_w + cls_w_spec*spec + cls_w_overlap*tversky_w   (convex)
        """
        c = self.cfg
        V = len(self.vocab)
        gt_p = gt.present & self.vocab
        pred_v = pred.present & self.vocab

        def w(f: str) -> float:                          # prominence weight in [~0.2, 1]
            p = gt.prominence.get(f)
            return (p / 5.0) if p else c.prom_default

        # prominence-weighted recall, with SUBTYPE partial credit (alpha): a predicted tag in the same
        # clinical family as a missed GT tag scores alpha*w instead of 0 and is an alpha-FP not a full FP.
        W_tp, W_gt, fp_full, fp_part = self._match_weights(pred_v, gt_p, w)
        fp_v = fp_full + fp_part                         # (spec is unused when cls_w_spec=0)

        sens = (W_tp / W_gt) if W_gt > 0 else 1.0        # prominence-weighted recall (partial-credited)
        tn = V - len(gt_p | pred_v)
        spec = tn / (tn + fp_v) if (tn + fp_v) > 0 else 1.0
        youden = sens + spec - 1.0

        # asymmetric Tversky over the FULL present sets, same partial-credit (FP incl. out-of-vocab).
        W_tp_all, W_gt_all, fpf_all, fpp_all = self._match_weights(pred.present, gt.present, w)
        W_fn_all = W_gt_all - W_tp_all
        W_fp_all = c.prom_default * (fpf_all + c.subtype_credit * fpp_all)
        denom = W_tp_all + c.tversky_fn * W_fn_all + c.tversky_fp * W_fp_all
        tversky = W_tp_all / denom if denom > 0 else 1.0

        wsum = c.cls_w_sens + c.cls_w_spec + c.cls_w_overlap
        reward = (c.cls_w_sens * sens + c.cls_w_spec * spec + c.cls_w_overlap * tversky) / wsum
        return reward, sens, spec, youden

    # -- BBOX: DIoU (GT box) + label precision/recall + anti-degeneracy --------- #
    def _bbox_reward(self, pred: ParsedReport, gt: ParsedReport) -> Optional[float]:
        c = self.cfg
        boxed = [f for f in pred.findings if f.bbox is not None]
        gt_box = {g.label: g.bbox for g in gt.findings if g.bbox is not None}
        if not boxed and not gt_box:
            return None                                  # no boxes on either side -> term inactive

        # (a) localization vs GT boxes (matched by exact label) where annotation exists
        loc = [_loc_reward(diou(f.bbox, gt_box[f.label])) for f in boxed if f.label in gt_box]
        localize = sum(loc) / len(loc) if loc else 0.0

        # (b) label alignment: predicted-box labels vs GT present cls (F-beta, recall-favored)
        box_labels = {f.label for f in boxed}
        gt_cls = gt.present
        if box_labels and gt_cls:
            inter = len(box_labels & gt_cls)
            precision = inter / len(box_labels)
            recall = inter / len(gt_cls)
            b2 = c.box_align_beta ** 2
            align = ((1 + b2) * precision * recall / (b2 * precision + recall)
                     if (b2 * precision + recall) > 0 else 0.0)
        elif not gt_cls and not box_labels:
            align = 1.0                                 # normal study, no boxes -> correct
        else:
            align = 0.0

        # (c) anti-degeneracy: tiny / near-whole-image / duplicate boxes -> in [0,1]
        degen = 0
        for f in boxed:
            a = _area(f.bbox)
            if a < c.box_area_min or a > c.box_area_max:
                degen += 1
        for i in range(len(boxed)):
            for j in range(i + 1, len(boxed)):
                if boxed[i].label != boxed[j].label and diou(boxed[i].bbox, boxed[j].bbox) >= c.box_dup_diou:
                    degen += 1
        degen = min(1.0, degen / (len(boxed) + 1.0))

        wsum = c.box_w_loc + c.box_w_align + c.box_w_degen
        return (c.box_w_loc * localize + c.box_w_align * align + c.box_w_degen * (1.0 - degen)) / wsum

    # -- TOKEN accuracy: task-aware -------------------------------------------- #
    def _prom_w(self, gt: ParsedReport, f: str) -> float:
        """Prominence weight for a finding: 1 + token_gain*(prominence/5); default if none."""
        p = gt.prominence.get(f)
        return (1.0 + self.cfg.token_gain * (p / 5.0)) if p else self.cfg.token_default

    def _token_reward(self, task: str, pred: ParsedReport, gt: ParsedReport) -> float:
        t = _norm_tag(task or '')
        if t == 'cls_tokens':
            return self._token_cls_tokens(pred, gt)
        if t == 'cls_json':
            return self._token_cls_json(pred, gt)
        return self._token_plain(pred, gt)               # ontology / default: unweighted

    def _token_cls_tokens(self, pred: ParsedReport, gt: ParsedReport) -> float:
        """Each abnormality token weighted by prominence -> weighted F1 over token sets."""
        P, G = pred.present, gt.present
        if not P and not G:
            return 1.0
        matched = P & G
        rec_den = sum(self._prom_w(gt, f) for f in G)
        prec_den = sum(self._prom_w(gt, f) for f in P)
        num = sum(self._prom_w(gt, f) for f in matched)
        recall = num / rec_den if rec_den > 0 else 1.0
        precision = num / prec_den if prec_den > 0 else (1.0 if not G else 0.0)
        return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    def _token_cls_json(self, pred: ParsedReport, gt: ParsedReport) -> float:
        """Weighted accuracy of the PRESENCE BOOLEAN per finding (the decision token),
        weighted by prominence. Unmentioned findings are treated as presence=false."""
        gtmap = gt.presence_map or {f: True for f in gt.present}
        if not gtmap:
            return 1.0
        num = den = 0.0
        for f, val in gtmap.items():
            w = self._prom_w(gt, f)
            pred_val = pred.presence_map.get(f, False)
            num += w * (1.0 if pred_val == val else 0.0)
            den += w
        return num / den if den > 0 else 1.0

    def _token_plain(self, pred: ParsedReport, gt: ParsedReport) -> float:
        """Plain unweighted bag-of-words F1 over findings + impression text."""
        ref = _tokens(gt.findings_text) | _tokens(gt.impression_text)
        hyp = _tokens(pred.findings_text) | _tokens(pred.impression_text)
        if not ref:
            return 1.0 if not hyp else 0.0
        matched = ref & hyp
        recall = len(matched) / len(ref)
        precision = (len(matched) / len(hyp)) if hyp else 0.0
        return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # -- EMBEDDING: per-section cosine -> (cos+1)/2 in [0,1], convex over sections -- #
    def _embedding_rewards(self, preds: List[ParsedReport], gts: List[ParsedReport]
                           ) -> List[Optional[float]]:
        """Per rollout: weighted mean over findings/impression/reasoning of (cos+1)/2.

        Encodes pred vs GT for each section with the fine-tuned radiology encoder.
        Returns None where no section is scorable or the embedder is unavailable
        (that rollout drops the embedding term from its convex combination).
        """
        n = len(preds)
        if self.embedder is None:
            return [None] * n
        c = self.cfg
        sec_w = {'findings': c.emb_w_findings, 'impression': c.emb_w_impression,
                 'reasoning': c.emb_w_reasoning}
        cos_by_sec: Dict[str, Dict[int, float]] = {s: {} for s in sec_w}
        for sec, getter in (('findings', lambda r: r.findings_text),
                            ('impression', lambda r: r.impression_text),
                            ('reasoning', lambda r: r.reasoning_text)):
            idx, pa, ga = [], [], []
            for i, (p, g) in enumerate(zip(preds, gts)):
                pt, gtt = getter(p), getter(g)
                if p.ok and pt and gtt:
                    idx.append(i); pa.append(pt); ga.append(gtt)
            if not idx:
                continue
            cos = self.embedder.cosine_pairs(pa, ga)
            if not cos:
                return [None] * n                        # embedder failed to load -> disable
            for k, i in enumerate(idx):
                cos_by_sec[sec][i] = (cos[k] + 1.0) / 2.0    # cosine [-1,1] -> [0,1]
        out: List[Optional[float]] = [None] * n
        for i in range(n):
            num = den = 0.0
            for sec, w in sec_w.items():
                if i in cos_by_sec[sec]:
                    num += w * cos_by_sec[sec][i]; den += w
            out[i] = (num / den) if den > 0 else None
        return out

    @staticmethod
    def _group_relative(emb: List[Optional[float]], keys: List) -> List[Optional[float]]:
        """Re-express each rollout's embedding score as a within-GROUP z-score mapped to [0,1], so GSPO
        sees variance (and thus gradient) from embedding even when the group's absolute cosines are ~flat.
        Rollouts sharing a key (same prompt = reference+task) form a group; a group with <2 scorable
        rollouts or ~zero spread keeps the absolute score (don't amplify pure noise)."""
        groups: Dict = {}
        for i, k in enumerate(keys):
            if emb[i] is not None:
                groups.setdefault(k, []).append(i)
        out = list(emb)
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            vals = [emb[i] for i in idxs]
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            sd = var ** 0.5
            if sd < 1e-6:
                continue
            for i in idxs:
                z = (emb[i] - m) / sd
                out[i] = min(1.0, max(0.0, 0.5 + 0.25 * z))   # z in [-2,2] -> [0,1], centered at 0.5
        return out

    # -- CONSISTENCY (NEW): impression must not invent findings ----------------- #
    def _mentions(self, text: Optional[str]) -> Set[str]:
        """Vocabulary labels whose words all appear in the text (rough mention detector)."""
        toks = _tokens(text)
        if not toks:
            return set()
        return {v for v in self.vocab if all(w in toks for w in v.split('_'))}

    def _consistency_reward(self, pred: ParsedReport) -> float:
        """Self-consistency: every abnormality asserted in the IMPRESSION must be
        established in the findings (structured tags or findings prose). Penalizes the
        'conclusion invents a finding' failure. Rule-based -> hard to reward-hack. 1.0 if
        the impression introduces nothing new (or there is no impression)."""
        impr = self._mentions(pred.impression_text)
        if not impr:
            return 1.0
        scope = set(pred.present) | self._mentions(pred.findings_text)
        unfaithful = impr - scope
        return 1.0 - len(unfaithful) / len(impr)

    # -- format / length as [0,1] goodness -------------------------------------- #
    def _format_reward(self, pred: ParsedReport) -> float:
        """1.0 for clean parseable structure; drops lambda_fmt per schema violation.
        (Fully unparseable JSON is scored 0 upstream in __call__.)"""
        return max(0.0, min(1.0, 1.0 - self.cfg.lambda_fmt * pred.n_violations))

    def _length_reward(self, pred: ParsedReport, gt: ParsedReport) -> float:
        """1.0 while within (1+alpha)*GT finding count; decays for over-long reports."""
        l_ref = max(1, len(gt.findings))
        excess = max(0.0, len(pred.findings) / l_ref - (1.0 + self.cfg.len_alpha))
        return max(0.0, 1.0 - self.cfg.len_slope * excess)

    # ------------------------------------------------------------------ #
    def __call__(self, completions, **kwargs) -> List[float]:
        c = self.cfg
        n = len(completions)

        gt_raw = kwargs.get(self.gt_key)
        if gt_raw is None:
            for k in ('solution', 'gt', 'ground_truth', 'label', 'answer'):
                if k in kwargs:
                    gt_raw = kwargs[k]
                    break
        if gt_raw is None:
            raise ValueError(f"radiology_report needs a GT column (tried '{self.gt_key}', 'solution', ...).")
        gt_raw = [gt_raw] * n if isinstance(gt_raw, str) else list(gt_raw)

        # GT source of truth is the pre-extracted `reference` (normalized finding_set,
        # prominence, absent_set, section texts). Fall back to parsing raw `solution`
        # only when reference is missing/unusable.
        refs = kwargs.get('reference')
        refs = [refs] * n if isinstance(refs, str) else (list(refs) if refs is not None else [None] * n)
        tasks = _col(kwargs, 'task', n)

        # cls_tokens emits `<TOKEN>` (not JSON) -> parse it accordingly; else JSON buckets.
        preds = [parse_cls_tokens(completions[i]) if _norm_tag(tasks[i] or '') == 'cls_tokens'
                 else parse_report(completions[i], self.vocab) for i in range(n)]
        gts = []
        for i in range(n):
            g = gt_from_reference(refs[i], self.vocab) if refs[i] is not None else None
            gts.append(g if g is not None else parse_report(gt_raw[i], self.vocab))

        # GT abnormality count per row (drives the dynamic embedding weight). Prefer the
        # precomputed `n_findings` dataset column; fall back to parsed GT present-set size.
        n_find_col = kwargs.get('n_findings')
        if isinstance(n_find_col, (int, float, str)) or n_find_col is None:
            n_find_col = None
        else:
            n_find_col = list(n_find_col)

        # per-row GT reliability -> scales the embedding weight (see _safety_col)
        safety = _safety_col(kwargs, n)

        # token-probability inputs for the calibration (Brier) term (cls_json; needs a vLLM rollout,
        # which attaches response_token_ids + rollout_logprobs to each row -> reward kwargs).
        use_cal = c.cal_mode == 'brier' and c.w_cal > 0
        tok_col = list(kwargs['response_token_ids']) if (use_cal and 'response_token_ids' in kwargs) else None
        lp_col = list(kwargs['rollout_logprobs']) if (use_cal and 'rollout_logprobs' in kwargs) else None
        true_ids, false_ids = self._get_bool_ids() if use_cal else (frozenset(), frozenset())

        emb = self._embedding_rewards(preds, gts)        # per-rollout [0,1] or None
        if c.emb_group_relative:                         # z-score embedding within each prompt-group
            grp_keys = [(refs[i], _norm_tag(tasks[i] or '')) if refs[i] is not None else ('__noref__', i)
                        for i in range(n)]
            emb = self._group_relative(emb, grp_keys)

        rewards = []
        comp = {k: [] for k in ('cls', 'bbox', 'token', 'embedding', 'consistency', 'format',
                                'length', 'sensitivity', 'specificity', 'youden', 'safety',
                                'recall', 'precision', 'calibration', 'attr', 'beta')}
        # Recall is UNDEFINED when a study has no in-vocab GT finding (W_gt=0): _cls_reward then returns
        # sens=1.0 by convention. Counting that 1.0 in the dual-ascent recall mean lets SILENCE on such
        # studies (truly-normal, or GT findings outside the vocab) inflate measured recall above target,
        # which relaxes lambda_recall exactly when it should push -> drives the all-normal collapse. We
        # keep sens=1.0 in each sample's own reward (correct for a normal study) but EXCLUDE W_gt=0
        # samples from the lambda_recall update, so the constraint price reflects recall only on studies
        # that actually have a scorable finding. rec_defined[i] = (this study had in-vocab GT).
        rec_defined: List[bool] = []
        for i in range(n):
            if not preds[i].ok:                          # unparseable -> reward floor (0.0)
                rewards.append(c.reward_floor)
                for k in comp:
                    comp[k].append(0.0)
                rec_defined.append(bool(gts[i].present & self.vocab))   # a miss on a scorable study counts
                continue
            r_cls, sens, spec, youden = self._cls_reward(preds[i], gts[i])
            r_box = self._bbox_reward(preds[i], gts[i]) if c.enable_bbox else None
            r_tok = self._token_reward(tasks[i], preds[i], gts[i])
            r_con = self._consistency_reward(preds[i])
            r_fmt = self._format_reward(preds[i])
            r_len = self._length_reward(preds[i], gts[i])
            r_emb = emb[i]

            # verifiable presence -> constraint terms (fixed-lambda now = weighted members of the combo;
            # a dual-ascent updater can later overwrite c.w_recall/c.w_precision per step):
            #   R_w = prominence-weighted recall (== cls sensitivity), P = precision.
            R_w = sens
            P = _precision(preds[i].present & self.vocab, gts[i].present & self.vocab)

            # embedding reliability weight: beta(n)=n/(n+k) (if enabled) REPLACES the legacy
            # abnormality-count scaling; both are further gated by safety (GT completeness).
            n_abn = int(n_find_col[i]) if n_find_col is not None else len(gts[i].present)
            beta = _beta(n_abn, c.beta_k)
            if c.beta_k > 0:
                w_emb_eff = c.w_emb * safety[i] * beta
            else:
                w_emb_eff = (c.w_emb * (1.0 + c.emb_abn_gain * min(n_abn, c.emb_abn_cap) / c.emb_abn_cap)
                             * safety[i])
            # benign-dip cap: presence perfect -> any embedding dip is phrasing; floor S to a baseline.
            if r_emb is not None and c.emb_baseline > 0 and P >= 0.999 and R_w >= 0.999:
                r_emb = max(r_emb, c.emb_baseline)

            # token-probability calibration (weighted Brier) + attribute agreement (both optional).
            r_cal = None
            if use_cal and tok_col is not None and lp_col is not None and i < len(tok_col) and i < len(lp_col):
                lpi = lp_col[i]
                if lpi and isinstance(lpi[0], (list, tuple)):    # single-turn wrap [[...]] -> [...]
                    lpi = lpi[0]
                r_cal = _weighted_brier(completions[i], tok_col[i], lpi, true_ids, false_ids,
                                        gts[i].present, c.cal_true_weight)
            r_attr = _attr_agreement(preds[i], gts[i], self.embedder) if c.w_attr > 0 else None

            # constraint prices: fixed weights, or (dual-ascent) the running Lagrange multipliers.
            # The current-batch value is applied to the whole batch; lambdas update after the loop.
            wR = self._lam_R if c.dual_ascent else c.w_recall
            wP = self._lam_P if c.dual_ascent else c.w_precision

            # convex combination over ACTIVE sub-rewards (all in [0,1]) -> total in [0,1]
            parts = [(c.w_cls, r_cls), (c.w_bbox, r_box), (c.w_token, r_tok),
                     (w_emb_eff, r_emb), (c.w_consistency, r_con),
                     (c.w_format, r_fmt), (c.w_length, r_len),
                     (wR, R_w), (wP, P),
                     (c.w_cal, r_cal), (c.w_attr, r_attr)]
            num = sum(w * r for w, r in parts if r is not None and w > 0)
            den = sum(w for w, r in parts if r is not None and w > 0)
            rewards.append(float(num / den) if den > 0 else 0.0)

            comp['cls'].append(r_cls); comp['bbox'].append(r_box if r_box is not None else 0.0)
            comp['token'].append(r_tok); comp['embedding'].append(r_emb if r_emb is not None else 0.0)
            comp['consistency'].append(r_con); comp['format'].append(r_fmt); comp['length'].append(r_len)
            comp['sensitivity'].append(sens); comp['specificity'].append(spec); comp['youden'].append(youden)
            comp['safety'].append(safety[i])
            comp['recall'].append(R_w); comp['precision'].append(P)
            comp['calibration'].append(r_cal if r_cal is not None else 0.0)
            comp['attr'].append(r_attr if r_attr is not None else 0.0)
            comp['beta'].append(beta)
            rec_defined.append(bool(gts[i].present & self.vocab))   # recall defined iff study has in-vocab GT

        # dual-ascent update (PPO-Lagrangian): lam <- clip(lam + eta*(target - measured)). Raise the
        # price of a constraint that is UNDER target (measured < target -> positive step), relax it once
        # satisfied. Uses this batch's mean recall/precision; the new lambdas apply to the NEXT batch.
        if c.dual_ascent and comp['recall']:
            # measure recall ONLY over studies where it is defined (W_gt>0); silence on normal / no-in-vocab
            # -GT studies no longer counts as free recall=1.0 and can't relax lambda_recall (collapse fix).
            rec_vals = [r for r, d in zip(comp['recall'], rec_defined) if d]
            mR = (sum(rec_vals) / len(rec_vals)) if rec_vals else c.recall_target   # none scorable -> no step
            mP = sum(comp['precision']) / len(comp['precision'])
            self._lam_R = min(c.lambda_max, max(c.lambda_min, self._lam_R + c.dual_lr * (c.recall_target - mR)))
            self._lam_P = min(c.lambda_max, max(c.lambda_min, self._lam_P + c.dual_lr * (c.precision_target - mP)))
        # log the (constant-per-batch) multipliers alongside the components for wandb
        comp['lambda_recall'] = [self._lam_R] * len(comp['recall'])
        comp['lambda_precision'] = [self._lam_P] * len(comp['precision'])

        self.last_components = comp
        return rewards


orms['radiology_report'] = RadiologyReportReward


# =============================================================================== #
#  Knowledge-graph + comparison rewards (separate tasks; same [0,1] framework).
# =============================================================================== #
def _embedder_from_args(args) -> Optional[EmbeddingScorer]:
    p = getattr(args, 'reward_embedding_model', None) or os.getenv('RADREWARD_EMBED_MODEL')
    return EmbeddingScorer(p) if p else None


def tversky(pred: Set[str], gt: Set[str], fn_w: float = 1.5, fp_w: float = 1.0) -> float:
    """Asymmetric set overlap in [0,1] (1=perfect). FN weighted heavier than FP by default."""
    tp = len(pred & gt); fn = len(gt - pred); fp = len(pred - gt)
    d = tp + fn_w * fn + fp_w * fp
    return tp / d if d > 0 else 1.0


# The graph data serializes triples in several styles; handle all of them.
_TRI_PATTERNS = [
    # (s) -[r]-> (o)  |  (s) --[r]--> (o)  |  (s) – [r] – (o)   (dash/arrow + brackets)
    re.compile(r'\(\s*([^()]{1,90}?)\s*\)\s*[-–—]+\s*>?\s*\[\s*([^\]]{1,60}?)\s*\]\s*[-–—]*\s*>?\s*\(\s*([^()]{1,90}?)\s*\)'),
    # (s, r, o)
    re.compile(r'\(\s*([^(),]{1,60}?)\s*,\s*([^(),]{1,60}?)\s*,\s*([^()]{1,90}?)\s*\)'),
]
_FINDING_NODE_RE = re.compile(r'Finding:\s*([^\n│├└─]+)')


def parse_triples(text: str) -> List[Tuple[str, str, str]]:
    """Extract (subject, relation, object) triples across the graph serialization
    styles seen in the data: '(s) – [r] – (o)', '(s) --[r]--> (o)', '(s, r, o)',
    and N-triple 's r o .' lines."""
    if not text:
        return []
    for pat in _TRI_PATTERNS:
        found = pat.findall(text)
        if found:
            return [(a.strip(), b.strip(), c.strip()) for a, b, c in found]
    # N-triples fallback: "subject predicate object ." lines in the triples section
    i = text.lower().rfind('triple')
    seg = text[i:] if i >= 0 else text
    out = []
    for ln in seg.splitlines():
        ln = ln.strip().rstrip('.').strip()
        parts = ln.split()
        if 3 <= len(parts) <= 8 and re.match(r'^[a-z0-9_]+$', parts[0]) and re.match(r'^[a-z0-9_]+$', parts[1]):
            out.append((parts[0], parts[1], ' '.join(parts[2:])))
    return out


def _triple_str(t: Tuple[str, str, str]) -> str:
    return f'{t[0]} {t[1]} {t[2]}'


class KnowledgeGraphReward(ORM):
    """Hierarchical positive-findings graph + triples reward (dense, all sub-rewards [0,1]).

      triple    : soft precision/recall of predicted vs GT triples, matched by embedding
                  similarity on "subject relation object" (the encoder saw ASP scene
                  graphs, so this generalizes). Falls back to exact overlap if no embedder.
      finding   : soft overlap of graph Finding nodes vs GT finding nodes.
      structure : parseable graph with >=1 well-formed triple.
      holistic  : (cos+1)/2 of the whole graph text vs GT.
    GT triples/findings come from `reference`; if absent, parsed from the GT `solution`.
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.embedder = _embedder_from_args(args)
        self.gt_key = os.getenv('RADREWARD_GT_KEY', 'solution')
        self.w = dict(triple=0.45, finding=0.20, structure=0.10, holistic=0.25)
        self.last_components: Dict[str, List[float]] = {}

    def _soft_f(self, pred_strs: List[str], gt_strs: List[str]) -> float:
        if not gt_strs and not pred_strs:
            return 1.0
        if not gt_strs or not pred_strs:
            return 0.0
        if self.embedder is not None:
            sims = self.embedder.matrix(pred_strs, gt_strs)      # [len(pred)][len(gt)]
            if sims:
                recall = sum(max(col) for col in zip(*sims)) / len(gt_strs)
                precision = sum(max(row) for row in sims) / len(pred_strs)
                recall = (recall + 1) / 2; precision = (precision + 1) / 2   # cos -> [0,1]
                return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return tversky(set(pred_strs), set(gt_strs), 1.0, 1.0)

    def __call__(self, completions, **kwargs) -> List[float]:
        n = len(completions)
        gt_raw = kwargs.get(self.gt_key) or kwargs.get('solution')
        gt_raw = [gt_raw] * n if isinstance(gt_raw, str) else list(gt_raw or [''] * n)
        refs = kwargs.get('reference')
        refs = [refs] * n if isinstance(refs, str) else list(refs or ['{}'] * n)

        rewards, comp = [], {k: [] for k in ('triple', 'finding', 'structure', 'holistic')}
        for i in range(n):
            pred = completions[i] or ''
            try:
                ref = json.loads(refs[i]) if isinstance(refs[i], str) else (refs[i] or {})
            except Exception:
                ref = {}
            gt_triples = [tuple(t) for t in ref.get('triples', [])] or parse_triples(gt_raw[i])
            gt_findings = {_norm_tag(x) for x in ref.get('finding_set', [])}
            p_triples = parse_triples(pred)
            p_findings = {_norm_tag(t[0]) for t in p_triples} | {_norm_tag(m) for m in _FINDING_NODE_RE.findall(pred)}

            tri = self._soft_f([_triple_str(t) for t in p_triples], [_triple_str(t) for t in gt_triples])
            fnd = (self._soft_f(sorted(p_findings), sorted(gt_findings))
                   if (p_findings or gt_findings) else 1.0)
            struct = 1.0 if p_triples else (0.3 if pred.strip() else 0.0)
            hol = 0.5
            if self.embedder is not None and pred.strip() and gt_raw[i]:
                cp = self.embedder.cosine_pairs([pred[:4000]], [gt_raw[i][:4000]])
                if cp:
                    hol = (cp[0] + 1) / 2
            w = self.w
            parts = [(w['triple'], tri), (w['finding'], fnd), (w['structure'], struct), (w['holistic'], hol)]
            den = sum(x for x, _ in parts)
            rewards.append(float(sum(x * r for x, r in parts) / den))
            comp['triple'].append(tri); comp['finding'].append(fnd)
            comp['structure'].append(struct); comp['holistic'].append(hol)
        self.last_components = comp
        return rewards


orms['radiology_kgraph'] = KnowledgeGraphReward


class ComparisonReward(ORM):
    """Two-image comparison reward (dense, [0,1]). `reference` carries per-image finding
    sets DERIVED from the tag data (verifiable, not the generated text):
        {set1, set2, shared, unique1, unique2, cot_text, summary_text}.

      coverage : recall that findings present across the two images are mentioned.
      diff     : correctly reflects shared vs differing findings.
      cot      : (cos+1)/2 of the predicted CoT section vs GT CoT.
      summary  : (cos+1)/2 of the predicted summary/JSON section vs GT.
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.embedder = _embedder_from_args(args)
        self.vocab = load_vocab()
        self.w = dict(cov=0.30, diff=0.25, cot=0.25, summary=0.20)
        self.last_components: Dict[str, List[float]] = {}

    def _mentions(self, text: str) -> Set[str]:
        toks = _tokens(text)
        return {v for v in self.vocab if toks and all(w in toks for w in v.split('_'))}

    @staticmethod
    def _section(text: str, which: str) -> str:
        parts = re.split(r'SECTION\s*\d+[^\n]*', text or '')
        if which == 'cot':
            return parts[1] if len(parts) > 1 else (text or '')
        return parts[2] if len(parts) > 2 else (parts[-1] if parts else (text or ''))

    def _emb01(self, a: str, b: str) -> Optional[float]:
        if self.embedder is None or not a or not b:
            return None
        cp = self.embedder.cosine_pairs([a[:4000]], [b[:4000]])
        return (cp[0] + 1) / 2 if cp else None

    def __call__(self, completions, **kwargs) -> List[float]:
        n = len(completions)
        refs = kwargs.get('reference')
        refs = [refs] * n if isinstance(refs, str) else list(refs or ['{}'] * n)
        rewards, comp = [], {k: [] for k in ('coverage', 'diff', 'cot', 'summary')}
        for i in range(n):
            pred = completions[i] or ''
            try:
                ref = json.loads(refs[i]) if isinstance(refs[i], str) else (refs[i] or {})
            except Exception:
                ref = {}
            g = lambda k: {_norm_tag(x) for x in ref.get(k, [])}
            set1, set2 = g('set1'), g('set2')
            shared = g('shared') or (set1 & set2)
            uniq = (g('unique1') | g('unique2')) or (set1 ^ set2)
            allf = set1 | set2
            pm = self._mentions(pred)
            cov = tversky(pm & allf, allf, 1.5, 0.0) if allf else 1.0        # recall of union (FP-free)
            diff = 0.5 * ((len(pm & uniq) / len(uniq) if uniq else 1.0)
                          + (len(pm & shared) / len(shared) if shared else 1.0))
            cot = self._emb01(self._section(pred, 'cot'), ref.get('cot_text', ''))
            summ = self._emb01(self._section(pred, 'sum'), ref.get('summary_text', ''))
            w = self.w
            parts = [(w['cov'], cov), (w['diff'], diff), (w['cot'], cot), (w['summary'], summ)]
            den = sum(x for x, r in parts if r is not None)
            num = sum(x * r for x, r in parts if r is not None)
            rewards.append(float(num / den) if den > 0 else 0.0)
            comp['coverage'].append(cov); comp['diff'].append(diff)
            comp['cot'].append(cot if cot is not None else 0.0)
            comp['summary'].append(summ if summ is not None else 0.0)
        self.last_components = comp
        return rewards


orms['radiology_comparison'] = ComparisonReward


# =============================================================================== #
#  ASP reward (task='asp') + BBox reward (task='bbox') + task router.
# =============================================================================== #
def _col(kwargs, key, n):
    v = kwargs.get(key)
    if isinstance(v, str) or v is None:
        return [v] * n
    v = list(v)
    return v if len(v) == n else [v[0] if v else None] * n


def _safety_col(kwargs, n):
    """Per-row GT-reliability weight (`safety_score` dataset column) in [0,1]. Multiplies the
    embedding sub-reward weight: complete GT (safety=1) -> full embedding trust; sparse GT
    (safety low) -> embedding down-weighted, verifiable terms (cls/FP, bbox, atoms) dominate via
    the convex-combo renormalization. Defaults to 1.0 when the column is absent or a row is
    unscored (e.g. bbox / comparison images not in the tag-extraction JSON) -> no change there."""
    s = kwargs.get('safety_score')
    if s is None:
        return [1.0] * n
    if isinstance(s, (int, float)):
        return [float(s)] * n
    s = list(s)
    if len(s) != n:
        return [1.0] * n
    out = []
    for x in s:
        try:
            out.append(min(1.0, max(0.0, float(x))) if x is not None else 1.0)
        except (TypeError, ValueError):
            out.append(1.0)
    return out


# The model tokenizer has |<ASP>| (id 248077) and |</ASP>| (id 248078) as SINGLE special tokens
# (examples/train/liger/token.txt), so a trained model emits those at the ASP block boundary. The
# older gemini SFT data used the plain 5-token string <|ASP|> with no closing tag. Match BOTH the
# special-token form and the legacy string so ASP extraction works for standalone `asp` and for a
# consolidated `asp,ontology` rollout (where the ASP block must be sliced out of the full report).
_ASP_OPEN_RE = re.compile(r'\|<ASP>\||<\|ASP\|>')
_ASP_CLOSE_RE = re.compile(r'\|</ASP>\||</\|ASP\|>')


def _asp_body(text: str) -> str:
    """The ASP block: text after the opening |<ASP>| (fallback <|ASP|>), truncated at the closing
    |</ASP>| when present. Returns the whole text if no opening token is found (legacy behaviour)."""
    t = text or ''
    mo = _ASP_OPEN_RE.search(t)
    body = t[mo.end():] if mo else t
    mc = _ASP_CLOSE_RE.search(body)
    return body[:mc.start()] if mc else body


class AspReward(ORM):
    """Dense ASP reward (all sub-rewards [0,1]).

      sections    : per-section (cos+1)/2 vs GT (scene_graph/explanation/cot/inference) -- correctness.
      progression : within-rollout logical flow -- (cos+1)/2 of OWN consecutive sections.
      finding     : Tversky of predicted finding() atoms vs GT finding_set.
      absent      : Tversky of predicted absent() atoms vs GT absent_set -- REAL specificity.
      structure   : parseable symbolic graph (finding/absent atoms present).
    GT sections/atoms come from `reference`.
    """
    ORDER = ['scene_graph', 'explanation', 'cot', 'inference']

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.embedder = _embedder_from_args(args)
        self.w = dict(sections=0.40, progression=0.15, finding=0.20, absent=0.15, structure=0.10)
        self.last_components: Dict[str, List[float]] = {}

    @staticmethod
    def _split_steps(text: str) -> List[str]:
        body = _asp_body(text)
        parts = re.split(r'STEP\s*\d+[^\n]*', body)
        return [p.strip() for p in parts[1:]]           # ordered step contents

    @staticmethod
    def _atoms(text: str, pred: str) -> Set[str]:
        return {_norm_tag(x) for x in re.findall(rf'\b{pred}\(\s*([a-z0-9_ ]+?)\s*[,)]', text or '', re.I)}

    def __call__(self, completions, **kwargs) -> List[float]:
        n = len(completions)
        refs = _col(kwargs, 'reference', n)
        G = []
        for i in range(n):
            try:
                G.append(json.loads(refs[i]) if isinstance(refs[i], str) else (refs[i] or {}))
            except Exception:
                G.append({})
        P = [self._split_steps(c) for c in completions]
        safety = _safety_col(kwargs, n)   # scales the embedding terms (sections/progression) per row

        # batched per-section similarity (pred section k vs GT section k)
        seg = []
        for i in range(n):
            gtsec = G[i].get('sections') or {}
            for k, name in enumerate(self.ORDER):
                gt, pt = gtsec.get(name, ''), (P[i][k] if k < len(P[i]) else '')
                if gt and pt:
                    seg.append((i, pt[:4000], gt[:4000]))
        sec_scores: Dict[int, List[float]] = {i: [] for i in range(n)}
        if self.embedder and seg:
            cos = self.embedder.cosine_pairs([a for _, a, _ in seg], [b for _, _, b in seg])
            for (i, _, _), c in zip(seg, cos):
                sec_scores[i].append((c + 1) / 2)

        # batched within-rollout progression (own section k vs k+1)
        prg = []
        for i in range(n):
            for k in range(len(P[i]) - 1):
                if P[i][k] and P[i][k + 1]:
                    prg.append((i, P[i][k][:2000], P[i][k + 1][:2000]))
        prog_scores: Dict[int, List[float]] = {i: [] for i in range(n)}
        if self.embedder and prg:
            cos = self.embedder.cosine_pairs([a for _, a, _ in prg], [b for _, _, b in prg])
            for (i, _, _), c in zip(prg, cos):
                prog_scores[i].append((c + 1) / 2)

        rewards, comp = [], {k: [] for k in ('sections', 'progression', 'finding', 'absent', 'structure')}
        for i in range(n):
            pf = self._atoms(completions[i], 'finding')
            pa = self._atoms(completions[i], 'absent')
            gf = {_norm_tag(x) for x in G[i].get('finding_set', [])}
            ga = {_norm_tag(x) for x in G[i].get('absent_set', [])}
            sec = (sum(sec_scores[i]) / len(sec_scores[i])) if sec_scores[i] else None
            prog = (sum(prog_scores[i]) / len(prog_scores[i])) if prog_scores[i] else None
            finding = tversky(pf, gf, 1.5, 1.0) if (pf or gf) else 1.0
            absent = tversky(pa, ga, 1.0, 1.0) if (pa or ga) else 1.0
            structure = 1.0 if (pf or pa) else (0.3 if completions[i].strip() else 0.0)
            w = self.w
            # sections/progression are embedding (cosine) terms -> down-weight by GT reliability;
            # finding/absent/structure are verifiable atom-overlap terms -> unscaled (dominate when safety low)
            parts = [(w['sections'] * safety[i], sec), (w['progression'] * safety[i], prog),
                     (w['finding'], finding), (w['absent'], absent), (w['structure'], structure)]
            den = sum(x for x, r in parts if r is not None)
            num = sum(x * r for x, r in parts if r is not None)
            rewards.append(float(num / den) if den > 0 else 0.0)
            comp['sections'].append(sec if sec is not None else 0.0)
            comp['progression'].append(prog if prog is not None else 0.0)
            comp['finding'].append(finding); comp['absent'].append(absent); comp['structure'].append(structure)
        self.last_components = comp
        return rewards


orms['radiology_asp'] = AspReward


# messy internal bbox labels -> canonical (mirror of the data-build map)
_BBOX_LABEL_MAP = {
    'peffusion': 'pleural_effusion', 'atelactasis': 'atelectasis', 'ribfracture': 'rib_fracture',
    'pleuralthickening': 'pleural_thickening', 'hilarlymphadenopathy': 'hilar_lymphadenopathy',
    'enlarged_hila': 'hilar_enlargement', 'artefacts': 'artifact', 'tinynodule': 'nodule',
    'reticular_opacity': 'reticular_opacities', 'cvc': 'central_venous_catheter', 'ett': 'endotracheal_tube',
}


def _canon_bbox_label(lab: str) -> str:
    n = _norm_tag(lab)
    return _BBOX_LABEL_MAP.get(n, n)


class BboxTaskReward(ORM):
    """BBox reward for the grounding task (all sub-rewards [0,1]). Parses the real output
    format `[{'bbox_2d': [x1,y1,x2,y2], 'label': '...'}]` at 0-1000, normalizes to [0,1],
    canonicalizes labels, and scores against `reference.boxes`:
      loc    : mean DIoU-localization of predicted boxes vs same-label GT boxes.
      align  : F-beta of predicted box labels vs GT box labels (recall-favored).
      degen  : (1 - fraction of tiny / whole-image / near-duplicate / excess boxes).
      format : output is a parseable, properly-closed list of box dicts (not truncated / looping).
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.w = dict(loc=0.45, align=0.35, degen=0.1, format=0.1)
        self.beta = 2.0
        self.area_min, self.area_max, self.dup = 5e-4, 0.5, 0.9
        self.box_cap = 18          # a CXR rarely has >~18 findings+devices; boxes beyond this are degenerate
        self.dup_scan = 60         # bound the O(n^2) duplicate scan (excess handled by the count penalty)
        self.last_components: Dict[str, List[float]] = {}

    @staticmethod
    def _format_score(text: str, n_valid: int) -> float:
        """1.0 for a clean, closed list of box dicts; low for unparseable or truncated/looping output
        (e.g. a greedy repetition that runs to max_tokens never closes its bracket)."""
        s = (text or '').strip()
        if '[' not in s or ']' not in s:
            return 0.0
        closed = s.rfind(']') > s.rfind('[')          # a ']' after the last '[' => list actually closed
        attempts = len(re.findall(r'bbox_2d|["\']box["\']|["\']bbox["\']', s))
        if attempts == 0:
            return 1.0 if n_valid == 0 and closed else 0.3   # a clean empty list is valid format
        base = min(1.0, n_valid / attempts)            # fraction of attempted boxes that parsed cleanly
        return base if closed else base * 0.3          # truncated / unclosed -> heavily penalized

    @staticmethod
    def _parse_boxes(text: str) -> List[Tuple[str, List[float]]]:
        obj = None
        s = (text or '').strip()
        for parse in (ast.literal_eval, json.loads):
            try:
                obj = parse(s); break
            except Exception:
                pass
        if not isinstance(obj, list):
            i, j = s.find('['), s.rfind(']')
            if i >= 0 and j > i:
                for parse in (ast.literal_eval, json.loads):
                    try:
                        obj = parse(s[i:j + 1]); break
                    except Exception:
                        pass
        boxes = []
        if isinstance(obj, list):
            for it in obj:
                if not isinstance(it, dict):
                    continue
                b = it.get('bbox_2d') or it.get('box') or it.get('bbox')
                lab = it.get('label')
                if b is None or lab is None:
                    continue
                if b and isinstance(b[0], list):
                    b = b[0]
                if not (isinstance(b, (list, tuple)) and len(b) == 4):
                    continue
                try:
                    box = [max(0.0, min(1.0, float(v) / 1000.0)) for v in b]
                except Exception:
                    continue
                boxes.append((_canon_bbox_label(lab), box))
        return boxes

    def __call__(self, completions, **kwargs) -> List[float]:
        n = len(completions)
        refs = _col(kwargs, 'reference', n)
        rewards, comp = [], {k: [] for k in ('loc', 'align', 'degen', 'format')}
        for i in range(n):
            try:
                gt = [(_canon_bbox_label(b['label']), b['box'])
                      for b in (json.loads(refs[i]).get('boxes', []) if isinstance(refs[i], str) else [])]
            except Exception:
                gt = []
            pred = self._parse_boxes(completions[i])

            gt_by = {}
            for lab, box in gt:
                gt_by.setdefault(lab, []).append(box)
            locs = [max(_loc_reward(diou(box, gb)) for gb in gt_by[lab]) for lab, box in pred if lab in gt_by]
            loc = sum(locs) / len(locs) if locs else 0.0

            pl, gl = {l for l, _ in pred}, {l for l, _ in gt}
            if pl and gl:
                inter = len(pl & gl)
                precision, recall = inter / len(pl), inter / len(gl)
                b2 = self.beta ** 2
                align = ((1 + b2) * precision * recall / (b2 * precision + recall)
                         if (b2 * precision + recall) > 0 else 0.0)
            elif not pl and not gl:
                align = 1.0
            else:
                align = 0.0

            degen = 0
            for _, box in pred:
                a = abs(box[2] - box[0]) * abs(box[3] - box[1])
                if a < self.area_min or a > self.area_max:
                    degen += 1
            # near-duplicate boxes of ANY label (same-label spam included) -- bounded O(n^2) scan
            scan = min(len(pred), self.dup_scan)
            for x in range(scan):
                for y in range(x + 1, scan):
                    if diou(pred[x][1], pred[y][1]) >= self.dup:
                        degen += 1
            # excess-box penalty: boxes beyond a plausible cap are degenerate (catches long loops)
            if len(pred) > self.box_cap:
                degen += (len(pred) - self.box_cap)
            degen = min(1.0, degen / (len(pred) + 1.0))

            fmt = self._format_score(completions[i], len(pred))

            w = self.w
            reward = (w['loc'] * loc + w['align'] * align
                      + w['degen'] * (1.0 - degen) + w['format'] * fmt) / sum(w.values())
            rewards.append(float(reward))
            comp['loc'].append(loc); comp['align'].append(align)
            comp['degen'].append(1.0 - degen); comp['format'].append(fmt)
        self.last_components = comp
        return rewards


orms['radiology_bbox'] = BboxTaskReward


# --- compound-rollout segmentation ------------------------------------------- #
def _json_objects(text: str) -> List[Tuple[int, int, str]]:
    """All top-level balanced {...} objects as (start, end, text). Quote/escape aware."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] == '{':
            depth, j, instr, esc, q = 0, i, False, '', False
            while j < n:
                ch = text[j]
                if instr:
                    if esc:
                        esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == q:
                        instr = False
                elif ch in '"\'':
                    instr, q = True, ch
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        out.append((i, j + 1, text[i:j + 1]))
                        i = j
                        break
                j += 1
        i += 1
    return out


_ONT_KEY_RE = re.compile(r"""['"](?:Findings|Summary|Prominence_Score|Complete Reasoning|Impression)['"]""", re.I)
_CLSJ_KEY_RE = re.compile(r"""['"](?:general|presence|other_findings)['"]""", re.I)


def _task_start(text: str, task: str) -> Optional[int]:
    """Start offset of a NON-JSON sub-task by its format signature (JSON tasks are
    handled separately via balanced-object classification)."""
    if task == 'asp':
        m = _ASP_OPEN_RE.search(text)
        return m.start() if m else None
    if task == 'bbox':
        m = re.search(r'\[\s*\{[^{}]*bbox_2d', text)
        return m.start() if m else None
    if task == 'cls_tokens':
        m = re.search(r'<[A-Z][A-Z0-9_]*>', text)
        return m.start() if m else None
    if task == 'comparison':
        m = re.search(r'SECTION\s*1', text, re.I)
        return m.start() if m else None
    if task in ('kgraph', 'kg'):
        m = re.search(r'Positive Findings Graph|Triples', text, re.I)
        return m.start() if m else None
    return None


def segment_completion(text: str, subtasks: List[str]) -> Dict[str, str]:
    """Split one completion into per-sub-task text. Preferred: explicit
    `<task>...</task>` delimiters. Else: classify balanced JSON objects by their keys
    for ontology/cls_json, locate the rest by format signature, slice in order.
    Last resort: whole text for any task not found."""
    text = text or ''
    segs: Dict[str, str] = {}
    # 0) ASP special-token mode: the model emits |<ASP>|...|</ASP>| (single tokens), so slice the ASP
    # block by those tokens directly. Handled before the generic <task> delimiter loop, whose regex
    # would otherwise match <ASP> inside the |<ASP>| token and leave stray pipes.
    if 'asp' in subtasks and _ASP_OPEN_RE.search(text):
        segs['asp'] = _asp_body(text).strip()
    # 1) delimiter mode
    for t in subtasks:
        if t in segs:
            continue
        m = re.search(rf'<\s*{re.escape(t)}\s*>(.*?)</\s*{re.escape(t)}\s*>', text, re.DOTALL | re.I)
        if m:
            segs[t] = m.group(1).strip()
    if all(t in segs for t in subtasks):
        return segs
    # 2) JSON tasks: classify balanced {...} objects by keys
    json_tasks = [t for t in subtasks if t in ('ontology', 'cls_json') and t not in segs]
    if json_tasks:
        for a, b, o in _json_objects(text):
            if 'ontology' in json_tasks and 'ontology' not in segs and _ONT_KEY_RE.search(o):
                segs['ontology'] = o
            elif 'cls_json' in json_tasks and 'cls_json' not in segs and _CLSJ_KEY_RE.search(o):
                segs['cls_json'] = o
    # 3) remaining (non-JSON) tasks by signature, sliced in text order
    remaining = [t for t in subtasks if t not in segs]
    if remaining:
        found = sorted((p, t) for p, t in ((_task_start(text, t), t) for t in remaining) if p is not None)
        for k, (p, t) in enumerate(found):
            end = found[k + 1][0] if k + 1 < len(found) else len(text)
            segs.setdefault(t, text[p:end].strip())
    # 4) last resort
    for t in subtasks:
        segs.setdefault(t, text)
    return segs


class RadiologyRouter(ORM):
    """Single reward that dispatches per row on the dataset `task` column.

    Supports COMPOUND rollouts: if `task` is a comma-separated list (e.g.
    "cls_tokens,ontology"), the completion is segmented into per-sub-task text
    (delimiters preferred, format-signature fallback), each is scored by its task
    reward, and the results are averaged. The `reference` for a compound row may be a
    task-keyed dict `{"cls_tokens": {...}, "ontology": {...}}`; a flat reference is
    shared across sub-tasks.

    Routes: ontology/cls_json/cls_tokens -> report; asp/bbox/kgraph/comparison -> own.
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self._report = RadiologyReportReward(args)
        self._sub = {
            'report': self._report,
            'asp': AspReward(args),
            'bbox': BboxTaskReward(args),
            'kgraph': KnowledgeGraphReward(args),
            'comparison': ComparisonReward(args),
        }
        # How a compound rollout's sub-task scores combine. 'mean' lets the policy drop one sub-task
        # (e.g. the ASP block in asp,ontology) and recover the loss by acing the other. 'geomean'
        # (product^(1/n)) and 'min' both collapse the compound reward toward 0 if ANY sub-task is
        # neglected, so every sub-task must stay up. 1_oct uses geomean. (RADREWARD_COMPOUND_AGG)
        self._agg_mode = os.getenv('RADREWARD_COMPOUND_AGG', 'mean')
        # cls_tokens format enforcement: a cls_tokens answer (or the cls_tokens segment of a compound
        # rollout) with NO <TAG> tokens -- empty, ASP-only, or an ontology/JSON report written instead --
        # has its cls_tokens score multiplied by this. 1.0 = off. Without it an empty answer scores ~0.94
        # on normal studies (GT <NORMAL>), so "say nothing" is nearly optimal there.
        self._notag_mult = float(os.getenv('RADREWARD_CLS_NOTAG_MULT', '1.0') or 1.0)
        # ASP length penalty (hinge): rows whose subtasks include one of RADREWARD_ASP_PEN_TASKS get their
        # total reward multiplied by 1 - MAX * min(1, max(0, L_asp - FREE) / RAMP), L_asp = ASP block
        # tokens. An ASP block under FREE tokens is free (ties with no ASP), so the group-relative
        # advantage pushes long blocks shorter without pushing ASP to zero. MAX=0 = off.
        self._asp_pen_max = float(os.getenv('RADREWARD_ASP_PEN_MAX', '0') or 0.0)
        self._asp_free = float(os.getenv('RADREWARD_ASP_FREE_TOK', '1500') or 1500)
        self._asp_ramp = max(1.0, float(os.getenv('RADREWARD_ASP_RAMP_TOK', '2500') or 2500))
        self._asp_pen_tasks = {_norm_tag(s) for s in os.getenv('RADREWARD_ASP_PEN_TASKS', 'cls_tokens').split(',')
                               if s.strip()}
        if self._notag_mult < 1.0 or self._asp_pen_max > 0:
            logger.info(f'radiology router: cls_tokens no-tag mult={self._notag_mult} | ASP length penalty '
                        f'max={self._asp_pen_max} free={self._asp_free:.0f} ramp={self._asp_ramp:.0f} '
                        f'tasks={sorted(self._asp_pen_tasks)}')

    def _asp_tokens(self, text: str) -> int:
        """Token length of the ASP block in `text` (0 if none), with the policy tokenizer when available."""
        span = asp_block_span(text)
        if span is None:
            return 0
        seg = text[span[0]:span[1]]
        self._report._get_bool_ids()                     # lazily loads the tokenizer (RADREWARD_MODEL_PATH)
        tok = self._report._tokenizer
        if tok is not None:
            return len(tok.encode(seg, add_special_tokens=False))
        return int(len(seg) / 3.2)                       # ~chars per token for this tokenizer on ASP text

    def _asp_mult(self, n_tok: int) -> float:
        if self._asp_pen_max <= 0 or n_tok <= 0:
            return 1.0
        excess = max(0.0, n_tok - self._asp_free) / self._asp_ramp
        return 1.0 - self._asp_pen_max * min(1.0, excess)

    def _aggregate(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        if self._agg_mode == 'min':
            return float(min(scores))
        if self._agg_mode == 'geomean':
            p = 1.0
            for s in scores:
                p *= max(0.0, float(s))
            return float(p ** (1.0 / len(scores)))
        return float(sum(scores) / len(scores))

    @staticmethod
    def _route(task: str) -> str:
        t = _norm_tag(task)
        if t == 'asp':
            return 'asp'
        if t == 'bbox':
            return 'bbox'
        if t in ('kgraph', 'kg'):
            return 'kgraph'
        if t in ('comparison', 'compare'):
            return 'comparison'
        return 'report'                                  # ontology / cls_json / cls_tokens / default

    @staticmethod
    def _subtasks(task_str) -> List[str]:
        return [_norm_tag(s) for s in re.split(r'[;,]', str(task_str or '')) if s.strip()]

    def _score_one(self, completion: str, task: str, ref, base_kwargs: dict, i: int, n: int) -> float:
        """Score a single (completion, task) with the right sub-reward (1-element batch)."""
        kw = {}
        for k, v in base_kwargs.items():
            kw[k] = [v[i]] if (isinstance(v, (list, tuple)) and len(v) == n) else v
        kw['task'] = [task]
        kw['reference'] = [ref if isinstance(ref, str) else json.dumps(ref or {})]
        return self._sub[self._route(task)]([completion], **kw)[0]

    def __call__(self, completions, **kwargs) -> List[float]:
        n = len(completions)
        tasks = _col(kwargs, 'task', n)
        refs = _col(kwargs, 'reference', n)
        out = [0.0] * n
        agg: Dict[str, List[float]] = {}        # 'label/component' -> per-row values (for wandb)

        def _grab(label, sub, idxs=None):
            """Collect a sub-reward's last_components into agg under `label/<comp>`."""
            lc = getattr(sub, 'last_components', None)
            if not lc:
                return
            for ck, cv in lc.items():
                if not isinstance(cv, list) or not cv:
                    continue
                vals = cv if idxs is None else cv                # bucket: full batch; compound: 1-elt
                agg.setdefault(f'{label}/{ck}', []).extend(float(x) for x in vals)

        # single-task rows: batch by route (efficient). compound rows: handle per-row.
        buckets: Dict[str, List[int]] = {}
        compound: List[int] = []
        subs_of: Dict[int, List[str]] = {}
        for i, t in enumerate(tasks):
            subs = self._subtasks(t)
            subs_of[i] = subs
            if len(subs) <= 1:
                buckets.setdefault(self._route(subs[0] if subs else ''), []).append(i)
            else:
                compound.append(i)

        for key, idxs in buckets.items():
            sub_completions = [completions[i] for i in idxs]
            sub_kwargs = {}
            for k, v in kwargs.items():
                sub_kwargs[k] = [v[i] for i in idxs] if (isinstance(v, (list, tuple)) and len(v) == n) else v
            res = self._sub[key](sub_completions, **sub_kwargs)
            for j, i in enumerate(idxs):
                out[i] = res[j]
                if subs_of[i] == ['cls_tokens']:                  # standalone cls_tokens: format enforcement
                    notag = not has_cls_tag_tokens(completions[i])
                    agg.setdefault('format/cls_tokens_notag', []).append(float(notag))
                    if notag and self._notag_mult < 1.0:
                        out[i] *= self._notag_mult
            _grab(key, self._sub[key])                            # whole-batch components for this route

        for i in compound:
            subs = subs_of[i]
            segs = segment_completion(completions[i], subs)
            try:
                ref_obj = json.loads(refs[i]) if isinstance(refs[i], str) else (refs[i] or {})
            except Exception:
                ref_obj = {}
            scores = []
            for t in subs:
                # task-keyed reference if present, else the shared reference
                sub_ref = ref_obj.get(t) if isinstance(ref_obj, dict) and t in ref_obj else ref_obj
                seg_text = segs.get(t, completions[i])
                s = self._score_one(seg_text, t, sub_ref, kwargs, i, n)
                if t == 'cls_tokens':                             # compound: penalise only the cls_tokens part
                    notag = not has_cls_tag_tokens(seg_text)
                    agg.setdefault('format/cls_tokens_notag', []).append(float(notag))
                    if notag and self._notag_mult < 1.0:
                        s *= self._notag_mult
                scores.append(s)
                _grab(t, self._sub[self._route(t)])               # this row's components, keyed by subtask
            out[i] = self._aggregate(scores)

        # ASP length penalty on the whole rollout (hinge above FREE tokens), for the configured tasks.
        if self._asp_pen_max > 0:
            for i in range(n):
                if not (set(subs_of[i]) & self._asp_pen_tasks):
                    continue
                n_tok = self._asp_tokens(completions[i])
                mult = self._asp_mult(n_tok)
                out[i] *= mult
                agg.setdefault('format/asp_present', []).append(float(n_tok > 0))
                agg.setdefault('format/asp_tokens', []).append(float(n_tok))
                agg.setdefault('format/asp_pen_mult', []).append(mult)

        # per-task-string reward means + all sub-reward component means -> wandb (rank0, best-effort)
        for i, t in enumerate(tasks):
            agg.setdefault(f'task_reward/{t}', []).append(out[i])
        self._log_wandb(agg, out, kwargs)
        return out

    def _log_wandb(self, agg: Dict[str, List[float]], out: List[float], kwargs) -> None:
        """Log per-task rewards + every sub-reward component mean to wandb. Rank-0 only, commit=False
        (attaches to the trainer's next logged step), and never raises into the training loop."""
        try:
            if str(os.environ.get('RANK', '0')) != '0':
                return
            import wandb
            if wandb.run is None:
                return
            data = {'reward/overall_mean': (sum(out) / len(out)) if out else 0.0}
            for k, v in agg.items():
                if v:
                    data[f'reward/{k}'] = sum(v) / len(v)
            wandb.log(data, commit=False)
        except Exception:
            pass


orms['radiology'] = RadiologyRouter
