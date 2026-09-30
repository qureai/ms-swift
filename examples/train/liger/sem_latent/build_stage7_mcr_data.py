#!/usr/bin/env python
"""Stage-7 data: MIMIC-CXR + CheXpert + ReXGradient ONLY ("mcr").

The user's rule for stage 7: "Use only mimic chexpert and rex gradient data". This script writes NEW files
(nothing existing is modified) under --out_dir (default
/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/stage7_mcr/):

  <stem>.mcr.jsonl   for each of the 5 train + 2 val files of the stage-6c mixture: the SAME raw lines (byte for
                     byte, original order) of the rows whose images ALL come from MIMIC-CXR or CheXpert
  REX_TRAIN.jsonl    ReXGradient-160K, official TRAIN split only, one row per study
  REX_VAL.jsonl      ReXGradient-160K, official VALID split, <= --rex_val_rows studies (one per patient)
  MANIFEST.json      sources (size/mtime), predicates, per-file kept/dropped counts, per-family counts, ReX
                     funnel + leakage assertions, seed, created, the launcher's DATASETS / VAL_DATASETS strings
  README.md          the same, for humans
The ReXGradient TEST split never appears anywhere (asserted, see LEAKAGE below).

----------------------------------------------------------------------------------------------------------------
(a) MIMIC/CheXpert FILTER over the stage-6c mixture (the rows are copied verbatim, only rows are dropped)
    inputs  A_TRAIN_120k, B3_TRAIN, 15_stage5_headguided_TRAIN, 17_refresher_multiturn_TRAIN, RET_diverse60k_TRAIN;
            val A_VAL, B3_VAL (all JSONL even when named .json; one row per line; checked from the first byte)
    image source of a path (== the scout predicate scratchpad/mcr/mc_filter.py, which mirrors
    swift_plugins/sem_targets.normalise_image_path): normpath, drop swift's image_<W>x<H>_ basename prefix, strip
    the storage root /localstorage | /fastdata3 | /fastdata4, then
        /cxr_data/opensource_datasets/mimic/...     -> mimic
        /cxr_data/opensource_datasets/chexpert/...  -> chexpert
        /cxr_data/opensource_datasets/rexgradient/  -> rexgradient   (0 rows in the mixture; dropped here anyway:
                                                                      ReX enters only through REX_TRAIN/REX_VAL)
        /cxr_data/internal_data/training/images/images/{mimic.,mim.,chex.}* -> mimic_int / chexpert_int
             (internal RE-EXPORTS of MIMIC/CheXpert films; not on disk on this host, and none carries a report
              turn -- dropped: KEEP is strictly the two open-source trees)
        anything else (segmed, internal, fake_data, B3 crops, relative paths) -> other
    KEEP a row  <=>  it has >= 1 image AND every image is mimic or chexpert. Rows with 0 images (text-only) are
    dropped (there are none). Drop reasons are counted: no_images | mixed (a MIMIC/CheXpert image next to another
    source) | internal_copy_only | internal_copy_mixed | rexgradient | other_only | parse_error.
    Checked on the kept rows: every distinct image path exists under some storage root (resolved exactly like
    swift/template/vision_utils._resolve_storage_path), a seeded sample decodes with PIL, no ReXGradient id /
    path appears in any kept line.

(b) ReXGradient-160K rows
    sources  $REX/metadata/{train,valid}_metadata_view_position.json (the HF files with ImageViewPosition filled;
             the plain *_metadata.json have it empty for 207,688 train images), $REX/rex_cases_frontal.jsonl (the
             frontal pick: PA > AP > UNKNOWN/None > first), $REX/rex_cases_frontal_hires.jsonl (its image:
             png_hires/<key>.png = 8-bit, percentile 0.5-99.5 stretch, Lanczos upscale to min side 1500,
             /fastdata4/sahil_work/convert8bit_hires.py) -- EXACTLY the image and path string the ckpt-608
             public-test benchmark fed the model (run_rex_hires.sh CASES). The raw deid_png files are I;16 and
             swift's load_image .convert('RGB') saturates 54% of their pixels: never used.
    study filters, in this order (each drop counted in MANIFEST rex.<split>.funnel):
        view of the picked image not frontal (LATERAL, LL, RAO, OTHER, ...)      -> drop
        body part of the picked image not in {Chest, ''}                        -> drop
        StudyDescription non-chest (sternum, abdomen/KUB/UGI/VCUG, bone survey, extremities, spine, ...) or
        ribs-only, or unknown ('[ID]', 'PAPERORDER', '') with a non-Chest body part -> drop
        de-identification token ([DATE] [HOSPITAL] [AGE] [URL] [IP_ADDRESS] ...) in Findings or Impression -> drop
        non-ASCII left after mapping ' ' - - " " and dropping U+FFFD / combining marks   -> drop
        no Findings sentence left after removing pure view/technique sentences ("PA and lateral chest
        radiographs are provided.", "Two views.", ...), junk ("Continued on next page", "Insert normal check",
        "Not explicitly stated"), header-only comparison/indication lines and pure dated comparison references,
        or Findings shorter than 10 characters                                  -> drop
        png_hires image missing                                                 -> drop
        canonical-target self-check failed (below)                              -> drop
        more than --rex_findings_cap studies share the identical cleaned Findings -> the extra ones (seeded
        order) are dropped. The 12,953-copy "The heart size and mediastinal contours are within normal limits.
        Both lungs are clear. ..." macro is the only string above 1,000 copies (MANIFEST records the effect of
        other caps).
    Impression pointer units ("Please see above.", "As discussed above.") are removed; a study whose Impression
    is then empty keeps its Findings and is routed to the single-turn sec_findings family (impression-free).
    ONE per-study transform, reused by every family, so the canonical SEM target is identical across families
    for the same study (sem_targets.py assumption):
        Findings  whitespace collapsed, leading "Findings:" header stripped, split into sentences
                  ((?<=[.!?])\\s+(?=[A-Z(]); a sequential "1. 2. 3." list is split into its items first), each
                  sentence one "* " bullet, radiologist's order and negatives kept (the stage-6c answers are
                  Gemini "* " bullets: sec_findings 98.8%, medrep/onto/fict Findings 98.7-99.9%)
        Impression prose (inline numbering kept, renumbered after removals; a single item loses its "1."),
                  sentences with recommendation wording (recommend, follow-up, consider, clinical correlation,
                  CT ..., nipple markers) MOVED to Recommendations as Gemini did for MIMIC -- never emptying it
        Recommendations  the moved sentences, else the fixed stage-6c sentence
                  "No further imaging recommendations based on current findings." (medrep 92.3%, fict 88.4%)
    answer serialisation = the stage-6c formats (formats-scout census):
        sec_findings  "\\n" + bullets          (73% of MIMIC sec_findings answers start with "\\n* ")
        sec_impression  Impression
        medrep        json.dumps({"Findings": "\\n"+bullets, "Impression": I, "Recommendations": R}, indent=2)
        fict          "Findings:\\n{bullets}\\n\\nImpression:\\n{I}\\n\\nRecommendations:\\n{R}"
        cls           the detailed cls-JSON first turn: the Gemini-3-Flash tag extraction of THIS study's report
                      ($LITELLM/10_april_chexpert_rexgradient_tag_extraction_200k_unique_reports_outputs,
                      joined on the whitespace-normalised "Indication/Comparison/Findings/Impression" report),
                      passed through normalize_json_target -- a verbatim copy of
                      vlm_inference_analysis/scripts/build_cls_json_multiturn_data.py:normalize_json_target, the
                      function that built every stage-6c cls answer (404/404 MIMIC A_VAL cls answers reproduced
                      byte for byte while writing this). Studies whose report has two different extractions, or
                      none that parses to the 7 ontology categories, get no cls turn.
    prompts: the exact stage-6c strings (asserted: each one must occur as a cleaned T1/cls prompt in the
    filtered MIMIC/CheXpert files); <image> once, at the end of the first user turn after "\\n".
    family mix (--rex_mix; one row per study, drawn from sha256(seed|key), renormalised over the families the
    study can take):
        fict          30%  [fict]                                  (what a ReX fict eval would ask)
        cls_fict      20%  [cls, fict]                             (HG "cls, then report" shape)
        cls_sec2      20%  [cls, two of {sec_findings, sec_impression, medrep}, sec_* before medrep]
                           (template-G shape; medrep after sec_findings is SEM-leak-masked but still CE-trained)
        sec_findings  20%  [sec_findings]                          (clean image -> Findings SEM signal)
        medrep        10%  [medrep]
    Excluded on purpose: single-turn sec_impression (49% duplicate Impressions, median 7 words), sec_summary /
    onto / reasoning / prominence / sjson (their content would have to be invented), Indication/Comparison in the
    prompt (never seen in training; prior images unavailable).
    rows: {"messages": [{"role", "content"}...], "images": ["/localstorage/sahil_work/ReXGradient-160K/png_hires/
    <key>.png"], "source": "rexgradient", "rex_key", "rex_split", "rex_family"} -- no system message (the launcher
    passes --system), no tools key; the extra columns are dropped by swift's loader (remove_useless_columns).
    Output order: sorted by sha256(seed|order|key) (so a swift '#N' prefix is an unbiased sample).
    Canonical self-check: every answer of every family is run through sem_targets.turn_target with its prompt;
    each must classify as its own family and give the same Findings (and Impression) body.

LEAKAGE (asserted; any violation aborts before anything is renamed into place)
    REX_TRAIN: every key in the TRAIN metadata; PatientID / AccessionNumber / StudyInstanceUid not in the VALID or
    TEST metadata; PatientID not in audit_datasplit/rexgradient_clean_patients.txt (the 13,771 valid+test
    patients). REX_VAL: every key in the VALID metadata; ids not in TRAIN or TEST. Both: the study key, patient
    id and every GRDN* token of the written line are not a TEST key/patient/accession; the image path (storage
    root stripped) is not a test png_hires / png8 / deid_png / opensource deid_png / rexgradient_eval img8 path;
    no key in agentic_paper_metrics/rextest_gt.jsonl, rextest_keys_*.json, audit_datasplit/rex_test_clean_ids.json,
    rexgradient_eval/test_images_from_vqa.json (by study path) or rexgradient_eval/img8/<patient>. The written
    files are re-read and re-checked. The MCR files: 0 ReXGradient paths and 0 GRDN tokens. NOTE: the ckpt-608
    "all 160k" ReX generations (reports_*, hr_*, r8_*, rex_chexbert.py all_f1) are contaminated for the train
    split once a model trains on REX_TRAIN; report ReX only on the public test split (and valid).

ASL / labels: ReX images have no 132-tag labels in cls_label_sidecar (checked: no basename / last-3-component
collision), so mldecoder_joint_plugin masks their ASL term per image. The licence of ReXGradient-160K is
non-commercial research only (README.md there): keep these files internal, a checkpoint trained on them is
research-only.

DETERMINISM: every random choice is a function of (--seed, study key); the MCR pass is order-preserving.
Re-running with the same inputs and flags writes byte-identical files (MANIFEST['outputs'][*]['sha256']).

USAGE (CPU only; ~5 min on 64 workers; refuses to overwrite existing outputs without --overwrite)
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<repo> <env python> examples/train/liger/sem_latent/build_stage7_mcr_data.py
    ... --skip_mcr (ReX only) | --skip_rex (MCR only) | --out_dir <scratch> (a dry build somewhere else)
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import platform
import random
import re
import shutil
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Tuple

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from swift_plugins import sem_targets as st  # noqa: E402  (pure: stdlib + numpy)

DATA_DIR = '/fastdata4/sahil_work/VLM_qure/data_preparation/training_data'
S6 = f'{DATA_DIR}/stage6_toolcalling'
DEFAULT_TRAIN = [f'{S6}/A_TRAIN_120k.jsonl', f'{S6}/B3_TRAIN.jsonl', f'{DATA_DIR}/15_stage5_headguided_TRAIN.json',
                 f'{DATA_DIR}/17_refresher_multiturn_TRAIN.json', f'{S6}/RET_diverse60k_TRAIN.jsonl']
DEFAULT_VAL = [f'{S6}/A_VAL.jsonl', f'{S6}/B3_VAL.jsonl']
DEFAULT_OUT = f'{DATA_DIR}/stage7_mcr'
REX_ROOT = '/fastdata4/sahil_work/ReXGradient-160K'
REX_IMAGE_PREFIX = '/localstorage/sahil_work/ReXGradient-160K/png_hires'   # == rex_cases_frontal_hires.jsonl
TAG_DIR = ('/fastdata4/sahil_work/litellm_inference/'
           '10_april_chexpert_rexgradient_tag_extraction_200k_unique_reports_outputs')
CLS_BUILDER = f'{_REPO}/vlm_inference_analysis/scripts/build_cls_json_multiturn_data.py'
SW = '/fastdata4/sahil_work'
EVAL_TEST_KEY_FILES = [f'{SW}/agentic_paper_metrics/rextest_gt.jsonl', f'{SW}/audit_datasplit/rex_test_clean_ids.json']
EVAL_TEST_KEY_GLOB = f'{SW}/agentic_paper_metrics/rextest_keys_*.json'
EVAL_TEST_VQA_IMAGES = f'{SW}/rexgradient_eval/test_images_from_vqa.json'
EVAL_TEST_IMG8 = f'{SW}/rexgradient_eval/img8'
CLEAN_PATIENTS = f'{SW}/audit_datasplit/rexgradient_clean_patients.txt'
CLS_LABEL_SIDECAR = f'{DATA_DIR}/cls_label_sidecar/labels.parquet'
DEFAULT_SEED = 20260930
DEFAULT_MIX = 'fict=0.30,cls_fict=0.20,cls_sec2=0.20,sec_findings=0.20,medrep=0.10'

# ---- prompts: the exact stage-6c strings (asserted against the filtered files at run time) --------------------
P_FICT = 'Generate the Findings, Impression, and CT / follow-up recommendations for the given chest X-ray.'
P_FIND = 'Generate the Findings of the given chest X-ray.'
P_IMP = 'Generate the Impression of the given chest X-ray.'
P_MED = 'Generate a medical report of the given X-ray with Findings, Impression, and Recommendations.'
P_CLS = ('Classify the abnormalities in the given chest x-ray as a detailed JSON report. Return a single JSON object '
         'with the ontology categories general, lung_cancer, copd, ild, pulmonary_hypertension, edema and '
         'other_findings. Under each category, report every finding of that category as an object with the keys '
         'presence, size, location and characteristics. Set presence to true when the finding is present in the '
         'image and false when it is absent, and report every finding including the absent ones. For an absent '
         'finding set size, location and characteristics to null; for a present finding give a very short '
         'characterization in each, or null where it does not apply.')
FAMILY_PROMPT = {'fict': P_FICT, 'sec_findings': P_FIND, 'sec_impression': P_IMP, 'medrep': P_MED}
CLS_CATEGORIES = ('general', 'lung_cancer', 'copd', 'ild', 'pulmonary_hypertension', 'edema', 'other_findings')
REC_BOILER = 'No further imaging recommendations based on current findings.'
SYSTEM = 'You are a helpful medical assistant.'   # 20_stage7_sem_latent.sh --system (for the leak statistics)

# ============================================================================================================
# (a) source predicate (== scratchpad/mcr/mc_filter.py)
# ============================================================================================================
STORAGE_ROOTS = ('/localstorage', '/fastdata3', '/fastdata4')
_WXH_PREFIX_RE = re.compile(r'^image_\d+x\d+_(.+)$')
_OPEN_RE = re.compile(r'^/cxr_data/opensource_datasets/(mimic|chexpert|rexgradient)/')
_INTERNAL_RE = re.compile(r'^/cxr_data/internal_data/training/images/images/(mimic|mim|chex)\.')
_INTERNAL_SRC = {'mimic': 'mimic_int', 'mim': 'mimic_int', 'chex': 'chexpert_int'}
KEEP = frozenset({'mimic', 'chexpert'})
INTERNAL = frozenset({'mimic_int', 'chexpert_int'})
GRDN_RE = re.compile(r'GRDN[0-9A-Z]{12}')
REX_SUBSTR_RE = re.compile(r'rexgradient', re.I)


def norm_path(path: Any) -> Optional[str]:
    """Storage-root-free normalised path (None: no usable path / unknown root)."""
    if isinstance(path, dict):
        path = path.get('path')
    if not isinstance(path, str) or not path.strip():
        return None
    p = os.path.normpath(path.strip())
    d, b = os.path.split(p)
    m = _WXH_PREFIX_RE.match(b)
    if m:
        p = os.path.join(d, m.group(1))
    for root in STORAGE_ROOTS:
        if p.startswith(root + '/'):
            return p[len(root):]
    return None


def image_source(path: Any) -> Optional[str]:
    p = norm_path(path)
    if p is None:
        return None
    m = _OPEN_RE.match(p)
    if m:
        return m.group(1)
    m = _INTERNAL_RE.match(p)
    if m:
        return _INTERNAL_SRC[m.group(1)]
    return 'other'


def row_images(row: Dict[str, Any]) -> List[Any]:
    imgs = row.get('images', row.get('image'))
    if isinstance(imgs, (str, dict)):
        imgs = [imgs]
    return list(imgs or ())


def classify_row(row: Dict[str, Any]) -> Tuple[str, List[Optional[str]]]:
    """('keep' | drop reason, per-image sources)."""
    srcs = [image_source(x) for x in row_images(row)]
    if not srcs:
        return 'no_images', srcs
    if all(s in KEEP for s in srcs):
        return 'keep', srcs
    if any(s == 'rexgradient' for s in srcs):
        return 'rexgradient', srcs
    if any(s in KEEP for s in srcs):
        return 'mixed', srcs
    if all(s in INTERNAL for s in srcs):
        return 'internal_copy_only', srcs
    if any(s in INTERNAL for s in srcs):
        return 'internal_copy_mixed', srcs
    return 'other_only', srcs


def resolve_storage_path(path: str) -> Optional[str]:
    """== swift/template/vision_utils._resolve_storage_path (exists as given, else the same suffix under another
    storage root, else None)."""
    if os.path.exists(path):
        return path
    for root in STORAGE_ROOTS:
        if path.startswith(root + '/'):
            suffix = path[len(root):]
            for alt in STORAGE_ROOTS:
                if alt != root and os.path.exists(alt + suffix):
                    return alt + suffix
            return None
    return None


def row_t1(row: Dict[str, Any], system: str = SYSTEM) -> List[Tuple[str, str, bool]]:
    """(family, canonical, leaked) per raw assistant message with a T1 target: exactly the targets of
    sem_targets.iter_row_targets, with the plugin's leak rule -- prior = the system prompt + every message BEFORE
    the target's merged assistant turn (sem_targets.assistant_turn_groups; sem_latent_plugin.row_block_targets sees
    swift's merged messages, so B3 tool traces inside the same turn are NOT prior context)."""
    msgs = st.row_messages(row)
    start_of = {}
    for g in st.assistant_turn_groups(msgs):
        for i in g:
            start_of[i] = g[0]
    out = []
    last_user = None
    for i, m in enumerate(msgs):
        if m['role'] == 'user':
            last_user = m['content']
            continue
        if m['role'] != 'assistant':
            continue
        tt = st.turn_target(last_user if last_user is not None else '', m['content'])
        if tt is not None:
            prior = [system] + [msgs[k]['content'] for k in range(start_of.get(i, i))]
            out.append((tt[0], tt[1], bool(st.is_leaked(tt[1], prior))))
    return out


# ---- MCR worker ------------------------------------------------------------------------------------------------
def _chunks(path: str, chunk_bytes: int) -> List[Tuple[int, int]]:
    size = os.path.getsize(path)
    out, s = [], 0
    while s < size:
        e = min(size, s + chunk_bytes)
        out.append((s, e))
        s = e
    return out or [(0, 0)]


def _iter_lines(path: str, start: int, end: int):
    """Lines whose FIRST byte lies in [start, end) (so byte-range chunks partition the file exactly)."""
    with open(path, 'rb') as f:
        if start > 0:
            f.seek(start - 1)
            f.readline()               # consumes through the newline at/after start-1
        while True:
            pos = f.tell()
            if pos >= end:
                break
            line = f.readline()
            if not line:
                break
            yield pos, line


def mcr_worker(job):
    fi, path, start, end, part_path = job
    c = collections.Counter()
    src_combo = collections.Counter()
    tmpl = collections.Counter()
    t1 = collections.Counter()
    t1_nl = collections.Counter()
    t1_keys = collections.defaultdict(set)
    prompts = collections.Counter()
    cls_prompts = collections.Counter()
    images = set()
    n_img = collections.Counter()
    grdn, rexsub = set(), 0
    with open(part_path, 'wb') as out:
        for pos, raw in _iter_lines(path, start, end):
            if not raw.strip():
                c['blank_lines'] += 1
                continue
            c['rows'] += 1
            try:
                row = json.loads(raw)
            except Exception:  # noqa: BLE001
                c['drop:parse_error'] += 1
                continue
            if not isinstance(row, dict):
                c['drop:parse_error'] += 1
                continue
            why, srcs = classify_row(row)
            combo = '+'.join(sorted(set(s or 'none' for s in srcs))) or 'no_images'
            src_combo[combo] += 1
            if why != 'keep':
                c[f'drop:{why}'] += 1
                continue
            c['kept'] += 1
            for s in srcs:
                c[f'kept_images:{s}'] += 1
            n_img[len(srcs)] += 1
            for x in row_images(row):
                images.add(x.get('path') if isinstance(x, dict) else x)
            t = row.get('template') or row.get('family')
            if t is not None:
                tmpl[str(t)] += 1
            if row.get('tools'):
                c['kept_rows_with_tools'] += 1
            for fam, can, leaked in row_t1(row):
                t1[fam] += 1
                t1_keys[fam].add(st.target_key(can))
                if not leaked:
                    t1_nl[fam] += 1
            last_user = None
            for m in st.row_messages(row):
                if m['role'] == 'user':
                    last_user = m['content']
                    cp = st.clean_user_prompt(m['content'])
                    if cp.startswith('Classify the abnormalities'):
                        cls_prompts[cp] += 1
                elif m['role'] == 'assistant':
                    fam = st.classify_turn(last_user if last_user is not None else '', m['content'])
                    if fam is not None:
                        prompts[(fam, st.clean_user_prompt(last_user))] += 1
            text = raw.decode('utf-8', 'replace')
            grdn.update(GRDN_RE.findall(text))
            rexsub += bool(REX_SUBSTR_RE.search(text))
            out.write(raw if raw.endswith(b'\n') else raw + b'\n')
    return {'fi': fi, 'counts': dict(c), 'src_combo': dict(src_combo), 'template': dict(tmpl), 't1': dict(t1),
            't1_nonleaked': dict(t1_nl), 't1_keys': {k: list(v) for k, v in t1_keys.items()},
            'prompts': [[k[0], k[1], v] for k, v in prompts.items()], 'cls_prompts': dict(cls_prompts),
            'images': list(images), 'n_images_per_row': dict(n_img), 'grdn': sorted(grdn), 'rexsub': rexsub,
            'start': start, 'end': end}


def _first_byte(path: str) -> str:
    with open(path, 'rb') as f:
        head = f.read(4096).lstrip()
    return chr(head[0]) if head else ''


def _stat(path: str) -> Dict[str, Any]:
    s = os.stat(path)
    return {'path': path, 'size': s.st_size, 'mtime': int(s.st_mtime)}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def _decode_check(paths: List[str]) -> Dict[str, Any]:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    modes, bad, sizes = collections.Counter(), [], []
    for p in paths:
        q = resolve_storage_path(p)
        try:
            with Image.open(q) as im:
                im.load()
                modes[im.mode] += 1
                sizes.append(min(im.size))
        except Exception as e:  # noqa: BLE001
            bad.append(f'{p}: {type(e).__name__}: {e}')
    sizes.sort()
    return {'n': len(paths), 'ok': len(paths) - len(bad), 'modes': dict(modes), 'bad': bad[:10],
            'min_side_min': sizes[0] if sizes else None, 'min_side_p50': sizes[len(sizes) // 2] if sizes else None}


def _exist_check(paths: List[str], threads: int) -> Dict[str, Any]:
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(resolve_storage_path, paths, chunksize=256))
    missing = [p for p, r in zip(paths, res) if r is None]
    roots = collections.Counter(next((rt for rt in STORAGE_ROOTS if (r or '').startswith(rt + '/')), 'none')
                                for r in res)
    return {'n': len(paths), 'missing': len(missing), 'missing_examples': missing[:10],
            'resolved_root': dict(roots)}


def run_mcr(args, pool, out_dir: str, tmp_dir: str, manifest: Dict[str, Any], log) -> Dict[str, Any]:
    files = [(p, 'train') for p in args.train_files] + [(p, 'val') for p in args.val_files]
    jobs, meta = [], []
    for fi, (p, role) in enumerate(files):
        p = st.resolve_storage_path(p)
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        if _first_byte(p) != '{':
            raise ValueError(f'{p}: not JSONL (first byte {_first_byte(p)!r}); only JSONL inputs are supported')
        stem = os.path.basename(p).rsplit('.', 1)[0]
        outp = os.path.join(out_dir, f'{stem}.mcr.jsonl')
        meta.append({'input': _stat(p), 'role': role, 'stem': stem, 'out': outp})
        pdir = os.path.join(tmp_dir, 'parts', f'{fi:02d}_{stem}')
        os.makedirs(pdir, exist_ok=True)
        for ci, (s, e) in enumerate(_chunks(p, args.chunk_mb << 20)):
            jobs.append((fi, p, s, e, os.path.join(pdir, f'{ci:05d}.jsonl')))
    log(f'MCR: {len(files)} files, {len(jobs)} byte-range chunks, {args.workers} workers')
    t0 = time.time()
    res = list(pool.imap(mcr_worker, jobs, chunksize=1))
    log(f'MCR: scanned in {time.time() - t0:.0f}s')
    per_file = []
    all_prompts = collections.Counter()
    all_cls_prompts = collections.Counter()
    for fi, m in enumerate(meta):
        rs = [r for r in res if r['fi'] == fi]
        rs.sort(key=lambda r: r['start'])
        agg = {k: collections.Counter() for k in ('counts', 'src_combo', 'template', 't1', 't1_nonleaked',
                                                   'cls_prompts', 'n_images_per_row')}
        keys = collections.defaultdict(set)
        imgs, grdn, rexsub = set(), set(), 0
        for r in rs:
            for k in agg:
                agg[k].update(r[k])
            for fam, ks in r['t1_keys'].items():
                keys[fam].update(ks)
            imgs.update(r['images'])
            grdn.update(r['grdn'])
            rexsub += r['rexsub']
            for fam, pr, v in r['prompts']:
                all_prompts[(fam, pr)] += v
        all_cls_prompts.update(agg['cls_prompts'])
        cnt = agg['counts']
        # rows are partitioned exactly by the byte ranges; cross-check with a newline count
        with open(m['input']['path'], 'rb') as f:
            n_lines = sum(b.count(b'\n') for b in iter(lambda: f.read(1 << 24), b''))
        m.update({
            'rows': cnt['rows'], 'kept': cnt['kept'], 'kept_frac': round(cnt['kept'] / max(cnt['rows'], 1), 4),
            'newlines_in_input': n_lines,
            'dropped': {k[5:]: v for k, v in sorted(cnt.items()) if k.startswith('drop:')},
            'kept_images_by_source': {k[12:]: v for k, v in sorted(cnt.items()) if k.startswith('kept_images:')},
            'kept_rows_with_tools': cnt['kept_rows_with_tools'],
            'kept_images_per_row': {str(k): v for k, v in sorted(agg['n_images_per_row'].items())},
            'source_combos_all_rows': dict(agg['src_combo'].most_common()),
            'kept_by_template_or_family': dict(sorted(agg['template'].items())),
            't1_turns_kept': dict(agg['t1']), 't1_turns_kept_nonleaked': dict(agg['t1_nonleaked']),
            't1_unique_texts_kept': {f: len(v) for f, v in keys.items()},
            'distinct_images_kept': len(imgs), 'grdn_tokens_in_kept_lines': sorted(grdn)[:20],
            'n_grdn_tokens': len(grdn), 'kept_lines_mentioning_rexgradient': rexsub,
        })
        m['_images'] = imgs
        m['_keys'] = keys
        m['_parts'] = [j[4] for j in jobs if j[0] == fi]
        m['_parts'].sort()
        per_file.append(m)
        log(f"  {m['stem']}: rows {m['rows']:,} kept {m['kept']:,} ({m['kept_frac']:.1%}) T1 {sum(agg['t1'].values()):,}"
            f" dropped {m['dropped']}")
        if cnt['rows'] + cnt['blank_lines'] != n_lines:
            raise RuntimeError(f"{m['stem']}: parsed {cnt['rows']} rows + {cnt['blank_lines']} blank != {n_lines} lines")
        if cnt.get('drop:parse_error'):
            raise RuntimeError(f"{m['stem']}: {cnt['drop:parse_error']} unparseable rows")
        if grdn or rexsub:
            raise RuntimeError(f"{m['stem']}: kept MIMIC/CheXpert lines mention ReXGradient ({len(grdn)} GRDN tokens, "
                               f'{rexsub} lines) -- refusing')
    # images of the kept rows
    all_imgs = sorted(set().union(*[m['_images'] for m in per_file]))
    ex = _exist_check(all_imgs, args.io_threads)
    log(f"MCR images: {ex['n']:,} distinct, missing {ex['missing']}, roots {ex['resolved_root']}")
    if ex['missing']:
        raise RuntimeError(f"MCR: {ex['missing']} kept images do not exist: {ex['missing_examples']}")
    rng = random.Random(args.seed)
    dec = _decode_check(rng.sample(all_imgs, min(args.decode_sample, len(all_imgs))))
    log(f"MCR decode sample: {dec}")
    if dec['ok'] != dec['n']:
        raise RuntimeError(f'MCR decode failures: {dec["bad"]}')
    # val/train overlap (inherited from the stage-6c split; reported, not changed)
    tr_imgs = set().union(*[m['_images'] for m in per_file if m['role'] == 'train'])
    tr_norm = {norm_path(p) for p in tr_imgs}
    tr_keys = collections.defaultdict(set)
    for m in per_file:
        if m['role'] == 'train':
            for f, v in m['_keys'].items():
                tr_keys[f].update(v)
    for m in per_file:
        if m['role'] == 'val':
            vn = {norm_path(p) for p in m['_images']}
            vk = set().union(*m['_keys'].values()) if m['_keys'] else set()
            tk = set().union(*tr_keys.values()) if tr_keys else set()
            m['val_images_also_in_train'] = len(vn & tr_norm)
            m['val_t1_texts_also_in_train'] = len(vk & tk)
    return {'files': per_file, 'image_exist_check': ex, 'image_decode_check': dec, 'prompts': all_prompts,
            'cls_prompts': all_cls_prompts,
            't1_unique_texts_train_total': {f: len(v) for f, v in tr_keys.items()}}


def finalize_mcr(mcr: Dict[str, Any], log) -> None:
    for m in mcr['files']:
        tmp = m['out'] + '.tmp'
        with open(tmp, 'wb') as out:
            for pp in m['_parts']:
                with open(pp, 'rb') as f:
                    shutil.copyfileobj(f, out, 1 << 24)
            out.flush()
            os.fsync(out.fileno())
        n = 0
        with open(tmp, 'rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''):
                n += b.count(b'\n')
        if n != m['kept']:
            raise RuntimeError(f"{tmp}: {n} lines != kept {m['kept']}")
        m['_tmp'] = tmp


# ============================================================================================================
# (b) ReXGradient
# ============================================================================================================
SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z(])')
HDR_RE = re.compile(r'(?i)^\s*(?:findings|impression|conclusions?|summary)\s*:\s*')
DEID_RE = re.compile(r'\[[A-Z][A-Z_ ]*\]')
DATE_RE = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
NUM_ITEM_RE = re.compile(r'(?:^|(?<=\s))\(?(\d{1,2})[.)]\s+(?=\S)')
ASCII_MAP = {'’': "'", '‘': "'", '–': '-', '—': '-', '“': '"', '”': '"',
             '�': '', ' ': ' ', '•': '*'}
REC_RE = re.compile(  # == scratchpad/mcr/rex_transform_proto.py REC (formats scout: moves 6.2% of Impressions)
    r'(?i)\b(recommend\w*|follow[- ]?up (is|could|may|can|with|imaging|radiograph|chest|study|ct|recommended)|consider\w*|'
    r'suggest(ed)? (ct|further|follow|repeat|correlation)|clinical(ly)? correlat\w*|further (evaluation|imaging)|'
    r'repeat (radiograph|imaging|chest|pa)|(chest )?ct (scan )?(could|may|is|of the chest|would|can)|nipple markers)\b')
POINTER_RE = re.compile(
    r'(?i)^(?:please\s+)?(?:(?:see|refer\s+to)\s+(?:the\s+)?(?:above|findings(?:\s+above)?|body\s+of\s+(?:the\s+)?report|'
    r'report(?:\s+above)?|discussion(?:\s+above)?)|as\s+(?:discussed\s+|described\s+|noted\s+|detailed\s+|stated\s+|'
    r'dictated\s+|mentioned\s+)?above|same\s+as\s+above)\s*\.?$')
JUNK_SENT_RE = re.compile(r'(?i)^(?:continued\s+on\s+next\s+page|insert\s+normal(?:\s+check)?|not\s+(?:explicitly\s+)?stated|'
                          r'see\s+(?:impression|below|above)|none|n/?a)\s*\.?$')
META_LINE_RE = re.compile(r'(?i)^(?:comparison|indication|history|clinical\s+history|technique)\s*:')
TECH_WORDS = frozenset((
    'pa ap and lateral lat frontal single one two three four 1 2 3 4 portable upright erect semi-upright semi '
    'semiupright seated sitting supine view views of the chest radiograph radiographs radiography film films image '
    'images projection projections study studies exam examination is are was were obtained provided submitted '
    'performed reviewed acquired taken a an left right decubitus oblique cone-down rib ribs bilateral as well also '
    'cxr x-ray xray x ray').split())
TECH_NOUNS = frozenset('view views radiograph radiographs film films image images projection projections cxr chest'.split())
FRONTAL_VIEWS = frozenset({'AP', 'PA', 'POSTERO_ANTERIOR', 'ANTERO_POSTERIOR', 'UNKNOWN', 'N/A', 'None', '', None})
# a whole non-chest SERIES (dropped even when "chest" is in the name: the picked frontal may be the abdomen film)
SERIES_NONCHEST_RE = re.compile(
    r'\b(?:STERNUM|BONE\s+SURVEY|BONE\s+LENGTH|VCUG|UGI|ESOPHAG\w*|BE\s+W/|KUB|A\.?A\.?S|ABDOMEN\s+ACUTE|ABD\s+ACUTE|'
    r'ABD\s+PORTABLE|ABD\s+EXAM|FOREIGN\s+BODY)\b', re.I)
# another body part (dropped only when the description has no chest marker: "CXR 2V/HIP FX PREOP" stays)
BODYPART_NONCHEST_RE = re.compile(
    r'\b(?:KNEE|ANKLE|FOOT|TIBIA|FIBULA|OS\s+CALCIS|ORBITS?|NECK|BREAST|SPINE|SHOULDER|HUMERUS|FOREARM|WRIST|HAND|'
    r'ELBOW|PELVIS|HIP|FEMUR|SKULL|SINUS\w*|ABDOMEN|ABD)\b', re.I)
CHEST_DESC_RE = re.compile(r'CHEST|CXR|THORA', re.I)
RIBS_DESC_RE = re.compile(r'\bRIBS?\b|RIBSLT|RIBSRT', re.I)
UNKNOWN_DESCS = frozenset({'', '[ID]', 'PAPERORDER', 'PA'})


def normalize_json_target(s):
    """VERBATIM copy of vlm_inference_analysis/scripts/build_cls_json_multiturn_data.py:normalize_json_target --
    the function that produced every stage-6c detailed cls-JSON answer (fences stripped, json/python-repr parsed,
    json.dumps(indent=2, ensure_ascii=False)). Checked against the original at run time when it is importable."""
    import ast as _ast
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip()
    if t.startswith('```'):
        t = t.removeprefix('```json').removeprefix('```').removesuffix('```').strip()
    for loader in (json.loads, _ast.literal_eval):
        try:
            obj = loader(t)
        except Exception:
            continue
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, indent=2, ensure_ascii=False)
    return None


def _h64(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256('|'.join(str(p) for p in parts).encode()).digest()[:8], 'big')


def to_ascii(t: str) -> str:
    t = ''.join(ASCII_MAP.get(c, c) for c in t)
    if any(ord(c) > 127 for c in t):
        t = ''.join(c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c))
    return t


def clean_field(t: Any) -> str:
    t = ' '.join(to_ascii(t if isinstance(t, str) else '').split())
    while True:
        m = HDR_RE.match(t)
        if not m:
            break
        t = t[m.end():]
    return t.strip()


def split_numbered(text: str) -> Optional[List[str]]:
    """Items of a SEQUENTIAL '1. ... 2. ... 3. ...' list (numbers must count up from 1), else None."""
    if not re.match(r'^\(?1[.)]\s+', text):
        return None
    starts, expect = [], 1
    for m in NUM_ITEM_RE.finditer(text):
        if int(m.group(1)) == expect:
            starts.append(m)
            expect += 1
    if not starts:
        return None
    items = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        items.append(text[m.end():end].strip())
    return [x for x in items if x]


def sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def is_technique(s: str) -> bool:
    words = re.findall(r'[a-z0-9-]+', s.lower())
    return bool(words) and all(w in TECH_WORDS for w in words) and any(w in TECH_NOUNS for w in words)


def findings_bullets(F: str, stats: collections.Counter) -> List[str]:
    items = split_numbered(F)
    if items is not None:
        stats['findings_numbered'] += 1
    sents = [s for it in (items if items is not None else [F]) for s in sentences(it)]
    out = []
    for s in sents:
        s = HDR_RE.sub('', s, count=1).strip()
        if not s:
            continue
        if is_technique(s):
            stats['findings_sent_dropped:technique'] += 1
            continue
        if JUNK_SENT_RE.match(s):
            stats['findings_sent_dropped:junk'] += 1
            continue
        if META_LINE_RE.match(s):
            stats['findings_sent_dropped:meta_header'] += 1
            continue
        if DATE_RE.search(s) and re.match(r'(?i)^(comparison|compared|prior)', s):
            stats['findings_sent_dropped:dated_comparison'] += 1
            continue
        out.append(s)
    return out


def impression_parts(I: str, stats: collections.Counter) -> Tuple[str, List[str]]:
    """-> (impression text or '', moved recommendation sentences)."""
    items = split_numbered(I)
    numbered = items is not None
    units = items if numbered else [I]
    keep_units, rec, n_sent = [], [], 0
    for u in units:
        ss = sentences(u)
        kept = []
        for s in ss:
            n_sent += 1
            if POINTER_RE.match(s) or JUNK_SENT_RE.match(s):
                stats['impression_sent_dropped:pointer'] += 1
                continue
            if REC_RE.search(s):
                rec.append(s)
                continue
            kept.append(s)
        if kept:
            keep_units.append(' '.join(kept))
    if not keep_units and rec:              # never empty the Impression by moving recommendations
        keep_units = [' '.join(rec)]
        rec = []
    if not keep_units:
        return '', []
    if numbered and len(keep_units) > 1:
        text = ' '.join(f'{i}. {u}' for i, u in enumerate(keep_units, 1))
    else:
        text = ' '.join(keep_units)
    return text.strip(), rec


def desc_ok(desc: Any, body_part: Any) -> Tuple[bool, str]:
    d = (desc or '').strip()
    if SERIES_NONCHEST_RE.search(d):
        return False, 'desc_nonchest_series'
    if CHEST_DESC_RE.search(d):
        return True, ''
    if BODYPART_NONCHEST_RE.search(d):
        return False, 'desc_nonchest_bodypart'
    if RIBS_DESC_RE.search(d):
        return False, 'desc_ribs_only'
    if d.upper() in UNKNOWN_DESCS:
        return (body_part == 'Chest'), 'desc_unknown_nonchest_bodypart'
    return False, 'desc_other'


def build_answers(bullets: List[str], imp: str, rec: List[str]) -> Dict[str, str]:
    F = '\n'.join('* ' + s for s in bullets)
    R = ' '.join(rec) if rec else REC_BOILER
    a = {'sec_findings': '\n' + F}
    if imp:
        a['sec_impression'] = imp
        a['medrep'] = json.dumps({'Findings': '\n' + F, 'Impression': imp, 'Recommendations': R}, indent=2)
        a['fict'] = f'Findings:\n{F}\n\nImpression:\n{imp}\n\nRecommendations:\n{R}'
    return a


def self_check(answers: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Every answer must classify as its own family through sem_targets with its prompt, and the Findings /
    Impression bodies must agree across families. -> {family: canonical} or None."""
    cans, secs = {}, {}
    for fam, ans in answers.items():
        tt = st.turn_target(FAMILY_PROMPT[fam] + '\n<image>', ans)
        if tt is None or tt[0] != fam:
            return None
        cans[fam] = tt[1]
        secs[fam] = st.split_canonical(tt[1])
    fb = {s['Findings'] for f, s in secs.items() if 'Findings' in s}
    ib = {s['Impression'] for f, s in secs.items() if 'Impression' in s}
    if len(fb) != 1 or len(ib) > 1:
        return None
    if 'sec_findings' not in secs or 'Findings' not in secs['sec_findings']:
        return None
    if 'medrep' in secs and set(secs['medrep']) != {'Findings', 'Impression'}:
        return None
    if 'fict' in secs and set(secs['fict']) != {'Findings', 'Impression'}:
        return None
    return cans


def parse_mix(s: str) -> Dict[str, float]:
    mix = {}
    for part in s.split(','):
        k, v = part.split('=')
        mix[k.strip()] = float(v)
    bad = set(mix) - {'fict', 'cls_fict', 'cls_sec2', 'sec_findings', 'medrep'}
    if bad or abs(sum(mix.values()) - 1) > 1e-6 or any(v < 0 for v in mix.values()):
        raise ValueError(f'--rex_mix must name fict, cls_fict, cls_sec2, sec_findings, medrep with weights summing to 1: {s}')
    return mix


def pick_family(key: str, seed: int, mix: Dict[str, float], has_cls: bool, has_imp: bool) -> str:
    allowed = {}
    for f, w in mix.items():
        if not has_imp and f != 'sec_findings':
            continue
        if not has_cls and f.startswith('cls_'):
            continue
        allowed[f] = w
    if not allowed or sum(allowed.values()) <= 0:
        return 'sec_findings'
    u = (_h64(seed, 'family', key) / 2 ** 64) * sum(allowed.values())
    acc = 0.0
    for f in sorted(allowed):
        acc += allowed[f]
        if u < acc:
            return f
    return sorted(allowed)[-1]


def make_row(key: str, split: str, family: str, answers: Dict[str, str], cls_ans: Optional[str], seed: int,
             image: str) -> Dict[str, Any]:
    def turns(seq):
        msgs = []
        for i, (q, a) in enumerate(seq):
            msgs.append({'role': 'user', 'content': f'{q}\n<image>' if i == 0 else q})
            msgs.append({'role': 'assistant', 'content': a})
        return msgs
    if family in ('fict', 'sec_findings', 'medrep'):
        seq = [(FAMILY_PROMPT[family], answers[family])]
    elif family == 'cls_fict':
        seq = [(P_CLS, cls_ans), (P_FICT, answers['fict'])]
    elif family == 'cls_sec2':
        pairs = [('sec_findings', 'sec_impression'), ('sec_impression', 'sec_findings'),
                 ('sec_findings', 'medrep'), ('sec_impression', 'medrep')]
        a, b = pairs[_h64(seed, 'sec2', key) % len(pairs)]
        seq = [(P_CLS, cls_ans), (FAMILY_PROMPT[a], answers[a]), (FAMILY_PROMPT[b], answers[b])]
    else:
        raise ValueError(family)
    return {'messages': turns(seq), 'images': [image], 'source': 'rexgradient', 'rex_key': key, 'rex_split': split,
            'rex_family': family if family != 'cls_sec2' else f'cls_sec2:{a}+{b}'}


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def load_cases(path: str) -> Dict[str, str]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r['key']] = r['images'][0]
    return out


def ids_of(meta: Dict[str, Any]) -> Dict[str, set]:
    s = {'keys': set(meta), 'patients': set(), 'accessions': set(), 'sius': set(), 'img_rel': set()}
    for m in meta.values():
        s['patients'].add(m['PatientID'])
        s['accessions'].add(m['AccessionNumber'])
        s['sius'].add(m['StudyInstanceUid'])
        for p in m['ImagePath']:
            s['img_rel'].add(p.split('deid_png/', 1)[-1])
    return s


KEY_RE = re.compile(r'^p(GRDN[0-9A-Z]{12})_a(GRDN[0-9A-Z]{12})_s(.+)$')


def rex_study_rows(split: str, meta: Dict[str, Any], frontal: Dict[str, str], hires: Dict[str, str],
                   cls_by_report: Dict[str, List[str]], args, io_threads: int, log) -> Tuple[List[Dict], Dict]:
    funnel = collections.Counter()
    tstats = collections.Counter()
    cand = []   # (key, bullets, imp, rec, answers, cans, cls_ans, image)
    # image existence first (NFS; threaded)
    keys = list(meta)
    imgs = {k: hires.get(k) for k in keys}
    with ThreadPoolExecutor(io_threads) as ex:
        exists = dict(zip(keys, ex.map(lambda k: imgs[k] is not None and resolve_storage_path(imgs[k]) is not None,
                                       keys, chunksize=256)))
    for key in keys:
        m = meta[key]
        funnel['studies'] += 1
        km = KEY_RE.match(key)
        if not km or km.group(1) != m['PatientID'] or km.group(2) != m['AccessionNumber'] or km.group(3) != m['StudyInstanceUid']:
            raise RuntimeError(f'{split}: key {key} does not match its metadata ids')
        pick = frontal.get(key)
        if pick is None or key not in hires:
            funnel['drop:no_frontal_pick'] += 1
            continue
        rel = '../deid_png/' + pick.split('/deid_png/', 1)[1]
        try:
            i = m['ImagePath'].index(rel)
        except ValueError:
            funnel['drop:pick_not_in_metadata'] += 1
            continue
        view = m['ImageViewPosition'][i] if m.get('ImageViewPosition') else None
        body = m['ImageBodyPart'][i] if m.get('ImageBodyPart') else ''
        if view not in FRONTAL_VIEWS:
            funnel[f'drop:view_{view}'] += 1
            continue
        if body not in ('Chest', '', None):
            funnel['drop:bodypart_nonchest'] += 1
            continue
        ok, why = desc_ok(m.get('StudyDescription'), body)
        if not ok:
            funnel[f'drop:{why}'] += 1
            continue
        F_raw, I_raw = m.get('Findings') or '', m.get('Impression') or ''
        if DEID_RE.search(F_raw) or DEID_RE.search(I_raw):
            funnel['drop:deid_token'] += 1
            continue
        F, I = clean_field(F_raw), clean_field(I_raw)
        if any(ord(c) > 127 for c in F + I):
            funnel['drop:non_ascii'] += 1
            continue
        if F_raw != F or I_raw != I:
            tstats['fields_normalised'] += 1
        bullets = findings_bullets(F, tstats)
        if not bullets or len(' '.join(bullets)) < 10:
            funnel['drop:findings_empty_or_junk'] += 1
            continue
        imp, rec = impression_parts(I, tstats)
        if not imp or len(imp) < st.MIN_BODY_CHARS:
            imp, rec = '', []
            tstats['impression_empty_after_pointer_removal'] += 1
        if rec:
            tstats['impression_rec_moved'] += 1
        answers = build_answers(bullets, imp, rec)
        cans = self_check(answers)
        if cans is None:
            funnel['drop:canonical_self_check'] += 1
            continue
        if not exists[key]:
            funnel['drop:image_missing'] += 1
            continue
        rk = ' '.join(f"Indication: {m['Indication']}\nComparison: {m['Comparison']}\nFindings: {m['Findings']}\n"
                      f"Impression: {m['Impression']}".split())
        cls_ans = None
        tg = cls_by_report.get(rk)
        if tg is None:
            tstats['cls:no_extraction'] += 1
        else:
            vals = {t for t in tg if t is not None}
            if len(vals) != 1:
                tstats['cls:unparseable_or_conflicting'] += 1
            else:
                v = next(iter(vals))
                try:
                    obj = json.loads(v)
                except Exception:  # noqa: BLE001
                    obj = None
                if not isinstance(obj, dict) or tuple(obj) != CLS_CATEGORIES:
                    tstats['cls:not_7_categories'] += 1
                else:
                    cls_ans = v
                    tstats['cls:ok'] += 1
        funnel['pass_filters'] += 1
        cand.append({'key': key, 'patient': m['PatientID'], 'F': answers['sec_findings'], 'answers': answers,
                     'cans': cans, 'cls': cls_ans, 'image': hires[key], 'has_imp': bool(imp), 'view': view})
    # duplicate cap on the identical cleaned Findings (seeded order)
    cap = args.rex_findings_cap
    by_f = collections.defaultdict(list)
    for c in cand:
        by_f[c['F']].append(c)
    kept, dup_stats = [], {}
    for F, cs in by_f.items():
        cs.sort(key=lambda c: _h64(args.seed, 'dup', c['key']))
        if cap and len(cs) > cap:
            dup_stats[F[:120]] = len(cs)
            funnel['drop:findings_dup_cap'] += len(cs) - cap
            cs = cs[:cap]
        kept.extend(cs)
    caps_effect = {}
    for c_ in (0, 200, 500, 1000, 2000):
        caps_effect[str(c_)] = sum(min(len(v), c_) if c_ else len(v) for v in by_f.values())
    info = {'funnel': dict(funnel), 'transform_stats': dict(tstats),
            'findings_dup_cap': cap, 'n_distinct_findings': len(by_f),
            'strings_over_cap': dict(sorted(dup_stats.items(), key=lambda kv: -kv[1])),
            'rows_if_cap': caps_effect}
    log(f'REX {split}: {funnel["studies"]:,} studies -> {funnel["pass_filters"]:,} pass -> {len(kept):,} after dup cap {cap}')
    return kept, info


def rex_leak_sets(args, log) -> Dict[str, Any]:
    R = args.rex_root
    test = load_json(f'{R}/metadata/test_metadata_view_position.json')
    ts = ids_of(test)
    ev_keys = set()
    for f in EVAL_TEST_KEY_FILES + sorted(glob.glob(EVAL_TEST_KEY_GLOB)):
        if not os.path.isfile(f):
            continue
        if f.endswith('.jsonl'):
            with open(f) as fh:
                ev_keys.update(json.loads(line)['key'] for line in fh)
        else:
            ev_keys.update(load_json(f))
    vqa = set()
    if os.path.isfile(EVAL_TEST_VQA_IMAGES):
        vqa = set(load_json(EVAL_TEST_VQA_IMAGES))
    img8_pat = set(os.listdir(EVAL_TEST_IMG8)) if os.path.isdir(EVAL_TEST_IMG8) else set()
    eval_example = set()
    ex_dir = f'{SW}/rexgradient_eval/meta/example'
    if os.path.isdir(ex_dir):
        eval_example = set(os.listdir(ex_dir))
    clean = set()
    if os.path.isfile(CLEAN_PATIENTS):
        with open(CLEAN_PATIENTS) as f:
            clean = {x.strip() for x in f if x.strip()}
    # every storage-root-free path a test image can have
    test_paths = set()
    for k in ts['keys']:
        for d in ('/sahil_work/ReXGradient-160K/png_hires', '/sahil_work/ReXGradient-160K/png8'):
            test_paths.add(f'{d}/{k}.png')
    for rel in ts['img_rel'] | vqa:
        test_paths.add(f'/sahil_work/ReXGradient-160K/deid_png/{rel}')
        test_paths.add(f'/cxr_data/opensource_datasets/rexgradient/deid_png/{rel}')
        test_paths.add(f'/sahil_work/rexgradient_eval/img8/{rel}')
    vqa_studies = set()
    for rel in vqa:
        parts = rel.split('/')
        if len(parts) >= 4:
            vqa_studies.add((parts[0], parts[1], parts[3]))
    info = {'test_studies': len(ts['keys']), 'test_patients': len(ts['patients']),
            'eval_test_keys': len(ev_keys), 'eval_test_keys_subset_of_test_metadata': ev_keys <= ts['keys'],
            'vqa_test_images': len(vqa), 'img8_patients': len(img8_pat), 'clean_patients(valid+test)': len(clean),
            'test_image_paths_checked': len(test_paths)}
    log(f'REX leak sets: {info}')
    if not info['eval_test_keys_subset_of_test_metadata']:
        raise RuntimeError('eval test keys are not a subset of the test metadata: unknown eval set, refusing')
    return {'test': ts, 'ev_keys': ev_keys, 'vqa_studies': vqa_studies, 'img8': img8_pat | eval_example,
            'clean': clean, 'test_paths': test_paths, 'info': info}


def assert_rex_rows(rows_or_lines, split: str, own: Dict[str, set], others: Dict[str, Dict[str, set]], L: Dict[str, Any],
                    ) -> Dict[str, int]:
    """Leakage assertions over REX rows (dicts) or written lines (str). Raises on any violation."""
    ts = L['test']
    c = collections.Counter()
    bad = []
    test_ids = ts['patients'] | ts['accessions']
    for x in rows_or_lines:
        if isinstance(x, str):
            line = x
            r = json.loads(x)
        else:
            r = x
            line = json.dumps(x)
        c['rows'] += 1
        key = r['rex_key']
        km = KEY_RE.match(key)
        pid, acc, siu = km.group(1), km.group(2), km.group(3)
        img = r['images'][0]
        npth = norm_path(img)
        checks = {
            'split_field': r.get('rex_split') == split,
            'key_in_own_split': key in own['keys'],
            'key_not_test': key not in ts['keys'],
            'patient_not_test': pid not in ts['patients'],
            'accession_not_test': acc not in ts['accessions'],
            'siu_not_test': siu not in ts['sius'],
            'key_not_eval_scored': key not in L['ev_keys'],
            'not_vqa_test_study': (pid, acc, siu) not in L['vqa_studies'],
            'patient_not_img8': pid not in L['img8'],
            'image_is_own_png_hires': npth == f'/sahil_work/ReXGradient-160K/png_hires/{key}.png',
            'image_not_test_path': npth not in L['test_paths'],
            'grdn_tokens_not_test': not (set(GRDN_RE.findall(line)) & test_ids),
            'one_image': len(r['images']) == 1,
        }
        for o, s in others.items():
            checks[f'patient_not_{o}'] = pid not in s['patients']
            checks[f'accession_not_{o}'] = acc not in s['accessions']
            checks[f'siu_not_{o}'] = siu not in s['sius']
        if split == 'train':
            checks['patient_not_clean_valid_test_list'] = pid not in L['clean']
        for k, ok in checks.items():
            if ok:
                c[f'pass:{k}'] += 1
            else:
                c[f'FAIL:{k}'] += 1
                if len(bad) < 10:
                    bad.append((k, key))
    if bad:
        raise RuntimeError(f'REX {split} leakage assertion FAILED: {bad}')
    return dict(c)


def run_rex(args, out_dir: str, manifest: Dict[str, Any], mcr_prompts, mcr_cls_prompts, log) -> Dict[str, Any]:
    R = args.rex_root
    mix = parse_mix(args.rex_mix)
    # 0. the prompt strings must be the stage-6c ones
    seen = {(f, p) for (f, p) in mcr_prompts} if mcr_prompts is not None else None
    prompt_check = {}
    for fam, p in FAMILY_PROMPT.items():
        prompt_check[fam] = None if seen is None else ((fam, p) in seen)
    prompt_check['cls'] = None if mcr_cls_prompts is None else (P_CLS in mcr_cls_prompts)
    log(f'REX prompts present in the filtered stage-6c files: {prompt_check}')
    if any(v is False for v in prompt_check.values()):
        raise RuntimeError(f'a REX prompt string is not a stage-6c prompt: {prompt_check}')
    # 1. metadata + picks
    t0 = time.time()
    meta = {s: load_json(f'{R}/metadata/{s}_metadata_view_position.json') for s in ('train', 'valid')}
    frontal = load_cases(f'{R}/rex_cases_frontal.jsonl')
    hires = load_cases(f'{R}/rex_cases_frontal_hires.jsonl')
    for k, p in hires.items():
        if not p.startswith(REX_IMAGE_PREFIX + '/') or os.path.basename(p) != f'{k}.png':
            raise RuntimeError(f'unexpected hires path form: {k} -> {p}')
    ids = {s: ids_of(meta[s]) for s in meta}
    L = rex_leak_sets(args, log)
    ts = L['test']
    for a, b in (('train', 'valid'),):
        for fld in ('keys', 'patients', 'accessions', 'sius'):
            if ids[a][fld] & ids[b][fld]:
                raise RuntimeError(f'{a}/{b} metadata share {fld}')
    for s in ids:
        for fld in ('keys', 'patients', 'accessions', 'sius'):
            if ids[s][fld] & ts[fld]:
                raise RuntimeError(f'{s}/test metadata share {fld}')
    log(f'REX metadata loaded in {time.time() - t0:.0f}s: train {len(meta["train"]):,} valid {len(meta["valid"]):,} '
        f'test {len(ts["keys"]):,}; frontal picks {len(frontal):,} hires {len(hires):,}')
    # 2. cls extraction (joined on the whitespace-normalised report)
    import pandas as pd
    fs = sorted(glob.glob(f'{args.tag_dir}/*.parquet'))
    df = pd.concat([pd.read_parquet(f, columns=['report', 'model_response_content']) for f in fs], ignore_index=True)
    try:   # identity with the original normaliser (when importable)
        import importlib.util
        spec = importlib.util.spec_from_file_location('_cls_builder', CLS_BUILDER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        smp = df['model_response_content'].iloc[:200].tolist()
        norm_identity = all(mod.normalize_json_target(s) == normalize_json_target(s) for s in smp)
    except Exception as e:  # noqa: BLE001
        norm_identity = f'not checked: {type(e).__name__}: {e}'
    if norm_identity is False:
        raise RuntimeError('normalize_json_target copy differs from build_cls_json_multiturn_data.py')
    wanted = set()
    for s in ('train', 'valid'):
        for m in meta[s].values():
            wanted.add(' '.join(f"Indication: {m['Indication']}\nComparison: {m['Comparison']}\nFindings: {m['Findings']}\n"
                                f"Impression: {m['Impression']}".split()))
    cls_by_report = collections.defaultdict(list)
    for rep, resp in zip(df['report'].tolist(), df['model_response_content'].tolist()):
        k = ' '.join(str(rep).split())
        if k in wanted:
            cls_by_report[k].append(normalize_json_target(resp))
    log(f'REX cls extractions: {len(df):,} parquet rows ({len(fs)} files), {len(cls_by_report):,} joined reports, '
        f'normaliser identity {norm_identity}')
    # 3. label-sidecar collision check (ASL must be masked for ReX images)
    label_collisions = None
    if os.path.isfile(CLS_LABEL_SIDECAR):
        lk = set(pd.read_parquet(CLS_LABEL_SIDECAR, columns=['key'])['key'].tolist())
        label_collisions = 0
        for k, p in hires.items():
            parts = p.strip('/').split('/')
            label_collisions += (os.path.basename(p) in lk) or ('/'.join(parts[-3:]) in lk)
        log(f'REX images colliding with cls_label_sidecar keys: {label_collisions}')
        if label_collisions:
            raise RuntimeError('ReX image paths collide with cls_label_sidecar keys: the ASL term would train on '
                               'another image\'s labels')
    # 4. rows
    out = {'splits': {}, 'prompt_check': prompt_check, 'cls_normaliser_identity': norm_identity,
           'cls_parquet': {'dir': args.tag_dir, 'files': len(fs), 'rows': len(df)},
           'label_sidecar_collisions': label_collisions, 'mix': mix, 'leak_sets': L['info']}
    rows_by_split = {}
    for split in ('train', 'valid'):
        kept, info = rex_study_rows(split, meta[split], frontal, hires, cls_by_report, args, args.io_threads, log)
        if split == 'valid':
            # one study per patient, then the first --rex_val_rows in seeded order
            best = {}
            for c in kept:
                h = _h64(args.seed, 'valpick', c['key'])
                if c['patient'] not in best or h < best[c['patient']][0]:
                    best[c['patient']] = (h, c)
            kept = [c for _, c in sorted(best.values(), key=lambda t: t[0])][:args.rex_val_rows]
            info['val_one_per_patient_then_first_n'] = args.rex_val_rows
        elif args.rex_max_train_rows:
            kept.sort(key=lambda c: _h64(args.seed, 'trainpick', c['key']))
            kept = kept[:args.rex_max_train_rows]
            info['train_row_cap'] = args.rex_max_train_rows
        rows = []
        fam_counts = collections.Counter()
        for c in kept:
            fam = pick_family(c['key'], args.seed, mix, c['cls'] is not None, c['has_imp'])
            rows.append(make_row(c['key'], split, fam, c['answers'], c['cls'], args.seed, c['image']))
            fam_counts[rows[-1]['rex_family']] += 1
        rows.sort(key=lambda r: _h64(args.seed, 'order', r['rex_key']))
        # T1 per family (+ plugin leak rule), unique texts, views
        t1, t1_nl, t1_keys, turns = collections.Counter(), collections.Counter(), collections.defaultdict(set), 0
        for r in rows:
            tts = row_t1(r)
            if not tts:
                raise RuntimeError(f'REX row without a T1 target: {r["rex_key"]}')
            for fam, can, leaked in tts:
                t1[fam] += 1
                t1_keys[fam].add(st.target_key(can))
                t1_nl[fam] += not leaked
            turns += sum(m['role'] == 'assistant' for m in r['messages'])
        others = {s: ids[s] for s in ids if s != split}
        leak = assert_rex_rows(rows, split, ids[split], others, L)
        info.update({'rows': len(rows), 'patients': len({c['patient'] for c in kept}),
                     'families': dict(sorted(fam_counts.items())),
                     'families_coarse': dict(collections.Counter(f.split(':')[0] for f in fam_counts.elements())),
                     'views': dict(collections.Counter(c['view'] if c['view'] is not None else 'null' for c in kept)),
                     'assistant_turns': turns, 't1_turns': dict(t1), 't1_turns_nonleaked': dict(t1_nl),
                     't1_unique_texts': {f: len(v) for f, v in t1_keys.items()},
                     'rows_with_cls_turn': sum(r['rex_family'].startswith('cls_') for r in rows),
                     'leak_assertions_in_memory': leak})
        out['splits'][split] = info
        rows_by_split[split] = rows
    # 5. write (tmp; renamed into place by main() after every check passed)
    names = {'train': 'REX_TRAIN.jsonl', 'valid': 'REX_VAL.jsonl'}
    for split, rows in rows_by_split.items():
        path = os.path.join(out_dir, names[split])
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
            f.flush()
            os.fsync(f.fileno())
        with open(tmp) as f:
            leak = assert_rex_rows((line for line in f), split, ids[split], {s: ids[s] for s in ids if s != split}, L)
        out['splits'][split]['leak_assertions_written_file'] = leak
        out['splits'][split]['out'] = path
        out['splits'][split]['_tmp'] = tmp
        log(f'REX {split}: wrote {len(rows):,} rows (re-read: {leak["rows"]:,} rows, every assertion passed)')
    # decode sample of the REX images
    rng = random.Random(args.seed)
    allimg = [r['images'][0] for r in rows_by_split['train']]
    out['image_decode_check'] = _decode_check(rng.sample(allimg, min(args.decode_sample, len(allimg))))
    log(f"REX decode sample: {out['image_decode_check']}")
    if out['image_decode_check']['ok'] != out['image_decode_check']['n']:
        raise RuntimeError('REX decode failures')
    out['examples'] = {}
    for r in rows_by_split['train']:
        f = r['rex_family'].split(':')[0]
        if f not in out['examples'] and (f != 'cls_sec2' or 'medrep' not in r['rex_family']):
            out['examples'][f] = r
        if len(out['examples']) == 5:
            break
    return out


# ============================================================================================================
# main
# ============================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_dir', default=DEFAULT_OUT)
    ap.add_argument('--train_files', nargs='+', default=DEFAULT_TRAIN)
    ap.add_argument('--val_files', nargs='*', default=DEFAULT_VAL)
    ap.add_argument('--rex_root', default=REX_ROOT)
    ap.add_argument('--tag_dir', default=TAG_DIR, help='Gemini tag-extraction parquets of the ReX reports (cls turn)')
    ap.add_argument('--rex_mix', default=DEFAULT_MIX)
    ap.add_argument('--rex_findings_cap', type=int, default=1000,
                    help='max studies per identical cleaned Findings text (0 = no cap)')
    ap.add_argument('--rex_val_rows', type=int, default=2000)
    ap.add_argument('--rex_max_train_rows', type=int, default=0, help='0 = every train study that passes')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--workers', type=int, default=min(64, os.cpu_count() or 1))
    ap.add_argument('--io_threads', type=int, default=64)
    ap.add_argument('--chunk_mb', type=int, default=64)
    ap.add_argument('--decode_sample', type=int, default=256)
    ap.add_argument('--skip_mcr', action='store_true')
    ap.add_argument('--skip_rex', action='store_true')
    ap.add_argument('--overwrite', action='store_true', help='replace existing outputs in --out_dir')
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''   # CPU only, always
    out_dir = os.path.abspath(args.out_dir)
    t_start = time.time()

    def log(msg):
        print(f'[stage7-mcr {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)

    inputs = [os.path.abspath(st.resolve_storage_path(p)) for p in args.train_files + args.val_files]
    if any(os.path.dirname(p) == out_dir for p in inputs):
        raise ValueError(f'--out_dir {out_dir} holds an input file: refusing (outputs must be NEW files)')
    os.makedirs(out_dir, exist_ok=True)
    planned = []
    if not args.skip_mcr:
        planned += [os.path.join(out_dir, os.path.basename(p).rsplit('.', 1)[0] + '.mcr.jsonl') for p in inputs]
    if not args.skip_rex:
        planned += [os.path.join(out_dir, n) for n in ('REX_TRAIN.jsonl', 'REX_VAL.jsonl')]
    planned += [os.path.join(out_dir, n) for n in ('MANIFEST.json', 'README.md')]
    existing = [p for p in planned if os.path.exists(p)]
    if existing and not args.overwrite:
        raise FileExistsError(f'outputs exist (pass --overwrite to replace): {existing}')
    tmp_dir = os.path.join(out_dir, f'.build_tmp_{os.getpid()}')
    os.makedirs(tmp_dir, exist_ok=True)

    manifest: Dict[str, Any] = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'host': platform.node(), 'seed': args.seed,
        'argv': sys.argv, 'builder': {'path': os.path.abspath(__file__), 'sha256': _sha256(os.path.abspath(__file__))},
        'python': sys.version.split()[0], 'sem_targets_rule_version': st.RULE_VERSION,
        'user_rule': 'Use only mimic chexpert and rex gradient data (stage 7)',
        'predicates': {
            'mcr_keep': 'row has >= 1 image and EVERY image source is mimic|chexpert '
                        '(/cxr_data/opensource_datasets/{mimic,chexpert}/ after storage-root strip)',
            'mcr_drop_reasons': ['no_images', 'mixed', 'internal_copy_only', 'internal_copy_mixed', 'rexgradient',
                                 'other_only', 'parse_error'],
            'rex_frontal_views': sorted(str(v) for v in FRONTAL_VIEWS),
            'rex_nonchest_series_regex': SERIES_NONCHEST_RE.pattern,
            'rex_nonchest_bodypart_regex': BODYPART_NONCHEST_RE.pattern, 'rex_rec_regex': REC_RE.pattern,
            'rex_pointer_regex': POINTER_RE.pattern, 'rex_sentence_split': SENT_RE.pattern,
            'rex_findings_cap': args.rex_findings_cap, 'rex_mix': args.rex_mix, 'rex_val_rows': args.rex_val_rows,
            'rex_image': 'png_hires/<key>.png via rex_cases_frontal_hires.jsonl (the ckpt-608 public-test input)',
        },
        'prompts': {'fict': P_FICT, 'sec_findings': P_FIND, 'sec_impression': P_IMP, 'medrep': P_MED, 'cls': P_CLS},
    }
    ctx = get_context('fork')
    mcr = None
    with ctx.Pool(args.workers) as pool:
        if not args.skip_mcr:
            mcr = run_mcr(args, pool, out_dir, tmp_dir, manifest, log)
            finalize_mcr(mcr, log)
    rex = None
    if not args.skip_rex:
        rex = run_rex(args, out_dir, manifest,
                      mcr['prompts'] if mcr else None, mcr['cls_prompts'] if mcr else None, log)

    # ---- rename into place, hashes, manifest -----------------------------------------------------------------
    outputs = []
    if mcr:
        for m in mcr['files']:
            os.replace(m['_tmp'], m['out'])
            outputs.append({'path': m['out'], 'role': m['role'], 'rows': m['kept'], 'kind': 'mcr'})
    if rex:
        for split, info in rex['splits'].items():
            os.replace(info['_tmp'], info['out'])
            outputs.append({'path': info['out'], 'role': 'train' if split == 'train' else 'val', 'rows': info['rows'],
                            'kind': 'rexgradient'})
    for o in outputs:
        o.update(_stat(o['path']))
        o['sha256'] = _sha256(o['path'])
    shutil.rmtree(tmp_dir, ignore_errors=True)
    manifest['outputs'] = outputs
    if mcr:
        manifest['mcr'] = {
            'files': [{k: v for k, v in m.items() if not k.startswith('_')} for m in mcr['files']],
            'image_exist_check': mcr['image_exist_check'], 'image_decode_check': mcr['image_decode_check'],
            't1_unique_texts_train_total': mcr['t1_unique_texts_train_total'],
            't1_prompts': [[f, p, v] for (f, p), v in mcr['prompts'].most_common()],
            'cls_prompts': dict(mcr['cls_prompts'].most_common()),
        }
    if rex:
        manifest['rex'] = {k: v for k, v in rex.items() if k != 'splits'}
        manifest['rex']['splits'] = {s: {k: v for k, v in i.items() if not k.startswith('_')} for s, i in rex['splits'].items()}
        manifest['rex']['sources'] = {
            'metadata': [_stat(f'{args.rex_root}/metadata/{s}_metadata_view_position.json') for s in ('train', 'valid', 'test')],
            'frontal_cases': _stat(f'{args.rex_root}/rex_cases_frontal.jsonl'),
            'hires_cases': _stat(f'{args.rex_root}/rex_cases_frontal_hires.jsonl'),
            'licence': f'{args.rex_root}/README.md (non-commercial research use only; no redistribution)'}
    tr = [o['path'] for o in outputs if o['role'] == 'train']
    va = [o['path'] for o in outputs if o['role'] == 'val']
    manifest['launcher'] = {'DATASETS': ' '.join(tr), 'VAL_DATASETS': ' '.join(va),
                            'train_rows': sum(o['rows'] for o in outputs if o['role'] == 'train'),
                            'val_rows': sum(o['rows'] for o in outputs if o['role'] == 'val'),
                            'note': '20_stage7_sem_latent.sh default (MIX=mcr); MIX=6c = the old stage-6c mixture'}
    if mcr and rex:   # what the stage-7 run sees per epoch
        tot = {'t1_turns': collections.Counter(), 't1_turns_nonleaked': collections.Counter()}
        for m in mcr['files']:
            if m['role'] == 'train':
                tot['t1_turns'].update(m['t1_turns_kept'])
                tot['t1_turns_nonleaked'].update(m['t1_turns_kept_nonleaked'])
        tot['t1_turns'].update(rex['splits']['train']['t1_turns'])
        tot['t1_turns_nonleaked'].update(rex['splits']['train']['t1_turns_nonleaked'])
        n_rows = manifest['launcher']['train_rows']
        manifest['mixture_train_totals'] = {
            'rows': n_rows, 'rows_mimic_chexpert': sum(m['kept'] for m in mcr['files'] if m['role'] == 'train'),
            'rows_rexgradient': rex['splits']['train']['rows'],
            't1_turns': dict(tot['t1_turns']), 't1_turns_nonleaked': dict(tot['t1_turns_nonleaked']),
            't1_nonleaked_per_1024_rows': round(1024 * sum(tot['t1_turns_nonleaked'].values()) / max(n_rows, 1), 1),
            'optimizer_steps_at_1024': -(-n_rows // 1024)}
    manifest['seconds'] = round(time.time() - t_start, 1)
    mpath = os.path.join(out_dir, 'MANIFEST.json')
    with open(mpath + '.tmp', 'w') as f:
        json.dump(manifest, f, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    os.replace(mpath + '.tmp', mpath)
    with open(os.path.join(out_dir, 'README.md.tmp'), 'w') as f:
        f.write(render_readme(manifest))
    os.replace(os.path.join(out_dir, 'README.md.tmp'), os.path.join(out_dir, 'README.md'))
    log(f'done in {manifest["seconds"]:.0f}s -> {out_dir}')
    print(json.dumps(manifest['launcher'], indent=1))


def render_readme(M: Dict[str, Any]) -> str:
    L = []
    L.append('# stage7_mcr -- MIMIC-CXR + CheXpert + ReXGradient only (stage 7)\n')
    L.append(f"Built {M['created']} on {M['host']} by `{M['builder']['path']}` (sha256 {M['builder']['sha256'][:16]}), "
             f"seed {M['seed']}, {M['seconds']:.0f} s. The machine-readable record is `MANIFEST.json`; this file is "
             'generated from it. Nothing outside this directory was modified.\n')
    L.append('User rule: *"Use only mimic chexpert and rex gradient data"* for stage 7.\n')
    L.append('## Launcher strings\n')
    L.append('```bash')
    L.append(f"DATASETS=\"{M['launcher']['DATASETS']}\"")
    L.append(f"VAL_DATASETS=\"{M['launcher']['VAL_DATASETS']}\"")
    L.append('```')
    L.append(f"{M['launcher']['train_rows']:,} train rows, {M['launcher']['val_rows']:,} val rows. These are the "
             '`20_stage7_sem_latent.sh` defaults (`MIX=mcr`); `MIX=6c` restores the stage-6c mixture.\n')
    if 'mixture_train_totals' in M:
        T = M['mixture_train_totals']
        L.append(f"Per epoch: {T['rows']:,} rows ({T['rows_mimic_chexpert']:,} MIMIC/CheXpert + {T['rows_rexgradient']:,} "
                 f"ReXGradient), {T['optimizer_steps_at_1024']:,} optimizer steps at global batch 1024, "
                 f"{sum(T['t1_turns_nonleaked'].values()):,} SEM-supervised (non-leaked) T1 turns = "
                 f"{T['t1_nonleaked_per_1024_rows']} per 1024-row step. Non-leaked T1 per family: " +
                 ', '.join(f'{k} {v:,}' for k, v in sorted(T['t1_turns_nonleaked'].items())) + '.\n')
    L.append('## Files\n')
    L.append('| file | role | rows | bytes | sha256 |')
    L.append('|---|---|---|---|---|')
    for o in M['outputs']:
        L.append(f"| `{os.path.basename(o['path'])}` | {o['role']} | {o['rows']:,} | {o['size']:,} | `{o['sha256'][:16]}` |")
    if 'mcr' in M:
        L.append('\n## (a) MIMIC/CheXpert filter of the stage-6c mixture\n')
        L.append(M['predicates']['mcr_keep'] + '. Kept rows are the original lines, byte for byte, in the original order. '
                 'Internal re-exports of MIMIC/CheXpert films (`internal_data/training/images/images/{mimic.,mim.,chex.}*`) '
                 'are NOT kept (not on disk here, no report turns).\n')
        L.append('| input | rows | kept | % | dropped (reason: rows) | T1 turns kept (non-leaked) | unique T1 texts |')
        L.append('|---|---|---|---|---|---|---|')
        for f in M['mcr']['files']:
            t1 = sum(f['t1_turns_kept'].values())
            nl = sum(f['t1_turns_kept_nonleaked'].values())
            ut = sum(f['t1_unique_texts_kept'].values())
            dr = ', '.join(f'{k}: {v:,}' for k, v in f['dropped'].items())
            L.append(f"| `{os.path.basename(f['input']['path'])}` | {f['rows']:,} | {f['kept']:,} | {100 * f['kept_frac']:.1f} | "
                     f"{dr} | {t1:,} ({nl:,}) | {ut:,} |")
        for role in ('train', 'val'):
            tot, nl = collections.Counter(), collections.Counter()
            for f in M['mcr']['files']:
                if f['role'] == role:
                    tot.update(f['t1_turns_kept'])
                    nl.update(f['t1_turns_kept_nonleaked'])
            L.append(f'\nT1 turns kept per family, {role} files (non-leaked): ' + ', '.join(
                f'{k} {v:,} ({nl.get(k, 0):,})' for k, v in sorted(tot.items())) + '.')
        L.append('Per-file family counts, per-template/family row counts and image-source combos: MANIFEST.json `mcr`.')
        ex = M['mcr']['image_exist_check']
        dc = M['mcr']['image_decode_check']
        L.append(f"\nImages of kept rows: {ex['n']:,} distinct, {ex['missing']} missing (resolved roots {ex['resolved_root']}); "
                 f"decode sample {dc['ok']}/{dc['n']} (modes {dc['modes']}).")
        for f in M['mcr']['files']:
            if f['role'] == 'val':
                L.append(f"`{os.path.basename(f['out'])}`: {f.get('val_images_also_in_train', 0):,} of its "
                         f"{f['distinct_images_kept']:,} images and {f.get('val_t1_texts_also_in_train', 0):,} of its T1 texts "
                         'also occur in the kept train files (inherited from the stage-6c split; not changed). Use '
                         '`val_sem_unseen/*` and REX_VAL for generalisation.')
    if 'rex' in M:
        R = M['rex']
        L.append('\n## (b) ReXGradient-160K rows\n')
        L.append('Official TRAIN split -> `REX_TRAIN.jsonl`, VALID split -> `REX_VAL.jsonl` (one study per patient, '
                 f"first {M['predicates']['rex_val_rows']:,} in seeded order). TEST split: never used. One row per study, one "
                 'image per row: `png_hires/<key>.png` (8-bit, min side 1500 px), the exact image and path string the '
                 'ckpt-608 public-test benchmark used. Answers are the stage-6c formats built from ONE per-study transform '
                 '(Findings -> `* ` bullets in the radiologist\'s order; Impression prose; recommendation sentences moved '
                 'to Recommendations, else the fixed stage-6c sentence); the cls turn is the Gemini tag extraction of the '
                 'same report through the stage-6c `normalize_json_target`.\n')
        for s, i in R['splits'].items():
            L.append(f"### {s}: {i['rows']:,} rows, {i['patients']:,} patients\n")
            L.append('Funnel: ' + ', '.join(f'{k} {v:,}' for k, v in i['funnel'].items()) + '.\n')
            L.append('Families: ' + ', '.join(f'{k} {v:,}' for k, v in i['families'].items()) + '.\n')
            L.append('T1 turns (non-leaked): ' + ', '.join(
                f"{k} {v:,} ({i['t1_turns_nonleaked'].get(k, 0):,})" for k, v in sorted(i['t1_turns'].items())) +
                '; unique T1 texts: ' + ', '.join(f'{k} {v:,}' for k, v in sorted(i['t1_unique_texts'].items())) + '.\n')
            L.append(f"cls turn available: {i['transform_stats'].get('cls:ok', 0):,} studies; rows with a cls turn "
                     f"{i['rows_with_cls_turn']:,}. Findings duplicate cap {i['findings_dup_cap']} "
                     f"(rows if cap = {i['rows_if_cap']}).\n")
            la = i['leak_assertions_written_file']
            fails = {k: v for k, v in la.items() if k.startswith('FAIL')}
            L.append(f"Leakage assertions on the written file ({la['rows']:,} rows): "
                     + ', '.join(f"{k[5:]}" for k in la if k.startswith('pass:')) + f" -- failures: {fails or 'none'}.\n")
        L.append(f"Test-side sets checked against: {R['leak_sets']}.\n")
        L.append(f"Prompts present in the filtered stage-6c files: {R['prompt_check']}. cls normaliser identical to "
                 f"build_cls_json_multiturn_data.py: {R['cls_normaliser_identity']}. ReX image keys colliding with "
                 f"cls_label_sidecar: {R['label_sidecar_collisions']} (so the ASL term is masked for ReX images).\n")
        L.append('**Contamination note.** The ckpt-608 ReX generations over all 160k studies (`reports_*`, `hr_*`, '
                 '`r8_*`, `rex_chexbert.py` all_f1) include the train split: once a model trains on REX_TRAIN, report ReX '
                 'numbers on the public test split (`agentic_paper_metrics/rextest_gt.jsonl`) and valid only. The ReX '
                 'rows teach the ReX style under the fict / medrep / sec_* / cls prompts, not under the ontology prompt '
                 'the ckpt-608 benchmark used.\n')
        L.append('**Licence.** ReXGradient-160K is non-commercial research use only, no redistribution '
                 '(`/fastdata4/sahil_work/ReXGradient-160K/README.md`): keep these files internal (never push them to HF); '
                 'a checkpoint trained on them is research-only.\n')
    L.append('## Rebuild\n')
    L.append('```bash')
    L.append('CUDA_VISIBLE_DEVICES="" PYTHONPATH=/fastdata4/sahil_work/VLM_qure/vlm/swift \\')
    L.append('  /home/ubuntu/miniconda3/envs/ms-swift-may-s4/bin/python \\')
    L.append('  /fastdata4/sahil_work/VLM_qure/vlm/swift/examples/train/liger/sem_latent/build_stage7_mcr_data.py --overwrite')
    L.append('```')
    L.append('Same inputs + flags -> byte-identical files (compare the sha256 column).\n')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
