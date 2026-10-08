"""Conceptual regression tests for the constrained-RL reward changes.

These are NOT end-to-end training tests; they pin the *conceptual* properties that the architecture
changes could silently break -- the kind of bug that would train a wrong policy without crashing:

  1. backwards-compat: new terms OFF -> reward == the legacy renormalized combo (no leak)
  2. learning signal: reward orders perfect>partial>garbage and varies across a group (GRPO gradient)
  3. precision/recall are DECOUPLED (the whole reason we split them into two constraints)
  4. weighted Brier penalizes hedging (present-token weight w) + reads RADREWARD_TRUE_WEIGHT
  5. Brier uses P(true)=1-P(false) and pairs with the GT label correctly
  6. Brier is a graceful no-op on malformed output (no crash, combo renormalizes)
  7. A_attr contains case-2: GT-silent additions are neutral; match>contradiction
  8. beta(n)*safety scales embedding reliability in the right direction
  9. total reward stays in [0,1] for arbitrary knob combos
 10. exp1 data integrity + reward runs on a consolidated reference; uncovered safety defaults to 1.0

Run:  python test_radiology_reward.py   (or: pytest test_radiology_reward.py)
CPU-only; the embedder/tokenizer are faked so no model download is needed.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radiology_reward as R  # noqa: E402

TID, FID = 1000, 1001         # fake 'true'/'false' token ids


class FakeEmbedder:
    """cosine_pairs(a,b): 1.0 for identical strings, else 0.0 -> (cos+1)/2 = 1.0 match, 0.5 mismatch."""
    def cosine_pairs(self, a, b):
        return [1.0 if x == y else 0.0 for x, y in zip(a, b)]

    def matrix(self, a, b):
        return [[1.0 if x == y else 0.0 for y in b] for x in a]


def cls_json(tags):
    """tags: list of (name, present_bool[, attrs]) -> a fenced cls-json completion string."""
    body = {'general': {}}
    for t in tags:
        name, present = t[0], t[1]
        attrs = t[2] if len(t) > 2 else {}
        body['general'][name] = {'presence': present, 'size': attrs.get('size'),
                                 'location': attrs.get('location'),
                                 'characteristics': attrs.get('characteristics')}
    return '```json\n' + json.dumps(body) + '\n```'


def ref(finding_set, prominence=None, attributes=None, findings_text=None, impression_text=None):
    d = {'finding_set': finding_set, 'absent_set': [], 'prominence': prominence or {}}
    if attributes is not None:
        d['attributes'] = attributes
    if findings_text is not None:
        d['findings_text'] = findings_text
    if impression_text is not None:
        d['impression_text'] = impression_text
    return json.dumps(d)


def make_reward(embedder=None, **cfg):
    r = R.RadiologyReportReward(args=None)
    r.embedder = embedder
    for k, v in cfg.items():
        setattr(r.cfg, k, v)
    r._bool_ids = (frozenset({TID}), frozenset({FID}))   # bypass tokenizer for the Brier tests
    return r


def score(r, completion, reference, task='cls_json', token_ids=None, logprobs=None):
    kw = {'reference': [reference], 'solution': [reference], 'task': [task]}
    if token_ids is not None:
        kw['response_token_ids'] = [token_ids]
        kw['rollout_logprobs'] = [[logprobs]]   # single-turn wrap [[...]]
    return r([completion], **kw)[0]


# ---------------------------------------------------------------------------------------------- #
def test_backwards_compat_no_leak():
    """Defaults (all new knobs off) -> total == renormalized combo of ONLY the legacy terms."""
    r = make_reward()               # embedder None -> no embedding term
    comp = cls_json([('nodules', True), ('consolidation', False)])
    gt = ref(['nodules'], prominence={'nodules': 4})
    pred = R.parse_report(comp, r.vocab)
    gtp = R.gt_from_reference(gt, r.vocab)
    c = r.cfg
    r_cls, *_ = r._cls_reward(pred, gtp)
    parts = [(c.w_cls, r_cls), (c.w_bbox, r._bbox_reward(pred, gtp)), (c.w_token, r._token_reward('cls_json', pred, gtp)),
             (c.w_consistency, r._consistency_reward(pred)), (c.w_format, r._format_reward(pred)),
             (c.w_length, r._length_reward(pred, gtp))]
    num = sum(w * v for w, v in parts if v is not None and w > 0)
    den = sum(w for w, v in parts if v is not None and w > 0)
    expected = num / den
    got = score(r, comp, gt)
    assert abs(got - expected) < 1e-6, f'backwards-compat leak: {got} != legacy {expected}'
    # and the new components are logged (as 0 / inert)
    assert r.cfg.w_recall == 0 and r.cfg.w_cal == 0 and r.cfg.cal_mode == 'none'


def test_learning_signal_ordering_and_variance():
    """Reward must order good>partial>garbage and differ across completions (else GRPO has no grad)."""
    r = make_reward(w_recall=0.4, w_precision=0.4)
    gt = ref(['nodules', 'effusion', 'cardiomegaly'], prominence={'nodules': 5, 'effusion': 4, 'cardiomegaly': 3})
    perfect = cls_json([('nodules', True), ('effusion', True), ('cardiomegaly', True)])
    partial = cls_json([('nodules', True), ('effusion', False), ('cardiomegaly', False)])
    garbage = cls_json([('pneumothorax', True), ('emphysema', True)])   # all wrong (FP), no TP
    rp, rq, rg = score(r, perfect, gt), score(r, partial, gt), score(r, garbage, gt)
    assert rp > rq > rg, f'ordering broken: perfect={rp} partial={rq} garbage={rg}'
    assert len({round(rp, 6), round(rq, 6), round(rg, 6)}) == 3, 'no variance -> zero gradient in-group'


def test_precision_recall_decoupled():
    """Hallucination must drop precision but NOT recall; omission must drop recall but NOT precision.
    This is the core reason we split cls into two independent constraint terms."""
    r = make_reward()
    gt = ref(['nodules', 'effusion'], prominence={'nodules': 4, 'effusion': 4})
    base = cls_json([('nodules', True), ('effusion', True)])
    hallucinate = cls_json([('nodules', True), ('effusion', True), ('pneumothorax', True)])  # +1 FP
    omit = cls_json([('nodules', True), ('effusion', False)])                                 # -1 FN

    def RP(comp):
        pred = R.parse_report(comp, r.vocab)
        gtp = R.gt_from_reference(gt, r.vocab)
        _, sens, _, _ = r._cls_reward(pred, gtp)
        return sens, R._precision(pred.present & r.vocab, gtp.present & r.vocab)
    R_base, P_base = RP(base)
    R_hall, P_hall = RP(hallucinate)
    R_omit, P_omit = RP(omit)
    assert P_hall < P_base and abs(R_hall - R_base) < 1e-9, 'hallucination should hit P only'
    assert R_omit < R_base and abs(P_omit - P_base) < 1e-9, 'omission should hit R_w only'


def test_weighted_brier_penalizes_hedging_and_env():
    """Present tags weighted w -> a hedge-everything-false policy cannot win the calibration reward;
    and RADREWARD_TRUE_WEIGHT actually changes w."""
    tags = [('nodules', None), ('effusion', None), ('consolidation', None)]  # names only; presence via tokens
    comp = cls_json([('nodules', True), ('effusion', True), ('consolidation', False)])
    gt_present = {'nodules', 'effusion'}
    tids, fids = frozenset({TID}), frozenset({FID})
    # confident-correct: true(0.95) for the two present, false(0.95) for the absent
    good_tok = [TID, TID, FID]; good_lp = [math.log(0.95)] * 3
    # hedge: says false confidently everywhere (misses both present findings)
    hedge_tok = [FID, FID, FID]; hedge_lp = [math.log(0.95)] * 3
    good = R._weighted_brier(comp, good_tok, good_lp, tids, fids, gt_present, 6.0)
    hedge = R._weighted_brier(comp, hedge_tok, hedge_lp, tids, fids, gt_present, 6.0)
    assert good > hedge, f'hedging should not beat confident-correct: good={good} hedge={hedge}'
    # w=1 (symmetric) shrinks the gap vs w=6 (present-weighted) -> weighting matters
    hedge_w1 = R._weighted_brier(comp, hedge_tok, hedge_lp, tids, fids, gt_present, 1.0)
    assert hedge < hedge_w1, 'higher true-weight must penalize missed present findings MORE'
    # env plumbing
    os.environ['RADREWARD_TRUE_WEIGHT'] = '9'
    try:
        assert R.RewardConfig().cal_true_weight == 9.0
    finally:
        del os.environ['RADREWARD_TRUE_WEIGHT']


def test_brier_complement_and_labels():
    """p_true = exp(lp) if 'true' sampled else 1-exp(lp); paired with GT label. Confident-false on an
    ABSENT tag should score well; confident-false on a PRESENT tag should score badly."""
    comp = cls_json([('nodules', False)])       # one tag
    tids, fids = frozenset({TID}), frozenset({FID})
    tok, lp = [FID], [math.log(0.99)]           # sampled 'false' with p=0.99 -> p_true ~ 0.01
    r_absent = R._weighted_brier(comp, tok, lp, tids, fids, set(), 6.0)          # y=0 -> good
    r_present = R._weighted_brier(comp, tok, lp, tids, fids, {'nodules'}, 6.0)   # y=1 -> bad
    assert r_absent > 0.95 and r_present < 0.05, f'complement/label wrong: absent={r_absent} present={r_present}'


def test_brier_misalignment_is_noop():
    """Malformed output (boolean-token count != parsed tag count) -> None (skip), never a crash;
    and the combo renormalizes so the total reward is still valid."""
    comp = cls_json([('nodules', True), ('effusion', False)])   # 2 tags
    tids, fids = frozenset({TID}), frozenset({FID})
    assert R._weighted_brier(comp, [TID, FID, FID], [0.0, 0.0, 0.0], tids, fids, {'nodules'}, 6.0) is None
    # in the full reward, a mismatch just drops the calibration term
    r = make_reward(cal_mode='brier', w_cal=0.5, w_recall=0.4)
    got = score(r, comp, ref(['nodules']), token_ids=[TID, FID, FID], logprobs=[0.0, 0.0, 0.0])
    assert 0.0 <= got <= 1.0 and r.last_components['calibration'][0] == 0.0


def test_attr_agreement_contains_case2():
    """A_attr compares ONLY where BOTH sides have a value: a GT-silent addition is neutral (not
    penalized, not rewarded); a match beats a contradiction."""
    emb = FakeEmbedder()
    gt = R.ParsedReport(ok=True, present={'nodules'},
                        attrs_map={'nodules': {'size': '2cm', 'location': 'RUL', 'characteristics': None}})
    match = R.ParsedReport(ok=True, present={'nodules'},
                           attrs_map={'nodules': {'size': '2cm', 'location': 'RUL', 'characteristics': 'spiculated'}})
    contradict = R.ParsedReport(ok=True, present={'nodules'},
                                attrs_map={'nodules': {'size': '9cm', 'location': 'LLL', 'characteristics': None}})
    a_match = R._attr_agreement(match, gt, emb)          # size+location compared (both match) -> 1.0
    a_contra = R._attr_agreement(contradict, gt, emb)    # size+location compared (both differ) -> 0.5
    assert abs(a_match - 1.0) < 1e-9 and abs(a_contra - 0.5) < 1e-9, (a_match, a_contra)
    # case-2: 'match' added characteristics='spiculated' that GT lacks -> it must NOT be counted
    # (GT characteristics is None -> pair skipped), so a_match stays 1.0 not diluted. Verified above.
    # no comparable attributes -> None (term skipped), not 0
    empty = R.ParsedReport(ok=True, present={'nodules'}, attrs_map={'nodules': {'size': None}})
    assert R._attr_agreement(empty, gt, emb) is None


def test_beta_safety_scale_embedding():
    """beta(n)*safety must scale the embedding term's influence: on a rich report (large n) or high
    safety the embedding moves the reward more than on a sparse / low-safety one."""
    assert R._beta(1, 4) < R._beta(10, 4) < 1.0 and R._beta(0, 4) == 0.0
    emb = FakeEmbedder()
    # identical GT text vs a bad prediction; vary n via n_findings. Higher n -> embedding weighted more
    # -> a BAD embedding drags the reward down more.
    r = make_reward(embedder=emb, beta_k=4.0)
    gt = ref(['nodules'], prominence={'nodules': 4}, findings_text='dense right upper lobe nodule')
    comp = cls_json([('nodules', True)])   # findings_text of pred != GT -> cosine 0 -> S=0.5 (bad-ish)

    def emb_share(n):
        kw = {'reference': [gt], 'solution': [gt], 'task': ['cls_json'], 'n_findings': [n]}
        r([comp], **kw)
        return r.last_components['beta'][0]
    b1, b10 = emb_share(1), emb_share(10)
    assert b10 > b1, f'beta must rise with n: n1={b1} n10={b10}'
    # safety direction: lower safety -> lower embedding weight (beta logged same, but safety multiplies it)
    kw_hi = {'reference': [gt], 'solution': [gt], 'task': ['cls_json'], 'n_findings': [10], 'safety_score': [1.0]}
    kw_lo = {'reference': [gt], 'solution': [gt], 'task': ['cls_json'], 'n_findings': [10], 'safety_score': [0.2]}
    r([comp], **kw_hi); assert r.last_components['safety'][0] == 1.0
    r([comp], **kw_lo); assert r.last_components['safety'][0] == 0.2


def test_reward_bounded_unit_interval():
    """Total reward must stay in [0,1] for arbitrary knob combinations (renormalized convex combo)."""
    emb = FakeEmbedder()
    combos = [dict(), dict(w_recall=0.5, w_precision=0.5), dict(cal_mode='brier', w_cal=0.7),
              dict(w_attr=0.9, beta_k=4.0), dict(w_recall=1, w_precision=1, w_cal=1, w_attr=1, beta_k=2.0)]
    comp = cls_json([('nodules', True), ('effusion', False)])
    gt = ref(['nodules'], prominence={'nodules': 4}, attributes={'nodules': {'location': 'RUL'}},
             findings_text='nodule')
    for cfg in combos:
        r = make_reward(embedder=emb, **cfg)
        v = score(r, comp, gt, token_ids=[TID, FID], logprobs=[math.log(0.9), math.log(0.9)])
        assert 0.0 <= v <= 1.0, f'reward out of [0,1]={v} for {cfg}'


def test_exp1_data_and_safety_default():
    """exp1 rows: safety>=0.8, correct difficulty bucket, consolidated reference has both blocks with
    attributes for abnormals; and _safety_col defaults uncovered rows to 1.0 (bbox/comparison invariant)."""
    import glob
    import pandas as pd
    files = glob.glob('/cache/fast_data_nas71/janhavi/data/rl_training/exp1/exp1_*.parquet')
    if files:
        import os.path as osp
        tier = osp.basename(files[0]).split('_')[1].split('.')[0]
        d = pd.read_parquet(files[0])
        assert (d['safety_score'] >= 0.8 - 1e-9).all(), 'exp1 has rows below safety 0.8'
        assert (d['difficulty_bucket'] == tier).all(), 'difficulty bucket mismatch in curriculum stage'
        cj = d[d['task'] == 'cls_json,ontology']
        abn = [json.loads(x) for x in cj['reference'] if json.loads(x)['cls_json']['finding_set']]
        assert abn, 'no abnormal consolidated rows found'
        one = abn[0]
        assert 'cls_json' in one and 'ontology' in one
        assert one['cls_json'].get('attributes'), 'abnormal row missing per-tag attributes (A_attr GT)'
        assert one['ontology'].get('findings_text'), 'consolidated row missing report text (embedding GT)'
    # safety default: an uncovered row (no safety_score column) -> 1.0 (embedding weight unchanged)
    assert R._safety_col({}, 3) == [1.0, 1.0, 1.0]
    assert R._safety_col({'safety_score': [0.5, None, 0.9]}, 3) == [0.5, 1.0, 0.9]


def test_precision_empty_pred_not_free():
    """The all-normal reward-hack fix: an empty prediction must NOT earn free precision on an abnormal
    study. P(∅, findings)=0; P(∅,∅)=1 (normal study -> predicting nothing is correct)."""
    assert R._precision(set(), {'nodules', 'effusion'}) == 0.0, 'empty pred on abnormal must score P=0'
    assert R._precision(set(), set()) == 1.0, 'empty pred on normal study is correct -> P=1'
    assert R._precision({'nodules'}, {'nodules', 'effusion'}) == 1.0, 'TP-only stays precise'
    assert R._precision({'nodules', 'pneumothorax'}, {'nodules'}) == 0.5, '1 FP of 2 -> 0.5'
    # end-to-end: an all-absent cls prediction on an abnormal GT now gets P=0 (no longer 1.0)
    r = make_reward(w_recall=0.5, w_precision=0.2)
    allnormal = cls_json([('nodules', False), ('effusion', False)])
    score(r, allnormal, ref(['nodules', 'effusion'], prominence={'nodules': 4, 'effusion': 4}))
    assert r.last_components['precision'][0] == 0.0, 'all-normal on abnormal GT must log precision 0'


def test_dual_ascent_lambda_update():
    """Env-gated dual ascent (PPO-Lagrangian): lambda_R/P are running multipliers. A batch whose
    recall is BELOW target must RAISE lambda_recall; a satisfied constraint must not raise its
    multiplier; lambdas stay clipped to [min,max] and are logged. Off (default) keeps fixed weights."""
    # default: dual ascent OFF, fixed weights
    assert make_reward(w_recall=0.4).cfg.dual_ascent is False
    # env plumbing
    os.environ['RADREWARD_DUAL_ASCENT'] = '1'; os.environ['RADREWARD_RECALL_TARGET'] = '0.9'
    try:
        assert R.RewardConfig().dual_ascent is True and R.RewardConfig().recall_target == 0.9
    finally:
        del os.environ['RADREWARD_DUAL_ASCENT']; del os.environ['RADREWARD_RECALL_TARGET']

    r = make_reward(w_recall=0.3, w_precision=0.3)
    r.cfg.dual_ascent = True
    r.cfg.recall_target, r.cfg.precision_target = 0.9, 0.1   # recall under-target; precision satisfied
    r.cfg.dual_lr, r.cfg.lambda_min, r.cfg.lambda_max = 0.1, 0.0, 1.0
    r._lam_R, r._lam_P = 0.3, 0.3
    gt = ref(['nodules', 'effusion'], prominence={'nodules': 4, 'effusion': 4})
    omit = cls_json([('nodules', True), ('effusion', False)])   # recall 0.5<0.9, precision 1.0>0.1
    kw = {'reference': [gt, gt], 'solution': [gt, gt], 'task': ['cls_json', 'cls_json']}
    r([omit, omit], **kw)
    assert r._lam_R > 0.3, f'under-target recall must raise lambda_R (0.3 -> {r._lam_R})'
    assert r._lam_P < 0.3, f'satisfied precision must lower lambda_P (0.3 -> {r._lam_P})'
    assert r.last_components['lambda_recall'][0] == r._lam_R, 'lambda must be logged'
    # clipping: drive under-target repeatedly -> lambda_R saturates at lambda_max, never exceeds
    r._lam_R = 0.98
    for _ in range(10):
        r([omit, omit], **kw)
    assert r._lam_R == r.cfg.lambda_max, f'lambda_R must clip at max, got {r._lam_R}'


TESTS = [test_backwards_compat_no_leak, test_learning_signal_ordering_and_variance,
         test_precision_recall_decoupled, test_weighted_brier_penalizes_hedging_and_env,
         test_brier_complement_and_labels, test_brier_misalignment_is_noop,
         test_attr_agreement_contains_case2, test_beta_safety_scale_embedding,
         test_reward_bounded_unit_interval, test_exp1_data_and_safety_default,
         test_precision_empty_pred_not_free, test_dual_ascent_lambda_update]

if __name__ == '__main__':
    passed = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS  {t.__name__}')
            passed += 1
        except Exception as e:
            import traceback
            print(f'FAIL  {t.__name__}: {type(e).__name__}: {e}')
            traceback.print_exc()
    print(f'\n{passed}/{len(TESTS)} passed')
    sys.exit(0 if passed == len(TESTS) else 1)
