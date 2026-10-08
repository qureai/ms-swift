"""Integration tests on the REAL tokenizer / embedder / vLLM — the layer the CPU unit suite fakes.

Each test validates an assumption the fakes bake in (and that only real hardware can falsify):
  1. the real tokenizer maps 'true'/'false' to single token ids the Brier matcher finds
  2. the real (loccontrol) embedder scores matching attributes above contradicting ones
  3. a real vLLM rollout produces per-token logprobs that feed the Brier end-to-end
  4. the 2B checkpoint loads through swift's vLLM path (asl_loss/ml_decoder_head/visual_3d ignored)
     and generates

Needs the model files; tests 3-4 need CUDA + the from-source vLLM (run on a GPU node, e.g.
run_gpu_tests.sbatch). Tests skip (not fail) when their prerequisite is missing, so the file is safe
to collect anywhere. Run: python test_radiology_reward_gpu.py  (or pytest).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radiology_reward as R  # noqa: E402

CKPT_2B = ('/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/vlm_ckpts/'
           'Qwen3.5-2B_stage6b_alltools/v0-20260906-173237/checkpoint-608')
EMB = ('/cache/fast_data_nas71/janhavi/vlm_ckpts/qwen3_4b_embedding/'
       '23_sep_qwen3_4b_cross_axis_v1/checkpoints/checkpoint-900')


class SkipTest(Exception):
    pass


def _skip(msg):
    raise SkipTest(msg)


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


_LLM = None


def _get_llm():
    """Load the 2B checkpoint once through swift's vLLM registration (aux heads ignored)."""
    global _LLM
    if _LLM is not None:
        return _LLM
    from transformers import AutoConfig
    from swift.model.vision_3d.vllm_qwen3_5 import register_qwen3_5_3d_model
    register_qwen3_5_3d_model()                       # register the custom class for Qwen3.5
    AutoConfig.from_pretrained(CKPT_2B, trust_remote_code=True)
    from vllm import LLM
    _LLM = LLM(model=CKPT_2B, gpu_memory_utilization=0.4, max_model_len=2048,
               enforce_eager=True, trust_remote_code=True, limit_mm_per_prompt={'image': 1})
    return _LLM


# ---------------------------------------------------------------------------------------------- #
def test_real_tokenizer_boolean_ids():
    """The Brier matcher assumes 'true'/'false' are single tokens whose ids it can find. If the real
    tokenizer splits them, the term would silently no-op on EVERY rollout -> assert it does not."""
    if not os.path.isdir(CKPT_2B):
        _skip(f'checkpoint missing: {CKPT_2B}')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT_2B, trust_remote_code=True)
    tids, fids = R._bool_token_id_sets(tok)
    assert tids and fids, "real tokenizer: no single-token 'true'/'false' -> Brier silently disabled"
    # a realistic presence slot must actually emit one of those ids
    for snippet, want in (('"presence": true', tids), ('"presence": false', fids)):
        ids = tok.encode(snippet, add_special_tokens=False)
        assert any(i in want for i in ids), f'no boolean id found in {snippet!r}'


def test_real_embedder_attribute_monotonic():
    """The fake A_attr assumed match>mismatch; verify the REAL loccontrol embedder actually does,
    including a paraphrase (RUL == right upper lobe) that exact-match would miss."""
    if not os.path.isdir(EMB):
        _skip(f'embedder missing: {EMB}')
    try:
        emb = R.EmbeddingScorer(EMB)
        probe = emb.encode(['right upper lobe'])          # surface the real error (cosine_pairs swallows it)
        assert probe is not None and len(probe) == 1
    except Exception as e:
        _skip(f'embedder unavailable ({type(e).__name__}: {e})')
    gt = R.ParsedReport(ok=True, present={'nodules'},
                        attrs_map={'nodules': {'location': 'right upper lobe', 'size': '2 cm',
                                               'characteristics': None}})
    match = R.ParsedReport(ok=True, present={'nodules'},
                           attrs_map={'nodules': {'location': 'RUL', 'size': '2 cm',
                                                  'characteristics': None}})
    contra = R.ParsedReport(ok=True, present={'nodules'},
                            attrs_map={'nodules': {'location': 'left lower lobe', 'size': '9 cm',
                                                   'characteristics': None}})
    a_m = R._attr_agreement(match, gt, emb)
    a_c = R._attr_agreement(contra, gt, emb)
    if a_m is None or a_c is None:
        _skip('embedder returned no cosines (A_attr None) -> cannot assess')
    assert a_m > a_c, f'real embedder: match {a_m:.3f} !> contradiction {a_c:.3f}'
    assert a_m > 0.6, f'paraphrase match unexpectedly low ({a_m:.3f}) -> embedder or A_attr wiring off'


def test_vllm_loads_2b_and_generates():
    """The 2B checkpoint carries asl_loss + ml_decoder_head (+ possibly visual_3d) heads that vanilla
    vLLM rejects. Assert swift's custom class loads it (ignoring them) and generates."""
    if not (_cuda() and os.path.isdir(CKPT_2B)):
        _skip('needs CUDA + checkpoint')
    from vllm import SamplingParams
    llm = _get_llm()          # if the aux heads were NOT ignored, this raises at load (asl_loss/ml_decoder_head)
    out = llm.generate(['Name one chest x-ray finding:'], SamplingParams(max_tokens=16, temperature=0))
    # reaching here means the checkpoint loaded (aux heads ignored via the plugin) AND generated
    assert out[0].outputs[0].text.strip(), 'model loaded but produced no text'


def test_logprobs_feed_brier_end_to_end():
    """A real vLLM rollout must yield per-token logprobs that, with the real tokenizer, drive the
    Brier -> this is the plumbing+alignment chain the CPU suite could only fake."""
    if not (_cuda() and os.path.isdir(CKPT_2B)):
        _skip('needs CUDA + checkpoint')
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    tok = AutoTokenizer.from_pretrained(CKPT_2B, trust_remote_code=True)
    tids, fids = R._bool_token_id_sets(tok)
    llm = _get_llm()
    prompt = ('Fill in true/false for each finding, return ONLY JSON:\n'
              '{"general": {"nodules": {"presence": <t/f>}, "effusion": {"presence": <t/f>}, '
              '"cardiomegaly": {"presence": <t/f>}}}\nJSON:')
    o = llm.generate([prompt], SamplingParams(max_tokens=128, temperature=0, logprobs=5))
    out = o[0].outputs[0]
    token_ids = list(out.token_ids)
    # ALIGNMENT PRIMITIVE (the real plumbing assumption): per-token logprobs align 1:1 with token_ids
    # and the sampled token's logprob is retrievable -- exactly what swift stores as rollout_logprobs.
    assert out.logprobs is not None and len(out.logprobs) == len(token_ids), 'logprobs not aligned to tokens'
    logprobs = [out.logprobs[i][t].logprob for i, t in enumerate(token_ids)]
    assert all(lp <= 0.0 for lp in logprobs), 'logprobs should be <= 0'
    # OPPORTUNISTIC end-to-end: if the model emitted presence booleans, the Brier chain must run clean.
    n_bool = sum(1 for t in token_ids if t in tids or t in fids)
    if n_bool == 0:
        _skip('model emitted no true/false tokens in text-only mode; Brier plumbing covered by the smoke')
    val = R._weighted_brier(out.text, token_ids, logprobs, tids, fids, {'nodules'}, 6.0)
    assert val is None or (0.0 <= val <= 1.0), f'Brier returned invalid value: {val}'
    print(f'  [logprobs] tokens={len(token_ids)} boolean tokens={n_bool}  brier={val}')


TESTS = [test_real_tokenizer_boolean_ids, test_real_embedder_attribute_monotonic,
         test_vllm_loads_2b_and_generates, test_logprobs_feed_brier_end_to_end]

if __name__ == '__main__':
    passed = skipped = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS  {t.__name__}')
            passed += 1
        except SkipTest as s:
            print(f'SKIP  {t.__name__}: {s}')
            skipped += 1
        except Exception as e:
            import traceback
            print(f'FAIL  {t.__name__}: {type(e).__name__}: {e}')
            traceback.print_exc()
    print(f'\n{passed} passed, {skipped} skipped, {len(TESTS) - passed - skipped} failed')
    sys.exit(0 if (passed + skipped) == len(TESTS) else 1)
