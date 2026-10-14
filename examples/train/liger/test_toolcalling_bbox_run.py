#!/usr/bin/env python3
"""Pre-launch checks for the 13-Oct tool-calling + bbox warm-start run.

Run in the training env (CPU, no GPU, no full model load):
    conda activate swift_july
    python examples/train/liger/test_toolcalling_bbox_run.py

It parses the config script itself (so it can't drift), then verifies:
  1. all referenced files exist (config, tokens, WARM checkpoint, datasets)
  2. the config wires a WARM-START, not a resume, with the new tokens + AUROC on
  3. the 291 new special tokens really integrate into the checkpoint tokenizer (each -> 1 id)
  4. the checkpoint can be loaded + embedding-resized (vocab/tie/weights/mean_resizing preconditions)
  5. every dataset uses the expected ShareGPT schema
  6. every angle-bracket token the tool-calling data emits is registered in tools_tokens.txt
  7. the per-dataset channel mapping is as expected, and cls_json is present (=> AUROC fires on the val split)
  8. the trainer's pooled-AUROC + eval drop_last=False changes are present in the source

Exit code 0 = all pass, 1 = at least one failure.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWIFT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CONFIG = os.path.join(HERE, "19_oct_qwen3_5_2b_ct_toolcalling_bbox_fullft.sh")

_PASS, _FAIL = 0, 0


def check(name, cond, detail=""):
    global _PASS, _FAIL
    ok = bool(cond)
    _PASS += ok
    _FAIL += (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def _sh_var(text, name):
    m = re.search(rf'^{name}=(.+)$', text, re.MULTILINE)
    if not m:
        return None
    return m.group(1).split('#')[0].strip().strip('"').strip("'")


def _resolve_latest_ckpt(root):
    """Mirror the config's job-start WARM resolver: latest complete checkpoint across all v*/ version dirs."""
    import glob
    best, best_step = None, -1
    for d in glob.glob(os.path.join(root, "v*", "checkpoint-*")):
        if not os.path.isfile(os.path.join(d, "config.json")):
            continue
        try:
            step = int(d.rsplit("checkpoint-", 1)[1])
        except ValueError:
            continue
        if step > best_step:
            best, best_step = d, step
    return best


def parse_config(text):
    """Pull WARM / TOKENS / MODEL / TC / DATA and the DATASET_PATHS entries (vars expanded). WARM may be a
    job-start command substitution; if so, resolve it the same way the config does (latest complete checkpoint)."""
    env = {k: _sh_var(text, k) for k in ("MODEL", "WARM", "TOKENS", "TC", "DATA", "OUTPUT_DIR", "CKPT_ROOT")}
    env["WARM_DYNAMIC"] = bool(env.get("WARM") and env["WARM"].lstrip().startswith("$("))
    if env["WARM_DYNAMIC"] and env.get("CKPT_ROOT"):
        env["WARM"] = _resolve_latest_ckpt(env["CKPT_ROOT"])
    block = re.search(r'DATASET_PATHS=\(\n(.*?)\n\)', text, re.DOTALL)
    paths = []
    if block:
        for ln in block.group(1).splitlines():
            m = re.search(r'"([^"]+)"', ln)
            if not m:
                continue
            p = m.group(1).split('#')[0]
            for k, v in env.items():
                if isinstance(v, str) and v:
                    p = p.replace(f'${k}', v).replace(f'${{{k}}}', v)
            paths.append(p)
    return env, paths


def main():
    if not os.path.isfile(CONFIG):
        print(f"[FAIL] config script missing: {CONFIG}")
        sys.exit(1)
    text = open(CONFIG).read()
    # command text = non-comment lines only (so flag checks don't match the header comment block)
    cmd = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    env, datasets = parse_config(text)
    warm, tokens_file = env["WARM"], env["TOKENS"]

    # ---- 1. files exist -------------------------------------------------------------------------------------
    check("config script exists", os.path.isfile(CONFIG))
    check("tokens file exists", tokens_file and os.path.isfile(tokens_file), tokens_file)
    if env.get("WARM_DYNAMIC"):
        check("WARM is resolved at job-start (dynamic latest-checkpoint)", True,
              f"root={env.get('CKPT_ROOT')}; resolves NOW to {warm or '<none yet>'}")
    check("WARM checkpoint dir exists (currently-latest; re-resolved at launch)", warm and os.path.isdir(warm), warm)
    check("WARM has config.json", warm and os.path.isfile(os.path.join(warm, "config.json")))
    check("WARM has safetensors index", warm and os.path.isfile(os.path.join(warm, "model.safetensors.index.json")))
    check("config parsed 7 datasets", len(datasets) == 7, f"{len(datasets)} found")
    for d in datasets:
        n = 0
        if os.path.isfile(d):
            with open(d) as f:
                for n, _ in enumerate(f, 1):
                    if n >= 1:
                        break
            n = 1  # exists + at least 1 line readable
        check(f"dataset exists & non-empty: {os.path.basename(d)}", os.path.isfile(d) and os.path.getsize(d) > 0)

    # ---- 2. warm-start, not resume; tokens + AUROC wired ----------------------------------------------------
    check("uses --model \"$WARM\" (warm-start from checkpoint)", '--model "$WARM"' in cmd)
    check("NO active --resume_from_checkpoint flag (token resize forbids resume)",
          not re.search(r'^\s*--resume_from_checkpoint', cmd, re.MULTILINE))
    check("--new_special_tokens wired to tokens file", '--new_special_tokens "$TOKENS"' in cmd)
    check("--eval_cls_auc true present", "--eval_cls_auc true" in cmd)
    check("CT_DATASET_CHANNELS exported (per-channel loss)", "CT_DATASET_CHANNELS=1" in text)
    check("freeze_llm false (embeddings/lm_head trainable so new rows learn)", "--freeze_llm false" in cmd)

    # ---- 3. tokens really integrate into the checkpoint tokenizer -------------------------------------------
    file_tokens = [ln.strip() for ln in open(tokens_file) if ln.strip()] if (tokens_file and os.path.isfile(tokens_file)) else []
    check("tokens file has 291 entries", len(file_tokens) == 291, f"{len(file_tokens)}")
    check("tokens file entries are unique", len(set(file_tokens)) == len(file_tokens))
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(warm, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        check("load checkpoint tokenizer", False, f"{type(e).__name__}: {e}")
    if tok is not None:
        base_len = len(tok)
        n_added = tok.add_special_tokens({"additional_special_tokens": file_tokens})
        check("all 291 tokens are NEW to the checkpoint tokenizer", n_added == 291, f"added {n_added}")
        check("tokenizer length grew by 291", len(tok) == base_len + 291, f"{base_len} -> {len(tok)}")
        # every new token must encode to exactly ONE id (i.e. it is atomic, not BPE-split)
        bad = [t for t in file_tokens if len(tok.encode(t, add_special_tokens=False)) != 1]
        check("every new token encodes to a single id", not bad, f"non-atomic: {bad[:5]}")
        import math
        padded = math.ceil(len(tok) / 128) * 128
        check("resize target is a 128-multiple >= new length", padded >= len(tok) and padded % 128 == 0,
              f"len {len(tok)} -> pad {padded}")

    # ---- 4. checkpoint load/resize preconditions -----------------------------------------------------------
    cfg = {}
    try:
        cfg = json.load(open(os.path.join(warm, "config.json")))
    except Exception as e:  # noqa: BLE001
        check("read checkpoint config.json", False, str(e))
    tcfg = cfg.get("text_config", cfg)
    vocab = cfg.get("vocab_size", tcfg.get("vocab_size"))
    check("checkpoint vocab_size == 248320", vocab == 248320, str(vocab))
    check("tie_word_embeddings True (single resize covers lm_head)", cfg.get("tie_word_embeddings") is True)
    try:
        idx = json.load(open(os.path.join(warm, "model.safetensors.index.json")))["weight_map"]
        keys = list(idx)
        check("checkpoint has embed_tokens weight", any("embed_tokens" in k for k in keys))
        check("checkpoint has lm_head weight", any("lm_head" in k for k in keys))
        check("checkpoint has trained visual_3d tower (warm-start loads it)",
              sum("visual_3d" in k for k in keys) > 100, f"{sum('visual_3d' in k for k in keys)} tensors")
    except Exception as e:  # noqa: BLE001
        check("read safetensors index", False, str(e))
    try:
        import inspect
        from transformers import PreTrainedModel
        sig = inspect.signature(PreTrainedModel.resize_token_embeddings)
        check("transformers resize supports mean_resizing (good new-row init)", "mean_resizing" in sig.parameters)
    except Exception as e:  # noqa: BLE001
        check("inspect resize_token_embeddings", False, str(e))

    # ---- 5/6. dataset schema + tool-calling token consistency --------------------------------------------
    TOK_RE = re.compile(r'\|</?ASP>\||</?[A-Z][A-Z0-9_]*>')
    token_set = set(file_tokens)

    def read_jsonl(path, k):
        out = []
        with open(path) as f:
            for i, ln in enumerate(f):
                if i >= k:
                    break
                ln = ln.strip()
                if ln:
                    out.append(json.loads(ln))
        return out

    for d in datasets:
        if not os.path.isfile(d):
            continue
        base = os.path.basename(d)
        try:
            recs = read_jsonl(d, 3)
        except Exception as e:  # noqa: BLE001
            check(f"schema: {base}", False, f"parse error {e}")
            continue
        ok_schema = all(
            isinstance(r.get("videos"), str) and r["videos"].startswith("s3://")
            and isinstance(r.get("messages"), list) and r["messages"]
            and all(set(m) >= {"from", "value"} for m in r["messages"])
            and any(m["from"] in ("gpt", "assistant") for m in r["messages"])
            for r in recs)
        check(f"schema ShareGPT {{videos:s3, messages:[from,value]}}: {base}", ok_schema)

    # tool-calling: every angle-bracket token emitted by the assistant must be registered
    tc = [d for d in datasets if "tool_calling" in d]
    if tc and os.path.isfile(tc[0]):
        recs = read_jsonl(tc[0], 2000)
        emitted, unregistered = set(), set()
        for r in recs:
            for m in r["messages"]:
                if m.get("from") in ("gpt", "assistant"):
                    for t in TOK_RE.findall(m.get("value", "")):
                        emitted.add(t)
                        if t not in token_set:
                            unregistered.add(t)
        check("tool-calling emits special tokens", len(emitted) > 0, f"{len(emitted)} distinct")
        check("EVERY emitted token is in tools_tokens.txt (else it would BPE-split)",
              not unregistered, f"unregistered: {sorted(unregistered)[:8]}")

    # ---- 7. channel mapping + AUROC-won't-fire expectation ------------------------------------------------
    try:
        sys.path.insert(0, SWIFT_ROOT)
        from swift.dataset.ct_channels import ct_channel_for_path
        expect = {"tool_calling": "tool", "complete_cls_json": "cls_json", "knowledge_graph": "kg",
                  "segmed_nodule_bbox": "nod_bbox", "omnict_anatomy_localization": "anat_loc",
                  "omnict_label_bbox": "omnict_bbox", "segmed_anatomy_grounded_abnormality": "seg_anat_abn"}
        chans = set()
        for d in datasets:
            ch = ct_channel_for_path(d)
            chans.add(ch)
            for sub, exp in expect.items():
                if sub in d:
                    check(f"channel({os.path.basename(d)[:28]}) == {exp}", ch == exp, f"got {ch}")
        check("no 'other' fallbacks (every dataset mapped)", "other" not in chans, f"channels={sorted(chans)}")
        check("cls_json channel PRESENT -> AUROC will fire (presence_ce_channel match)", "cls_json" in chans)
    except Exception as e:  # noqa: BLE001
        check("import ct_channel_for_path", False, f"{type(e).__name__}: {e}")

    # ---- 8. trainer AUROC + eval drop_last source checks -------------------------------------------------
    try:
        tr = open(os.path.join(SWIFT_ROOT, "swift", "trainers", "seq2seq_trainer.py")).read()
        check("AUROC pools across ranks (all_gather_object)", "all_gather_object" in tr
              and "_cls_auc_compute_and_log" in tr)
        check("AUROC panel adds micro/prevw/per-tag", all(s in tr for s in
              ("cls_micro_auc", "cls_prevw_auc", "cls_per_tag_auc")))
        check("AUROC handles empty buffer safely", "if not pooled:" in tr)
        mx = open(os.path.join(SWIFT_ROOT, "swift", "trainers", "mixin.py")).read()
        check("eval dataloader forces drop_last=False", "dataloader_drop_last = False" in mx)
    except Exception as e:  # noqa: BLE001
        check("read trainer sources", False, str(e))

    print(f"\n{'='*60}\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
