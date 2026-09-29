#!/usr/bin/env python
"""Gradio viewer for Qwen3.5-2B eval predictions, with GT text search.

Browse the eval: full-resolution image + prompt + ground-truth report + prediction.
Type a term in "Search ground truth" (e.g. "nodule") to filter to cases whose GT report
contains that term, then page through the matches. Reads the (possibly growing) result JSONL.
"""
import argparse
import json
import os
import re

import gradio as gr

DEFAULT_RESULT = "/fastdata4/sahil_work/VLM_qure/eval_outputs/Qwen3.5-2B_ckpt5191_140k_high_score_eval/result.jsonl"
STORAGE_ROOTS = ("/localstorage", "/fastdata3", "/fastdata4")
_THINK_PREFIX = re.compile(r"^\s*<think>\s*</think>\s*", re.S)


def resolve_image(path):
    if not path:
        return None
    cands = [path]
    for a in STORAGE_ROOTS:
        if path.startswith(a):
            for b in STORAGE_ROOTS:
                cands.append(path.replace(a, b, 1))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def get_prompt(rec):
    prompt = ""
    for m in rec.get("messages") or []:
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except Exception:
                continue
        if m.get("role") == "user":
            prompt = m.get("content", "")
    return (prompt or "").replace("<image>", "").strip()


def get_image_path(rec):
    imgs = rec.get("images") or []
    if not imgs:
        return None
    im = imgs[0]
    return (im.get("path") or im.get("bytes")) if isinstance(im, dict) else im


def load_records(result_path):
    recs = []
    if not os.path.exists(result_path):
        return recs
    with open(result_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    return recs


def load_gt_map(gt_parquet, image_col, report_col):
    if not gt_parquet or not os.path.exists(gt_parquet):
        return {}
    import pandas as pd
    df = pd.read_parquet(gt_parquet, columns=[image_col, report_col])
    return dict(zip(df[image_col].tolist(), df[report_col].tolist()))


def build_app(result_path, gt_map=None, strip_think_default=True):
    gt_map = gt_map or {}

    def index(recs):
        """Precompute per-record fields + lowercased GT for fast search."""
        paths, gts, gts_low = [], [], []
        for r in recs:
            p = get_image_path(r)
            gt = r.get("labels") or gt_map.get(p) or ""
            paths.append(p)
            gts.append(gt)
            gts_low.append((gt or "").lower())
        return paths, gts, gts_low

    recs = load_records(result_path)
    paths, gts, gts_low = index(recs)
    state = {"recs": recs, "paths": paths, "gts": gts, "low": gts_low, "filt": list(range(len(recs)))}

    def apply_search(query):
        q = (query or "").strip().lower()
        if not q:
            state["filt"] = list(range(len(state["recs"])))
        else:
            low = state["low"]
            state["filt"] = [i for i in range(len(low)) if q in low[i]]
        return state["filt"]

    def render(pos, strip_think, query):
        filt = state["filt"]
        n = len(filt)
        total = len(state["recs"])
        if n == 0:
            msg = (f"No records yet — click Reload." if total == 0
                   else f"0 / {total} match “{query}”. Try another term or clear the search.")
            return None, "", "", "", msg
        pos = max(0, min(int(pos), n - 1))
        i = filt[pos]
        r = state["recs"][i]
        img = resolve_image(state["paths"][i])
        prompt = get_prompt(r)
        gt = state["gts"][i]
        pred = r.get("response") or ""
        if strip_think:
            pred = _THINK_PREFIX.sub("", pred).strip()
        q = (query or "").strip()
        scope = f"match {pos + 1} / {n}" + (f" for “{q}”" if q else "") + f"   ·   (of {total} total)"
        info = f"{scope}   ·   image: {state['paths'][i] or '—'}" + (
            "   ·   (image not found)" if state['paths'][i] and not img else "")
        return img, prompt, gt, pred, info

    def do_search(query, strip_think):
        apply_search(query)
        n = len(state["filt"])
        return (*render(0, strip_think, query),
                gr.update(value=0, maximum=max(n - 1, 0), label=f"Match # (0 – {max(n - 1, 0)})"))

    def nav(pos, strip_think, query, delta):
        n = len(state["filt"])
        newpos = max(0, min(int(pos) + delta, max(n - 1, 0)))
        return (newpos, *render(newpos, strip_think, query))

    def reload(pos, strip_think, query):
        state["recs"] = load_records(result_path)
        state["paths"], state["gts"], state["low"] = index(state["recs"])
        apply_search(query)
        n = len(state["filt"])
        return (*render(pos, strip_think, query),
                gr.update(maximum=max(n - 1, 0), label=f"Match # (0 – {max(n - 1, 0)})"))

    with gr.Blocks(title="Qwen3.5-2B Eval Viewer") as demo:
        gr.Markdown("## Qwen3.5-2B — Eval Viewer  ·  140k high-score prod eval\n"
                    "Search the **ground-truth report** (e.g. `nodule`, `effusion`, `pneumothorax`) to filter cases, then page through matches.")
        with gr.Row():
            search = gr.Textbox(label="Search ground truth", placeholder="e.g. nodule", scale=4)
            search_btn = gr.Button("Search", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", scale=1)
            reload_btn = gr.Button("⟳ Reload data", scale=1)
        with gr.Row():
            pos = gr.Number(value=0, precision=0, label="Match #", scale=2)
            prev_btn = gr.Button("← Prev", scale=1)
            next_btn = gr.Button("Next →", scale=1)
            strip_ck = gr.Checkbox(value=strip_think_default, label="Strip empty <think> prefix", scale=2)
        info = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=5):
                image = gr.Image(label="Chest X-ray (full resolution)", type="filepath", height=680, show_download_button=True)
            with gr.Column(scale=6):
                prompt_box = gr.Textbox(label="Prompt", lines=3, max_lines=6, show_copy_button=True)
                with gr.Row():
                    gt_box = gr.Textbox(label="Ground truth (report)", lines=22, max_lines=40, show_copy_button=True)
                    pred_box = gr.Textbox(label="Prediction", lines=22, max_lines=40, show_copy_button=True)

        outs = [image, prompt_box, gt_box, pred_box, info]
        search_btn.click(do_search, [search, strip_ck], [*outs, pos])
        search.submit(do_search, [search, strip_ck], [*outs, pos])
        clear_btn.click(lambda s: ("", *do_search("", s)), [strip_ck], [search, *outs, pos])
        prev_btn.click(lambda p, s, q: nav(p, s, q, -1), [pos, strip_ck, search], [pos, *outs])
        next_btn.click(lambda p, s, q: nav(p, s, q, +1), [pos, strip_ck, search], [pos, *outs])
        pos.submit(lambda p, s, q: render(p, s, q), [pos, strip_ck, search], outs)
        strip_ck.change(lambda p, s, q: render(p, s, q), [pos, strip_ck, search], outs)
        reload_btn.click(reload, [pos, strip_ck, search], [*outs, pos])
        demo.load(lambda s, q: render(0, s, q), [strip_ck, search], outs)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_path", default=DEFAULT_RESULT)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--gt_parquet", default=None)
    ap.add_argument("--gt_image_col", default="neysa_image_paths")
    ap.add_argument("--gt_report_col", default="report")
    args = ap.parse_args()
    print(f"Loading results from: {args.result_path}", flush=True)
    gt_map = load_gt_map(args.gt_parquet, args.gt_image_col, args.gt_report_col)
    print(f"GT report map entries: {len(gt_map)}", flush=True)
    app = build_app(args.result_path, gt_map=gt_map)
    app.launch(server_name="0.0.0.0", server_port=args.port, show_error=True,
               allowed_paths=["/localstorage", "/fastdata3", "/fastdata4"])
