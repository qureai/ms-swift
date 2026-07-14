"""
Prepare a Qwen-VL checkpoint for serving with Dynamo + SGLang WITHOUT mutating the original.

Background
----------
ms-swift saves Qwen2.5-VL checkpoints whose config.json carries an explicit
`"model_type": "qwen2_5_vl_text"` inside the nested `text_config`. transformers >=4.57
proxies attribute reads on `Qwen2_5_VLConfig` to that sub-config, so `config.model_type`
resolves to `"qwen2_5_vl_text"` instead of `"qwen2_5_vl"`. SGLang's Qwen-VL mrope code
only recognizes `"qwen2_5_vl"`, so serving fails at runtime with
`RuntimeError: Unimplemented model type: qwen2_5_vl_text`.

Fix
---
Removing the nested `text_config.model_type` makes transformers fall back to the correct
top-level class attribute (`qwen2_5_vl`). This helper builds a "fixed" checkpoint directory
that symlinks every file from the original and rewrites only `config.json` — the original
checkpoint is never touched.

Usage
-----
# As a CLI (prints the path to pass to `dynamo.sglang --model-path`):
python prepare_qwen_checkpoint.py --model-path /path/to/checkpoint-8241

# In code:
from prepare_qwen_checkpoint import prepare_fixed_checkpoint
serve_path = prepare_fixed_checkpoint("/path/to/checkpoint-8241")
# python -m dynamo.sglang --model-path {serve_path} ...
"""

import os
import sys
import json
import copy
import argparse
from pathlib import Path
from typing import Optional


# Qwen VL top-level model types that use a nested text_config which can shadow model_type.
_QWEN_VL_PREFIXES = ("qwen2_vl", "qwen2_5_vl", "qwen3_vl")


def _needs_fix(raw_config: dict) -> bool:
    """True if this is a Qwen-VL config whose nested text_config.model_type shadows the top-level."""
    top = raw_config.get("model_type")
    if not top or not str(top).startswith("qwen"):
        return False
    if not any(str(top).startswith(p) for p in _QWEN_VL_PREFIXES):
        return False
    text_cfg = raw_config.get("text_config")
    return (
        isinstance(text_cfg, dict)
        and "model_type" in text_cfg
        and text_cfg["model_type"] != top
    )


def prepare_fixed_checkpoint(
    model_path: str,
    dest_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Return a directory safe to serve with SGLang.

    If ``model_path`` is a Qwen-VL checkpoint affected by the ``text_config.model_type``
    shadowing bug, create (or reuse) a sibling ``<name>_dynamo_fixed`` directory that
    symlinks the weights and holds a corrected ``config.json``, and return that path.
    Otherwise return ``model_path`` unchanged. The original checkpoint is never modified.
    """
    src = Path(model_path).resolve()
    config_path = src / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json under {src}")

    with open(config_path) as f:
        raw = json.load(f)

    if not _needs_fix(raw):
        return str(src)

    dest = Path(dest_dir).resolve() if dest_dir else src.parent / f"{src.name}_dynamo_fixed"

    # Reuse an already-prepared fixed dir unless --force.
    # `dest` is keyed on the checkpoint name, so guard against reusing a dir whose
    # name matches but whose symlinks point at a DIFFERENT (moved) source: only
    # reuse when config.json is already fixed AND every symlink resolves to this src.
    if dest.exists() and not force:
        fixed_cfg = dest / "config.json"
        if fixed_cfg.exists():
            with open(fixed_cfg) as f:
                cfg_ok = not _needs_fix(json.load(f))
            links_ok = True
            for entry in src.iterdir():
                if entry.name == "config.json":
                    continue
                link = dest / entry.name
                if not link.is_symlink() or os.path.realpath(link) != str(entry.resolve()):
                    links_ok = False
                    break
            if cfg_ok and links_ok:
                return str(dest)

    dest.mkdir(parents=True, exist_ok=True)
    # Symlink everything except config.json (which we rewrite).
    for entry in src.iterdir():
        if entry.name == "config.json":
            continue
        link = dest / entry.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(entry.resolve())

    fixed = copy.deepcopy(raw)
    fixed.get("text_config", {}).pop("model_type", None)
    with open(dest / "config.json", "w") as f:
        json.dump(fixed, f, indent=2)

    return str(dest)


def _report(model_path: str) -> None:
    """Best-effort before/after print of what transformers reports (requires transformers)."""
    try:
        from transformers import AutoConfig
    except Exception:
        return
    try:
        before = AutoConfig.from_pretrained(model_path).model_type
        print(f"[info] transformers reports model_type={before!r} for the ORIGINAL checkpoint", file=sys.stderr)
    except Exception as err:
        print(f"[warn] could not load original config via transformers: {err}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a Qwen-VL checkpoint for SGLang serving (non-destructive).")
    parser.add_argument("--model-path", required=True, help="Path to the original HF checkpoint directory.")
    parser.add_argument("--dest-dir", default=None, help="Where to write the fixed copy (default: <name>_dynamo_fixed alongside the original).")
    parser.add_argument("--force", action="store_true", help="Rebuild the fixed dir even if it already exists.")
    parser.add_argument("--verify", action="store_true", help="Also print the model_type transformers reports before/after.")
    args = parser.parse_args()

    if args.verify:
        _report(args.model_path)

    fixed_path = prepare_fixed_checkpoint(args.model_path, args.dest_dir, args.force)

    if args.verify:
        try:
            from transformers import AutoConfig
            print(f"[info] transformers reports model_type={AutoConfig.from_pretrained(fixed_path).model_type!r} for the SERVE path", file=sys.stderr)
        except Exception:
            pass

    if fixed_path == str(Path(args.model_path).resolve()):
        print(f"[info] no fix needed; serve the original path", file=sys.stderr)
    # stdout = just the path, so it can be captured in shell: CKPT=$(python prepare_qwen_checkpoint.py --model-path ...)
    print(fixed_path)
