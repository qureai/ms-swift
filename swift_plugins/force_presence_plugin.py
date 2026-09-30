#!/usr/bin/env python
"""Threshold-controlled generation: at each `"presence":` slot, force true/false by a
per-finding threshold on the LM's own P(true), so the whole report is generated consistently.

Env:
  MLD_FORCE_TAU        global default threshold on P(true)          (default 0.5 == greedy)
  MLD_FORCE_TAU_JSON   path to {finding_key: tau} for per-tag control (optional)
  MLD_FORCE_OUT        path to dump per-finding decisions jsonl      (optional)
Use --infer_backend pt --max_batch_size 1.

Inference-only and deliberately NOT mode-gated (installs under any qure_plugin_mode).
ONE COPY PER PROCESS (F1): registered under both `force_presence_plugin` and
`swift_plugins.force_presence_plugin`; _install() is once()-guarded (a second copy would append a
second forcing LogitsProcessor). QURE_PLUGIN_DEDUP=0 restores the legacy behaviour.
"""
# F1 -- must stay the first statements of the module (qure_plugin_mode.dedup_module)
try:
    from swift_plugins.qure_plugin_mode import dedup_module as _dedup_module, once as _once
except ImportError:  # repo root not on sys.path; swift put this plugin's own dir there
    from qure_plugin_mode import dedup_module as _dedup_module, once as _once
_dedup_module(__name__, 'force_presence_plugin')

import json, os, re  # noqa: E401,E402
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from swift.utils import get_logger

logger = get_logger()
TAU = float(os.environ.get("MLD_FORCE_TAU", "0.5"))
TAU_BY_KEY = json.load(open(os.environ["MLD_FORCE_TAU_JSON"])) if os.environ.get("MLD_FORCE_TAU_JSON") else {}
OUT = os.environ.get("MLD_FORCE_OUT")
_KEYRE = re.compile(r'"([a-z0-9_]+)":\s*\{\s*$')


class ForcePresence(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tok = tokenizer
        def one(s):
            ids = tokenizer.encode(s, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        self.id_true = one(" true") or one("true")
        self.id_false = one(" false") or one("false")
        self.buf = ""
        self.prev_len = 0
        self.records = []

    def __call__(self, input_ids, scores):
        seq = input_ids[0]
        # reset on a fresh generation
        if seq.shape[0] < self.prev_len:
            self.buf = ""
        self.prev_len = seq.shape[0]
        # append the most recently generated token's text
        self.buf += self.tok.decode(seq[-1:].tolist())
        # are we about to emit a `presence` value? (buf just ended with  presence":)
        if self.buf.replace(" ", "").endswith('presence":'):
            p = torch.softmax(scores[0, [self.id_true, self.id_false]].float(), dim=-1)
            p_true = float(p[0])
            m = None
            for mm in re.finditer(r'"([a-z0-9_]+)":\s*\{', self.buf):
                m = mm.group(1)
            key = m or "?"
            tau = float(TAU_BY_KEY.get(key, TAU))
            decision = p_true >= tau
            forced = self.id_true if decision else self.id_false
            scores[:] = float("-inf")
            scores[0, forced] = 0.0
            self.records.append({"finding": key, "p_true": round(p_true, 5),
                                 "tau": tau, "forced_present": bool(decision)})
        return scores


def _install():
    if not _once('force_presence_plugin._install'):
        return  # F1: already installed in this process
    from swift.template.base import Template
    orig = Template.prepare_generate_kwargs
    _procs = {}

    def patched(self, generate_kwargs, model=None):
        gk = orig(self, generate_kwargs, model=model)
        fp = ForcePresence(self.tokenizer)
        _procs["last"] = fp
        lp = gk.get("logits_processor") or LogitsProcessorList()
        lp.append(fp); gk["logits_processor"] = lp
        return gk

    Template.prepare_generate_kwargs = patched

    if OUT:
        from swift.template.templates.qwen import Qwen3_5Template
        odec = Qwen3_5Template.decode

        def decode(self, generate_ids, **kwargs):
            text = odec(self, generate_ids, **kwargs)
            fp = _procs.get("last")
            if fp is not None and fp.records:
                with open(OUT, "a") as f:
                    f.write(json.dumps({"decisions": fp.records,
                                        "n_present": sum(r["forced_present"] for r in fp.records)}) + "\n")
                fp.records = []
            return text
        Qwen3_5Template.decode = decode
    logger.info(f"[force-presence] installed | global tau={TAU} | per-key overrides={len(TAU_BY_KEY)} | out={OUT}")


_install()
logger.info("[force-presence] plugin loaded")
