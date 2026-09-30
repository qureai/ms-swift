#!/usr/bin/env python
"""Record the LM's P(true) at every `"presence":` slot during NORMAL (free-running) greedy
generation — does NOT modify logits. One JSON line per image: {finding_key: p_true, ...}.

Env: MLD_REC_OUT = output jsonl path.  Use --infer_backend pt.
Align rows to the input val_dataset by ORDER (run with --val_dataset_shuffle false).

Inference-only and deliberately NOT mode-gated (installs under any qure_plugin_mode).
ONE COPY PER PROCESS (F1): registered under both `record_presence_plugin` and
`swift_plugins.record_presence_plugin`; _install() is once()-guarded (a second copy would write
every row twice). QURE_PLUGIN_DEDUP=0 restores the legacy behaviour.
"""
# F1 -- must stay the first statements of the module (qure_plugin_mode.dedup_module)
try:
    from swift_plugins.qure_plugin_mode import dedup_module as _dedup_module, once as _once
except ImportError:  # repo root not on sys.path; swift put this plugin's own dir there
    from qure_plugin_mode import dedup_module as _dedup_module, once as _once
_dedup_module(__name__, 'record_presence_plugin')

import json, os, re  # noqa: E401,E402
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from swift.utils import get_logger

logger = get_logger()
OUT = os.environ.get("MLD_REC_OUT", "presence_logprobs.jsonl")


class RecordPresence(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tok = tokenizer
        def one(s):
            ids = tokenizer.encode(s, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        self.id_true = one(" true") or one("true")
        self.id_false = one(" false") or one("false")
        self.buf = ""
        self.prev_len = 0
        self.records = {}

    def __call__(self, input_ids, scores):
        seq = input_ids[0]
        if seq.shape[0] < self.prev_len:      # fresh generation
            self.buf = ""; self.records = {}
        self.prev_len = seq.shape[0]
        self.buf += self.tok.decode(seq[-1:].tolist())
        if self.buf.replace(" ", "").endswith('presence":'):
            p = torch.softmax(scores[0, [self.id_true, self.id_false]].float(), dim=-1)
            key = None
            for m in re.finditer(r'"([a-z0-9_]+)":\s*\{', self.buf):
                key = m.group(1)
            if key and key not in self.records:
                self.records[key] = round(float(p[0]), 5)
        return scores       # unchanged — greedy proceeds normally (free-running)


def _install():
    if not _once('record_presence_plugin._install'):
        return  # F1: already installed in this process
    from swift.template.base import Template
    from swift.template.templates.qwen import Qwen3_5Template
    orig = Template.prepare_generate_kwargs
    _proc = {}

    def patched(self, generate_kwargs, model=None):
        gk = orig(self, generate_kwargs, model=model)
        rp = RecordPresence(self.tokenizer); _proc["last"] = rp
        lp = gk.get("logits_processor") or LogitsProcessorList()
        lp.append(rp); gk["logits_processor"] = lp
        return gk
    Template.prepare_generate_kwargs = patched

    odec = Qwen3_5Template.decode

    def decode(self, generate_ids, **kwargs):
        text = odec(self, generate_ids, **kwargs)
        rp = _proc.get("last")
        if rp is not None:
            os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
            with open(OUT, "a") as f:
                f.write(json.dumps({"p_true": rp.records}) + "\n")
            rp.records = {}
        return text
    Qwen3_5Template.decode = decode
    logger.info(f"[record-presence] installed -> {OUT}")


_install()
logger.info("[record-presence] plugin loaded")
