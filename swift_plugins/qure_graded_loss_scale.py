#!/usr/bin/env python
"""Graded per-token AR loss weights for the cls-JSON / ASP / graph multi-turn data.

Registers loss_scale `qure_graded`:

  classification JSON turn
      0.10  JSON structure -- braces, quotes, commas, and the attribute keys
            "presence"/"size"/"location"/"characteristics", plus null values
      0.30  category and finding NAME tokens
      1.00  `false` presence value
      6.00  `true`  presence value        <- 6x the weight of false, as specified
  ASP turn
      3.00  the findings/observation facts
      1.00  everything else (headers, separators, boilerplate)
  hierarchical graph turn
      3.00  finding lines and triples naming a finding
      1.00  the tree scaffolding
  report turn
      1.00  uniform

`is_binary = False`, which is what makes graded weights possible -- and it is also why this
needs the chunked CE: a non-binary loss_scale routes the loss through a path that materializes
logits (248,320 vocab -> ~90 GB at bs=28), so it MUST be paired with the chunked weighted CE
rather than the stock loss. See qure_chunked_ce.py.

Weights are applied by SPLITTING the response string into segments, each returned with its
weight; swift then tokenizes each segment and repeats the weight across its tokens
(swift/template/base.py:_encode_context_list). So the split points must land on real
character boundaries in the response text.
"""
import os
import re
from typing import List, Tuple

from swift.loss_scale.base import LossScale
from swift.loss_scale.mapping import loss_scale_map
from swift.utils import get_logger

logger = get_logger()

W_FORMAT = float(os.environ.get('QURE_W_FORMAT', '0.10'))
W_NAME = float(os.environ.get('QURE_W_NAME', '0.30'))
W_FALSE = float(os.environ.get('QURE_W_FALSE', '1.0'))
W_TRUE = float(os.environ.get('QURE_W_TRUE', '6.0'))
W_ASP_FINDING = float(os.environ.get('QURE_W_ASP_FINDING', '3.0'))
W_GRAPH_FINDING = float(os.environ.get('QURE_W_GRAPH_FINDING', '3.0'))
W_DEFAULT = float(os.environ.get('QURE_W_DEFAULT', '1.0'))

ATTR_KEYS = ('presence', 'size', 'location', 'characteristics', 'characterization')

# a finding/category name is a quoted key followed by ':' ; the attribute keys above are
# structure, not names
_KEY = re.compile(r'"([A-Za-z0-9_]+)"(\s*:\s*)')
_TF = re.compile(r'\b(true|false)\b')
# ASP facts vs headers/prose/separators. A whitelist of predicate NAMES was wrong -- the real
# targets use predicates like tip_location(...) that no reasonable list would enumerate. Any
# `predicate(args).` line is a fact; ASCII-art separators, emoji step headers, `% comments`
# and free prose are not.
_ASP_FACT = re.compile(r'^\s*[a-z][a-z0-9_]*\([^()]*\)\s*\.?\s*$')
# The TRAINING graph targets look like:
#   **A. Hierarchical Positive Findings Graph**
#   - **Positive Finding 1: Mild right paratracheal soft tissue prominence**
#     - Status: present
# i.e. markdown bullets, NOT the box-drawing tree the model emits at inference. Matching the
# inference format found zero findings in the training data.
_GRAPH_FINDING = re.compile(r'^\s*[-*]?\s*\**\s*(?:Positive\s+)?Finding\s*\d*\s*:', re.I)


def _segments_cls_json(text: str) -> Tuple[List[str], List[float]]:
    """Split the classification JSON into (structure | name | true | false) segments."""
    segs, ws, i = [], [], 0
    events = []
    for m in _KEY.finditer(text):
        events.append((m.start(1), m.end(1), W_FORMAT if m.group(1) in ATTR_KEYS else W_NAME))
    for m in _TF.finditer(text):
        events.append((m.start(1), m.end(1), W_TRUE if m.group(1) == 'true' else W_FALSE))
    events.sort()
    for a, b, w in events:
        if a < i:            # overlapping match, skip
            continue
        if a > i:
            segs.append(text[i:a])
            ws.append(W_FORMAT)
        segs.append(text[a:b])
        ws.append(w)
        i = b
    if i < len(text):
        segs.append(text[i:])
        ws.append(W_FORMAT)
    return segs, ws


def _segments_by_line(text: str, is_content, w_content: float) -> Tuple[List[str], List[float]]:
    """Line-wise split: lines matching `is_content` get w_content, the rest W_DEFAULT."""
    segs, ws = [], []
    for line in text.splitlines(keepends=True):
        segs.append(line)
        ws.append(w_content if is_content(line) else W_DEFAULT)
    return (segs, ws) if segs else ([text], [W_DEFAULT])


class QureGradedLossScale(LossScale):
    """Graded weights over the assistant turns of the cls-JSON multi-turn data."""

    is_binary = False

    def get_loss_scale(self, context: str, **kwargs):
        # only assistant responses reach here with a positive base weight; swift handles the
        # prompt/response split via base_strategy
        t = context.lstrip()
        if '<|ASP|>' in context or t.startswith('%'):
            return _segments_by_line(context, lambda ln: bool(_ASP_FACT.match(ln)), W_ASP_FINDING)
        if 'Hierarchical Positive Findings Graph' in context or 'Positive Findings Triples' in context:
            return _segments_by_line(
                context,
                lambda ln: bool(_GRAPH_FINDING.match(ln)) or ', name,' in ln
                or 'finding_' in ln or ln.lstrip().startswith('(finding'),
                W_GRAPH_FINDING)
        if '"presence"' in context:
            return _segments_cls_json(context)
        return [context], [W_DEFAULT]


loss_scale_map['qure_graded'] = QureGradedLossScale
logger.info(f'[qure-loss-scale] registered `qure_graded` · format={W_FORMAT} name={W_NAME} '
            f'false={W_FALSE} true={W_TRUE} asp_finding={W_ASP_FINDING} '
            f'graph_finding={W_GRAPH_FINDING}')
