"""CT fork: map a training dataset file path to a short per-dataset `channel` name, so per-dataset loss
can be logged separately (consumed by Seq2SeqTrainer._log_channel_losses -> wandb `loss/<channel>`).

Activated by the env var CT_DATASET_CHANNELS (set in the training script). When on, the dataset loader
tags every dataset that does NOT already carry a `channel` column (the complete cls-json already sets its
own `channel` in-record, so it is left as-is). Rules are ordered (substring, channel); first substring
found in the path basename wins. A dataset matching nothing falls back to 'other' so ALL datasets end up
with a `channel` column (required for concatenation to succeed).
"""
import os
from typing import Optional

# Most-specific substrings first (first match wins on the path basename).
_CHANNEL_RULES = [
    ('complete_cls_json', 'cls_json'),                        # 28_Sep all-tags true/false roll-call
    ('tag_extraction_cls_tokens', 'cls_tags'),               # <TAG> multi-label bag
    ('cls_plus_mini_report_multiturn', 'cls_rep_mt'),        # tags -> mini report, multi-turn
    ('cls_plus_mini_report_combined', 'cls_rep'),            # tags -> mini report, single-turn
    ('full_structured_report', 'rep_full'),                  # full JSON report
    ('mini_structured_report', 'rep_mini'),                  # short JSON report
    ('knowledge_graph', 'kg'),                               # KG extraction
    ('pe_multiturn', 'pe'),                                  # pulmonary embolism grounding
    ('cac_multiturn', 'cac'),                                # coronary artery calcium (28_aug)
    ('cac_json', 'cac'),                                     # coronary artery calcium (30_sep json)
    ('nodule_fpr', 'fpr'),                                   # nodule false-positive reduction
    ('segmed_nodule_bbox', 'nod_bbox'),                      # nodule bounding boxes
    ('segmed_nodule_qa', 'nod_qa'),                          # nodule Q&A
    ('omnict_anatomy_grounded_abnormality', 'anat_abn'),     # anatomy-conditioned abnormality (omnict)
    ('segmed_anatomy_grounded_abnormality', 'seg_anat_abn'), # anatomy-conditioned abnormality (segmed)
    ('anatomy_localization', 'anat_loc'),                    # organ/anatomy localization
    ('omnict_label_bbox', 'omnict_bbox'),                    # omnict label + bounding boxes (multiturn)
    ('omnict_region_findings', 'omnict_qa'),                 # omnict region-findings Q&A
    ('omnict_abnormality', 'abn'),                           # abnormality grounding (omnict)
    ('report_comparison', 'rep_cmp'),                        # prior-vs-current report comparison
]


def ct_channel_for_path(path):
    """Short channel name for a dataset file path (by basename substring), or 'other' if unmatched."""
    if not path or not isinstance(path, str):
        return None
    base = os.path.basename(path)
    for sub, ch in _CHANNEL_RULES:
        if sub in base:
            return ch
    return 'other'
