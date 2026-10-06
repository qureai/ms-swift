#!/usr/bin/env bash
# Post-process the PHI audit: parse pass 1 -> retry the empty generations -> merge -> report.
# Waits for any in-flight `swift infer` to exit first, so it can be launched while the main
# run is still going.
set -euo pipefail

PY=/home/ubuntu/miniconda3/envs/ms-swift-may/bin/python
REPO=/fastdata4/sahil_work/VLM_qure/vlm/swift
SCR="$REPO/vlm_inference_analysis/scripts"
BASE=/fastdata4/sahil_work/VLM_qure/eval_outputs/Qwen3.5-27B_phi_audit_25k
S="$BASE/screened"

cd "$REPO"

while pgrep -f "swift/cli/infer.py" >/dev/null; do sleep 60; done
echo "== pass 1 finished: $(wc -l < "$S/result.jsonl") rows =="

echo "== parsing pass 1 =="
$PY "$SCR/parse_phi_results.py" --result "$S/result.jsonl" \
    --sidecar "$S/infer_input.ids.jsonl" --out-dir "$S"

echo "== building retry input =="
$PY "$SCR/build_phi_retry.py" --audit "$S/phi_audit.json" --out-dir "$S/retry"

if [[ $(wc -l < "$S/retry/infer_input.jsonl") -gt 0 ]]; then
    echo "== retry pass ($(wc -l < "$S/retry/infer_input.jsonl") images) =="
    "$REPO/examples/deploy/run_phi_audit_retry.sh" || echo "WARN retry pass failed; continuing"
    while pgrep -f "swift/cli/infer.py" >/dev/null; do sleep 30; done
    if [[ -s "$S/retry/result.jsonl" ]]; then
        $PY "$SCR/parse_phi_results.py" --result "$S/retry/result.jsonl" \
            --sidecar "$S/retry/infer_input.ids.jsonl" --out-dir "$S/retry"
    fi
fi

echo "== merging =="
RETRY_ARG=()
[[ -f "$S/retry/phi_audit.json" ]] && RETRY_ARG=(--retry-audit "$S/retry/phi_audit.json")
$PY "$SCR/merge_phi_results.py" --audit "$S/phi_audit.json" "${RETRY_ARG[@]}" \
    --corrupt "$S/corrupt_images.json" --out-dir "$BASE"

echo "== report =="
$PY "$SCR/report_phi_stats.py" --audit "$BASE/phi_audit_25k.json" \
    --summary "$BASE/phi_audit_25k_summary.json" \
    --out "$BASE/PHI_AUDIT_REPORT.md"

cp "$BASE/PHI_AUDIT_REPORT.md" "$REPO/vlm_inference_analysis/PHI_AUDIT_REPORT.md"
echo "== DONE =="
ls -la "$BASE"/phi_audit_25k*.json "$BASE/PHI_AUDIT_REPORT.md"
