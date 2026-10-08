#!/usr/bin/env bash
# SLURM wrapper that runs run_ct_infer.sh (in-process CT-volume inference, 1 process/GPU, bf16).
# Resources are sized to survive the scron reaper (kills jobs >60min that are under-utilized):
#   -c 24      -> <=25 CPUs, so the "CPU eff <10%" rule NEVER applies (exempt).
#   --mem=100G -> not ">100G", so the "RAM eff <10%" rule NEVER applies (exempt); covers model-load spike.
#   --gpus=4   -> whole l40s node (4 GPUs/node) => logical devices 0-3 match run_ct_infer.sh GPUS=(0 1 2 3).
# GPU rules DO apply (>1 GPU): active bf16 inference (batch 4, 256^3 volumes) keeps GPU mem-eff >20% and
# busy-eff >10%, so it won't be reaped while it's actually inferring. qrnd20/qrnd12 excluded (faulty GPUs).
#
# Submit:  sbatch slurm_ct_infer.sh     (edit run_ct_infer.sh for checkpoint/prompt/dataset)
#
#SBATCH --job-name=ct_infer_1308
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --exclude=qrnd20,qrnd12
#SBATCH --output=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/infer/ct_volume/logs/%x-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/infer/ct_volume/logs/%x-%j.err

set -euo pipefail
CT_DIR=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/infer/ct_volume
mkdir -p "$CT_DIR/logs"

# --- env: swift_july (transformers 5.9.0 -> supports the qwen3_5 checkpoint). Activating it makes
#     run_ct_infer.sh's `PY=python` resolve to swift_july's python. conda's cuda activate scripts
#     reference unset vars, so relax `set -u` around activation. ---
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-?} python=$(which python)"
bash "$CT_DIR/run_ct_infer.sh"
echo "[$(date)] done (exit $?)"
