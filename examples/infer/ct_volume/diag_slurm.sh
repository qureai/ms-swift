#!/usr/bin/env bash
#SBATCH --job-name=ct_diag
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=qrnd20,qrnd12
#SBATCH --output=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/infer/ct_volume/logs/ct_diag-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/infer/ct_volume/logs/ct_diag-%j.err
set -euo pipefail
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u
python /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/infer/ct_volume/diag_truncation.py
