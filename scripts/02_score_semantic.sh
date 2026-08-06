#!/usr/bin/env bash
#SBATCH --job-name=score_semantic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/score_%j.out
#SBATCH --error=logs/score_%j.err
#
# Scores both prediction files with CER/WER + misnomer semantic scoring.
# Requires results/qwen_predictions.jsonl and results/churro_predictions.jsonl
# to exist (produced by 01_infer_*.sh jobs).
#
# Submit after both inference jobs complete:
#   QWEN_JOB=$(sbatch --parsable scripts/01_infer_qwen.sh)
#   CHURRO_JOB=$(sbatch --parsable scripts/01_infer_churro.sh)
#   sbatch --dependency=afterok:${QWEN_JOB}:${CHURRO_JOB} scripts/02_score_semantic.sh
#
# Or submit manually once both prediction files exist:
#   sbatch scripts/02_score_semantic.sh

set -eo pipefail

mkdir -p logs results

module purge
module load anaconda3/2025.12
module load cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate churro_exp

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1

# Churro/IAM texts vary widely in length, which fragments PyTorch's CUDA
# caching allocator over a long run (many distinct tensor shapes) until an
# allocation fails despite plenty of nominally-free VRAM -- confirmed by the
# CUDA OOM message itself recommending this exact setting on 2026-07-24.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"

# Compute nodes have no internet access, so misnomer cannot be (re)installed
# from GitHub here. Upgrade it manually from a login/vis node beforehand:
#   pip install --upgrade --force-reinstall --no-deps \
#       "misnomer[model,multilingual,gates] @ git+https://github.com/apjanco/misnomer.git"
python -c "import misnomer; print('misnomer package ready')"

# Score sequentially, not concurrently: two concurrent processes each hold a
# full model stack (LM + embedder + dictionary) in GPU memory, which hit
# CUDA OOM (mid-run, not just at the end) when tried on 2026-07-24 with the
# post-v1.1 misnomer footprint. Sequential avoids the failure mode entirely.
python src/score.py \
    --input  results/qwen_predictions.jsonl \
    --output results/qwen_scored.jsonl \
    --log_every 50

python src/score.py \
    --input  results/churro_predictions.jsonl \
    --output results/churro_scored.jsonl \
    --log_every 50

echo "Semantic scoring complete."
echo "Qwen scored:   results/qwen_scored.jsonl"
echo "Churro scored: results/churro_scored.jsonl"
