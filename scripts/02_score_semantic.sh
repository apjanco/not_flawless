#!/usr/bin/env bash
#SBATCH --job-name=score_semantic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
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

echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"

# Score both files concurrently on the same GPU.
# misnomer keeps its models in GPU memory; two concurrent processes will share
# VRAM.  If OOM occurs, remove the & / wait and score sequentially.
python src/score.py \
    --input  results/qwen_predictions.jsonl \
    --output results/qwen_scored.jsonl \
    --log_every 50 &

python src/score.py \
    --input  results/churro_predictions.jsonl \
    --output results/churro_scored.jsonl \
    --log_every 50 &

wait

echo "Semantic scoring complete."
echo "Qwen scored:   results/qwen_scored.jsonl"
echo "Churro scored: results/churro_scored.jsonl"
