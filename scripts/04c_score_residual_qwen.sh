#!/usr/bin/env bash
#SBATCH --job-name=score_residual_qwen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/score_residual_qwen_%j.out
#SBATCH --error=logs/score_residual_qwen_%j.err
#
# One-off high-memory pass over the 71 qwen documents that repeatedly
# OOM'd across shard rounds -- same root cause as the churro residual
# (see "Known Limitations" in plan.md): substitution_surprisals can
# request a single tensor allocation unbounded in document length, which
# no amount of memory reliably fixes for the worst documents. Run
# specifically to recover matched qwen/churro pairs for the paired
# base-vs-fine-tuned comparison, not to chase 100% qwen coverage alone.

set -eo pipefail

mkdir -p logs results/shards

module purge
module load anaconda3/2025.12
module load cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate churro_exp

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"

python -c "import misnomer; print('misnomer package ready')"

python src/score.py \
    --input  results/shards/qwen_predictions_residual.jsonl \
    --output results/shards/qwen_scored_residual.jsonl \
    --log_every 5 \
    --reload_every 20

echo "Residual scoring complete."
