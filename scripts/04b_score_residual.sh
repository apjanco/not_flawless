#!/usr/bin/env bash
#SBATCH --job-name=score_residual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/score_residual_%j.out
#SBATCH --error=logs/score_residual_%j.err
#
# One-off high-memory pass over the ~74 churro documents that repeatedly
# OOM'd at 128G across two shard rounds -- longer/substitution-dense
# documents under misnomer v1.1's uncached per-substitution scoring
# (see "Known Limitations" in plan.md). If this still fails, these
# documents should be accepted as an unscored gap rather than escalated
# further.

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
    --input  results/shards/churro_predictions_residual.jsonl \
    --output results/shards/churro_scored_residual.jsonl \
    --log_every 5 \
    --reload_every 20

echo "Residual scoring complete."
