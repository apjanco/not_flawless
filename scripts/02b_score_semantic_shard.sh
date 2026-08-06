#!/usr/bin/env bash
#SBATCH --job-name=score_shard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/score_shard_%j.out
#SBATCH --error=logs/score_shard_%j.err
#
# One shard of the Churro/qwen semantic rescore. Requires SHARD_ID and
# NUM_SHARDS to be set via --export (see submit_semantic_shards.sh) and
# results/shards/*_shard{ID}of{N}.jsonl to exist (see src/shard_prep.py).

set -eo pipefail

: "${SHARD_ID:?SHARD_ID must be set}"
: "${NUM_SHARDS:?NUM_SHARDS must be set}"

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
echo "Shard: ${SHARD_ID} of ${NUM_SHARDS}"

python -c "import misnomer; print('misnomer package ready')"

python src/score.py \
    --input  "results/shards/qwen_predictions_shard${SHARD_ID}of${NUM_SHARDS}.jsonl" \
    --output "results/shards/qwen_scored_shard${SHARD_ID}of${NUM_SHARDS}.jsonl" \
    --log_every 20

python src/score.py \
    --input  "results/shards/churro_predictions_shard${SHARD_ID}of${NUM_SHARDS}.jsonl" \
    --output "results/shards/churro_scored_shard${SHARD_ID}of${NUM_SHARDS}.jsonl" \
    --log_every 20

echo "Shard ${SHARD_ID} of ${NUM_SHARDS} complete."
