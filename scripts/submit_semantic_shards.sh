#!/usr/bin/env bash
# Submit the Churro/qwen semantic rescore as N parallel shard jobs, each on
# its own GPU. Run src/shard_prep.py first to create the shard files.
#
# Usage: bash scripts/submit_semantic_shards.sh [num_shards]
# Example: bash scripts/submit_semantic_shards.sh 4

set -euo pipefail

NUM_SHARDS=${1:-4}

echo "Submitting ${NUM_SHARDS} semantic scoring shard jobs..."

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    JOB_ID=$(sbatch --parsable \
        --export=ALL,SHARD_ID=${SHARD_ID},NUM_SHARDS=${NUM_SHARDS} \
        scripts/02b_score_semantic_shard.sh)
    echo "  shard ${SHARD_ID}/${NUM_SHARDS} -> job ${JOB_ID}"
done

echo "Done. Monitor with: squeue -u \$USER"
