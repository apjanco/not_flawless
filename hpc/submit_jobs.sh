#!/bin/bash
# hpc/submit_jobs.sh
# Submit N parallel sharded evaluation jobs.
# Usage: bash hpc/submit_jobs.sh [num_shards]
# Example: bash hpc/submit_jobs.sh 4

NUM_SHARDS=${1:-4}

echo "Submitting ${NUM_SHARDS} sharded jobs..."

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    JOB_ID=$(sbatch \
        --export=ALL,QWEN_SHARD_ID=${SHARD_ID},QWEN_NUM_SHARDS=${NUM_SHARDS} \
        --job-name="ocr_eval_shard${SHARD_ID}of${NUM_SHARDS}" \
        hpc/submit_job.slurm | awk '{print $NF}')
    echo "  Shard ${SHARD_ID}/${NUM_SHARDS} -> job ${JOB_ID}"
done

echo "Done. Monitor with: squeue -u \$USER"
