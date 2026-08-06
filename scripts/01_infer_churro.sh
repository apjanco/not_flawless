#!/usr/bin/env bash
#SBATCH --job-name=infer_churro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/infer_churro_%j.out
#SBATCH --error=logs/infer_churro_%j.err
#
# Runs Churro 3B (fine-tuned) inference on the churro-dataset test split.
# This job is independent of 01_infer_qwen.sh and can run in parallel.
# Submit with:  sbatch scripts/01_infer_churro.sh   (do NOT run with bash directly)
# To chain with scoring:
#   CHURRO_JOB=$(sbatch --parsable scripts/01_infer_churro.sh)
#   sbatch --dependency=afterok:$CHURRO_JOB scripts/02_score_semantic.sh

set -eo pipefail

mkdir -p logs results

module purge
module load anaconda3/2025.12
module load cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate churro_exp

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/infer.py \
    --model stanford-oval/churro-3B \
    --output results/churro_predictions.jsonl \
    --max_new_tokens 20000 \
    --tensor_parallel_size 1 \
    --max_model_len 32768 \
    --log_every 50

echo "Churro inference complete."
