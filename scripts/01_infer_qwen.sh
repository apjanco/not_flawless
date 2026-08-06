#!/usr/bin/env bash
#SBATCH --job-name=infer_qwen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/infer_qwen_%j.out
#SBATCH --error=logs/infer_qwen_%j.err
#
# Runs zero-shot Qwen 2.5 VL 3B inference on the churro-dataset test split.
# Submit with:  sbatch scripts/01_infer_qwen.sh   (do NOT run with bash directly)
# To chain with scoring:
#   QWEN_JOB=$(sbatch --parsable scripts/01_infer_qwen.sh)
#   sbatch --dependency=afterok:$QWEN_JOB scripts/02_score_semantic.sh

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
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --output results/qwen_predictions.jsonl \
    --max_new_tokens 20000 \
    --tensor_parallel_size 1 \
    --max_model_len 32768 \
    --log_every 50

echo "Qwen inference complete."
