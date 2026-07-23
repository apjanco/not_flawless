#!/usr/bin/env bash
#SBATCH --job-name=analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err
#
# Generates summary tables and figures from the scored JSONL files.
# No GPU required — runs on a CPU node.
#
# Submit after scoring is complete:
#   SCORE_JOB=$(sbatch --parsable scripts/02_score_semantic.sh)
#   sbatch --dependency=afterok:$SCORE_JOB scripts/03_analyze.sh
#
# Or run locally once scored files exist:
#   bash scripts/03_analyze.sh

set -eo pipefail

mkdir -p logs results/figures

module purge 2>/dev/null || true
module load anaconda3/2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate churro_exp

echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"

python src/analyze.py \
    --qwen   results/qwen_scored.jsonl \
    --churro results/churro_scored.jsonl \
    --outdir results/figures

echo ""
echo "Analysis complete."
echo "Figures: results/figures/"
echo "Tables:  results/summary_table.csv  results/finetuning_delta.csv"
