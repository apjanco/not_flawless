#!/usr/bin/env bash
#SBATCH --job-name=score_iam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/score_iam_%j.out
#SBATCH --error=logs/score_iam_%j.err
#
# Adds misnomer semantic scoring to the six IAM result files (predictions
# were generated earlier on Adroit; this only rescores them, no inference).
# Requires results/iam_raw/*.jsonl to exist (copied from the Adroit project).
#
# Submit with:
#   sbatch scripts/04_score_iam.sh

set -eo pipefail

mkdir -p logs results/iam_scored

module purge
module load anaconda3/2025.12
module load cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate churro_exp

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1

echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"

# Compute nodes have no internet access, so misnomer cannot be (re)installed
# from GitHub here. Upgrade it manually from a login/vis node beforehand:
#   pip install --upgrade --force-reinstall --no-deps \
#       "misnomer[model,multilingual,gates] @ git+https://github.com/apjanco/misnomer.git"
python -c "import misnomer; print('misnomer package ready')"

python src/score_iam.py \
    --input-glob "results/iam_raw/*_results.jsonl" \
    --output-dir results/iam_scored \
    --log_every 200

echo "IAM semantic scoring complete. Output in results/iam_scored/"
