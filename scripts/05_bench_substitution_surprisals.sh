#!/usr/bin/env bash
#SBATCH --job-name=bench_subsurp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_subsurp_%j.out
#SBATCH --error=logs/bench_subsurp_%j.err
#
# Benchmarks legacy vs new substitution_surprisals on a real Churro document
# (image_path=628, ~2986 gt words / ~450 substitutions) that previously
# required the 256G residual pass to score without OOMing under misnomer
# v1.1's uncached per-substitution scoring (see plan.md "Known Limitations").
# 256G requested defensively so the legacy call (which allocates unbounded
# per-substitution tensors) has room to run to completion for comparison.

set -eo pipefail

mkdir -p logs

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
python -c "import misnomer, importlib.metadata as m; print('misnomer commit:', m.distribution('misnomer').read_text('direct_url.json'))"

python src/bench_substitution_surprisals.py
