#!/usr/bin/env bash
# scripts/00_setup.sh
# One-time environment setup on Della.
# Run interactively on a login node (no GPU needed):
#   bash scripts/00_setup.sh
#
# After completion, activate with:
#   conda activate churro_exp

set -eo pipefail

module purge
module load anaconda3/2025.12
module load cudatoolkit/12.9

# ── create env (python + pip only; fast) ─────────────────────────────────────
# If the env already exists, skip creation:
conda env create -f environment.yml || conda env update --prune -f environment.yml
conda activate churro_exp

# ── pip packages (driven directly for speed) ──────────────────────────────────
# 1. Pin torch + torchvision with the CUDA 12.8 variant tag.
#    Using --index-url (not --extra-index-url) so pip only looks at the CUDA
#    wheel index for these two packages, preventing it from finding the ancient
#    torchvision-0.1.6 on PyPI and backtracking through every torch version.
pip install \
    torch==2.7.1+cu128 \
    torchvision==0.22.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 2. Install vllm separately so its own torch constraint is resolved against
#    the already-installed 2.7.1+cu128 wheel, not from scratch.
pip install "vllm>=0.6.0"

# 3. Remaining dependencies (no torch/vllm in requirements.txt).
pip install -r requirements.txt

# ── pre-fetch dataset and models to HuggingFace cache on scratch ──────────────
export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface

echo "Downloading churro-dataset test split..."
python -c "
from datasets import load_dataset
ds = load_dataset('stanford-oval/churro-dataset', split='test')
print(f'  {len(ds)} examples loaded')
print(f'  columns: {ds.column_names}')
"

echo "Downloading Qwen/Qwen2.5-VL-3B-Instruct..."
hf download Qwen/Qwen2.5-VL-3B-Instruct

echo "Downloading stanford-oval/churro-3B..."
hf download stanford-oval/churro-3B

echo ""
echo "Setup complete.  Activate with:  conda activate churro_exp"
