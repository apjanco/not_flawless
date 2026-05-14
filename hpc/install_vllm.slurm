#!/bin/bash
# hpc/install_vllm.slurm
# Install vLLM into the not_flawless conda env on a compute node.
#
# Compute nodes have no internet access, so the wheel must be downloaded first
# on the login node. Run this once before submitting:
#
#   module load anaconda3/2025.6
#   conda activate not_flawless
#   VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest \
#     | python -c "import sys,json; print(json.load(sys.stdin)['tag_name'].lstrip('v'))")
#   mkdir -p /scratch/network/aj7878/not_flawless/setup/wheels
#   pip download \
#     "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu128-cp38-abi3-manylinux_2_35_x86_64.whl" \
#     -d /scratch/network/aj7878/not_flawless/setup/wheels
#
# Then submit: sbatch hpc/install_vllm.slurm

#SBATCH --job-name=install_vllm
#SBATCH --output=results/logs/install_%j.log
#SBATCH --error=results/logs/install_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=apjanco@princeton.edu

module load anaconda3/2025.6
module load cudatoolkit/12.8

mkdir -p results/logs

conda activate not_flawless

echo "Installing vLLM"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Time:   $(date)"
echo "CUDA:   $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "=========================================="

WHEEL_DIR="/scratch/network/aj7878/not_flawless/setup/wheels"
WHEEL=$(ls "${WHEEL_DIR}"/vllm-*.whl 2>/dev/null | head -1)

if [[ -z "$WHEEL" ]]; then
    echo "ERROR: No vLLM wheel found in ${WHEEL_DIR}"
    echo "Download it on the login node first (see comments at top of this script)"
    exit 1
fi

echo "Installing wheel: $WHEEL"
pip install --no-index --find-links="${WHEEL_DIR}" "$WHEEL"

echo "=========================================="
echo "Done: $(date)"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)' 2>&1)"
