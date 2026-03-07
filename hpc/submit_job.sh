#!/bin/bash
# hpc/submit_job.sh
# SLURM job submission script for Adroit HPC

#SBATCH --job-name=ocr_evaluation
#SBATCH --output=results/logs/job_%j.log
#SBATCH --error=results/logs/job_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=apjanco@princeton.edu

# Load modules
module load anaconda3/2025.6

# Create results directories
mkdir -p results/logs
mkdir -p results/metrics
mkdir -p results/visualizations

# Activate environment
conda activate not_flawless

# Run evaluation orchestrator
echo "Starting OCR/HTR Model Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "=========================================="
echo ""

cd "$(dirname "$SLURM_SCRIPT")"
python hpc/run_evaluation.py

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "Time: $(date)"
echo "Results saved to: results/"
