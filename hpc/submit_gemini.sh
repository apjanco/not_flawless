#!/bin/bash
# hpc/submit_gemini.sh
# SLURM job submission script for Gemini-only evaluation via Portkey

#SBATCH --job-name=gemini_eval
#SBATCH --output=results/logs/gemini_%j.log
#SBATCH --error=results/logs/gemini_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=apjanco@princeton.edu

# Load modules
module load anaconda3/2025.6

# Create results directories
mkdir -p results/logs
mkdir -p results/metrics

# Activate environment
conda activate not_flawless

echo "Starting Gemini evaluation via Portkey"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "=========================================="

# Ensure API keys are set (export them in your environment before submitting,
# or uncomment and fill in below):
# export PORTKEY_API_KEY="your-portkey-key"
# export GOOGLE_API_KEY="your-google-key"

cd "$(dirname "$SLURM_SCRIPT")/.."

python - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from evaluators import gemini_eval

metrics = gemini_eval.evaluate()
print("\nFinal metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
EOF

echo ""
echo "=========================================="
echo "Gemini Evaluation Complete"
echo "Time: $(date)"
echo "Results saved to: results/"
