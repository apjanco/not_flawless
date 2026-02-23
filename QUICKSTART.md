# QUICK START GUIDE

## Pre-Evaluation Setup (5 minutes)

### 1. Validate Project Setup
```bash
cd /home/apjanco/projects/not_flawless
python test_setup.py
```
All checks should pass ✓

### 2. Download IAM Database
```bash
bash setup/download_data.sh
```
This downloads ~15GB from Hugging Face. Takes 5-30 minutes depending on connection.

### 3. Install Dependencies
```bash
bash setup/setup_environment.sh
# Or manually:
pip install -r requirements.txt
```

## Running Batch Evaluations on Adroit

### Single Command to Submit All
```bash
sbatch hpc/submit_job.sh
```

### Monitor Progress
```bash
# Check job status
squeue -u $USER

# View job output (once submitted)
tail -f slurm-<JOBID>.out
```

### After Job Completes
```bash
# Copy results to local machine
scp -r adroit.princeton.edu:~/not_flawless/results ./local-backup/

# View results
ls results/
cat results/tesseract_results.jsonl | head -5
cat results/metrics/summary.csv
```

## Running API Evaluations (Interactive Node)

### Step 1: Request Interactive Node
```bash
# From Adroit login node
salloc --time=02:00:00 --cpus-per-task=4 --mem=16G --gres=gpu:1
```

### Step 2: Set API Credentials
```bash
# For Portkey (ChatGPT + Gemini)
export PORTKEY_API_KEY="your-key-here"

# For Google Vision
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

### Step 3: Run Evaluation
```bash
bash hpc/run_api_evaluation.sh
```

### Step 4: View Results
```bash
# While still on interactive node
ls results/
cat results/chatgpt_results.jsonl | head -5
```

## Understanding Results

### JSONL Format (One line per image)
```json
{
  "image_path": "/path/to/image.png",
  "ground_truth": "expected text",
  "predicted_text": "recognized text",
  "cer": 5.2,
  "wer": 0.0,
  "inference_time": 0.234,
  "error": null
}
```

### Summary Metrics (JSON)
```json
{
  "num_samples": 1000,
  "num_successful": 998,
  "mean_cer": 12.34,
  "median_cer": 8.5,
  "mean_inference_time": 0.234
}
```

## Common Tasks

### Load and Analyze Results
```python
import jsonlines
import pandas as pd

# Load JSONL results
with jsonlines.open('results/tesseract_results.jsonl') as reader:
    results = list(reader)

# Convert to DataFrame
df = pd.DataFrame(results)

# Statistics
print(df['cer'].describe())
print(df['wer'].describe())
print(f"Success rate: {(df['error'].isna().sum() / len(df)) * 100:.1f}%")
```

### Combine Multiple Models
```bash
# Create comparison CSV
python << 'EOF'
import json
import csv
from pathlib import Path

results_dir = Path('results/metrics')
summary = {}

for metrics_file in results_dir.glob('*_metrics.json'):
    model_name = metrics_file.stem.replace('_metrics', '')
    with open(metrics_file) as f:
        summary[model_name] = json.load(f)

# Create comparison table
print("Model Comparison:")
print("-" * 60)
for model, metrics in summary.items():
    print(f"{model:20} | CER: {metrics['mean_cer']:6.2f} | WER: {metrics['mean_wer']:6.2f}")
EOF
```

### Test with Small Sample (before full run)
```python
# In evaluators/tesseract_eval.py, change load call:
test_images, test_labels = load_iam_data(split="test", limit=100)
```

## Troubleshooting

### "IAM data directory not found"
```bash
# Check download completed
ls -lh data/iam/
# If empty, re-run download
bash setup/download_data.sh
```

### "Module not found" error
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### SLURM Job Failed
```bash
# Check error message
cat slurm-<JOBID>.err

# Common issues:
# - Module not loaded: add to submit_job.sh: module load cuda/12.2
# - Out of memory: increase --mem=32G in submit_job.sh
# - GPU not available: check with: sinfo --gres
```

### API Keys Not Working
```bash
# Verify credentials are set
echo $PORTKEY_API_KEY
echo $GOOGLE_APPLICATION_CREDENTIALS

# Test connectivity
python << 'EOF'
import requests
import os

api_key = os.getenv("PORTKEY_API_KEY")
print(f"Key set: {bool(api_key)}")
print(f"Key length: {len(api_key) if api_key else 0}")
EOF
```

## Performance Tips

1. **Start small**: Test with 100 images first
2. **Use GPU**: Request GPU node for faster models
3. **Monitor costs**: API models can be expensive - watch usage
4. **Batch processing**: Some models faster with batches (not yet implemented)
5. **Parallel runs**: Can run different models simultaneously on separate nodes

## Key Files

| File | Purpose |
|------|---------|
| `test_setup.py` | Validation before running |
| `setup/download_data.sh` | Download IAM database |
| `hpc/submit_job.sh` | SLURM batch job |
| `hpc/run_evaluation.py` | Orchestrates all models |
| `evaluators/<model>_eval.py` | Individual model evaluator |
| `results/<model>_results.jsonl` | Per-image results |
| `results/metrics/summary.csv` | All models comparison |

## Support

- **Data issues**: See `CODE_REVIEW.md` section "Known Limitations"
- **API setup**: See `API_EVALUATION.md` for detailed instructions
- **Project details**: See `SPEC.md` for full specification
- **Architecture**: See `README.md` for overview

---

**Status**: Production ready ✓
**Updated**: 2026-02-23
**Models Available**: 11 (6 batch + 3 API + 2 placeholder)
**Expected Runtime**: 
- Tesseract: ~500ms per image
- API models: ~2-5s per image
