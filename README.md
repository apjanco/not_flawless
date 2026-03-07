# OCR/HTR Models Evaluation on IAM Database

Comprehensive evaluation of OCR and Handwriting Text Recognition models on the IAM Handwriting Database, optimized for Princeton's Adroit HPC cluster.

## Quick Start

### 1. Setup (Run Locally or on Adroit)

```bash
# Make setup scripts executable
chmod +x setup/*.sh

# Download datasets (requires IAM credentials)
./setup/download_data.sh

# Download/prepare models
./setup/download_models.sh

# Install Python dependencies
./setup/setup_environment.sh
```

module load anaconda3/2025.6
conda activate not_flawless

### 2. Run Batch Evaluations on Adroit HPC

```bash
# From Adroit login node:
sbatch hpc/submit_job.sh

# Monitor job:
squeue -u $USER
```

### 3. Run API-Based Models on Interactive Node

See `API_EVALUATION.md` for detailed setup instructions.

```bash
# Request interactive node
salloc --time=02:00:00 --gres=gpu:1

# Set API credentials
export PORTKEY_API_KEY="your-key"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/creds.json"

# Run evaluations
bash hpc/run_api_evaluation.sh
```

### 4. Analyze Results

Download results directory from Adroit and run analysis notebooks in `notebooks/`:
- `01_data_exploration.ipynb` - Dataset overview
- `02_model_evaluation.ipynb` - Results comparison
- `03_results_analysis.ipynb` - Detailed analysis and recommendations

## Project Structure

See `SPEC.md` for detailed project structure and methodology.

## Model Evaluators

Each model has its own evaluation script in `evaluators/`:

### Local Models
| Model | Script | Type | Status |
|-------|--------|------|--------|
| Tesseract | `tesseract_eval.py` | OCR | Template |
| PyLaia | `pylaia_eval.py` | HTR | Template |
| Kraken | `kraken_eval.py` | OCR/HTR | Template |
| Qwen2-VL 8B | `qwen_eval.py` | Vision-Language | Template |
| DeepSeek OCR | `deepseek_eval.py` | OCR | Template |
| Chandra | `chandra_eval.py` | OCR | Template |

### API-Based Models (Interactive Node)
| Model | Script | Provider | Status |
|-------|--------|----------|--------|
| ChatGPT Vision | `chatgpt_eval.py` | OpenAI (via Portkey) | Template |
| Gemini Vision | `gemini_eval.py` | Google (via Portkey) | Template |
| Google Vision API | `google_vision_eval.py` | Google Cloud | Template |

## Requirements

- Python 3.8+
- GPU support (CUDA/cuDNN) for optimal performance
- See `requirements.txt` for Python package dependencies

## Data

- **IAM Handwriting Database**: Requires registration at https://fki.ics.unimaas.nl/databases/iam-handwriting-database/
- Other datasets: [To be specified]

## HPC Configuration

Optimized for Princeton's Adroit cluster:
- Edit `hpc/job_config.txt` for job parameters
- Default: GPU allocation, 4 hours walltime
- Modify `hpc/submit_job.sh` to adjust SLURM parameters

## Output

Results are written to `results/`:
- `results/metrics/` - CSV files with quantitative metrics
- `results/logs/` - Detailed evaluation logs
- `results/visualizations/` - Generated charts and plots
- `results/reports/` - Summary reports

## Contributors

[To be populated]
