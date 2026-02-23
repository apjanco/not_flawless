# API-Based Model Evaluation on Adroit Interactive Nodes

This guide explains how to set up and run API-based OCR models on Princeton's Adroit HPC cluster using interactive nodes.

## Supported API Models

1. **ChatGPT Vision** (via Portkey)
   - Model: `gpt-4-vision`
   - Provider: OpenAI
   - Setup: Portkey API key

2. **Google Gemini** (via Portkey)
   - Model: `gemini-2.0-flash-vision`
   - Provider: Google
   - Setup: Portkey API key

3. **Google Vision API**
   - Direct API access
   - Provider: Google Cloud
   - Setup: Service account credentials JSON

## Prerequisites

### Local Setup (before connecting to Adroit)

1. **Portkey Setup** (for ChatGPT and Gemini)
   ```bash
   # 1. Create account at https://www.portkey.ai/
   # 2. Get your API key from dashboard
   # 3. Keep it safe - you'll need it on Adroit
   ```

2. **Google Cloud Setup** (for Vision API)
   ```bash
   # 1. Go to https://cloud.google.com/
   # 2. Create a new project
   # 3. Enable Vision API
   # 4. Create service account with Vision API permissions
   # 5. Download credentials JSON file
   # 6. Transfer credentials to Adroit (or use cloud storage)
   ```

## Running on Adroit Interactive Node

### Step 1: Request Interactive Node

```bash
# SSH to Adroit
ssh <your-username>@adroit.princeton.edu

# Request an interactive GPU node (optional)
salloc --time=02:00:00 --cpus-per-task=4 --mem=16G --gres=gpu:1
```

### Step 2: Prepare Environment

```bash
# Load Python module
module load anaconda3/2024.2

# Navigate to project
cd /path/to/not_flawless

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Set Environment Variables

```bash
# For ChatGPT and Gemini (via Portkey)
export PORTKEY_API_KEY="your-portkey-api-key"

# For Google Vision API
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

**Note:** If credentials.json is on your local machine, copy it first:
```bash
scp credentials.json <username>@adroit.princeton.edu:~/
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/credentials.json"
```

### Step 4: Run Evaluation Script

```bash
# Run all API evaluators
bash hpc/run_api_evaluation.sh

# Or run individual models
python evaluators/chatgpt_eval.py
python evaluators/gemini_eval.py
python evaluators/google_vision_eval.py
```

### Step 5: Check Results

```bash
# View results
ls results/
ls results/metrics/

# Check individual model results
cat results/<model>_results.jsonl | head -5

# View summary
cat results/metrics/summary.csv
```

## Cost Considerations

| Model | Cost | Notes |
|-------|------|-------|
| ChatGPT-4 Vision | ~$0.01-0.03 per image | Via Portkey |
| Google Gemini | ~$0.001-0.002 per image | Via Portkey |
| Google Vision API | ~$1.50 per 1000 images | Document feature |

## Troubleshooting

### API Key Not Working
```bash
# Check if environment variable is set
echo $PORTKEY_API_KEY
echo $GOOGLE_APPLICATION_CREDENTIALS

# Try setting again
export PORTKEY_API_KEY="your-key"
```

### Credentials File Not Found
```bash
# Check file exists
ls -la $GOOGLE_APPLICATION_CREDENTIALS

# If not on Adroit, copy it
scp ~/credentials.json adroit.princeton.edu:~/
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/credentials.json"
```

### Rate Limiting
If you hit API rate limits:
- Use smaller dataset sample first
- Add delays between requests (modify evaluator)
- Consider upgrading API plan
- Stagger evaluations over time

### Timeout Issues
For slow internet connections:
- Increase timeout in evaluator scripts (default: 60s)
- Run on Adroit GPU nodes for faster processing
- Test with a small batch first

## Output Files

Each evaluator produces:

```
results/
├── <model>_results.jsonl          # Per-sample results
├── metrics/
│   ├── <model>_metrics.json       # Aggregated metrics
│   └── summary.csv                # All models comparison
└── logs/
    └── evaluation_log.json        # Evaluation timestamps
```

## JSONL Format Example

```json
{
  "image_path": "/path/to/image.png",
  "ground_truth": "sample text",
  "predicted_text": "sample text",
  "cer": 0.0,
  "wer": 0.0,
  "inference_time": 2.34,
  "error": null
}
```

## Tips for Efficient Evaluation

1. **Test with small sample first**
   ```bash
   # Modify load_iam_data() in utils.py to limit samples
   ```

2. **Monitor API usage**
   - Check dashboard: https://www.portkey.ai/ (Portkey)
   - Check dashboard: https://console.cloud.google.com/ (Google)

3. **Download results locally**
   ```bash
   scp -r adroit.princeton.edu:~/not_flawless/results ~/local-backup/
   ```

4. **Keep interactive session alive**
   - For long evaluations, use `tmux` or `screen`
   - Or increase `--time` allocation when requesting node

## Security Notes

- **Never commit API keys** to git
- **Delete credentials.json** after evaluation
- **Use IAM roles** for Google Cloud (least privilege)
- **Monitor API costs** regularly
- **Rotate keys** periodically

## Additional Resources

- [Portkey Documentation](https://portkey.ai/docs/)
- [Google Vision API Guide](https://cloud.google.com/vision/docs)
- [Adroit HPC Documentation](https://researchcomputing.princeton.edu/support/knowledge-base/adroit)
