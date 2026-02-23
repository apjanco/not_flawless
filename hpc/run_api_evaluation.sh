#!/bin/bash
# hpc/run_api_evaluation.sh
# Interactive testing script for API-based models on Adroit

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "API-Based Model Evaluation (Interactive)"
echo "=========================================="
echo ""

# Create results directories
mkdir -p "$PROJECT_ROOT/results/logs"
mkdir -p "$PROJECT_ROOT/results/metrics"
mkdir -p "$PROJECT_ROOT/results/visualizations"

# Check for environment variables
echo "Checking for required API credentials..."
echo ""

if [ -z "$PORTKEY_API_KEY" ]; then
    echo "⚠ Warning: PORTKEY_API_KEY not set"
    echo "  To use ChatGPT and Gemini evaluators, set:"
    echo "    export PORTKEY_API_KEY=<your-portkey-key>"
    echo ""
else
    echo "✓ PORTKEY_API_KEY is set"
fi

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "⚠ Warning: GOOGLE_APPLICATION_CREDENTIALS not set"
    echo "  To use Google Vision evaluator, set:"
    echo "    export GOOGLE_APPLICATION_CREDENTIALS=<path-to-credentials-json>"
    echo ""
else
    echo "✓ GOOGLE_APPLICATION_CREDENTIALS is set"
fi

echo ""
echo "=========================================="
echo "Available API Evaluators"
echo "=========================================="
echo ""
echo "1. ChatGPT Vision (via Portkey)"
echo "   Script: chatgpt_eval.py"
echo "   Requires: PORTKEY_API_KEY"
echo ""
echo "2. Google Gemini (via Portkey)"
echo "   Script: gemini_eval.py"
echo "   Requires: PORTKEY_API_KEY"
echo ""
echo "3. Google Vision API"
echo "   Script: google_vision_eval.py"
echo "   Requires: GOOGLE_APPLICATION_CREDENTIALS"
echo ""

echo "=========================================="
echo "Setup Instructions"
echo "=========================================="
echo ""
echo "1. On your local machine, get API keys:"
echo "   - Portkey: https://www.portkey.ai/ (for ChatGPT and Gemini)"
echo "   - Google Cloud: https://cloud.google.com/docs/authentication"
echo ""
echo "2. On Adroit interactive node, set environment variables:"
echo "   export PORTKEY_API_KEY=<your-key>"
echo "   export GOOGLE_APPLICATION_CREDENTIALS=<path-to-json>"
echo ""
echo "3. Run individual evaluators:"
echo "   python evaluators/chatgpt_eval.py"
echo "   python evaluators/gemini_eval.py"
echo "   python evaluators/google_vision_eval.py"
echo ""
echo "4. Results will be saved to: results/"
echo "   - results/<model>_results.jsonl (per-sample results)"
echo "   - results/metrics/<model>_metrics.json (aggregated metrics)"
echo "   - results/metrics/summary.csv (all models)"
echo ""

echo "=========================================="
echo "Running Python Orchestrator"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"

# Run Python evaluation script
python3 << 'PYTHON_EOF'
import os
import sys
import subprocess
from pathlib import Path

project_root = Path.cwd()
evaluators = [
    "evaluators.chatgpt_eval",
    "evaluators.gemini_eval",
    "evaluators.google_vision_eval"
]

print("\nStarting API-based model evaluations...")
print("=" * 60)

for evaluator in evaluators:
    print(f"\nEvaluating: {evaluator}")
    print("-" * 60)
    
    try:
        module = __import__(evaluator, fromlist=['evaluate'])
        if hasattr(module, 'evaluate'):
            result = module.evaluate(str(project_root))
            if result.get('error'):
                print(f"⚠ {evaluator}: {result['error']}")
            else:
                print(f"✓ {evaluator}: Completed successfully")
        else:
            print(f"✗ {evaluator}: Missing evaluate() function")
    except Exception as e:
        print(f"✗ {evaluator}: {type(e).__name__}: {str(e)}")

print("\n" + "=" * 60)
print("Evaluation complete. Results saved to: results/")
PYTHON_EOF

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
