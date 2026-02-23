#!/usr/bin/env python3
"""
hpc/run_evaluation.py
Orchestrator script to run all model evaluations sequentially on HPC
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add evaluators to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import evaluators
from evaluators import (
    tesseract_eval,
    paddleocr_eval,
    easyocr_eval,
    pylaia_eval,
    kraken_eval,
    qwen_eval,
    deepseek_eval,
    chandra_eval
)

def setup_results_dir():
    """Create results directory structure"""
    results_dir = PROJECT_ROOT / "results"
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (results_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    return results_dir

def log_evaluation(evaluator_name, status, duration, metrics=None):
    """Log evaluation results"""
    results_dir = PROJECT_ROOT / "results"
    log_file = results_dir / "logs" / "evaluation_log.json"
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "evaluator": evaluator_name,
        "status": status,
        "duration_seconds": duration,
        "metrics": metrics
    }
    
    # Load existing log or create new
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []
    
    log_data.append(entry)
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

def run_evaluation(evaluator_module, evaluator_name):
    """Run a single model evaluator"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {evaluator_name}")
    print(f"{'='*60}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        if hasattr(evaluator_module, 'evaluate'):
            results = evaluator_module.evaluate(str(PROJECT_ROOT))
            duration = time.time() - start_time
            
            print(f"\n✓ {evaluator_name} completed successfully")
            print(f"Duration: {duration:.2f} seconds")
            
            log_evaluation(evaluator_name, "success", duration, results)
            return True
        else:
            print(f"✗ {evaluator_name} missing evaluate() function")
            log_evaluation(evaluator_name, "error", time.time() - start_time)
            return False
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n✗ {evaluator_name} failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        print(f"Duration: {duration:.2f} seconds")
        
        log_evaluation(evaluator_name, "error", duration, {"error": str(e)})
        return False

def main():
    """Main orchestration function"""
    print("\n" + "="*60)
    print("OCR/HTR Models Evaluation - Orchestrator")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup results directory
    results_dir = setup_results_dir()
    print(f"Results directory: {results_dir}")
    
    # List of evaluators to run
    evaluators = [
        # (module, name)
        # Uncomment as evaluator scripts are implemented
        (tesseract_eval, "Tesseract"),
        (pylaia_eval, "PyLaia"),
        (kraken_eval, "Kraken"),
        (qwen_eval, "Qwen2-VL-8B"),
        (deepseek_eval, "DeepSeek-OCR"),
        (chandra_eval, "Chandra"),
    ]
    
    print(f"\nScheduled evaluators: {len(evaluators)}")
    for _, name in evaluators:
        print(f"  - {name}")
    
    if not evaluators:
        print("\n⚠ No evaluators configured to run")
        print("Update the evaluators list in run_evaluation.py to add models")
        return
    
    # Run evaluations
    total_start = time.time()
    results_summary = {
        "success": 0,
        "failed": 0,
        "total": len(evaluators)
    }
    
    for evaluator_module, evaluator_name in evaluators:
        if run_evaluation(evaluator_module, evaluator_name):
            results_summary["success"] += 1
        else:
            results_summary["failed"] += 1
    
    # Summary
    total_duration = time.time() - total_start
    
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Total evaluators: {results_summary['total']}")
    print(f"Successful: {results_summary['success']}")
    print(f"Failed: {results_summary['failed']}")
    print(f"Total duration: {total_duration/3600:.2f} hours ({total_duration/60:.2f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
