"""
evaluators/utils.py
Shared utility functions for model evaluators
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import numpy as np
from torchmetrics.text import CharErrorRate, WordErrorRate
from datasets import load_dataset

def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent

def get_data_dir() -> Path:
    """Get data directory"""
    return get_project_root() / "data"

def get_results_dir() -> Path:
    """Get results directory"""
    results_dir = get_project_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def load_iam_data(split: str = "test", limit: int = None):
    """
    Load IAM database using Hugging Face Datasets from Teklia/IAM-line
    
    Args:
        split: Data split (train/validation/test) or None to load all splits combined
        limit: Maximum number of samples to load (None = all)
    
    Returns:
        Hugging Face Dataset object
    """
    try:
        log_info("load_iam_data", f"Loading IAM dataset split: {split}")
        
        if split is None:
            # Load all splits and combine them
            train = load_dataset("Teklia/IAM-line", split="train", trust_remote_code=True)
            validation = load_dataset("Teklia/IAM-line", split="validation", trust_remote_code=True)
            test = load_dataset("Teklia/IAM-line", split="test", trust_remote_code=True)
            
            from datasets import concatenate_datasets
            dataset = concatenate_datasets([train, validation, test])
            log_info("load_iam_data", f"Combined all splits: {len(dataset)} total samples")
        else:
            dataset = load_dataset("Teklia/IAM-line", split=split, trust_remote_code=True)
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
            log_info("load_iam_data", f"Limited to {len(dataset)} samples")
        
        log_info("load_iam_data", f"Loaded {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        log_error("load_iam_data", f"Failed to load dataset: {str(e)}")
        raise

# Initialize torchmetrics CER and WER calculators
_cer_metric = CharErrorRate()
_wer_metric = WordErrorRate()

def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) using torchmetrics
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        CER as percentage (0-100)
    """
    if len(reference) == 0:
        return 100.0 if len(hypothesis) > 0 else 0.0
    
    # torchmetrics returns CER as a fraction (0-1), convert to percentage
    cer = _cer_metric([hypothesis], [reference])
    return float(cer.item() * 100)

def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) using torchmetrics
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        WER as percentage (0-100)
    """
    ref_words = reference.split()
    
    if len(ref_words) == 0:
        return 100.0 if len(hypothesis.split()) > 0 else 0.0
    
    # torchmetrics returns WER as a fraction (0-1), convert to percentage
    wer = _wer_metric([hypothesis], [reference])
    return float(wer.item() * 100)

def save_metrics(model_name: str, metrics: Dict[str, Any]) -> Path:
    """
    Save evaluation metrics to CSV
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metrics
    
    Returns:
        Path to saved metrics file
    """
    results_dir = get_results_dir()
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = metrics_dir / f"{model_name}_metrics.json"
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return output_file

def append_metrics_csv(model_name: str, metrics: Dict[str, Any]) -> Path:
    """
    Append metrics to a summary CSV file
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metrics
    
    Returns:
        Path to CSV file
    """
    results_dir = get_results_dir()
    csv_file = results_dir / "metrics" / "summary.csv"
    
    # Prepare row
    row = {"model": model_name}
    row.update(metrics)
    
    # Write or append to CSV
    fieldnames = ["model"] + sorted([k for k in metrics.keys()])
    
    file_exists = csv_file.exists()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    return csv_file

def save_results_jsonl(model_name: str, results: List[Dict[str, Any]]) -> Path:
    """
    Save per-sample evaluation results to JSONL file in results directory
    
    Args:
        model_name: Name of the model
        results: List of result dictionaries (one per sample)
    
    Returns:
        Path to JSONL file
    """
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / f"{model_name}_results.jsonl"
    
    # Write results efficiently
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                # Ensure all values are JSON serializable
                clean_result = {k: (float(v) if isinstance(v, (float, np.floating)) else v) 
                               for k, v in result.items()}
                f.write(json.dumps(clean_result, ensure_ascii=False) + '\n')
        log_info(model_name, f"Saved {len(results)} results to {output_file.name}")
    except Exception as e:
        log_error(model_name, f"Failed to save JSONL: {str(e)}")
        raise
    
    return output_file

def log_info(model_name: str, message: str):
    """Log informational message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{model_name}] {message}")

def log_error(model_name: str, message: str):
    """Log error message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [ERROR] [{model_name}] {message}", file=sys.stderr)

def log_warning(model_name: str, message: str):
    """Log warning message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [WARNING] [{model_name}] {message}")
