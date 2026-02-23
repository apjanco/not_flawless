"""
evaluators/utils.py
Shared utility functions for model evaluators
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np

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

def load_iam_data(split: str = "test", limit: int = None) -> Tuple[List[str], List[str]]:
    """
    Load IAM database from Teklia/IAM-line Hugging Face dataset
    
    Args:
        split: Data split (train/val/test)
        limit: Maximum number of samples to load (None = all)
    
    Returns:
        Tuple of (image_paths, ground_truth_labels)
    """
    data_dir = get_data_dir() / "iam"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"IAM data directory not found: {data_dir}")
    
    image_paths = []
    labels = []
    
    # Load from splits directory structure
    # Expected structure: data/iam/splits/[train|val|test]/
    split_dir = data_dir / "splits" / split
    
    if not split_dir.exists():
        # Try alternative structure: data/iam/[split]/
        split_dir = data_dir / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"IAM split directory not found: {split_dir}")
    
    # Look for images and metadata
    # Teklia/IAM-line typically has: images + metadata JSON
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif'}
    
    # Collect all image files
    image_files = sorted([
        f for f in split_dir.rglob('*')
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    
    # Load corresponding labels
    # Try to load from a metadata file first
    metadata_file = split_dir.parent / f"{split}_metadata.json"
    metadata = {}
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            log_warning("load_iam_data", f"Could not load metadata: {str(e)}")
    
    # Process images
    for img_path in image_files:
        # Try to get label from metadata
        img_key = img_path.stem
        label = metadata.get(img_key, "") or metadata.get(str(img_path), "")
        
        # If no metadata label, try to load from .txt file with same name
        if not label:
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        label = f.read().strip()
                except Exception:
                    label = ""
        
        image_paths.append(str(img_path))
        labels.append(label)
        
        if limit and len(image_paths) >= limit:
            break
    
    if not image_paths:
        log_warning("load_iam_data", f"No images found in {split_dir}")
    
    return image_paths, labels

def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER)
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        CER as percentage (0-100)
    """
    if len(reference) == 0:
        return 100.0 if len(hypothesis) > 0 else 0.0
    
    # Simple implementation using edit distance
    errors = _levenshtein_distance(reference, hypothesis)
    return (errors / len(reference)) * 100

def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER)
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        WER as percentage (0-100)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 100.0 if len(hyp_words) > 0 else 0.0
    
    errors = _levenshtein_distance(ref_words, hyp_words)
    return (errors / len(ref_words)) * 100

def _levenshtein_distance(s1: List[str], s2: List[str]) -> int:
    """
    Calculate Levenshtein distance between two sequences
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

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
