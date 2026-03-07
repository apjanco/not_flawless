"""
evaluators/tesseract_eval.py
Tesseract OCR model evaluator
"""

import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datasets import load_from_disk

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

from evaluators.utils import (
    get_data_dir, get_results_dir, load_iam_data,
    character_error_rate, word_error_rate,
    save_metrics, append_metrics_csv, save_results_jsonl,
    log_info, log_error, log_warning
)

MODEL_NAME = "tesseract"

def check_dependencies():
    """Check if Tesseract is installed"""
    if pytesseract is None or Image is None:
        return False
    
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def evaluate(project_root: str = None) -> Dict[str, Any]:
    """
    Main evaluation function
    
    Args:
        project_root: Path to project root
    
    Returns:
        Dictionary of evaluation metrics
    """
    log_info(MODEL_NAME, "Starting evaluation")
    
    # Check dependencies
    if not check_dependencies():
        log_error(MODEL_NAME, "Tesseract not installed or not found in PATH")
        return {"error": "Tesseract not available"}
    
    log_info(MODEL_NAME, f"Tesseract version: {pytesseract.get_tesseract_version()}")
    
    # Load test data
    try:
        ds = load_from_disk('/scratch/network/aj7878/not_flawless/data/iam')
        test_images = [i['image'] for i in ds]
        test_labels = [i['text'] for i in ds]
        log_info(MODEL_NAME, f"Loaded {len(test_images)} images")
    except Exception as e:
        log_error(MODEL_NAME, f"Failed to load test data: {str(e)}")
        return {"error": str(e)}
    
    if not test_images:
        log_warning(MODEL_NAME, "No test images found")
        return {"error": "No test data"}
    
    # Run evaluation
    metrics = _run_evaluation(test_images, test_labels)
    
    # Save results
    save_metrics(MODEL_NAME, metrics)
    append_metrics_csv(MODEL_NAME, metrics)
    
    log_info(MODEL_NAME, "Evaluation complete")
    return metrics

def _run_evaluation(images: List[Image], ground_truths: List[str]) -> Dict[str, Any]:
    """
    Run evaluation on test set
    
    Args:
        image_paths: List of image file paths
        ground_truths: List of ground truth text
    
    Returns:
        Dictionary of metrics
    """
    results = []
    cer_values = []
    wer_values = []
    inference_times = []
    num_errors = 0
    
    log_info(MODEL_NAME, f"Processing {len(images)} images")
    
    for idx, (img, ground_truth) in enumerate(zip(images, ground_truths)):
        result = {
            "image_path": idx,
            "ground_truth": ground_truth,
            "predicted_text": None,
            "cer": None,
            "wer": None,
            "inference_time": None,
            "error": None
        }
        
        try:
            # Load image
            image = img
            
            # Perform OCR
            start_time = time.time()
            predicted_text = pytesseract.image_to_string(image)
            inference_time = time.time() - start_time
            
            # Calculate metrics
            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)
            
            result["predicted_text"] = predicted_text
            result["cer"] = cer
            result["wer"] = wer
            result["inference_time"] = inference_time
            
            cer_values.append(cer)
            wer_values.append(wer)
            inference_times.append(inference_time)
            
            if (idx + 1) % 50 == 0:
                avg_time = sum(inference_times) / len(inference_times)
                log_info(MODEL_NAME, f"Processed {idx + 1}/{len(images)} | Avg time: {avg_time:.3f}s")
        
        except Exception as e:
            log_warning(MODEL_NAME, f"Error processing image {images}: {str(e)}")
            result["error"] = str(e)
            num_errors += 1
        
        results.append(result)
    
    # Save JSONL results
    save_results_jsonl(MODEL_NAME, results)
    
    # Calculate aggregated metrics
    metrics = {
        "num_samples": len(images),
        "num_successful": len(images) - num_errors,
        "num_errors": num_errors,
    }
    
    if cer_values:
        metrics["mean_cer"] = sum(cer_values) / len(cer_values)
        metrics["median_cer"] = sorted(cer_values)[len(cer_values) // 2]
        metrics["min_cer"] = min(cer_values)
        metrics["max_cer"] = max(cer_values)
    else:
        metrics["mean_cer"] = None
        metrics["median_cer"] = None
        metrics["min_cer"] = None
        metrics["max_cer"] = None
    
    if wer_values:
        metrics["mean_wer"] = sum(wer_values) / len(wer_values)
        metrics["median_wer"] = sorted(wer_values)[len(wer_values) // 2]
        metrics["min_wer"] = min(wer_values)
        metrics["max_wer"] = max(wer_values)
    else:
        metrics["mean_wer"] = None
        metrics["median_wer"] = None
        metrics["min_wer"] = None
        metrics["max_wer"] = None
    
    if inference_times:
        metrics["mean_inference_time"] = sum(inference_times) / len(inference_times)
        metrics["total_inference_time"] = sum(inference_times)
        metrics["min_inference_time"] = min(inference_times)
        metrics["max_inference_time"] = max(inference_times)
    else:
        metrics["mean_inference_time"] = None
        metrics["total_inference_time"] = None
        metrics["min_inference_time"] = None
        metrics["max_inference_time"] = None
    
    return metrics

if __name__ == "__main__":
    # Allow running evaluator directly
    project_root = Path(__file__).parent.parent
    result = evaluate(str(project_root))
