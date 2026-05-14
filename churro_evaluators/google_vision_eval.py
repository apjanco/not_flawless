"""
evaluators/google_vision_eval.py
Google Cloud Vision API OCR evaluator
"""

import time
import os
from pathlib import Path
from typing import Dict, Any, List

try:
    from google.cloud import vision
except ImportError:
    vision = None

from evaluators.utils import (
    get_data_dir, get_results_dir, load_iam_data,
    character_error_rate, word_error_rate,
    save_metrics, append_metrics_csv, save_results_jsonl,
    log_info, log_error, log_warning
)

MODEL_NAME = "google-vision"

def check_dependencies():
    """Check if required packages are installed"""
    return vision is not None

def get_gcp_credentials():
    """Get Google Cloud credentials from environment"""
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        log_warning(MODEL_NAME, "GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    return creds

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
        log_error(MODEL_NAME, "google-cloud-vision not installed")
        return {"error": "google-cloud-vision not available"}
    
    # Check credentials
    creds_path = get_gcp_credentials()
    if not creds_path:
        log_error(MODEL_NAME, "GOOGLE_APPLICATION_CREDENTIALS not configured")
        return {"error": "Credentials not configured"}
    
    if not os.path.exists(creds_path):
        log_error(MODEL_NAME, f"Credentials file not found: {creds_path}")
        return {"error": "Credentials file not found"}
    
    # Initialize client
    try:
        client = vision.ImageAnnotatorClient()
        log_info(MODEL_NAME, "Google Vision client initialized")
    except Exception as e:
        log_error(MODEL_NAME, f"Failed to initialize client: {str(e)}")
        return {"error": str(e)}
    
    # Load test data
    try:
        test_images, test_labels = load_iam_data(split="test")
        log_info(MODEL_NAME, f"Loaded {len(test_images)} test images")
    except Exception as e:
        log_error(MODEL_NAME, f"Failed to load test data: {str(e)}")
        return {"error": str(e)}
    
    if not test_images:
        log_warning(MODEL_NAME, "No test images found")
        return {"error": "No test data"}
    
    # Run evaluation
    metrics = _run_evaluation(client, test_images, test_labels)
    
    # Save results
    save_metrics(MODEL_NAME, metrics)
    append_metrics_csv(MODEL_NAME, metrics)
    
    log_info(MODEL_NAME, "Evaluation complete")
    return metrics

def _run_evaluation(client, image_paths: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """
    Run evaluation on test set
    
    Args:
        client: Google Vision API client
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
    
    for idx, (image_path, ground_truth) in enumerate(zip(image_paths, ground_truths)):
        result = {
            "image_path": image_path,
            "ground_truth": ground_truth,
            "predicted_text": None,
            "cer": None,
            "wer": None,
            "inference_time": None,
            "error": None
        }
        
        try:
            # Read image file
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Perform OCR
            start_time = time.time()
            response = client.text_detection(image=image)
            inference_time = time.time() - start_time
            
            if response.error.message:
                raise Exception(f"Vision API error: {response.error.message}")
            
            # Extract text
            texts = response.text_annotations
            if texts:
                # First annotation is the full text
                predicted_text = texts[0].description.strip()
            else:
                predicted_text = ""
            
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
            
            if (idx + 1) % 10 == 0:
                log_info(MODEL_NAME, f"Processed {idx + 1}/{len(image_paths)} images")
        
        except Exception as e:
            log_warning(MODEL_NAME, f"Error processing image {image_path}: {str(e)}")
            result["error"] = str(e)
            num_errors += 1
        
        results.append(result)
    
    # Save JSONL results
    save_results_jsonl(MODEL_NAME, results)
    
    # Calculate aggregated metrics
    metrics = {
        "num_samples": len(image_paths),
        "num_successful": len(image_paths) - num_errors,
        "num_errors": num_errors,
    }
    
    if cer_values:
        metrics["mean_cer"] = sum(cer_values) / len(cer_values)
        metrics["median_cer"] = sorted(cer_values)[len(cer_values) // 2]
    else:
        metrics["mean_cer"] = None
        metrics["median_cer"] = None
    
    if wer_values:
        metrics["mean_wer"] = sum(wer_values) / len(wer_values)
        metrics["median_wer"] = sorted(wer_values)[len(wer_values) // 2]
    else:
        metrics["mean_wer"] = None
        metrics["median_wer"] = None
    
    if inference_times:
        metrics["mean_inference_time"] = sum(inference_times) / len(inference_times)
        metrics["total_inference_time"] = sum(inference_times)
    else:
        metrics["mean_inference_time"] = None
        metrics["total_inference_time"] = None
    
    return metrics

if __name__ == "__main__":
    # Allow running evaluator directly
    project_root = Path(__file__).parent.parent
    result = evaluate(str(project_root))
