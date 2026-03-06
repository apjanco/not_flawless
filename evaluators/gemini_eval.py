"""
evaluators/gemini_eval.py
Google Gemini (via Vertex AI) OCR evaluator with logprobs support
"""

import time
import os
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

try:
    from google import genai
    from google.genai.types import GenerateContentConfig
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .utils import (
    get_data_dir, get_results_dir, load_iam_data,
    character_error_rate, word_error_rate,
    save_metrics, append_metrics_csv, save_results_jsonl,
    log_info, log_error, log_warning
)

MODEL_NAME = "gemini-vision"
MODEL_ID = "gemini-2.5-flash"  # or "gemini-2.0-flash"

def check_dependencies():
    """Check if required packages are installed"""
    return GENAI_AVAILABLE

def get_gcp_config():
    """Get GCP project configuration from environment"""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "global")
    
    if not project_id:
        log_warning(MODEL_NAME, "GOOGLE_CLOUD_PROJECT environment variable not set")
    
    return project_id, location

def evaluate(project_root: str = None, top_logprobs: int = 5) -> Dict[str, Any]:
    """
    Main evaluation function
    
    Args:
        project_root: Path to project root
        top_logprobs: Number of top alternative tokens to return (1-20)
    
    Returns:
        Dictionary of evaluation metrics
    """
    log_info(MODEL_NAME, "Starting evaluation")
    
    # Check dependencies
    if not check_dependencies():
        log_error(MODEL_NAME, "google-genai library not installed. Run: pip install google-genai")
        return {"error": "google-genai not available"}
    
    # Check GCP config
    project_id, location = get_gcp_config()
    if not project_id:
        log_error(MODEL_NAME, "GOOGLE_CLOUD_PROJECT environment variable not set")
        return {"error": "GCP project not configured"}
    
    # Initialize client
    try:
        client = genai.Client(vertexai=True, project=project_id, location=location)
        log_info(MODEL_NAME, f"Initialized Vertex AI client for project: {project_id}")
    except Exception as e:
        log_error(MODEL_NAME, f"Failed to initialize Vertex AI client: {str(e)}")
        return {"error": str(e)}
    
    # Load test data
    try:
        dataset = load_iam_data(split="test")
        log_info(MODEL_NAME, f"Loaded {len(dataset)} test samples")
    except Exception as e:
        log_error(MODEL_NAME, f"Failed to load test data: {str(e)}")
        return {"error": str(e)}
    
    if not dataset or len(dataset) == 0:
        log_warning(MODEL_NAME, "No test data found")
        return {"error": "No test data"}
    
    # Run evaluation
    metrics = _run_evaluation(client, dataset, top_logprobs)
    
    # Save results
    save_metrics(MODEL_NAME, metrics)
    append_metrics_csv(MODEL_NAME, metrics)
    
    log_info(MODEL_NAME, "Evaluation complete")
    return metrics

def _run_evaluation(client, dataset, top_logprobs: int = 5) -> Dict[str, Any]:
    """
    Run evaluation on test set with logprobs
    
    Args:
        client: Google GenAI client
        dataset: HuggingFace dataset with 'image' and 'text' fields
        top_logprobs: Number of top alternative tokens to return
    
    Returns:
        Dictionary of metrics
    """
    results = []
    cer_values = []
    wer_values = []
    inference_times = []
    num_errors = 0
    
    # Configure generation with logprobs
    generation_config = GenerateContentConfig(
        max_output_tokens=1024,
        response_logprobs=True,
        logprobs=top_logprobs,  # Get top N alternative tokens
    )
    
    prompt = "Please read and transcribe all text in this image. Return only the transcribed text, nothing else."
    
    for idx, example in enumerate(dataset):
        result = {
            "index": idx,
            "ground_truth": example.get('text', ''),
            "predicted_text": None,
            "cer": None,
            "wer": None,
            "inference_time": None,
            "logprobs": None,
            "mean_logprob": None,
            "confidence": None,
            "error": None
        }
        
        ground_truth = example.get('text', '')
        
        try:
            # Get PIL image from dataset
            pil_image = example['image']
            
            # Prepare content with image and prompt
            contents = [pil_image, prompt]
            
            # Call Gemini API with logprobs
            start_time = time.time()
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=generation_config,
            )
            inference_time = time.time() - start_time
            
            # Extract text response
            predicted_text = response.text.strip() if response.text else ""
            
            # Extract logprobs
            logprobs_data = _extract_logprobs(response)
            
            # Calculate mean logprob and confidence
            mean_logprob = None
            confidence = None
            if logprobs_data:
                logprob_values = [t["logprob"] for t in logprobs_data if t.get("logprob") is not None]
                if logprob_values:
                    mean_logprob = sum(logprob_values) / len(logprob_values)
                    # Convert mean logprob to confidence percentage
                    confidence = math.exp(mean_logprob) * 100
            
            # Calculate metrics
            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)
            
            result["predicted_text"] = predicted_text
            result["cer"] = cer
            result["wer"] = wer
            result["inference_time"] = inference_time
            result["logprobs"] = logprobs_data
            result["mean_logprob"] = mean_logprob
            result["confidence"] = confidence
            
            cer_values.append(cer)
            wer_values.append(wer)
            inference_times.append(inference_time)
            
            if (idx + 1) % 10 == 0:
                log_info(MODEL_NAME, f"Processed {idx + 1}/{len(dataset)} samples")
        
        except Exception as e:
            log_warning(MODEL_NAME, f"Error processing sample {idx}: {str(e)}")
            result["error"] = str(e)
            num_errors += 1
        
        results.append(result)
    
    # Save JSONL results
    save_results_jsonl(MODEL_NAME, results)
    
    # Calculate aggregated metrics
    metrics = {
        "num_samples": len(dataset),
        "num_successful": len(dataset) - num_errors,
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
    
    if wer_values:
        metrics["mean_wer"] = sum(wer_values) / len(wer_values)
        metrics["median_wer"] = sorted(wer_values)[len(wer_values) // 2]
        metrics["min_wer"] = min(wer_values)
        metrics["max_wer"] = max(wer_values)
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

def _extract_logprobs(response) -> Optional[List[Dict[str, Any]]]:
    """
    Extract logprobs from Gemini response
    
    Args:
        response: Gemini API response
    
    Returns:
        List of token logprob data with top alternatives
    """
    try:
        if not response.candidates or not response.candidates[0].logprobs_result:
            return None
        
        logprobs_result = response.candidates[0].logprobs_result
        logprobs_data = []
        
        for i, chosen_candidate in enumerate(logprobs_result.chosen_candidates):
            token_entry = {
                "token": chosen_candidate.token,
                "logprob": chosen_candidate.log_probability,
                "prob": math.exp(chosen_candidate.log_probability) if chosen_candidate.log_probability else None,
                "top_tokens": []
            }
            
            # Extract top alternative tokens
            if i < len(logprobs_result.top_candidates):
                top_alternatives = logprobs_result.top_candidates[i].candidates
                for alt_token_info in top_alternatives:
                    token_entry["top_tokens"].append({
                        "token": alt_token_info.token,
                        "logprob": alt_token_info.log_probability,
                        "prob": math.exp(alt_token_info.log_probability) if alt_token_info.log_probability else None
                    })
            
            logprobs_data.append(token_entry)
        
        return logprobs_data
    
    except Exception as e:
        log_warning(MODEL_NAME, f"Failed to extract logprobs: {str(e)}")
        return None

if __name__ == "__main__":
    # Allow running evaluator directly
    project_root = Path(__file__).parent.parent
    result = evaluate(str(project_root))
