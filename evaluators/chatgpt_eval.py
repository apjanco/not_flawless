"""
evaluators/chatgpt_eval.py
ChatGPT (via OpenAI API) OCR evaluator
"""

import time
import os
import math
from pathlib import Path
from typing import Dict, Any, List
import base64

try:
    import requests
except ImportError:
    requests = None

from evaluators.utils import (
    get_data_dir, get_results_dir, load_iam_data,
    character_error_rate, word_error_rate,
    save_metrics, append_metrics_csv, save_results_jsonl,
    log_info, log_error, log_warning
)

MODEL_NAME = "chatgpt-vision"

def check_dependencies():
    """Check if required packages are installed"""
    return requests is not None

def get_portkey_key():
    """Get Portkey API key from environment"""
    api_key = os.getenv("PORTKEY_API_KEY")
    if not api_key:
        log_warning(MODEL_NAME, "PORTKEY_API_KEY environment variable not set")
    return api_key

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
        log_error(MODEL_NAME, "requests library not installed")
        return {"error": "requests not available"}
    
    # Check API key
    api_key = get_portkey_key()
    if not api_key:
        log_error(MODEL_NAME, "PORTKEY_API_KEY environment variable not set")
        return {"error": "API key not configured"}
    
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
    metrics = _run_evaluation(api_key, test_images, test_labels)
    
    # Save results
    save_metrics(MODEL_NAME, metrics)
    append_metrics_csv(MODEL_NAME, metrics)
    
    log_info(MODEL_NAME, "Evaluation complete")
    return metrics

def _encode_image(image_path: str) -> str:
    """Encode image to base64 for API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def _run_evaluation(api_key: str, image_paths: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """
    Run evaluation on test set
    
    Args:
        api_key: Portkey API key
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
    
    headers = {
        "x-portkey-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    for idx, (image_path, ground_truth) in enumerate(zip(image_paths, ground_truths)):
        result = {
            "image_path": image_path,
            "ground_truth": ground_truth,
            "predicted_text": None,
            "cer": None,
            "wer": None,
            "inference_time": None,
            "logprobs": None,
            "error": None
        }
        
        try:
            # Encode image
            image_data = _encode_image(image_path)
            
            # Prepare request with logprobs enabled
            payload = {
                "model": "gpt-4-vision",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please read and transcribe all text in this image. Return only the transcribed text."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1024,
                "logprobs": True,
                "top_logprobs": 5  # Get top 5 token predictions per position
            }
            
            # Call API via Portkey
            start_time = time.time()
            response = requests.post(
                "https://api.portkey.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            inference_time = time.time() - start_time
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            response_data = response.json()
            predicted_text = response_data["choices"][0]["message"]["content"].strip()
            
            # Extract logprobs if available
            # Structure: list of {token, logprob, top_logprobs: [{token, logprob}, ...]}
            logprobs_data = None
            if "logprobs" in response_data["choices"][0]:
                raw_logprobs = response_data["choices"][0]["logprobs"]
                if raw_logprobs and "content" in raw_logprobs:
                    logprobs_data = []
                    for token_info in raw_logprobs["content"]:
                        chosen_logprob = token_info.get("logprob", 0)
                        token_entry = {
                            "token": token_info.get("token", ""),
                            "logprob": chosen_logprob,
                            "prob": math.exp(chosen_logprob) if chosen_logprob is not None else None,
                            "top_tokens": []
                        }
                        # Extract top token predictions with probabilities
                        if "top_logprobs" in token_info:
                            for top in token_info["top_logprobs"]:
                                top_logprob = top.get("logprob", 0)
                                token_entry["top_tokens"].append({
                                    "token": top.get("token", ""),
                                    "logprob": top_logprob,
                                    "prob": math.exp(top_logprob) if top_logprob is not None else None
                                })
                        logprobs_data.append(token_entry)
            
            # Calculate metrics
            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)
            
            result["predicted_text"] = predicted_text
            result["cer"] = cer
            result["wer"] = wer
            result["inference_time"] = inference_time
            result["logprobs"] = logprobs_data
            
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
