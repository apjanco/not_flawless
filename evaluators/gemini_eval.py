"""
evaluators/gemini_eval.py
Google Gemini (direct Google API) OCR evaluator with logprobs and tokenizer support
Uses async requests for faster evaluation.
"""

import asyncio
import time
import os
import math
import json
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import io

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from datasets import load_from_disk
except ImportError:
    load_from_disk = None

try:
    from tqdm import tqdm
    from tqdm.asyncio import tqdm as atqdm
except ImportError:
    tqdm = None
    atqdm = None

from .utils import (
    get_data_dir, get_results_dir, load_iam_data,
    character_error_rate, word_error_rate,
    save_metrics, append_metrics_csv, save_results_jsonl,
    log_info, log_error, log_warning
)

MODEL_NAME = "gemini-3-pro-preview"
MODEL_ID = "gemini-3-pro-preview"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 25
MAX_REQUESTS_PER_DAY = 250
MAX_CONCURRENT_REQUESTS = 5  # Conservative to stay under RPM limit
REQUEST_DELAY = 60.0 / MAX_REQUESTS_PER_MINUTE  # ~2.4 seconds between requests

def check_dependencies():
    """Check if required packages are installed"""
    if not AIOHTTP_AVAILABLE:
        log_warning(MODEL_NAME, "aiohttp not installed, falling back to sync requests")
    return (REQUESTS_AVAILABLE or AIOHTTP_AVAILABLE) and GENAI_AVAILABLE


def _get_checkpoint_path() -> Path:
    """Get path to checkpoint file."""
    results_dir = get_results_dir()
    return results_dir / f"{MODEL_NAME}_checkpoint.json"


def _load_checkpoint() -> Dict[str, Any]:
    """
    Load checkpoint from disk.
    
    Returns:
        Dictionary with 'completed_indices' (set of processed indices),
        'results' (list of results), and 'last_run_date' (date string).
    """
    checkpoint_path = _get_checkpoint_path()
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            # Convert list back to set for efficient lookup
            data['completed_indices'] = set(data.get('completed_indices', []))
            log_info(MODEL_NAME, f"Loaded checkpoint: {len(data['completed_indices'])} samples already processed")
            return data
        except Exception as e:
            log_warning(MODEL_NAME, f"Failed to load checkpoint: {e}")
    return {'completed_indices': set(), 'results': [], 'last_run_date': None}


def _save_checkpoint(completed_indices: set, results: List[Dict], run_date: str):
    """
    Save checkpoint to disk.
    
    Args:
        completed_indices: Set of indices that have been processed
        results: List of result dictionaries
        run_date: Date string of current run
    """
    checkpoint_path = _get_checkpoint_path()
    try:
        data = {
            'completed_indices': list(completed_indices),
            'results': results,
            'last_run_date': run_date,
            'total_processed': len(completed_indices)
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f)
        log_info(MODEL_NAME, f"Checkpoint saved: {len(completed_indices)} samples processed")
    except Exception as e:
        log_warning(MODEL_NAME, f"Failed to save checkpoint: {e}")


def _get_daily_request_count() -> int:
    """
    Get number of requests made today from checkpoint.
    
    Returns:
        Number of requests made today, or 0 if new day.
    """
    checkpoint = _load_checkpoint()
    today = time.strftime('%Y-%m-%d')
    if checkpoint.get('last_run_date') == today:
        # Same day - count requests made today
        return len([r for r in checkpoint.get('results', []) 
                    if r.get('run_date') == today])
    return 0

def get_google_api_key():
    """Get Google API key from environment"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log_warning(MODEL_NAME, "GOOGLE_API_KEY environment variable not set")
    return api_key

def get_gemini_tokenizer():
    """
    Get Gemini tokenizer via google-genai library (v1.66.0+)
    
    Uses the new official Google Gen AI SDK instead of deprecated google-generativeai.
    
    Returns:
        Client configured for the Gemini API with tokenizer capability, or None if unavailable
    """
    try:
        api_key = get_google_api_key()
        if not api_key:
            log_warning(MODEL_NAME, "Cannot initialize tokenizer: GOOGLE_API_KEY not set")
            return None
        
        # Create client using new google-genai SDK
        client = genai.Client(api_key=api_key)
        log_info(MODEL_NAME, "Initialized Gemini tokenizer with google-genai v1.66.0+")
        return client
    except Exception as e:
        log_warning(MODEL_NAME, f"Failed to initialize Gemini tokenizer: {str(e)}")
        return None

def evaluate(project_root: str = None, top_logprobs: int = 5, max_requests: int = None) -> Dict[str, Any]:
    """
    Main evaluation function with checkpointing and rate limiting.
    
    Supports incremental evaluation - run once per day to process up to 250 samples.
    Progress is saved to checkpoint file and resumes from where it left off.
    
    Args:
        project_root: Path to project root
        top_logprobs: Number of top alternative tokens to return (1-20)
        max_requests: Maximum requests for this run (default: MAX_REQUESTS_PER_DAY)
    
    Returns:
        Dictionary of evaluation metrics
    """
    log_info(MODEL_NAME, "Starting Gemini evaluation via Google API (direct)")
    
    # Check dependencies
    if not check_dependencies():
        log_error(MODEL_NAME, "requests and/or google-genai not installed")
        return {"error": "Dependencies not available"}
    
    # Check GOOGLE_API_KEY (required for both inference and tokenizer)
    google_api_key = get_google_api_key()
    if not google_api_key:
        log_error(MODEL_NAME, "GOOGLE_API_KEY environment variable is REQUIRED but not set")
        log_error(MODEL_NAME, "Please set: export GOOGLE_API_KEY='your-google-api-key'")
        return {"error": "GOOGLE_API_KEY not configured - evaluation cannot proceed"}
    
    # Initialize tokenizer with validated API key
    tokenizer_model = get_gemini_tokenizer()
    if not tokenizer_model:
        log_error(MODEL_NAME, "Failed to initialize Gemini tokenizer")
        log_error(MODEL_NAME, "Ensure GOOGLE_API_KEY is set correctly and the Gemini API is accessible")
        return {"error": "Tokenizer initialization failed - evaluation cannot proceed"}
    
    # Load full IAM dataset from local disk
    try:
        ds = load_from_disk('/scratch/network/aj7878/not_flawless/data/iam')
        test_images = [i['image'] for i in ds]
        test_labels = [i['text'] for i in ds]
        log_info(MODEL_NAME, f"Loaded {len(test_images)} images from full IAM dataset")
    except Exception as e:
        log_error(MODEL_NAME, f"Failed to load test data: {str(e)}")
        return {"error": str(e)}
    
    if not test_images:
        log_warning(MODEL_NAME, "No test images found")
        return {"error": "No test data"}
    
    # Load checkpoint to resume from previous run
    checkpoint = _load_checkpoint()
    completed_indices = checkpoint['completed_indices']
    existing_results = checkpoint['results']
    
    # Determine how many requests we can make
    if max_requests is None:
        max_requests = MAX_REQUESTS_PER_DAY
    
    # Calculate remaining quota
    remaining_indices = [i for i in range(len(test_images)) if i not in completed_indices]
    requests_to_make = min(len(remaining_indices), max_requests)
    
    if not remaining_indices:
        log_info(MODEL_NAME, "All samples already processed! Generating final metrics.")
        metrics = _aggregate_metrics(existing_results, len(test_images))
        save_metrics(MODEL_NAME, metrics)
        append_metrics_csv(MODEL_NAME, metrics)
        return metrics
    
    log_info(MODEL_NAME, f"Progress: {len(completed_indices)}/{len(test_images)} samples completed")
    log_info(MODEL_NAME, f"Will process {requests_to_make} samples this run (daily limit: {MAX_REQUESTS_PER_DAY})")
    
    # Select samples to process this run
    indices_to_process = remaining_indices[:requests_to_make]
    images_to_process = [test_images[i] for i in indices_to_process]
    labels_to_process = [test_labels[i] for i in indices_to_process]
    
    # Run evaluation (sync only for rate limiting control)
    new_results = _run_evaluation_with_checkpointing(
        google_api_key, images_to_process, labels_to_process,
        indices_to_process, tokenizer_model, top_logprobs,
        completed_indices, existing_results
    )
    
    # Merge results
    all_results = existing_results + new_results
    completed_indices.update(r['index'] for r in new_results if not r.get('error'))
    
    # Save final checkpoint
    today = time.strftime('%Y-%m-%d')
    _save_checkpoint(completed_indices, all_results, today)
    
    # Generate metrics
    metrics = _aggregate_metrics(all_results, len(test_images))
    metrics['checkpoint_info'] = {
        'total_samples': len(test_images),
        'processed_samples': len(completed_indices),
        'remaining_samples': len(test_images) - len(completed_indices),
        'is_complete': len(completed_indices) >= len(test_images)
    }
    
    # Save results
    save_metrics(MODEL_NAME, metrics)
    append_metrics_csv(MODEL_NAME, metrics)
    save_results_jsonl(MODEL_NAME, all_results)
    
    log_info(MODEL_NAME, f"Evaluation run complete. Progress: {len(completed_indices)}/{len(test_images)}")
    if len(completed_indices) < len(test_images):
        log_info(MODEL_NAME, f"Run again tomorrow to continue ({len(test_images) - len(completed_indices)} samples remaining)")
    
    return metrics


def _run_evaluation_with_checkpointing(
    google_api_key: str,
    test_images: List,
    test_labels: List,
    indices: List[int],
    tokenizer_model,
    top_logprobs: int,
    completed_indices: set,
    existing_results: List[Dict]
) -> List[Dict[str, Any]]:
    """
    Run evaluation with rate limiting and incremental checkpointing.
    
    Processes images one at a time with delays to respect rate limits,
    saving checkpoint after each successful request.
    
    Args:
        google_api_key: Google API key
        test_images: List of PIL Image objects to process
        test_labels: List of ground truth text labels
        indices: Original indices of these samples in full dataset
        tokenizer_model: Gemini client for tokenization
        top_logprobs: Number of top alternative tokens to return
        completed_indices: Set of already completed indices (for checkpoint)
        existing_results: Existing results from checkpoint
    
    Returns:
        List of new result dictionaries
    """
    results = []
    api_url = GEMINI_API_URL.format(model=MODEL_ID)
    headers = {"Content-Type": "application/json"}
    prompt = "Please read and transcribe all text in this image. Return only the transcribed text, nothing else."
    
    num_samples = len(test_images)
    today = time.strftime('%Y-%m-%d')
    
    # Track timing for rate limiting
    request_times = []
    
    # Create iterator with tqdm progress bar if available
    iterator = enumerate(zip(indices, test_images, test_labels))
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"{MODEL_NAME}", total=num_samples, unit="img")
    
    for batch_idx, (original_idx, pil_image, ground_truth) in iterator:
        result = {
            "index": original_idx,
            "ground_truth": ground_truth,
            "predicted_text": None,
            "cer": None,
            "wer": None,
            "semantic_error": None,
            "kl_divergence": None,
            "entropy": None,
            "mean_gt_rank": None,
            "top5_accuracy": None,
            "inference_time": None,
            "logprobs": None,
            "mean_logprob": None,
            "confidence": None,
            "error": None,
            "run_date": today
        }
        
        # Rate limiting: ensure we don't exceed 25 requests per minute
        current_time = time.time()
        # Remove timestamps older than 60 seconds
        request_times = [t for t in request_times if current_time - t < 60]
        
        if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
            # Wait until oldest request is 60 seconds old
            wait_time = 60 - (current_time - request_times[0]) + 0.1
            if wait_time > 0:
                log_info(MODEL_NAME, f"Rate limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                request_times = request_times[1:]  # Remove oldest
        
        try:
            # Encode PIL image to base64
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_data = base64.b64encode(buf.getvalue()).decode()
            
            # Build native Gemini REST payload with logprobs enabled
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 1024,
                    "responseLogprobs": True,
                    "logprobs": top_logprobs
                }
            }
            
            # Call Google Gemini API directly
            start_time = time.time()
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                params={"key": google_api_key},
                timeout=60
            )
            inference_time = time.time() - start_time
            request_times.append(time.time())
            
            if response.status_code == 429:
                # Rate limited - wait and retry once
                log_warning(MODEL_NAME, "Rate limited (429), waiting 60s before retry...")
                time.sleep(60)
                response = requests.post(
                    api_url,
                    json=payload,
                    headers=headers,
                    params={"key": google_api_key},
                    timeout=60
                )
                inference_time = time.time() - start_time
                request_times.append(time.time())
            
            if response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text}")
            
            response_data = response.json()
            
            # Extract text response from native Gemini format
            predicted_text = ""
            candidates = response_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    predicted_text = parts[0].get("text", "").strip()
            
            # Extract logprobs from response
            logprobs_data = _extract_logprobs_from_response(response_data)
            
            # Calculate mean logprob and confidence
            mean_logprob = None
            confidence = None
            if logprobs_data:
                logprob_values = [t["logprob"] for t in logprobs_data if t.get("logprob") is not None]
                if logprob_values:
                    mean_logprob = sum(logprob_values) / len(logprob_values)
                    confidence = math.exp(mean_logprob) * 100
            
            # Calculate metrics
            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)
            semantic_metrics = get_semantic_error(
                logprobs_data, ground_truth, predicted_text,
                tokenizer_model=tokenizer_model
            )
            
            result["predicted_text"] = predicted_text
            result["cer"] = cer
            result["wer"] = wer
            result["inference_time"] = inference_time
            result["logprobs"] = logprobs_data
            result["mean_logprob"] = mean_logprob
            result["confidence"] = confidence
            result["semantic_error"] = semantic_metrics.get("semantic_error")
            result["kl_divergence"] = semantic_metrics.get("kl_divergence")
            result["entropy"] = semantic_metrics.get("entropy")
            result["mean_gt_rank"] = semantic_metrics.get("mean_gt_rank")
            result["top5_accuracy"] = semantic_metrics.get("top5_accuracy")
            
            # Update completed indices and save checkpoint incrementally
            completed_indices.add(original_idx)
            
        except Exception as e:
            log_warning(MODEL_NAME, f"Error processing sample {original_idx}: {str(e)}")
            result["error"] = str(e)
        
        results.append(result)
        
        # Save checkpoint every 10 samples or on last sample
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_samples - 1:
            all_results = existing_results + results
            _save_checkpoint(completed_indices, all_results, today)
    
    return results


async def _process_single_image(
    session: "aiohttp.ClientSession",
    semaphore: asyncio.Semaphore,
    idx: int,
    pil_image,
    ground_truth: str,
    api_url: str,
    google_api_key: str,
    prompt: str,
    top_logprobs: int,
    tokenizer_model
) -> Dict[str, Any]:
    """
    Process a single image asynchronously.
    
    Args:
        session: aiohttp session
        semaphore: Semaphore to limit concurrency
        idx: Image index
        pil_image: PIL Image object
        ground_truth: Ground truth text
        api_url: Gemini API URL
        google_api_key: Google API key
        prompt: OCR prompt
        top_logprobs: Number of top logprobs to request
        tokenizer_model: Gemini tokenizer model
    
    Returns:
        Result dictionary for this image
    """
    result = {
        "index": idx,
        "ground_truth": ground_truth,
        "predicted_text": None,
        "cer": None,
        "wer": None,
        "semantic_error": None,
        "kl_divergence": None,
        "entropy": None,
        "mean_gt_rank": None,
        "top5_accuracy": None,
        "inference_time": None,
        "logprobs": None,
        "mean_logprob": None,
        "confidence": None,
        "error": None
    }
    
    async with semaphore:
        try:
            # Encode PIL image to base64
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_data = base64.b64encode(buf.getvalue()).decode()
            
            # Build native Gemini REST payload with logprobs enabled
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 1024,
                    "responseLogprobs": True,
                    "logprobs": top_logprobs
                }
            }
            
            # Call Google Gemini API directly
            start_time = time.time()
            async with session.post(
                api_url,
                json=payload,
                params={"key": google_api_key},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                inference_time = time.time() - start_time
                
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"API error {response.status}: {text}")
                
                response_data = await response.json()
            
            # Extract text response from native Gemini format
            predicted_text = ""
            candidates = response_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    predicted_text = parts[0].get("text", "").strip()
            
            # Extract logprobs from response
            logprobs_data = _extract_logprobs_from_response(response_data)
            
            # Calculate mean logprob and confidence
            mean_logprob = None
            confidence = None
            if logprobs_data:
                logprob_values = [t["logprob"] for t in logprobs_data if t.get("logprob") is not None]
                if logprob_values:
                    mean_logprob = sum(logprob_values) / len(logprob_values)
                    confidence = math.exp(mean_logprob) * 100
            
            # Calculate metrics
            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)
            semantic_metrics = get_semantic_error(
                logprobs_data, ground_truth, predicted_text, 
                tokenizer_model=tokenizer_model
            )
            
            result["predicted_text"] = predicted_text
            result["cer"] = cer
            result["wer"] = wer
            result["inference_time"] = inference_time
            result["logprobs"] = logprobs_data
            result["mean_logprob"] = mean_logprob
            result["confidence"] = confidence
            result["semantic_error"] = semantic_metrics.get("semantic_error")
            result["kl_divergence"] = semantic_metrics.get("kl_divergence")
            result["entropy"] = semantic_metrics.get("entropy")
            result["mean_gt_rank"] = semantic_metrics.get("mean_gt_rank")
            result["top5_accuracy"] = semantic_metrics.get("top5_accuracy")
            
        except Exception as e:
            log_warning(MODEL_NAME, f"Error processing sample {idx}: {str(e)}")
            result["error"] = str(e)
    
    return result


async def _run_evaluation_async(
    google_api_key: str, 
    test_images: List, 
    test_labels: List, 
    tokenizer_model, 
    top_logprobs: int = 5
) -> Dict[str, Any]:
    """
    Run evaluation on test set calling Google Gemini API directly using async requests.
    
    Args:
        google_api_key: Google API key
        test_images: List of PIL Image objects
        test_labels: List of ground truth text labels
        tokenizer_model: Gemini client for tokenization (optional)
        top_logprobs: Number of top alternative tokens to return
    
    Returns:
        Dictionary of metrics
    """
    api_url = GEMINI_API_URL.format(model=MODEL_ID)
    prompt = "Please read and transcribe all text in this image. Return only the transcribed text, nothing else."
    num_samples = len(test_images)
    
    log_info(MODEL_NAME, f"Starting async evaluation with {MAX_CONCURRENT_REQUESTS} concurrent requests")
    
    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            _process_single_image(
                session, semaphore, idx, pil_image, ground_truth,
                api_url, google_api_key, prompt, top_logprobs, tokenizer_model
            )
            for idx, (pil_image, ground_truth) in enumerate(zip(test_images, test_labels))
        ]
        
        # Use tqdm for progress bar if available
        if atqdm is not None:
            results = await atqdm.gather(
                *tasks,
                desc=f"{MODEL_NAME}",
                total=num_samples,
                unit="img"
            )
        else:
            results = await asyncio.gather(*tasks)
    
    # Sort results by index to maintain order
    results = sorted(results, key=lambda x: x["index"])
    
    # Save JSONL results
    save_results_jsonl(MODEL_NAME, results)
    
    # Aggregate metrics
    return _aggregate_metrics(results, num_samples)


def _run_evaluation_sync(google_api_key: str, test_images: List, test_labels: List, tokenizer_model, top_logprobs: int = 5) -> Dict[str, Any]:
    """
    Run evaluation on test set calling Google Gemini API directly (synchronous fallback).
    
    Args:
        google_api_key: Google API key
        test_images: List of PIL Image objects
        test_labels: List of ground truth text labels
        tokenizer_model: Gemini client for tokenization (optional)
        top_logprobs: Number of top alternative tokens to return
    
    Returns:
        Dictionary of metrics
    """
    results = []
    
    api_url = GEMINI_API_URL.format(model=MODEL_ID)
    headers = {"Content-Type": "application/json"}
    
    prompt = "Please read and transcribe all text in this image. Return only the transcribed text, nothing else."
    
    num_samples = len(test_images)
    
    # Create iterator with tqdm progress bar if available
    iterator = enumerate(zip(test_images, test_labels))
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"{MODEL_NAME} (sync)", total=num_samples, unit="img")
    
    for idx, (pil_image, ground_truth) in iterator:
        result = {
            "index": idx,
            "ground_truth": ground_truth,
            "predicted_text": None,
            "cer": None,
            "wer": None,
            "semantic_error": None,
            "kl_divergence": None,
            "entropy": None,
            "mean_gt_rank": None,
            "top5_accuracy": None,
            "inference_time": None,
            "logprobs": None,
            "mean_logprob": None,
            "confidence": None,
            "error": None
        }
        
        try:
            # Encode PIL image to base64
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_data = base64.b64encode(buf.getvalue()).decode()
            
            # Build native Gemini REST payload with logprobs enabled
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 1024,
                    "responseLogprobs": True,
                    "logprobs": top_logprobs
                }
            }
            
            # Call Google Gemini API directly
            start_time = time.time()
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                params={"key": google_api_key},
                timeout=60
            )
            inference_time = time.time() - start_time
            
            if response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text}")
            
            response_data = response.json()
            
            # Extract text response from native Gemini format
            # Structure: candidates[0].content.parts[0].text
            predicted_text = ""
            candidates = response_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    predicted_text = parts[0].get("text", "").strip()
            
            # Extract logprobs from response
            logprobs_data = _extract_logprobs_from_response(response_data)
            
            # Calculate mean logprob and confidence
            mean_logprob = None
            confidence = None
            if logprobs_data:
                logprob_values = [t["logprob"] for t in logprobs_data if t.get("logprob") is not None]
                if logprob_values:
                    mean_logprob = sum(logprob_values) / len(logprob_values)
                    confidence = math.exp(mean_logprob) * 100
            
            # Calculate metrics
            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)
            semantic_metrics = get_semantic_error(
                logprobs_data, ground_truth, predicted_text, 
                tokenizer_model=tokenizer_model
            )
            
            result["predicted_text"] = predicted_text
            result["cer"] = cer
            result["wer"] = wer
            result["inference_time"] = inference_time
            result["logprobs"] = logprobs_data
            result["mean_logprob"] = mean_logprob
            result["confidence"] = confidence
            result["semantic_error"] = semantic_metrics.get("semantic_error")
            result["kl_divergence"] = semantic_metrics.get("kl_divergence")
            result["entropy"] = semantic_metrics.get("entropy")
            result["mean_gt_rank"] = semantic_metrics.get("mean_gt_rank")
            result["top5_accuracy"] = semantic_metrics.get("top5_accuracy")
        
        except Exception as e:
            log_warning(MODEL_NAME, f"Error processing sample {idx}: {str(e)}")
            result["error"] = str(e)
        
        results.append(result)
    
    # Save JSONL results
    save_results_jsonl(MODEL_NAME, results)
    
    # Aggregate and return metrics
    return _aggregate_metrics(results, num_samples)

def _aggregate_metrics(results: List[Dict[str, Any]], num_samples: int) -> Dict[str, Any]:
    """
    Aggregate individual results into summary metrics.
    
    Args:
        results: List of result dictionaries from processing
        num_samples: Total number of samples
    
    Returns:
        Dictionary of aggregated metrics
    """
    cer_values = []
    wer_values = []
    inference_times = []
    semantic_errors = []
    kl_divergences = []
    entropies = []
    mean_gt_ranks = []
    top5_accuracies = []
    num_errors = 0
    
    for r in results:
        if r.get("error"):
            num_errors += 1
        else:
            if r.get("cer") is not None:
                cer_values.append(r["cer"])
            if r.get("wer") is not None:
                wer_values.append(r["wer"])
            if r.get("inference_time") is not None:
                inference_times.append(r["inference_time"])
            if r.get("semantic_error") is not None:
                semantic_errors.append(r["semantic_error"])
            if r.get("kl_divergence") is not None:
                kl_divergences.append(r["kl_divergence"])
            if r.get("entropy") is not None:
                entropies.append(r["entropy"])
            if r.get("mean_gt_rank") is not None:
                mean_gt_ranks.append(r["mean_gt_rank"])
            if r.get("top5_accuracy") is not None:
                top5_accuracies.append(r["top5_accuracy"])
    
    metrics = {
        "num_samples": num_samples,
        "num_successful": num_samples - num_errors,
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
    
    if semantic_errors:
        metrics["mean_semantic_error"] = sum(semantic_errors) / len(semantic_errors)
    if kl_divergences:
        metrics["mean_kl_divergence"] = sum(kl_divergences) / len(kl_divergences)
    if entropies:
        metrics["mean_entropy"] = sum(entropies) / len(entropies)
    if mean_gt_ranks:
        metrics["mean_gt_rank"] = sum(mean_gt_ranks) / len(mean_gt_ranks)
    if top5_accuracies:
        metrics["mean_top5_accuracy"] = sum(top5_accuracies) / len(top5_accuracies)
    
    return metrics


def _extract_logprobs_from_response(response_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Extract logprobs from a native Google Gemini API response.

    The Gemini REST API returns logprobs under:
      candidates[0].logprobsResult.chosenCandidates   — the chosen token at each position
      candidates[0].logprobsResult.topCandidates[i]   — list of top-k alternatives per position

    Each entry in chosenCandidates / topCandidates has the shape:
      { "token": str, "tokenId": int, "logProbability": float }

    We normalise this into the same internal format used by the rest of the pipeline:
      { "token": str, "logprob": float, "prob": float, "top_tokens": [...] }

    Args:
        response_data: Parsed JSON response from the Gemini REST API

    Returns:
        List of token logprob data with top alternatives, or None if unavailable
    """
    try:
        candidates = response_data.get("candidates", [])
        if not candidates:
            return None

        logprobs_result = candidates[0].get("logprobsResult", {})
        chosen = logprobs_result.get("chosenCandidates", [])
        top_candidates = logprobs_result.get("topCandidates", [])

        if not chosen:
            return None

        logprobs_data = []
        for i, chosen_token in enumerate(chosen):
            logprob = chosen_token.get("logProbability")
            token_entry = {
                "token": chosen_token.get("token", ""),
                "logprob": logprob,
                "prob": math.exp(logprob) if logprob is not None else None,
                "top_tokens": []
            }

            # top_candidates[i] is a dict with a "candidates" list
            if i < len(top_candidates):
                for alt in top_candidates[i].get("candidates", []):
                    alt_logprob = alt.get("logProbability")
                    token_entry["top_tokens"].append({
                        "token": alt.get("token", ""),
                        "logprob": alt_logprob,
                        "prob": math.exp(alt_logprob) if alt_logprob is not None else None
                    })

            logprobs_data.append(token_entry)

        return logprobs_data if logprobs_data else None

    except Exception as e:
        log_warning(MODEL_NAME, f"Failed to extract logprobs from response: {str(e)}")
        return None

def get_semantic_error(logprobs_data: Optional[List[Dict[str, Any]]], ground_truth: str, predicted_text: str, tokenizer_model=None) -> Dict[str, float]:
    """
    Compute semantic error metrics from Gemini logprobs with optional Gemini tokenizer.
    
    Semantic Error = mean(log P(predicted_token) - log P(correct_token))
    
    This captures: How much more confident was the model in its prediction vs the correct answer?
    - Positive value: Model was more confident in wrong tokens
    - Negative value: Model was more confident in correct tokens (but still predicted wrong)
    - Zero: Equal confidence (or perfect prediction)
    
    Also computes:
    - KL divergence between predicted and ground truth token probabilities
    - Entropy: How uncertain was the model? (low = confident, high = uncertain)
    - Mean rank of ground truth tokens in the model's output
    - Top-5 accuracy: Fraction of GT tokens in model's top-5 predictions
    
    Args:
        logprobs_data: Extracted logprobs from Portkey API response
        ground_truth: The correct text
        predicted_text: The model's prediction
        tokenizer_model: Optional Gemini model for accurate tokenization
        
    Returns:
        Dictionary with semantic_error, kl_divergence, entropy, and rank metrics
    """
    if not logprobs_data:
        return {
            "semantic_error": None,
            "kl_divergence": None,
            "entropy": None,
            "mean_gt_rank": None,
            "top5_accuracy": None
        }
    
    try:
        # Try to tokenize using Gemini tokenizer if available, fall back to character-level
        gt_tokens = None
        pred_tokens = None
        use_token_level = False
        
        if tokenizer_model:
            try:
                # Use Gemini's tokenizer (google-genai v1.66.0+) for accurate token-level analysis
                gt_response = tokenizer_model.models.count_tokens(
                    model=MODEL_ID,
                    contents=ground_truth
                )
                pred_response = tokenizer_model.models.count_tokens(
                    model=MODEL_ID,
                    contents=predicted_text
                )
                
                # Extract token count - google-genai returns total_tokens
                if hasattr(gt_response, 'total_tokens') and hasattr(pred_response, 'total_tokens'):
                    # For token-level metrics, we use the count as a proxy
                    # The actual token IDs are abstracted in google-genai
                    gt_tokens = list(range(gt_response.total_tokens))
                    pred_tokens = list(range(pred_response.total_tokens))
                    use_token_level = True
                    log_info(MODEL_NAME, f"Using token-level semantic error analysis (GT: {len(gt_tokens)}, Pred: {len(pred_tokens)} tokens)")
            except Exception as e:
                log_info(MODEL_NAME, f"Tokenizer model available but token extraction failed: {str(e)}, falling back to character-level")
        
        # Fall back to character-level if tokenizer unavailable
        if not use_token_level:
            gt_tokens = list(ground_truth)
            pred_tokens = list(predicted_text)
            log_info(MODEL_NAME, "Using character-level semantic error analysis")
        
        # Align lengths for comparison (up to 50 tokens for efficiency)
        min_len = min(len(gt_tokens), len(pred_tokens), min(len(logprobs_data), 50))
        
        log_diffs = []
        kl_components = []
        gt_ranks = []
        entropies = []
        
        for i in range(min_len):
            if i >= len(logprobs_data):
                break
            
            logprob_entry = logprobs_data[i]
            gt_token = gt_tokens[i] if i < len(gt_tokens) else None
            pred_token = pred_tokens[i] if i < len(pred_tokens) else None
            
            # Get logprob for predicted token (the chosen one)
            log_p_pred = logprob_entry.get("logprob")
            if log_p_pred is None:
                continue
            
            # Look for ground truth token in top_tokens
            log_p_gt = None
            if gt_token and logprob_entry.get("top_tokens"):
                for alt_token in logprob_entry["top_tokens"]:
                    if alt_token.get("token", "").strip() == str(gt_token).strip():
                        log_p_gt = alt_token.get("logprob")
                        break
            
            # If GT token not found in top alternatives, estimate from main prediction
            if log_p_gt is None:
                if logprob_entry.get("token", "").strip() == str(gt_token).strip():
                    log_p_gt = log_p_pred
                else:
                    # Penalize missing GT token
                    log_p_gt = log_p_pred - 5.0  # Assume much lower probability
            
            # Semantic error: log(P_pred) - log(P_gt)
            if log_p_pred is not None and log_p_gt is not None:
                log_diff = log_p_pred - log_p_gt
                log_diffs.append(log_diff)
                
                # KL component: -log P(gt) when model chose something else
                if logprob_entry.get("token", "").strip() != str(gt_token).strip():
                    kl_components.append(-log_p_gt)
            
            # Compute entropy from the token's probability distribution
            if logprob_entry.get("top_tokens"):
                probs = []
                for alt_token in logprob_entry["top_tokens"]:
                    logp = alt_token.get("logprob")
                    if logp is not None:
                        probs.append(math.exp(logp))
                
                if probs and log_p_pred is not None:
                    # Entropy: -sum(p * log(p))
                    entropy = 0.0
                    for p in probs:
                        if p > 0:
                            entropy -= p * math.log(p)
                    entropies.append(entropy)
                    
                    # Rank of GT token in top tokens
                    gt_rank = None
                    for rank, alt_token in enumerate(logprob_entry["top_tokens"]):
                        if alt_token.get("token", "").strip() == str(gt_token).strip():
                            gt_rank = rank
                            break
                    
                    if gt_rank is not None:
                        gt_ranks.append(gt_rank)
        
        # Compute aggregated metrics
        if log_diffs:
            semantic_error = sum(log_diffs) / len(log_diffs)
        else:
            semantic_error = None
        
        if kl_components:
            kl_divergence = sum(kl_components) / len(kl_components)
        else:
            kl_divergence = None
        
        if entropies:
            entropy = sum(entropies) / len(entropies)
        else:
            entropy = None
        
        if gt_ranks:
            mean_gt_rank = sum(gt_ranks) / len(gt_ranks)
            top5_accuracy = sum(1 for r in gt_ranks if r < 5) / len(gt_ranks)
        else:
            mean_gt_rank = None
            top5_accuracy = None
        
        return {
            "semantic_error": semantic_error,
            "kl_divergence": kl_divergence,
            "entropy": entropy,
            "mean_gt_rank": mean_gt_rank,
            "top5_accuracy": top5_accuracy
        }
    
    except Exception as e:
        log_warning(MODEL_NAME, f"Failed to compute semantic error: {str(e)}")
        return {
            "semantic_error": None,
            "kl_divergence": None,
            "entropy": None,
            "mean_gt_rank": None,
            "top5_accuracy": None
        }

if __name__ == "__main__":
    # Allow running evaluator directly
    project_root = Path(__file__).parent.parent
    result = evaluate(str(project_root))
