"""
evaluators/claude_eval.py
Claude (via Anthropic API) OCR evaluator with checkpointing support.
Uses async requests for faster evaluation.
"""

import asyncio
import time
import os
import math
import io
import base64
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from dotenv import load_dotenv
    # Load .env file from project root
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed, rely on system env vars

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import requests
except ImportError:
    requests = None

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

from evaluators.utils import (
    get_data_dir, get_results_dir,
    character_error_rate, word_error_rate,
    save_metrics, append_metrics_csv, save_results_jsonl,
    log_info, log_error, log_warning
)

MODEL_NAME = "claude-vision"
MODEL_ID = "claude-opus-4-6"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 50
MAX_CONCURRENT_REQUESTS = 10  # Process 10 at a time within rate limit
BATCH_SIZE = 50  # Process 50, then checkpoint and wait if needed

def check_dependencies():
    """Check if required packages are installed"""
    if aiohttp is None:
        log_warning(MODEL_NAME, "aiohttp not installed, falling back to sync requests")
    return requests is not None or aiohttp is not None


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


def get_anthropic_api_key():
    """Get Anthropic API key from environment"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log_warning(MODEL_NAME, "ANTHROPIC_API_KEY environment variable not set")
    return api_key


def evaluate(project_root: str = None, max_requests: int = None) -> Dict[str, Any]:
    """
    Main evaluation function with checkpointing support.
    
    Supports incremental evaluation - progress is saved to checkpoint file
    and resumes from where it left off on subsequent runs.

    Args:
        project_root: Path to project root
        max_requests: Maximum requests for this run (default: all remaining)

    Returns:
        Dictionary of evaluation metrics
    """
    log_info(MODEL_NAME, "Starting evaluation")

    # Check dependencies
    if not check_dependencies():
        log_error(MODEL_NAME, "requests library not installed")
        return {"error": "requests not available"}

    # Check API key
    api_key = get_anthropic_api_key()
    if not api_key:
        log_error(MODEL_NAME, "ANTHROPIC_API_KEY environment variable is REQUIRED but not set")
        log_error(MODEL_NAME, "Please set: export ANTHROPIC_API_KEY='your-anthropic-key'")
        return {"error": "ANTHROPIC_API_KEY not configured - evaluation cannot proceed"}

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
        max_requests = len(test_images)  # No limit
    
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
    log_info(MODEL_NAME, f"Will process {requests_to_make} samples this run")
    
    # Select samples to process this run
    indices_to_process = remaining_indices[:requests_to_make]
    images_to_process = [test_images[i] for i in indices_to_process]
    labels_to_process = [test_labels[i] for i in indices_to_process]
    
    # Run evaluation with checkpointing (async if available)
    if aiohttp is not None:
        new_results = asyncio.run(_run_evaluation_async_with_checkpointing(
            api_key, images_to_process, labels_to_process,
            indices_to_process,
            completed_indices, existing_results
        ))
    else:
        new_results = _run_evaluation_with_checkpointing(
            api_key, images_to_process, labels_to_process,
            indices_to_process,
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
        log_info(MODEL_NAME, f"Run again to continue ({len(test_images) - len(completed_indices)} samples remaining)")
    
    return metrics


def _run_evaluation_with_checkpointing(
    api_key: str,
    test_images: List,
    test_labels: List,
    indices: List[int],
    completed_indices: set,
    existing_results: List[Dict]
) -> List[Dict[str, Any]]:
    """
    Run evaluation with incremental checkpointing (sync fallback).
    """
    results = []
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION
    }
    prompt = "Please read and transcribe all text in this image. Return only the transcribed text, nothing else."
    
    num_samples = len(test_images)
    today = time.strftime('%Y-%m-%d')
    
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
            "inference_time": None,
            "error": None,
            "run_date": today
        }
        
        try:
            # Encode PIL image to base64
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_data = base64.b64encode(buf.getvalue()).decode()
            
            # Build Anthropic API payload with vision
            payload = {
                "model": MODEL_ID,
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            # Call Anthropic API
            start_time = time.time()
            response = requests.post(
                ANTHROPIC_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            inference_time = time.time() - start_time
            
            if response.status_code == 429:
                # Rate limited - wait and retry once
                log_warning(MODEL_NAME, "Rate limited (429), waiting 60s before retry...")
                time.sleep(60)
                response = requests.post(
                    ANTHROPIC_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                inference_time = time.time() - start_time
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            response_data = response.json()
            
            # Extract text from Anthropic response format
            # Response: {"content": [{"type": "text", "text": "..."}], ...}
            predicted_text = ""
            for block in response_data.get("content", []):
                if block.get("type") == "text":
                    predicted_text += block.get("text", "")
            predicted_text = predicted_text.strip()
            
            # Calculate metrics
            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)
            
            result["predicted_text"] = predicted_text
            result["cer"] = cer
            result["wer"] = wer
            result["inference_time"] = inference_time
            
            # Update completed indices
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


async def _run_evaluation_async_with_checkpointing(
    api_key: str,
    test_images: List,
    test_labels: List,
    indices: List[int],
    completed_indices: set,
    existing_results: List[Dict]
) -> List[Dict[str, Any]]:
    """
    Run async evaluation with batch checkpointing.
    
    Processes images in batches concurrently for speed, saving checkpoint
    after each batch completes.
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION
    }
    prompt = "Please read and transcribe all text in this image. Return only the transcribed text, nothing else."
    
    num_samples = len(test_images)
    today = time.strftime('%Y-%m-%d')
    
    all_new_results = []
    batch_start_time = time.time()
    requests_this_minute = 0
    
    log_info(MODEL_NAME, f"Starting async evaluation with rate limit {MAX_REQUESTS_PER_MINUTE} RPM")
    
    # Process in batches
    for batch_start in range(0, num_samples, BATCH_SIZE):
        # Rate limiting: check if we need to wait before this batch
        elapsed = time.time() - batch_start_time
        if elapsed < 60 and requests_this_minute + min(BATCH_SIZE, num_samples - batch_start) > MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - elapsed + 0.5
            log_info(MODEL_NAME, f"Rate limit: waiting {wait_time:.1f}s before next batch...")
            await asyncio.sleep(wait_time)
            batch_start_time = time.time()
            requests_this_minute = 0
        elif elapsed >= 60:
            # Reset counter for new minute
            batch_start_time = time.time()
            requests_this_minute = 0
        batch_end = min(batch_start + BATCH_SIZE, num_samples)
        batch_indices = indices[batch_start:batch_end]
        batch_images = test_images[batch_start:batch_end]
        batch_labels = test_labels[batch_start:batch_end]
        
        log_info(MODEL_NAME, f"Processing batch {batch_start//BATCH_SIZE + 1}: samples {batch_start+1}-{batch_end} of {num_samples}")
        
        requests_this_minute += len(batch_indices)
        
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                _process_single_image_async(
                    session, semaphore, original_idx, pil_image, ground_truth,
                    headers, prompt, today
                )
                for original_idx, pil_image, ground_truth in zip(batch_indices, batch_images, batch_labels)
            ]
            
            # Use tqdm for progress bar if available
            if atqdm is not None:
                batch_results = await atqdm.gather(
                    *tasks,
                    desc=f"{MODEL_NAME} batch {batch_start//BATCH_SIZE + 1}",
                    total=len(tasks),
                    unit="img"
                )
            else:
                batch_results = await asyncio.gather(*tasks)
        
        # Update completed indices and results
        for r in batch_results:
            if not r.get('error'):
                completed_indices.add(r['index'])
        all_new_results.extend(batch_results)
        
        # Save checkpoint after each batch
        all_results = existing_results + all_new_results
        _save_checkpoint(completed_indices, all_results, today)
    
    return all_new_results


async def _process_single_image_async(
    session: "aiohttp.ClientSession",
    semaphore: asyncio.Semaphore,
    idx: int,
    pil_image,
    ground_truth: str,
    headers: Dict[str, str],
    prompt: str,
    run_date: str
) -> Dict[str, Any]:
    """
    Process a single image asynchronously.
    """
    result = {
        "index": idx,
        "ground_truth": ground_truth,
        "predicted_text": None,
        "cer": None,
        "wer": None,
        "inference_time": None,
        "error": None,
        "run_date": run_date
    }
    
    async with semaphore:
        try:
            # Encode PIL image to base64
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_data = base64.b64encode(buf.getvalue()).decode()

            # Build Anthropic API payload with vision
            payload = {
                "model": MODEL_ID,
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            start_time = time.time()
            async with session.post(
                ANTHROPIC_API_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                inference_time = time.time() - start_time
                
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"API error: {response.status} - {text}")

                response_data = await response.json()

            # Extract text from Anthropic response format
            predicted_text = ""
            for block in response_data.get("content", []):
                if block.get("type") == "text":
                    predicted_text += block.get("text", "")
            predicted_text = predicted_text.strip()

            # Calculate metrics
            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)

            result["predicted_text"] = predicted_text
            result["cer"] = cer
            result["wer"] = wer
            result["inference_time"] = inference_time

        except Exception as e:
            log_warning(MODEL_NAME, f"Error processing sample {idx}: {str(e)}")
            result["error"] = str(e)
    
    return result


def _aggregate_metrics(results: List[Dict], num_samples: int) -> Dict[str, Any]:
    """Aggregate metrics from individual results."""
    cer_values = [r["cer"] for r in results if r.get("cer") is not None]
    wer_values = [r["wer"] for r in results if r.get("wer") is not None]
    inference_times = [r["inference_time"] for r in results if r.get("inference_time") is not None]
    num_errors = sum(1 for r in results if r.get("error") is not None)

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

    return metrics


if __name__ == "__main__":
    # Run evaluation
    metrics = evaluate()
    print(f"Evaluation complete: {metrics}")
