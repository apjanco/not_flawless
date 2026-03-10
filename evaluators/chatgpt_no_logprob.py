"""
evaluators/chatgpt_eval.py
ChatGPT (via OpenAI API via Portkey) OCR evaluator with logprobs and semantic error analysis
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

MODEL_NAME = "chatgpt5"
MODEL_ID = "gpt-5"
MAX_CONCURRENT_REQUESTS = 10

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

def get_portkey_key():
    """Get Portkey API key from environment"""
    api_key = os.getenv("PORTKEY_API_KEY")
    if not api_key:
        log_warning(MODEL_NAME, "PORTKEY_API_KEY environment variable not set")
    return api_key

def evaluate(project_root: str = None, top_logprobs: int = 5, max_requests: int = None) -> Dict[str, Any]:
    """
    Main evaluation function with checkpointing support.
    
    Supports incremental evaluation - progress is saved to checkpoint file
    and resumes from where it left off on subsequent runs.

    Args:
        project_root: Path to project root
        top_logprobs: Number of top alternative tokens to return (1-20)
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
    api_key = get_portkey_key()
    if not api_key:
        log_error(MODEL_NAME, "PORTKEY_API_KEY environment variable is REQUIRED but not set")
        log_error(MODEL_NAME, "Please set: export PORTKEY_API_KEY='your-portkey-key'")
        return {"error": "PORTKEY_API_KEY not configured - evaluation cannot proceed"}

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
            indices_to_process, top_logprobs,
            completed_indices, existing_results
        ))
    else:
        new_results = _run_evaluation_with_checkpointing(
            api_key, images_to_process, labels_to_process,
            indices_to_process, top_logprobs,
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
    top_logprobs: int,
    completed_indices: set,
    existing_results: List[Dict]
) -> List[Dict[str, Any]]:
    """
    Run evaluation with incremental checkpointing.
    
    Processes images and saves checkpoint after each batch.
    
    Args:
        api_key: Portkey API key
        test_images: List of PIL Image objects to process
        test_labels: List of ground truth text labels
        indices: Original indices of these samples in full dataset
        top_logprobs: Number of top alternative tokens to return
        completed_indices: Set of already completed indices (for checkpoint)
        existing_results: Existing results from checkpoint
    
    Returns:
        List of new result dictionaries
    """
    results = []
    headers = {
        "x-portkey-api-key": api_key,
        "Content-Type": "application/json"
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
        
        try:
            # Encode PIL image to base64
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_data = base64.b64encode(buf.getvalue()).decode()
            
            payload = {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"}
                            }
                        ]
                    }
                ],
                "max_completion_tokens": 1024,
                "logprobs": False
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
            
            if response.status_code == 429:
                # Rate limited - wait and retry once
                log_warning(MODEL_NAME, "Rate limited (429), waiting 60s before retry...")
                time.sleep(60)
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
            logprobs_data = _extract_logprobs(response_data)
            
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
            semantic_metrics = get_semantic_error(logprobs_data, ground_truth, predicted_text)
            
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
    top_logprobs: int,
    completed_indices: set,
    existing_results: List[Dict]
) -> List[Dict[str, Any]]:
    """
    Run async evaluation with batch checkpointing.
    
    Processes images in batches concurrently for speed, saving checkpoint
    after each batch completes.
    """
    headers = {
        "x-portkey-api-key": api_key,
        "Content-Type": "application/json"
    }
    prompt = "Please read and transcribe all text in this image. Return only the transcribed text, nothing else."
    
    num_samples = len(test_images)
    today = time.strftime('%Y-%m-%d')
    batch_size = 50  # Process 50 images concurrently, then checkpoint
    
    all_new_results = []
    
    log_info(MODEL_NAME, f"Starting async evaluation with batch size {batch_size}")
    
    # Process in batches
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_indices = indices[batch_start:batch_end]
        batch_images = test_images[batch_start:batch_end]
        batch_labels = test_labels[batch_start:batch_end]
        
        log_info(MODEL_NAME, f"Processing batch {batch_start//batch_size + 1}: samples {batch_start+1}-{batch_end} of {num_samples}")
        
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                _process_single_image_with_idx(
                    session, semaphore, original_idx, pil_image, ground_truth,
                    headers, prompt, top_logprobs, today
                )
                for original_idx, pil_image, ground_truth in zip(batch_indices, batch_images, batch_labels)
            ]
            
            # Use tqdm for progress bar if available
            if atqdm is not None:
                batch_results = await atqdm.gather(
                    *tasks,
                    desc=f"{MODEL_NAME} batch {batch_start//batch_size + 1}",
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


async def _process_single_image_with_idx(
    session: "aiohttp.ClientSession",
    semaphore: asyncio.Semaphore,
    idx: int,
    pil_image,
    ground_truth: str,
    headers: Dict[str, str],
    prompt: str,
    top_logprobs: int,
    run_date: str
) -> Dict[str, Any]:
    """
    Process a single image asynchronously with original index preserved.
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
        "error": None,
        "run_date": run_date
    }
    
    async with semaphore:
        try:
            # Encode PIL image to base64
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_data = base64.b64encode(buf.getvalue()).decode()

            payload = {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"}
                            }
                        ]
                    }
                ],
                "max_completion_tokens": 1024,
                "logprobs": False
            }

            start_time = time.time()
            async with session.post(
                "https://api.portkey.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                inference_time = time.time() - start_time
                
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"API error: {response.status} - {text}")

                response_data = await response.json()

            predicted_text = response_data["choices"][0]["message"]["content"].strip()
            logprobs_data = _extract_logprobs(response_data)

            mean_logprob = None
            confidence = None
            if logprobs_data:
                logprob_values = [t["logprob"] for t in logprobs_data if t.get("logprob") is not None]
                if logprob_values:
                    mean_logprob = sum(logprob_values) / len(logprob_values)
                    confidence = math.exp(mean_logprob) * 100

            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)
            semantic_metrics = get_semantic_error(logprobs_data, ground_truth, predicted_text)

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


async def _process_single_image(
    session: "aiohttp.ClientSession",
    semaphore: asyncio.Semaphore,
    idx: int,
    pil_image,
    ground_truth: str,
    headers: Dict[str, str],
    prompt: str,
    top_logprobs: int
) -> Dict[str, Any]:
    """
    Process a single image asynchronously.
    
    Args:
        session: aiohttp session
        semaphore: Semaphore to limit concurrency
        idx: Image index
        pil_image: PIL Image object
        ground_truth: Ground truth text
        headers: Request headers
        prompt: OCR prompt
        top_logprobs: Number of top logprobs to request
    
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

            # Prepare request with logprobs enabled
            payload = {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"}
                            }
                        ]
                    }
                ],
                "max_completion_tokens": 1024,
                "logprobs": False,
                #"top_logprobs": top_logprobs
            }

            # Call API via Portkey
            start_time = time.time()
            async with session.post(
                "https://api.portkey.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                inference_time = time.time() - start_time
                
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"API error: {response.status} - {text}")

                response_data = await response.json()

            predicted_text = response_data["choices"][0]["message"]["content"].strip()

            # Extract logprobs
            logprobs_data = _extract_logprobs(response_data)

            # Calculate mean logprob and confidence
            mean_logprob = None
            confidence = None
            if logprobs_data:
                logprob_values = [t["logprob"] for t in logprobs_data if t.get("logprob") is not None]
                if logprob_values:
                    mean_logprob = sum(logprob_values) / len(logprob_values)
                    confidence = math.exp(mean_logprob) * 100

            # Calculate CER / WER
            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)

            # Calculate semantic error metrics
            semantic_metrics = get_semantic_error(logprobs_data, ground_truth, predicted_text)

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


def _extract_logprobs(response_data: Dict) -> Optional[List[Dict]]:
    """Extract logprobs from OpenAI response format."""
    raw_logprobs = response_data["choices"][0].get("logprobs", {})
    if not raw_logprobs or not raw_logprobs.get("content"):
        return None
    
    logprobs_data = []
    for token_info in raw_logprobs["content"]:
        chosen_logprob = token_info.get("logprob")
        token_entry = {
            "token": token_info.get("token", ""),
            "logprob": chosen_logprob,
            "prob": math.exp(chosen_logprob) if chosen_logprob is not None else None,
            "top_tokens": []
        }
        for top in token_info.get("top_logprobs", []):
            top_logprob = top.get("logprob")
            token_entry["top_tokens"].append({
                "token": top.get("token", ""),
                "logprob": top_logprob,
                "prob": math.exp(top_logprob) if top_logprob is not None else None
            })
        logprobs_data.append(token_entry)
    return logprobs_data


async def _run_evaluation_async(
    api_key: str, 
    test_images: List, 
    test_labels: List, 
    top_logprobs: int = 5
) -> Dict[str, Any]:
    """
    Run evaluation on test set via Portkey → OpenAI using async requests.

    Args:
        api_key: Portkey API key
        test_images: List of PIL Image objects
        test_labels: List of ground truth text labels
        top_logprobs: Number of top alternative tokens to request

    Returns:
        Dictionary of aggregated metrics
    """
    headers = {
        "x-portkey-api-key": api_key,
        "Content-Type": "application/json"
    }
    prompt = "Please read and transcribe all text in this image. Return only the transcribed text, nothing else."
    num_samples = len(test_images)
    
    log_info(MODEL_NAME, f"Starting async evaluation with {MAX_CONCURRENT_REQUESTS} concurrent requests")
    
    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            _process_single_image(
                session, semaphore, idx, pil_image, ground_truth,
                headers, prompt, top_logprobs
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


def _run_evaluation_sync(
    api_key: str, 
    test_images: List, 
    test_labels: List, 
    top_logprobs: int = 5
) -> Dict[str, Any]:
    """
    Run evaluation on test set via Portkey → OpenAI (synchronous fallback).

    Args:
        api_key: Portkey API key
        test_images: List of PIL Image objects
        test_labels: List of ground truth text labels
        top_logprobs: Number of top alternative tokens to request

    Returns:
        Dictionary of aggregated metrics
    """
    results = []
    headers = {
        "x-portkey-api-key": api_key,
        "Content-Type": "application/json"
    }
    prompt = "Please read and transcribe all text in this image. Return only the transcribed text, nothing else."
    num_samples = len(test_images)
    
    log_info(MODEL_NAME, "Running sync evaluation (install aiohttp for faster async)")

    # Use tqdm if available
    iterator = zip(test_images, test_labels)
    if tqdm is not None:
        iterator = tqdm(
            iterator, 
            desc=f"{MODEL_NAME}", 
            total=num_samples, 
            unit="img"
        )

    for idx, (pil_image, ground_truth) in enumerate(iterator):
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

            payload = {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 1024,
                "logprobs": True,
                "top_logprobs": top_logprobs
            }

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
            logprobs_data = _extract_logprobs(response_data)

            mean_logprob = None
            confidence = None
            if logprobs_data:
                logprob_values = [t["logprob"] for t in logprobs_data if t.get("logprob") is not None]
                if logprob_values:
                    mean_logprob = sum(logprob_values) / len(logprob_values)
                    confidence = math.exp(mean_logprob) * 100

            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)
            semantic_metrics = get_semantic_error(logprobs_data, ground_truth, predicted_text)

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

    save_results_jsonl(MODEL_NAME, results)
    return _aggregate_metrics(results, num_samples)


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


def get_semantic_error(
    logprobs_data: Optional[List[Dict[str, Any]]],
    ground_truth: str,
    predicted_text: str,
) -> Dict[str, Any]:
    """
    Compute semantic error metrics from OpenAI-format logprobs.

    Semantic Error = mean(log P(predicted_token) - log P(correct_token))

    This captures how much more confident the model was in its prediction
    versus the correct answer at each token position:
    - Positive: model favoured the wrong token
    - Negative: model leaned toward the correct token despite predicting wrong
    - Zero: equal confidence, or perfect prediction

    Also computes:
    - KL divergence proxy: mean -log P(gt) for positions where model was wrong
    - Entropy: mean token-level entropy from top-k distribution
    - Mean rank of ground-truth token in the model's top-k list
    - Top-5 accuracy: fraction of GT tokens that appear in top-5 predictions

    Args:
        logprobs_data: Extracted logprobs list (internal format with top_tokens)
        ground_truth: Correct transcription
        predicted_text: Model's prediction

    Returns:
        Dict with keys: semantic_error, kl_divergence, entropy,
                        mean_gt_rank, top5_accuracy
    """
    if not logprobs_data:
        return {
            "semantic_error": None,
            "kl_divergence": None,
            "entropy": None,
            "mean_gt_rank": None,
            "top5_accuracy": None,
        }

    try:
        # Character-level alignment (OpenAI BPE tokens ≠ characters, but we
        # compare against the chosen-token stream which is position-aligned)
        gt_tokens = list(ground_truth)
        pred_tokens = list(predicted_text)

        min_len = min(len(gt_tokens), len(pred_tokens), len(logprobs_data), 50)

        log_diffs = []
        kl_components = []
        gt_ranks = []
        entropies = []

        for i in range(min_len):
            logprob_entry = logprobs_data[i]
            gt_token = gt_tokens[i]

            log_p_pred = logprob_entry.get("logprob")
            if log_p_pred is None:
                continue

            # Search for GT token in the top-k alternatives
            log_p_gt = None
            for alt in logprob_entry.get("top_tokens", []):
                if alt.get("token", "").strip() == str(gt_token).strip():
                    log_p_gt = alt.get("logprob")
                    break

            # If not found, estimate
            if log_p_gt is None:
                if logprob_entry.get("token", "").strip() == str(gt_token).strip():
                    log_p_gt = log_p_pred          # predicted == GT
                else:
                    log_p_gt = log_p_pred - 5.0    # penalise unseen GT token

            log_diffs.append(log_p_pred - log_p_gt)
            if logprob_entry.get("token", "").strip() != str(gt_token).strip():
                kl_components.append(-log_p_gt)

            # Entropy and rank from top-k distribution
            top_tokens = logprob_entry.get("top_tokens", [])
            if top_tokens:
                probs = [math.exp(t["logprob"]) for t in top_tokens if t.get("logprob") is not None]
                if probs:
                    entropies.append(-sum(p * math.log(p) for p in probs if p > 0))

                for rank, alt in enumerate(top_tokens):
                    if alt.get("token", "").strip() == str(gt_token).strip():
                        gt_ranks.append(rank)
                        break

        return {
            "semantic_error": sum(log_diffs) / len(log_diffs) if log_diffs else None,
            "kl_divergence": sum(kl_components) / len(kl_components) if kl_components else None,
            "entropy": sum(entropies) / len(entropies) if entropies else None,
            "mean_gt_rank": sum(gt_ranks) / len(gt_ranks) if gt_ranks else None,
            "top5_accuracy": sum(1 for r in gt_ranks if r < 5) / len(gt_ranks) if gt_ranks else None,
        }

    except Exception as e:
        log_warning(MODEL_NAME, f"Failed to compute semantic error: {str(e)}")
        return {
            "semantic_error": None,
            "kl_divergence": None,
            "entropy": None,
            "mean_gt_rank": None,
            "top5_accuracy": None,
        }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    result = evaluate(str(project_root))
