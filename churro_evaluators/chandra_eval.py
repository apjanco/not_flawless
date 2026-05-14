"""
evaluators/chandra_eval.py
Chandra OCR model evaluator
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List

try:
    from datasets import load_from_disk
except ImportError:
    load_from_disk = None

try:
    from chandra.model import InferenceManager, BatchInputItem
    chandra_available = True
except ImportError:
    InferenceManager = None
    BatchInputItem = None
    chandra_available = False

from evaluators.utils import (
    get_data_dir, get_results_dir,
    character_error_rate, word_error_rate,
    save_metrics, append_metrics_csv, save_results_jsonl,
    log_info, log_error, log_warning
)

MODEL_NAME = "chandra"


def _get_checkpoint_path() -> Path:
    """Get path to checkpoint file."""
    return get_results_dir() / f"{MODEL_NAME}_checkpoint.jsonl"


def _load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint from disk."""
    checkpoint_path = _get_checkpoint_path()
    if checkpoint_path.exists():
        try:
            results = []
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    results.append(json.loads(line))
            completed_indices = {r['image_idx'] for r in results}
            log_info(MODEL_NAME, f"Loaded checkpoint: {len(completed_indices)} samples already processed")
            return {'completed_indices': completed_indices, 'results': results}
        except Exception as e:
            log_warning(MODEL_NAME, f"Failed to load checkpoint: {e}")
    return {'completed_indices': set(), 'results': []}


def _append_checkpoint(result: Dict[str, Any]):
    """Append a single result to the checkpoint file."""
    checkpoint_path = _get_checkpoint_path()
    try:
        with open(checkpoint_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    except Exception as e:
        log_warning(MODEL_NAME, f"Failed to append checkpoint: {e}")

def check_dependencies():
    """Check if Chandra OCR is installed"""
    return chandra_available

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
        log_error(MODEL_NAME, "Chandra OCR not installed")
        return {"error": "Chandra OCR not available"}
    
    # Initialize model
    try:
        log_info(MODEL_NAME, "Initializing Chandra OCR model (HuggingFace)")
        model = InferenceManager(method="hf")
        log_info(MODEL_NAME, "Model initialized successfully")
    except Exception as e:
        log_error(MODEL_NAME, f"Failed to initialize model: {str(e)}")
        return {"error": str(e)}
    
    # Load test data
    try:
        ds = load_from_disk('/scratch/network/aj7878/not_flawless/data/iam')
        test_images = [i['image'] for i in ds]
        test_labels = [i['text'] for i in ds]
        log_info(MODEL_NAME, f"Loaded {len(test_images)} images from IAM dataset")
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

    remaining_indices = [i for i in range(len(test_images)) if i not in completed_indices]

    if not remaining_indices:
        log_info(MODEL_NAME, "All samples already processed! Generating final metrics.")
        metrics = _aggregate_metrics(existing_results, len(test_images))
        save_metrics(MODEL_NAME, metrics)
        append_metrics_csv(MODEL_NAME, metrics)
        return metrics

    log_info(MODEL_NAME, f"Progress: {len(completed_indices)}/{len(test_images)} samples completed")
    log_info(MODEL_NAME, f"Will process {len(remaining_indices)} remaining samples")

    # Run evaluation
    new_results = _run_evaluation(
        [test_images[i] for i in remaining_indices],
        [test_labels[i] for i in remaining_indices],
        remaining_indices,
        model
    )

    all_results = existing_results + new_results

    # Save final JSONL and metrics
    save_results_jsonl(MODEL_NAME, all_results)
    metrics = _aggregate_metrics(all_results, len(test_images))
    save_metrics(MODEL_NAME, metrics)
    append_metrics_csv(MODEL_NAME, metrics)

    log_info(MODEL_NAME, "Evaluation complete")
    return metrics

def _run_evaluation(images: list, ground_truths: List[str], indices: List[int], model) -> List[Dict[str, Any]]:
    """
    Run evaluation on test set with incremental checkpointing.

    Args:
        images: List of PIL Images
        ground_truths: List of ground truth text
        indices: Original dataset indices for these images
        model: Initialized InferenceManager

    Returns:
        List of result dictionaries
    """
    results = []

    for batch_idx, (image, ground_truth, original_idx) in enumerate(zip(images, ground_truths, indices)):
        result = {
            "image_idx": original_idx,
            "ground_truth": ground_truth,
            "predicted_text": None,
            "cer": None,
            "wer": None,
            "inference_time": None,
            "error": None
        }

        try:
            # Perform OCR
            start_time = time.time()
            batch = [BatchInputItem(image=image, prompt_type="ocr")]
            outputs = model.generate(batch)
            predicted_text = outputs[0].markdown if not outputs[0].error else ""
            inference_time = time.time() - start_time

            # Calculate metrics
            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)

            result["predicted_text"] = predicted_text
            result["cer"] = cer
            result["wer"] = wer
            result["inference_time"] = inference_time

        except Exception as e:
            log_warning(MODEL_NAME, f"Error processing image {original_idx}: {str(e)}")
            result["error"] = str(e)

        results.append(result)
        _append_checkpoint(result)

        if (batch_idx + 1) % 100 == 0:
            log_info(MODEL_NAME, f"Processed {batch_idx + 1}/{len(images)} images")

    return results


def _aggregate_metrics(results: List[Dict[str, Any]], total_samples: int) -> Dict[str, Any]:
    """Aggregate per-sample results into summary metrics."""
    cer_values = [r["cer"] for r in results if r.get("cer") is not None]
    wer_values = [r["wer"] for r in results if r.get("wer") is not None]
    inference_times = [r["inference_time"] for r in results if r.get("inference_time") is not None]
    num_errors = sum(1 for r in results if r.get("error"))

    metrics = {
        "num_samples": total_samples,
        "num_successful": len(results) - num_errors,
        "num_errors": num_errors,
    }

    metrics["mean_cer"] = sum(cer_values) / len(cer_values) if cer_values else None
    metrics["median_cer"] = sorted(cer_values)[len(cer_values) // 2] if cer_values else None
    metrics["mean_wer"] = sum(wer_values) / len(wer_values) if wer_values else None
    metrics["median_wer"] = sorted(wer_values)[len(wer_values) // 2] if wer_values else None
    metrics["mean_inference_time"] = sum(inference_times) / len(inference_times) if inference_times else None
    metrics["total_inference_time"] = sum(inference_times) if inference_times else None

    return metrics

if __name__ == "__main__":
    # Allow running evaluator directly
    project_root = Path(__file__).parent.parent
    result = evaluate(str(project_root))
