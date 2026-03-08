"""
evaluators/qwen_eval.py
Qwen3-VL-8B-Instruct vision-language model evaluator for OCR
"""
import io
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, List, Set
from datasets import load_from_disk
import torch
from nnsight import NNsight
import torch.nn.functional as F


try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    import torch
except ImportError:
    Qwen3VLForConditionalGeneration = None
    AutoProcessor = None
    torch = None

from evaluators.utils import (
    get_data_dir, get_results_dir, load_iam_data,
    character_error_rate, word_error_rate,
    save_metrics, append_metrics_csv, save_results_jsonl,
    log_info, log_error, log_warning
)

MODEL_NAME = "Qwen3-VL-8B-Instruct"
# Local path for offline use on compute nodes (no internet access)
LOCAL_MODEL_PATH = "/scratch/network/aj7878/not_flawless/models/Qwen3-VL-8B-Instruct"
# Checkpoint file for resumable evaluation
CHECKPOINT_FILE = "/scratch/network/aj7878/not_flawless/results/Qwen3-VL-8B-Instruct_checkpoint.jsonl"

def check_dependencies():
    """Check if required packages are installed"""
    return Qwen3VLForConditionalGeneration is not None and torch is not None


def load_checkpoint() -> tuple[Set[int], List[Dict]]:
    """
    Load existing checkpoint data to resume evaluation.
    
    Returns:
        Tuple of (set of processed indices, list of existing results)
    """
    checkpoint_path = Path(CHECKPOINT_FILE)
    processed_indices = set()
    existing_results = []
    
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        existing_results.append(result)
                        processed_indices.add(result["image_path"])
            log_info(MODEL_NAME, f"Loaded checkpoint with {len(processed_indices)} already processed images")
        except Exception as e:
            log_warning(MODEL_NAME, f"Failed to load checkpoint: {str(e)}, starting fresh")
    
    return processed_indices, existing_results


def save_checkpoint_result(result: Dict):
    """
    Append a single result to the checkpoint file.
    
    Args:
        result: Result dictionary for one image
    """
    checkpoint_path = Path(CHECKPOINT_FILE)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(checkpoint_path, 'a') as f:
        f.write(json.dumps(result) + '\n')

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
        log_error(MODEL_NAME, "Required packages not installed (transformers, torch)")
        return {"error": "Dependencies not available"}
    
    # Initialize model
    try:
        log_info(MODEL_NAME, f"Loading Qwen3-VL 8B model from {LOCAL_MODEL_PATH}")
        processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )
        log_info(MODEL_NAME, "Model initialized successfully")
    except Exception as e:
        log_error(MODEL_NAME, f"Failed to initialize model: {str(e)}")
        return {"error": str(e)}
    
    # Load test data
    try:
        ds = load_from_disk('/scratch/network/aj7878/not_flawless/data/iam')
        test_images = [i['image'] for i in ds]
        test_labels = [i['text'] for i in ds]
        log_info(MODEL_NAME, f"Loaded {len(test_images)} test images")
    except Exception as e:
        log_error(MODEL_NAME, f"Failed to load test data: {str(e)}")
        return {"error": str(e)}
    
    if not test_images:
        log_warning(MODEL_NAME, "No test images found")
        return {"error": "No test data"}
    
    # Run evaluation
    metrics = _run_evaluation(model, processor, test_images, test_labels)
    
    # Only save final metrics if all samples are processed
    if metrics.get("num_successful", 0) + metrics.get("num_errors", 0) == metrics.get("num_samples", 0):
        save_metrics(MODEL_NAME, metrics)
        append_metrics_csv(MODEL_NAME, metrics)
        log_info(MODEL_NAME, "Evaluation complete - all samples processed")
    else:
        log_info(MODEL_NAME, f"Evaluation paused - {metrics.get('num_successful', 0) + metrics.get('num_errors', 0)}/{metrics.get('num_samples', 0)} samples processed so far")
    return metrics

def get_semantic_error(model, processor, inputs, ground_truth: str, predicted_text: str) -> Dict[str, float]:
    """
    Compute semantic error metrics comparing model confidence in predicted vs ground truth tokens.
    
    Semantic Error = mean(log P(predicted_token) - log P(correct_token))
    
    This captures: How much more confident was the model in its prediction vs the correct answer?
    - Positive value: Model was more confident in wrong tokens
    - Negative value: Model was more confident in correct tokens (but still predicted wrong)
    - Zero: Equal confidence (or perfect prediction)
    
    Also computes KL divergence between the distributions.
    Entropy tells you: Was the model sure about something?
    KL tells you: Was the model right about the correct answer?

    A model can be confidently wrong (low entropy, high KL) or uncertainly right (high entropy, low KL).
    
    Args:
        model: The Qwen3-VL model
        processor: The processor/tokenizer
        inputs: Pre-processed inputs (with image)
        ground_truth: The correct text
        predicted_text: The model's prediction
        
    Returns:
        Dictionary with semantic_error and kl_divergence metrics
    """
    # Wrap the model with nnsight
    nns_model = NNsight(model)
    
    # Forward pass to get last layer activations
    with torch.no_grad():
        with nns_model.trace(**inputs):
            # Get the last layer output
            last_hidden = nns_model.model.language_model.layers[-1].output[0].save()
    
    # Project hidden states to vocabulary logits using the lm_head
    # last_hidden shape: (batch, seq_len, hidden_dim)
    logits = model.lm_head(last_hidden)  # (batch, seq_len, vocab_size)
    
    # Apply softmax to get probabilities and log_softmax for log probs
    probs = F.softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
    
    # Tokenize ground truth and predicted text
    gt_tokens = processor.tokenizer.encode(ground_truth, add_special_tokens=False)
    pred_tokens = processor.tokenizer.encode(predicted_text, add_special_tokens=False)
    
    # Get probabilities at the last input position (first generation step)
    first_pos_probs = probs[0, -1, :]  # (vocab_size,)
    first_pos_log_probs = log_probs[0, -1, :]  # (vocab_size,)
    
    # Compute entropy of the distribution: H = -sum(P * log P)
    # High entropy = model is uncertain, low entropy = model is confident
    entropy = -torch.sum(first_pos_probs * first_pos_log_probs).item()
    
    # Get sorted indices for rank computation (descending by probability)
    sorted_indices = torch.argsort(first_pos_probs, descending=True)
    # Create rank lookup: token_id -> rank (0-indexed, so rank 0 = top prediction)
    rank_lookup = {token_id.item(): rank for rank, token_id in enumerate(sorted_indices)}
    
    # Compute log probability differences for aligned tokens
    # log P(pred) - log P(gt) for each position
    min_len = min(len(gt_tokens), len(pred_tokens), 50)  # Limit to first 50 tokens
    
    log_diffs = []
    kl_components = []
    gt_ranks = []
    
    # Clamp log probs to avoid extreme values (min ~= log(1e-10))
    min_log_prob = -23.0  # Approximately log(1e-10)
    
    for i in range(min_len):
        gt_token = gt_tokens[i]
        pred_token = pred_tokens[i]
        
        log_p_pred = max(first_pos_log_probs[pred_token].item(), min_log_prob)
        log_p_gt = max(first_pos_log_probs[gt_token].item(), min_log_prob)
        
        # Semantic error: log(P_pred) - log(P_gt)
        # Positive = model preferred predicted token
        # Negative = model actually preferred ground truth token
        log_diff = log_p_pred - log_p_gt
        log_diffs.append(log_diff)
        
        # Rank of ground truth token (0 = best, higher = worse)
        gt_rank = rank_lookup.get(gt_token, len(rank_lookup))
        gt_ranks.append(gt_rank)
        
        # KL divergence component: -log P(gt) for one-hot ground truth
        p_gt = first_pos_probs[gt_token].item()
        p_pred = first_pos_probs[pred_token].item()
        
        # Avoid log(0) issues
        eps = 1e-10
        if gt_token != pred_token:
            # KL between one-hot GT and model's distribution at pred position
            kl = -log_p_gt  # Since one-hot, this simplifies to -log P(gt)
            kl_components.append(kl)
    
    # Compute metrics
    if log_diffs:
        semantic_error = sum(log_diffs) / len(log_diffs)
    else:
        semantic_error = 0.0
    
    if kl_components:
        kl_divergence = sum(kl_components) / len(kl_components)
    else:
        kl_divergence = 0.0
    
    if gt_ranks:
        mean_gt_rank = sum(gt_ranks) / len(gt_ranks)
        # Also track how many GT tokens were in top-5
        top5_accuracy = sum(1 for r in gt_ranks if r < 5) / len(gt_ranks)
    else:
        mean_gt_rank = 0.0
        top5_accuracy = 0.0
    
    return {
        "semantic_error": semantic_error,
        "kl_divergence": kl_divergence,
        "entropy": entropy,
        "mean_gt_rank": mean_gt_rank,
        "top5_accuracy": top5_accuracy
    }


def _run_evaluation(model, processor, test_images: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """
    Run evaluation on test set with checkpoint/resume support.
    
    Args:
        model:Qwen3-VL-8B-Instruct model instance
        processor: Qwen3-VL-8B-Instruct processor instance
        image_paths: List of image file paths
        ground_truths: List of ground truth text
    
    Returns:
        Dictionary of metrics
    """
    # Load existing checkpoint
    processed_indices, existing_results = load_checkpoint()
    
    results = existing_results.copy()
    cer_values = []
    wer_values = []
    inference_times = []
    num_errors = 0
    
    # Extract metrics from existing results
    for r in existing_results:
        if r.get("cer") is not None:
            cer_values.append(r["cer"])
        if r.get("wer") is not None:
            wer_values.append(r["wer"])
        if r.get("inference_time") is not None:
            inference_times.append(r["inference_time"])
        if r.get("error") is not None:
            num_errors += 1
    
    new_processed = 0
    
    for idx, (pil_img, ground_truth) in enumerate(zip(test_images, ground_truths)):
        # Skip already processed images
        if idx in processed_indices:
            continue
        
        result = {
            "image_path": idx,
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
            "error": None
        }
        
        try:
            
            # Convert to base64 data URI
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
            
            # Prepare input for Qwen3-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "Extract all text."}
                    ],
                }
            ]
            
            # Perform OCR
            start_time = time.time()

            # Preparation for inference
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=800)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            predicted_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]  # Extract first element since we process one image at a time
            
            inference_time = time.time() - start_time
            
            # Calculate metrics
            cer = character_error_rate(ground_truth, predicted_text)
            wer = word_error_rate(ground_truth, predicted_text)
            semantic_metrics = get_semantic_error(model, processor, inputs, ground_truth, predicted_text)

            result["predicted_text"] = predicted_text
            result["cer"] = cer
            result["wer"] = wer
            result["semantic_error"] = semantic_metrics["semantic_error"]
            result["kl_divergence"] = semantic_metrics["kl_divergence"]
            result["entropy"] = semantic_metrics["entropy"]
            result["mean_gt_rank"] = semantic_metrics["mean_gt_rank"]
            result["top5_accuracy"] = semantic_metrics["top5_accuracy"]
            result["inference_time"] = inference_time
            
            cer_values.append(cer)
            wer_values.append(wer)
            inference_times.append(inference_time)
            
            if (idx + 1) % 100 == 0:
                log_info(MODEL_NAME, f"Processed {idx + 1}/{len(test_images)} images (total), {new_processed} new in this run")
        
        except Exception as e:
            log_warning(MODEL_NAME, f"Error processing image {idx}: {str(e)}")
            result["error"] = str(e)
            num_errors += 1
        
        # Save checkpoint after each image
        save_checkpoint_result(result)
        results.append(result)
        new_processed += 1
        
        # Log progress every 10 new images
        if new_processed % 10 == 0:
            log_info(MODEL_NAME, f"Progress: {len(results)}/{len(test_images)} total, {new_processed} new this run")
    
    # Save JSONL results (final version with all results)
    if len(results) == len(test_images):
        save_results_jsonl(MODEL_NAME, results)
        log_info(MODEL_NAME, "All images processed, saved final results")
    
    # Calculate aggregated metrics
    metrics = {
        "num_samples": len(test_images),
        "num_successful": len(test_images) - num_errors,
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
