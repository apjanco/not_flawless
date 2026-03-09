"""
evaluators/chatgpt_eval.py
ChatGPT (via OpenAI API via Portkey) OCR evaluator with logprobs and semantic error analysis
"""

import time
import os
import math
import io
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import requests
except ImportError:
    requests = None

try:
    from datasets import load_from_disk
except ImportError:
    load_from_disk = None

from evaluators.utils import (
    get_data_dir, get_results_dir,
    character_error_rate, word_error_rate,
    save_metrics, append_metrics_csv, save_results_jsonl,
    log_info, log_error, log_warning
)

MODEL_NAME = "chatgpt-vision"
MODEL_ID = "gpt-4o"

def check_dependencies():
    """Check if required packages are installed"""
    return requests is not None

def get_portkey_key():
    """Get Portkey API key from environment"""
    api_key = os.getenv("PORTKEY_API_KEY")
    if not api_key:
        log_warning(MODEL_NAME, "PORTKEY_API_KEY environment variable not set")
    return api_key

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

    # Run evaluation
    metrics = _run_evaluation(api_key, test_images, test_labels, top_logprobs)

    # Save results
    save_metrics(MODEL_NAME, metrics)
    append_metrics_csv(MODEL_NAME, metrics)

    log_info(MODEL_NAME, "Evaluation complete")
    return metrics

def _run_evaluation(api_key: str, test_images: List, test_labels: List, top_logprobs: int = 5) -> Dict[str, Any]:
    """
    Run evaluation on test set via Portkey → OpenAI.

    Args:
        api_key: Portkey API key
        test_images: List of PIL Image objects
        test_labels: List of ground truth text labels
        top_logprobs: Number of top alternative tokens to request

    Returns:
        Dictionary of aggregated metrics
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

    prompt = "Please read and transcribe all text in this image. Return only the transcribed text, nothing else."

    num_samples = len(test_images)
    for idx, (pil_image, ground_truth) in enumerate(zip(test_images, test_labels)):
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
                "max_tokens": 1024,
                "logprobs": True,
                "top_logprobs": top_logprobs
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

            # Extract logprobs (OpenAI format: choices[0].logprobs.content)
            logprobs_data = None
            raw_logprobs = response_data["choices"][0].get("logprobs", {})
            if raw_logprobs and raw_logprobs.get("content"):
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

            cer_values.append(cer)
            wer_values.append(wer)
            inference_times.append(inference_time)

            if (idx + 1) % 10 == 0:
                log_info(MODEL_NAME, f"Processed {idx + 1}/{num_samples} samples")

        except Exception as e:
            log_warning(MODEL_NAME, f"Error processing sample {idx}: {str(e)}")
            result["error"] = str(e)
            num_errors += 1

        results.append(result)

    # Save JSONL results
    save_results_jsonl(MODEL_NAME, results)

    # Calculate aggregated metrics
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
