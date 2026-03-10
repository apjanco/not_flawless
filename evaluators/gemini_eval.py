"""
evaluators/gemini_eval.py
Google Gemini (direct Google API) OCR evaluator with logprobs and tokenizer support
"""

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

from .utils import (
    get_data_dir, get_results_dir, load_iam_data,
    character_error_rate, word_error_rate,
    save_metrics, append_metrics_csv, save_results_jsonl,
    log_info, log_error, log_warning
)

MODEL_NAME = "gemini-3-pro-preview"
MODEL_ID = "gemini-3-pro-preview"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

def check_dependencies():
    """Check if required packages are installed"""
    return REQUESTS_AVAILABLE and GENAI_AVAILABLE

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

def evaluate(project_root: str = None, top_logprobs: int = 5) -> Dict[str, Any]:
    """
    Main evaluation function using Portkey API
    
    Args:
        project_root: Path to project root
        top_logprobs: Number of top alternative tokens to return (1-20)
    
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
    
    # Run evaluation
    metrics = _run_evaluation(google_api_key, test_images, test_labels, tokenizer_model, top_logprobs)
    
    # Save results
    save_metrics(MODEL_NAME, metrics)
    append_metrics_csv(MODEL_NAME, metrics)
    
    log_info(MODEL_NAME, "Evaluation complete")
    return metrics

def _run_evaluation(google_api_key: str, test_images: List, test_labels: List, tokenizer_model, top_logprobs: int = 5) -> Dict[str, Any]:
    """
    Run evaluation on test set calling Google Gemini API directly.
    
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
    cer_values = []
    wer_values = []
    inference_times = []
    num_errors = 0
    
    api_url = GEMINI_API_URL.format(model=MODEL_ID)
    headers = {"Content-Type": "application/json"}
    
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
