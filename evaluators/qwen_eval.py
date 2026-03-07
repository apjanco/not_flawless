"""
evaluators/qwen_eval.py
Qwen3-VL-8B-Instruct vision-language model evaluator for OCR
"""
import io
import base64
import time
from pathlib import Path
from typing import Dict, Any, List
from datasets import load_from_disk

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

def check_dependencies():
    """Check if required packages are installed"""
    return Qwen3VLForConditionalGeneration is not None and torch is not None

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
    
    # Save results
    save_metrics(MODEL_NAME, metrics)
    append_metrics_csv(MODEL_NAME, metrics)
    
    log_info(MODEL_NAME, "Evaluation complete")
    return metrics

def _run_evaluation(model, processor, test_images: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """
    Run evaluation on test set
    
    Args:
        model:Qwen3-VL-8B-Instruct model instance
        processor: Qwen3-VL-8B-Instruct processor instance
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
    
    for idx, (pil_img, ground_truth) in enumerate(zip(test_images, ground_truths)):
        result = {
            "image_path": idx,
            "ground_truth": ground_truth,
            "predicted_text": None,
            "cer": None,
            "wer": None,
            "inference_time": None,
            "error": None
        }
        
        try:
            
            # Convert to base64 data URI
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
            
            # Prepare input for Qwen2-VL
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
            
            result["predicted_text"] = predicted_text
            result["cer"] = cer
            result["wer"] = wer
            result["inference_time"] = inference_time
            
            cer_values.append(cer)
            wer_values.append(wer)
            inference_times.append(inference_time)
            
            if (idx + 1) % 100 == 0:
                log_info(MODEL_NAME, f"Processed {idx + 1}/{len(test_images)} images")
        
        except Exception as e:
            log_warning(MODEL_NAME, f"Error processing image {idx}: {str(e)}")
            result["error"] = str(e)
            num_errors += 1
        
        results.append(result)
    
    # Save JSONL results
    save_results_jsonl(MODEL_NAME, results)
    
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
