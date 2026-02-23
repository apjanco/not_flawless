"""
evaluators/qwen_eval.py
Qwen2-VL 8B vision-language model evaluator for OCR
"""

import time
from pathlib import Path
from typing import Dict, Any, List

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import torch
except ImportError:
    Qwen2VLForConditionalGeneration = None
    AutoProcessor = None
    torch = None

from evaluators.utils import (
    get_data_dir, get_results_dir, load_iam_data,
    character_error_rate, word_error_rate,
    save_metrics, append_metrics_csv, save_results_jsonl,
    log_info, log_error, log_warning
)

MODEL_NAME = "qwen2-vl-8b"

def check_dependencies():
    """Check if required packages are installed"""
    return Qwen2VLForConditionalGeneration is not None and torch is not None

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
        log_info(MODEL_NAME, "Initializing Qwen2-VL 8B model")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-8B-Instruct")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-8B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        log_info(MODEL_NAME, "Model initialized successfully")
    except Exception as e:
        log_error(MODEL_NAME, f"Failed to initialize model: {str(e)}")
        return {"error": str(e)}
    
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
    metrics = _run_evaluation(model, processor, test_images, test_labels)
    
    # Save results
    save_metrics(MODEL_NAME, metrics)
    append_metrics_csv(MODEL_NAME, metrics)
    
    log_info(MODEL_NAME, "Evaluation complete")
    return metrics

def _run_evaluation(model, processor, image_paths: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """
    Run evaluation on test set
    
    Args:
        model: Qwen2-VL model instance
        processor: Qwen2-VL processor instance
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
    
    for idx, (image_path, ground_truth) in enumerate(zip(image_paths, ground_truths)):
        result = {
            "image_path": image_path,
            "ground_truth": ground_truth,
            "predicted_text": None,
            "cer": None,
            "wer": None,
            "inference_time": None,
            "error": None
        }
        
        try:
            from PIL import Image
            
            # Load image
            image = Image.open(image_path)
            
            # Prepare input for Qwen2-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": "Please read and transcribe all text in this image."}
                    ],
                }
            ]
            
            # Perform OCR
            start_time = time.time()
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs = processor(images=[image], return_tensors="pt").to("cuda")
            text_input = processor(text=text, padding=True, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                output_ids = model.generate(
                    **{**image_inputs, **text_input},
                    max_new_tokens=1024,
                )
            
            predicted_text = processor.batch_decode(
                output_ids[:, text_input.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0].strip()
            
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
