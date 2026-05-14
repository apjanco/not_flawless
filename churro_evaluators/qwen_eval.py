"""
churro_evaluators/qwen_eval.py
Qwen3-VL-8B-Instruct vision-language model evaluator for OCR using vLLM
"""
import io
import json
import os
import base64
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Set

from datasets import load_dataset

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from evaluators.utils import (
    get_data_dir, get_results_dir,
    character_error_rate, word_error_rate,
    save_metrics, append_metrics_csv, save_results_jsonl,
    log_info, log_error, log_warning
)

MODEL_NAME = "Qwen3-VL-8B-Instruct-churro"
LOCAL_MODEL_PATH = "/scratch/network/aj7878/not_flawless/models/Qwen3-VL-8B-Instruct"
RESULTS_DIR = "/scratch/network/aj7878/not_flawless/results"

# Sharding: set QWEN_SHARD_ID (0-based) and QWEN_NUM_SHARDS via env vars.
# Each shard processes indices where idx % NUM_SHARDS == SHARD_ID.
SHARD_ID = int(os.environ.get("QWEN_SHARD_ID", "0"))
NUM_SHARDS = int(os.environ.get("QWEN_NUM_SHARDS", "1"))

_shard_suffix = f"_shard{SHARD_ID}of{NUM_SHARDS}" if NUM_SHARDS > 1 else ""
CHECKPOINT_FILE = f"{RESULTS_DIR}/{MODEL_NAME}{_shard_suffix}_checkpoint.jsonl"

BATCH_SIZE = 8
MAX_TOKENS = 800
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.9


def extract_text_from_transcription(xml_string):
    lines = []
    for line_content in re.findall(r'<Line>(.*?)</Line>', xml_string, re.DOTALL):
        text = re.sub(r'<[^>]+>', '', line_content).strip()
        if text:
            lines.append(text)
    return '\n'.join(lines)


def check_dependencies():
    return LLM is not None


def load_checkpoint() -> tuple[Set[int], List[Dict]]:
    processed_indices = set()
    existing_results = []

    # When sharding, also load the original single-shard checkpoint (if it exists)
    # so we don't re-process work done before sharding was introduced.
    candidate_files = [Path(CHECKPOINT_FILE)]
    if NUM_SHARDS > 1:
        original = Path(f"{RESULTS_DIR}/{MODEL_NAME}_checkpoint.jsonl")
        if original != Path(CHECKPOINT_FILE):
            candidate_files.append(original)

    for checkpoint_path in candidate_files:
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            if result["image_path"] not in processed_indices:
                                existing_results.append(result)
                                processed_indices.add(result["image_path"])
            except Exception as e:
                log_warning(MODEL_NAME, f"Failed to load checkpoint {checkpoint_path}: {str(e)}, skipping")

    if processed_indices:
        log_info(MODEL_NAME, f"Loaded checkpoint with {len(processed_indices)} already processed images")

    return processed_indices, existing_results


def save_checkpoint_result(result: Dict):
    checkpoint_path = Path(CHECKPOINT_FILE)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'a') as f:
        f.write(json.dumps(result) + '\n')


def evaluate(project_root: str = None) -> Dict[str, Any]:
    """
    Main evaluation function.

    Args:
        project_root: Path to project root

    Returns:
        Dictionary of evaluation metrics
    """
    log_info(MODEL_NAME, "Starting evaluation")

    if not check_dependencies():
        log_error(MODEL_NAME, "vllm is not installed")
        return {"error": "Dependencies not available"}

    try:
        log_info(MODEL_NAME, f"Initializing vLLM with model from {LOCAL_MODEL_PATH}")
        llm = LLM(
            model=LOCAL_MODEL_PATH,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            limit_mm_per_prompt={"image": 1},
        )
        log_info(MODEL_NAME, "vLLM model initialized successfully")
    except Exception as e:
        log_error(MODEL_NAME, f"Failed to initialize model: {str(e)}")
        return {"error": str(e)}

    try:
        ds = load_dataset("stanford-oval/churro-dataset", split="train")
        log_info(MODEL_NAME, f"Loaded dataset with {len(ds)} samples")
    except Exception as e:
        log_error(MODEL_NAME, f"Failed to load dataset: {str(e)}")
        return {"error": str(e)}

    if len(ds) == 0:
        log_warning(MODEL_NAME, "No test images found")
        return {"error": "No test data"}

    metrics = _run_evaluation(llm, ds)

    if metrics.get("num_successful", 0) + metrics.get("num_errors", 0) == metrics.get("num_samples", 0):
        save_metrics(MODEL_NAME, metrics)
        append_metrics_csv(MODEL_NAME, metrics)
        log_info(MODEL_NAME, "Evaluation complete - all samples processed")
    else:
        log_info(MODEL_NAME, f"Evaluation paused - {metrics.get('num_successful', 0) + metrics.get('num_errors', 0)}/{len(ds)} samples processed so far")

    return metrics


def _run_evaluation(llm, ds) -> Dict[str, Any]:
    """
    Run batched evaluation with checkpoint/resume support.

    Args:
        llm: Initialized vLLM LLM instance
        ds: HuggingFace dataset

    Returns:
        Dictionary of metrics
    """
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    processed_indices, existing_results = load_checkpoint()
    results = existing_results.copy()
    cer_values, wer_values, inference_times = [], [], []
    num_errors = 0

    for r in existing_results:
        if r.get("cer") is not None:
            cer_values.append(r["cer"])
        if r.get("wer") is not None:
            wer_values.append(r["wer"])
        if r.get("inference_time") is not None:
            inference_times.append(r["inference_time"])
        if r.get("error") is not None:
            num_errors += 1

    num_remaining = sum(
        1 for idx in range(len(ds))
        if idx % NUM_SHARDS == SHARD_ID and idx not in processed_indices
    )
    log_info(MODEL_NAME, f"Shard {SHARD_ID}/{NUM_SHARDS}: {num_remaining} samples remaining to process")

    batch_messages = []
    batch_meta = []  # (idx, ground_truth)
    batch_num = 0
    total_batches = (num_remaining + BATCH_SIZE - 1) // BATCH_SIZE

    def flush_batch():
        nonlocal batch_num, num_errors
        if not batch_messages:
            return
        batch_num += 1
        start_time = time.time()
        try:
            outputs = llm.chat(batch_messages, sampling_params)
            batch_time = time.time() - start_time
            per_image_time = batch_time / len(batch_messages)
        except Exception as e:
            log_error(MODEL_NAME, f"Batch {batch_num} inference failed: {e}")
            for idx, ground_truth in batch_meta:
                result = {
                    "image_path": idx,
                    "ground_truth": ground_truth,
                    "predicted_text": None,
                    "cer": None,
                    "wer": None,
                    "inference_time": None,
                    "error": str(e),
                }
                save_checkpoint_result(result)
                results.append(result)
                num_errors += 1
            batch_messages.clear()
            batch_meta.clear()
            return

        for (idx, ground_truth), output in zip(batch_meta, outputs):
            try:
                predicted_text = output.outputs[0].text.strip()
                cer = character_error_rate(ground_truth, predicted_text)
                wer = word_error_rate(ground_truth, predicted_text)
                result = {
                    "image_path": idx,
                    "ground_truth": ground_truth,
                    "predicted_text": predicted_text,
                    "cer": cer,
                    "wer": wer,
                    "inference_time": per_image_time,
                    "error": None,
                }
                cer_values.append(cer)
                wer_values.append(wer)
                inference_times.append(per_image_time)
            except Exception as e:
                log_warning(MODEL_NAME, f"Failed to process output for image {idx}: {e}")
                result = {
                    "image_path": idx,
                    "ground_truth": ground_truth,
                    "predicted_text": None,
                    "cer": None,
                    "wer": None,
                    "inference_time": None,
                    "error": str(e),
                }
                num_errors += 1
            save_checkpoint_result(result)
            results.append(result)

        log_info(MODEL_NAME, f"Batch {batch_num}/{total_batches} done — {len(results)}/{len(ds)} total processed")
        batch_messages.clear()
        batch_meta.clear()

    for idx, sample in enumerate(ds):
        if idx % NUM_SHARDS != SHARD_ID:
            continue
        if idx in processed_indices:
            continue

        pil_img = sample['image']
        ground_truth = extract_text_from_transcription(sample['cleaned_transcription'])

        # PNG requires RGB/RGBA; convert CMYK, P, L, etc.
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")

        # Resize so the longest side is at most MAX_IMAGE_DIM pixels.
        # Qwen2-VL uses 28x28 px per visual token; a 1120px image gives ~1600 tokens,
        # leaving plenty of headroom in the 8192-token context window.
        MAX_IMAGE_DIM = 1120
        w, h = pil_img.size
        if max(w, h) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(w, h)
            pil_img = pil_img.resize(
                (int(w * scale), int(h * scale)),
                resample=2,  # PIL.Image.LANCZOS
            )

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "Extract all text."},
                ],
            }
        ]
        batch_messages.append(messages)
        batch_meta.append((idx, ground_truth))

        if len(batch_messages) >= BATCH_SIZE:
            flush_batch()

    flush_batch()  # flush any remaining partial batch

    shard_model_name = f"{MODEL_NAME}{_shard_suffix}"
    shard_total = sum(1 for idx in range(len(ds)) if idx % NUM_SHARDS == SHARD_ID)
    if len(results) >= shard_total:
        save_results_jsonl(shard_model_name, results)
        log_info(MODEL_NAME, f"Shard {SHARD_ID}/{NUM_SHARDS} complete, saved final results")

    metrics = {
        "num_samples": len(ds),
        "num_successful": len(results) - num_errors,
        "num_errors": num_errors,
        "mean_cer": sum(cer_values) / len(cer_values) if cer_values else None,
        "median_cer": sorted(cer_values)[len(cer_values) // 2] if cer_values else None,
        "mean_wer": sum(wer_values) / len(wer_values) if wer_values else None,
        "median_wer": sorted(wer_values)[len(wer_values) // 2] if wer_values else None,
        "mean_inference_time": sum(inference_times) / len(inference_times) if inference_times else None,
        "total_inference_time": sum(inference_times) if inference_times else None,
    }
    return metrics


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    result = evaluate(str(project_root))
