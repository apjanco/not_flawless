"""
Inference runner for Churro experiment.

Runs either Qwen 2.5 VL 3B (base) or Churro 3B (fine-tuned) on the test split
of the stanford-oval/churro-dataset and writes per-example predictions to JSONL.

Usage:
    python src/infer.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --output results/qwen_predictions.jsonl

    python src/infer.py \
        --model stanford-oval/churro-3B \
        --output results/churro_predictions.jsonl \
        --tensor_parallel_size 1

Checkpointing: if --output already exists, already-processed image_path values
are skipped, so the job can be safely restarted after preemption.
"""

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Exact prompt from Churro paper Appendix B (Table 6)
PROMPT = (
    "You are an expert in diplomatic transcription of historical documents from "
    "various languages. Your task is to extract the full text from a given page. "
    "Only output the transcribed text between <answer> and </answer> tags.\n"
    "Follow these instructions:\n"
    "1. You will be provided with a scanned document page.\n"
    "2. Perform transcription on the entirety of the page, converting all visible "
    "text into the following format. Include handwritten and print text, if any. "
    "Include tables, captions, headers, main text and all other visible text.\n"
    "3. If you encounter any non-text elements, simply skip them without attempting "
    "to describe them.\n"
    "4. Do not modernize or standardize the text. For example, if the transcription "
    "is using \"\u017f\" instead of \"s\" or \"\u0430\" instead of \"a\", keep it "
    "that way.\n"
    "5. When you come across text in languages other than English, transcribe it as "
    "accurately as possible without translation.\n"
    "6. Output the OCR result in the following format:\n"
    "<answer>extracted text here</answer>"
)

# Metadata columns to carry through to output
META_COLS = ["main_language", "languages", "main_script", "scripts", "document_type"]


def extract_answer(text: str) -> str:
    """Pull text between <answer>…</answer>; fall back to full output."""
    import re
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def format_prompt(processor: AutoProcessor, image: Image.Image) -> str:
    """Apply the model's chat template around the image + instruction prompt."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        done.add(rec["image_path"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def main():
    parser = argparse.ArgumentParser(description="Run VLM inference on churro-dataset test split.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--max_new_tokens", type=int, default=20000,
        help="Max tokens to generate per example (20k covers the longest pages)",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=32768,
        help="Maximum sequence length (input + output tokens)",
    )
    parser.add_argument(
        "--log_every", type=int, default=50,
        help="Print progress every N examples",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset (test split)...")
    ds = load_dataset("stanford-oval/churro-dataset", split="test", trust_remote_code=True)
    print(f"  {len(ds)} examples")

    print(f"Loading processor for {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)

    print(f"Loading model {args.model} with vLLM...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        download_dir=None,  # use HF_HOME cache only
    )

    sampling_params = SamplingParams(
        temperature=0,  # greedy decoding (matches Churro paper)
        max_tokens=args.max_new_tokens,
    )

    done_ids = load_done_ids(out_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} examples already done, skipping.")

    n_done = 0
    n_errors = 0

    with open(out_path, "a") as fout:
        for i, example in enumerate(ds):
            # image_path may be an integer index or a string identifier
            img_id = example.get("image_path", i)

            if img_id in done_ids:
                continue

            image: Image.Image = example["image"]

            t0 = time.time()
            try:
                prompt_text = format_prompt(processor, image)
                outputs = llm.generate(
                    [{"prompt": prompt_text, "multi_modal_data": {"image": image}}],
                    sampling_params,
                )
                predicted = extract_answer(outputs[0].outputs[0].text)
                error = None
                n_done += 1
            except Exception as exc:
                predicted = ""
                error = str(exc)
                n_errors += 1

            elapsed = time.time() - t0

            record: dict = {
                "image_path": img_id,
                "ground_truth": example.get("cleaned_transcription", ""),
                "predicted_text": predicted,
                "inference_time": elapsed,
                "error": error,
            }
            for col in META_COLS:
                if col in example and example[col] is not None:
                    record[col] = example[col]

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if (i + 1) % args.log_every == 0:
                print(
                    f"[{i + 1}/{len(ds)}] done={n_done} errors={n_errors} "
                    f"last_time={elapsed:.1f}s"
                )

    print(f"\nFinished. {n_done} completed, {n_errors} errors.")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
