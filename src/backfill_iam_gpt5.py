"""
Backfill empty GPT-5 predictions in the IAM experiment.

2,285 of 10,373 IAM lines (22.0%) came back from GPT-5 (via Portkey) with an
empty predicted_text despite a 200 response -- error is null, inference_time
looks normal. GPT-5 is a reasoning model; the original evaluator
(not_flawless_adroit/evaluators/chatgpt_no_logprob.py) requested
max_completion_tokens=1024, which a reasoning model can burn entirely on
internal reasoning tokens before emitting any visible answer. This script
retries exactly those rows with a larger token budget so they can be scored
like everything else, instead of being excluded from semantic scoring.

Uses the same prompt and model id (gpt-5) as the original evaluator for a
fair retry, calling the OpenAI API directly (OPENAI_API_KEY) rather than
through Portkey -- the Portkey key on file returns 401 Invalid API Key, and
a direct OpenAI key was supplied instead. The only other change is a larger
max_completion_tokens.

Usage:
    python src/backfill_iam_gpt5.py \
        --results results/iam_raw/chatgpt5_results.jsonl \
        --iam-dataset /scratch/gpfs/MM4/apjanco/not_flawless_adroit/data/iam \
        --env-file /scratch/gpfs/MM4/apjanco/not_flawless_adroit/.env \
        --max-completion-tokens 4096

Checkpointing: appends to a sidecar backfill file and resumes by index, so
the job is safely restartable. After a run, merges any newly-successful
rows back into --results in place (original empty rows are superseded).
"""

import argparse
import asyncio
import base64
import io
import json
import os
import time
from pathlib import Path

import aiohttp
import jiwer
from datasets import load_from_disk
from dotenv import load_dotenv

MODEL_ID = "gpt-5"
PROMPT = "Please read and transcribe all text in this image. Return only the transcribed text, nothing else."
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MAX_CONCURRENT_REQUESTS = 10


def compute_cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 100.0
    return jiwer.cer(ref, pred) * 100.0


def compute_wer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 100.0
    return jiwer.wer(ref, pred) * 100.0


def load_deduped_by_index(path: Path) -> dict:
    by_index = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_index[rec["index"]] = rec  # last occurrence wins
    return by_index


async def retry_one(session, semaphore, idx, pil_image, ground_truth, headers, max_completion_tokens):
    result = {"index": idx, "ground_truth": ground_truth, "predicted_text": "", "error": None}
    async with semaphore:
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        image_data = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                    ],
                }
            ],
            "max_completion_tokens": max_completion_tokens,
            "logprobs": False,
        }

        try:
            start = time.time()
            async with session.post(
                OPENAI_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status == 429:
                    await asyncio.sleep(60)
                    async with session.post(
                        OPENAI_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)
                    ) as retry_response:
                        response = retry_response
                        body = await response.json()
                else:
                    body = await response.json()

                if response.status != 200:
                    raise RuntimeError(f"API error: {response.status} - {body}")

            predicted_text = body["choices"][0]["message"]["content"].strip()
            result["predicted_text"] = predicted_text
            result["cer"] = compute_cer(predicted_text, ground_truth)
            result["wer"] = compute_wer(predicted_text, ground_truth)
            result["inference_time"] = time.time() - start
        except Exception as exc:
            result["error"] = str(exc)

    return result


async def run_backfill(targets, images, labels, headers, max_completion_tokens, out_path, log_every):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    n_done = 0
    n_recovered = 0

    async with aiohttp.ClientSession() as session:
        tasks = [
            retry_one(session, semaphore, idx, img, gt, headers, max_completion_tokens)
            for idx, img, gt in zip(targets, images, labels)
        ]
        with open(out_path, "a") as fout:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
                if result.get("predicted_text"):
                    n_recovered += 1
                if log_every > 0 and n_done % log_every == 0:
                    print(f"  backfilled {n_done}/{len(targets)}, recovered {n_recovered}")

    print(f"Backfill run complete: {n_done} attempted, {n_recovered} recovered a non-empty prediction.")


def merge_backfill(results_path: Path, backfill_path: Path) -> None:
    """Overwrite empty rows in results_path with successful backfill rows, last-write-wins."""
    by_index = load_deduped_by_index(results_path)

    if backfill_path.exists():
        with open(backfill_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("predicted_text"):
                    # Preserve original schema fields not touched by the retry.
                    merged = dict(by_index[rec["index"]])
                    merged["predicted_text"] = rec["predicted_text"]
                    merged["cer"] = rec["cer"]
                    merged["wer"] = rec["wer"]
                    merged["inference_time"] = rec["inference_time"]
                    merged["error"] = None
                    merged["backfilled"] = True
                    by_index[rec["index"]] = merged

    with open(results_path, "w") as fout:
        for idx in sorted(by_index):
            fout.write(json.dumps(by_index[idx], ensure_ascii=False) + "\n")

    n_backfilled = sum(1 for r in by_index.values() if r.get("backfilled"))
    n_still_empty = sum(1 for r in by_index.values() if not (r.get("predicted_text") or "").strip())
    print(f"Merged: {n_backfilled} rows backfilled. {n_still_empty} rows still empty after backfill.")


def main():
    parser = argparse.ArgumentParser(description="Backfill empty GPT-5 IAM predictions.")
    parser.add_argument("--results", default="results/iam_raw/chatgpt5_results.jsonl")
    parser.add_argument("--iam-dataset", default="/scratch/gpfs/MM4/apjanco/not_flawless_adroit/data/iam")
    parser.add_argument("--env-file", default="/scratch/gpfs/MM4/apjanco/not_flawless_adroit/.env")
    parser.add_argument("--backfill-output", default="results/iam_raw/chatgpt5_backfill.jsonl")
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--merge-only", action="store_true", help="Skip API calls, just merge an existing backfill file.")
    args = parser.parse_args()

    load_dotenv(args.env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not args.merge_only:
        raise SystemExit("OPENAI_API_KEY not found (checked --env-file and environment).")

    results_path = Path(args.results)
    backfill_path = Path(args.backfill_output)

    by_index = load_deduped_by_index(results_path)
    empty_indices = sorted(i for i, r in by_index.items() if not (r.get("predicted_text") or "").strip())
    print(f"{len(empty_indices)} / {len(by_index)} rows have empty predictions.")

    already_attempted = set()
    if backfill_path.exists():
        with open(backfill_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    already_attempted.add(json.loads(line)["index"])
    targets = [i for i in empty_indices if i not in already_attempted]

    if args.merge_only:
        print("Skipping API calls (--merge-only).")
    elif not targets:
        print("Nothing to backfill (all empty rows already attempted).")
    else:
        print(f"Backfilling {len(targets)} rows (resuming; {len(already_attempted)} already attempted this run).")
        print("Loading IAM dataset images...")
        ds = load_from_disk(args.iam_dataset)
        images = [ds[i]["image"] for i in targets]
        labels = [ds[i]["text"] for i in targets]

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        asyncio.run(
            run_backfill(targets, images, labels, headers, args.max_completion_tokens, backfill_path, args.log_every)
        )

    merge_backfill(results_path, backfill_path)


if __name__ == "__main__":
    main()
