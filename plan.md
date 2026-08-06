# Plan: Semantic Error in Churro Experiment

## Overview

This experiment measures semantic error rates in historical document OCR/HTR by comparing:
- **Churro 3B** (`stanford-oval/churro-3B`): Qwen 2.5 VL fine-tuned on 97k historical pages
- **Qwen 2.5 VL 3B** (`Qwen/Qwen2.5-VL-3B-Instruct`): the zero-shot base model

Evaluation uses the test split of `stanford-oval/churro-dataset` (1,170 examples across 29 language clusters) and the `misnomer` library for semantic error scoring alongside standard CER/WER.

---

## Research Questions

1. Does fine-tuning reduce semantic error rates (not just CER/WER)?
2. How does semantic error vary across languages and scripts?
3. Do semantic errors correlate with, or diverge from, traditional CER/WER findings?
4. What error types appear distinctively in a multilingual historical corpus (hallucination, repetitive degeneration, reading-order errors, script normalization)?

---

## File Structure

```
not_flawless/
├── SPEC.md
├── plan.md                        # this file
├── environment.yml                # conda env spec
├── scripts/
│   ├── 00_setup.sh                # one-time env setup on Della
│   ├── 01_infer_qwen.sh           # SLURM job: Qwen 2.5 VL 3B inference
│   ├── 01_infer_churro.sh         # SLURM job: Churro 3B inference
│   ├── 02_score_semantic.sh       # SLURM job: misnomer scoring
│   └── 03_analyze.sh              # SLURM job: analysis / figures
├── src/
│   ├── infer.py                   # shared inference runner (model-agnostic)
│   ├── score.py                   # misnomer scoring + CER/WER
│   └── analyze.py                 # aggregation, plots, tables
├── results/
│   ├── qwen_predictions.jsonl     # per-example inference output
│   ├── churro_predictions.jsonl   # per-example inference output
│   ├── qwen_scored.jsonl          # predictions + semantic scores
│   ├── churro_scored.jsonl        # predictions + semantic scores
│   └── figures/
└── notebooks/
    └── analysis.ipynb             # exploratory analysis
```

---

## Step 0 — Environment Setup

### Conda / module environment on Della

```bash
# scripts/00_setup.sh
module purge
module load anaconda3/2025.12

# Create the environment from environment.yml (single source of truth).
# To update an existing env: conda env update --prune -f environment.yml
conda env create -f environment.yml
conda activate churro_exp
```

> **Note:** vLLM is used by the Churro paper for inference. It greatly speeds up batched generation and supports multi-GPU tensor parallelism. Confirm the Della GPU partition (e.g., `gpu` or `pli`) and available CUDA version before installing.

### HuggingFace cache

```bash
export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
huggingface-cli login   # only needed if model is gated
```

Pre-fetch the test split and both models before submitting jobs to avoid repeated downloads inside jobs:

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('stanford-oval/churro-dataset', split='test')
print(ds)
"
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
huggingface-cli download stanford-oval/churro-3B
```

---

## Step 1 — Inference

### `src/infer.py`

Single script parameterised by model ID so both jobs share the same code.

```python
"""
Usage:
    python src/infer.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --output results/qwen_predictions.jsonl \
        --batch_size 4 \
        --max_new_tokens 20000
"""
import argparse, json, time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

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
    "is using \"U+017F\" instead of \"s\" or \"U+0430\" instead of \"a\", keep it "
    "that way.\n"
    "5. When you come across text in languages other than English, transcribe it as "
    "accurately as possible without translation.\n"
    "6. Output the OCR result in the following format:\n"
    "<answer>extracted text here</answer>"
)

# --- metadata columns to retain ---
META_COLS = ["main_language", "languages", "main_script", "scripts"]

def extract_answer(text: str) -> str:
    import re
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=20000)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    ds = load_dataset("stanford-oval/churro-dataset", split="test")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 1},
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["image_path"])

    with open(out_path, "a") as fout:
        for i, example in enumerate(ds):
            img_id = example["image_path"] if "image_path" in example else i
            if img_id in done_ids:
                continue

            image = example["image"]
            t0 = time.time()
            try:
                outputs = llm.generate(
                    [{"prompt": PROMPT, "multi_modal_data": {"image": image}}],
                    sampling_params,
                )
                predicted = extract_answer(outputs[0].outputs[0].text)
                error = None
            except Exception as e:
                predicted = ""
                error = str(e)

            elapsed = time.time() - t0

            record = {
                "image_path": img_id,
                "ground_truth": example["ground_truth"],
                "predicted_text": predicted,
                "inference_time": elapsed,
                "error": error,
            }
            for col in META_COLS:
                if col in example:
                    record[col] = example[col]

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if i % 50 == 0:
                print(f"[{i}/{len(ds)}] done")

if __name__ == "__main__":
    main()
```

### SLURM job scripts

**`scripts/01_infer_qwen.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=infer_qwen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/infer_qwen_%j.out

module purge
module load anaconda3/2025.12
conda activate churro_exp

python src/infer.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --output results/qwen_predictions.jsonl \
    --batch_size 4 \
    --tensor_parallel_size 1
```

**`scripts/01_infer_churro.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=infer_churro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/infer_churro_%j.out

module purge
module load anaconda3/2025.12
conda activate churro_exp

python src/infer.py \
    --model stanford-oval/churro-3B \
    --output results/churro_predictions.jsonl \
    --batch_size 4 \
    --tensor_parallel_size 1
```

> **Tip:** Both jobs can run simultaneously on separate GPU allocations. Each takes an estimated 2–4 hours for 1,170 examples at ~4–5 s/example.

---

## Step 2 — CER/WER + Semantic Scoring

### `src/score.py`

```python
"""
Usage:
    python src/score.py \
        --input results/qwen_predictions.jsonl \
        --output results/qwen_scored.jsonl
"""
import argparse, json
from pathlib import Path
import jiwer
import misnomer
from misnomer import ScorerConfig

# Use the multilingual embedder since texts span 29 language clusters
SCORER_CONFIG = ScorerConfig(
    embedder_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    lm_model="Qwen/Qwen2.5-0.5B",
)

def cer(pred: str, ref: str) -> float:
    return jiwer.cer(ref, pred) * 100

def wer(pred: str, ref: str) -> float:
    return jiwer.wer(ref, pred) * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["image_path"])

    with open(in_path) as fin, open(out_path, "a") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec["image_path"] in done_ids:
                continue

            pred = rec.get("predicted_text", "") or ""
            gt = rec.get("ground_truth", "") or ""

            # Standard metrics
            rec["cer"] = cer(pred, gt)
            rec["wer"] = wer(pred, gt)

            # Semantic error scoring
            if pred and gt:
                try:
                    report = misnomer.score(
                        predicted=pred,
                        ground_truth=gt,
                        config=SCORER_CONFIG,
                    )
                    rec["semantic_error_count"] = report.semantic_error_count
                    rec["obvious_error_count"] = report.obvious_error_count
                    rec["semantic_has_error"] = report.semantic_error_count > 0
                    rec["semantic_document_score"] = report.document_score
                    rec["semantic_document_error_type"] = getattr(report, "document_error_type", None)
                    rec["semantic_document_embedding_similarity"] = getattr(
                        report, "document_embedding_similarity", None
                    )
                    rec["semantic_scorer_mode"] = report.scorer_mode
                    rec["semantic_scorer_version"] = getattr(report, "scorer_version", None)
                    rec["semantic_lm_model"] = getattr(report, "lm_model", None)
                    rec["semantic_embedder_model"] = getattr(report, "embedder_model", None)
                except Exception as e:
                    rec["semantic_score_error"] = str(e)
            else:
                rec["semantic_error_count"] = 0
                rec["obvious_error_count"] = 0
                rec["semantic_has_error"] = False

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

if __name__ == "__main__":
    main()
```

**`scripts/02_score_semantic.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=score_semantic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/score_%j.out

module purge
module load anaconda3/2025.12
conda activate churro_exp

python src/score.py --input results/qwen_predictions.jsonl --output results/qwen_scored.jsonl &
python src/score.py --input results/churro_predictions.jsonl --output results/churro_scored.jsonl &
wait
```

---

## Step 3 — Analysis

### `src/analyze.py`

Key analyses to run:

```python
import json, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

def load_jsonl(path):
    return pd.DataFrame([json.loads(l) for l in Path(path).read_text().splitlines()])

qwen = load_jsonl("results/qwen_scored.jsonl")
churro = load_jsonl("results/churro_scored.jsonl")

qwen["model"] = "Qwen 2.5 VL 3B (base)"
churro["model"] = "Churro 3B (fine-tuned)"
df = pd.concat([qwen, churro], ignore_index=True)

# --- 1. Aggregate summary table (mirrors Churro paper Table 1/2) ---
summary = df.groupby(["model", "main_language"]).agg(
    cer=("cer", "mean"),
    wer=("wer", "mean"),
    semantic_error_count=("semantic_error_count", "mean"),
    semantic_document_score=("semantic_document_score", "mean"),
    semantic_has_error_rate=("semantic_has_error", "mean"),
    n=("image_path", "count"),
).reset_index()

# --- 2. Fine-tuning delta: semantic error improvement ---
pivot_sem = summary.pivot(index="main_language", columns="model",
                           values="semantic_document_score")
pivot_sem["delta"] = pivot_sem["Churro 3B (fine-tuned)"] - pivot_sem["Qwen 2.5 VL 3B (base)"]
# positive delta = Churro is better (higher semantic document score = fewer errors)

# --- 3. Correlation: CER vs semantic error rate ---
# Does fine-tuning that improves CER always improve semantic error too?
# Are there cases where CER improves but semantic error does not (or vice versa)?

# --- 4. Script-level analysis ---
by_script = df.groupby(["model", "main_script"]).agg(
    cer=("cer", "mean"),
    semantic_document_score=("semantic_document_score", "mean"),
).reset_index()

# --- 5. Error taxonomy breakdown ---
# semantic vs obvious errors as proportion
df["semantic_ratio"] = df["semantic_error_count"] / (
    df["semantic_error_count"] + df["obvious_error_count"] + 1e-9
)
```

**Suggested plots:**
1. Side-by-side bar chart: mean semantic error rate by language, Qwen vs Churro
2. Scatter: CER vs `semantic_document_score` colored by model — tests whether CER predicts semantic quality
3. Heatmap: `semantic_has_error_rate` by `main_script` × model
4. Histogram: `semantic_error_count` distribution per model
5. Diverging bar: fine-tuning delta in semantic error rate by language (complement to Churro paper's Table 1/2)

---

## Output Schema

Each record in `*_scored.jsonl` matches the example in SPEC.md:

```json
{
  "image_path": 12044,
  "ground_truth": "...",
  "predicted_text": "...",
  "cer": 75.61,
  "wer": 86.44,
  "inference_time": 4.30,
  "error": null,
  "semantic_error_count": 157,
  "obvious_error_count": 9,
  "semantic_has_error": true,
  "semantic_document_score": 0.629,
  "semantic_document_error_type": "partial",
  "semantic_document_embedding_similarity": 0.814,
  "semantic_scorer_mode": "full",
  "semantic_scorer_version": "1.0",
  "semantic_lm_model": "Qwen/Qwen2.5-0.5B",
  "semantic_embedder_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "languages": ["German"],
  "main_language": "German",
  "main_script": "Latin",
  "scripts": ["Latin"]
}
```

---

## Suggestions for Improving the Experiment

### 1. Use the multilingual embedder from the start
The default misnomer embedder (`all-MiniLM-L6-v2`) is English-centric. The SPEC example already uses `paraphrase-multilingual-MiniLM-L12-v2`, which is appropriate here. Confirm this is set in `ScorerConfig` (see `score.py` above).

### 2. Stratify by document type (printed vs handwritten)
The Churro dataset includes a `document_type` or equivalent column (printed/handwritten). The Churro paper shows very different performance patterns between these two subsets. Splitting the semantic error analysis the same way will make comparison to Table 1 vs Table 2 more direct.

### 3. Track semantic errors that CER misses
Compute a "semantic error delta" — examples where CER is low but `semantic_error_count` is high. These are the most interesting cases: plausible-sounding but factually wrong transcriptions. A concrete definition:

```python
df["missed_by_cer"] = (df["cer"] < 10) & (df["semantic_error_count"] > 5)
```

These cases are the core novel contribution of the experiment.

### 4. Sample examples for qualitative analysis
After scoring, identify representative examples in each error regime for human inspection and HTML highlighting via `misnomer.highlight()`. Suggested strata:
- Low CER, high semantic error (fine-tuning "fixed" surface but introduced semantic drift)
- High CER, low semantic error (many typos but semantically coherent)
- Both models agree / disagree on semantic quality

### 5. Normalize for text length
`semantic_error_count` is an absolute count; divide by ground-truth word count to get a semantic error rate per word, enabling fair comparison across short and long pages.

### 6. Consider adding a third model for context
If compute allows, adding Qwen 2.5 VL 72B (also benchmarked in the Churro paper) would test whether scale alone reduces semantic errors, independent of fine-tuning.

### 7. Checkpointing is critical
Both inference and scoring are long-running jobs. The scripts above append to output files and skip already-processed IDs on restart. Use `--dependency=afterok:<job_id>` in SLURM to chain steps automatically.

### 8. Della-specific considerations
- Request `--gres=gpu:a100:1` or `gpu:1` depending on partition config. A100 80GB is preferred for vLLM with 3B models.
- Set `TRANSFORMERS_CACHE` and `HF_HOME` to scratch space to avoid filling `/home`.
- The scoring step (`score.py`) also needs a GPU for the transformer LM; it cannot run on a CPU-only node in `full` mode.
- Consider using `--array` jobs if you want to parallelize scoring across language groups.

---

## Known Limitations — misnomer v1.1 scoring performance (2026-07-25)

Semantic scoring on Churro documents is dramatically slower under misnomer
v1.1 than the original "3–5h" estimate below assumed, and slower than the
same scoring step ran under misnomer's pre-v1.1 code (see below). Root
cause, mitigation applied, and why we're not fixing the underlying issue
right now:

**Root cause**: `LMScorer.substitution_surprisals` (`misnomer/src/misnomer/models/lm.py`)
computes a full-word, teacher-forced Δ (delta) for every word-level
substitution between prediction and ground truth. For each substituted
word it rebuilds the ground-truth prefix from scratch (`" ".join(ground_truth_words[:i])`)
and runs an independent full forward pass over it — no KV-cache or hidden-state
reuse across substitution positions within the same document. Cost is
roughly O(substitutions × document_length) per document. This code path is
gated by `cfg.gates_active` (`scorer_version >= "1.1"`, our default), not by
`use_surprisal_contrast` as its docstring implies — that flag only affects
how the *result* is used afterward, not whether the expensive computation
runs.

IAM never hit this (lines are short, tens of words). Churro documents are
full pages (predicted-text length up to 94,840 characters), so this became
the practical bottleneck: an isolated measurement (job 11573709, 2026-07-24,
128G mem, no OOM) processed only **173 qwen rows in 8h05m** (≈21.4 rows/hour
on 1 A100) — compare to a June 2026 Adroit-cluster run of an older,
pre-v1.1 misnomer that scored 21,320+ rows in a single job. That older run
used a different (larger) dataset and predates the v1.1 delta machinery
entirely, so it isn't a strict apples-to-apples comparison, but it confirms
the slowness is specific to the new per-substitution computation, not an
inherent property of scoring long documents.

Zero-shot Qwen2.5-VL-3B predictions are hit hardest: worse zero-shot
transcription quality means more word-level substitutions per document,
and the cost scales with substitution count. Churro-3B's predictions (fewer
errors, shorter median length: 517 vs. 1,671 chars) score noticeably faster.

**The efficient fix** (not implemented — see below) would be a single
teacher-forced forward pass over the full ground-truth sequence per
document, computing per-position logits once, then a cheap incremental
lookup per substitution for the predicted word's probability at that
position under the shared cached prefix — the same approach the pre-v1.1
`word_perplexities` path already uses correctly. `substitution_surprisals`
appears to have regressed away from that.

**Mitigation applied**: sharded scoring across 4 parallel single-GPU SLURM
jobs (`src/shard_prep.py`, `scripts/02b_score_semantic_shard.sh`,
`scripts/submit_semantic_shards.sh`), splitting `qwen_predictions.jsonl`
and `churro_predictions.jsonl` by `image_path % 4` and pre-seeding each
shard's output from already-valid scored rows so no completed work is
redone. This does not fix the per-row cost, only parallelizes around it.

**Decision**: this is a research project — we're accepting the slow path to
keep misnomer v1.1's gate taxonomy (`numeric_error_count`, `garble_count`,
`homoglyph_count`, `segmentation_count`) and the `early-modern-german-fraktur`
tradition profile for Churro, both of which require `scorer_version >= 1.1`.
Falling back to `scorer_version="1.0"` would restore the old speed but lose
both. Revisit the caching fix (either upstream in misnomer, or a local
workaround) if this needs to run again at this document length.

### Update (2026-07-27): precise mechanism confirmed, final coverage accepted

Sharding (4 parallel jobs, `--mem=128G`) got Qwen and Churro most of the way,
but a residual set on each side kept failing even at 128G. Bumping a
one-off pass to `--mem=256G` (`scripts/04b_score_residual.sh`,
`scripts/04c_score_residual_qwen.sh`) surfaced the actual error underneath
the generic "OOM" symptom: real CUDA allocation failures, with requested
single-tensor sizes up to **180.71 GiB on an 80GB A100**. This confirms the
mechanism exactly: `substitution_surprisals` batches up to `chunk_size=16`
candidates per forward pass and pads the whole batch to the *longest*
prefix in that chunk; the resulting `logits` tensor is
`(chunk_size × max_len × vocab_size)` in float32, and for a substitution
positioned late in a long ground-truth document this is unbounded — no
realistic amount of memory fixes the worst case, only avoids it for
shorter documents.

Practical result of the 256G residual passes: Churro recovered 40 of 74
remaining documents (54%); Qwen recovered **0 of 71** (its zero-shot
predictions are far more substitution-dense per document, so the same
documents are more expensive to score for Qwen than for Churro-3B). Final
coverage:

| | Valid | Coverage |
|---|---|---|
| Qwen (zero-shot) | 1,099 / 1,170 | 93.9% |
| Churro-3B (fine-tuned) | 1,136 / 1,170 | 97.1% |
| **Matched pairs (both valid)** | **1,088 / 1,170** | **93.0%** |

Since the project's comparison is paired (base vs. fine-tuned on the same
document), the matched-pairs set is what matters, not either model's
individual coverage — recorded in `results/matched_pairs_image_paths.json`
(1,088 `image_path` values). Any Qwen-vs-Churro comparative analysis should
restrict to this set; each model's own `*_scored.jsonl` still carries its
full individual valid set (1,099 / 1,136) for single-model statistics.
Accepted as final — see the decision above for why we stopped here rather
than continuing to escalate resources.

### Update (2026-07-28): root cause fixed upstream in misnomer

Implemented a fix in `misnomer` (branch
`fix/substitution-surprisals-unbounded-memory`) that replaces the per-substitution
padded-batch recomputation with a single full-document forward pass (giving every
position's ground-truth log-probs for free, exactly like the pre-v1.1 method) plus
one cheap incremental KV-cache step per substitution, reusing the cached context
from that single pass. This removes the unbounded `(chunk_size × max_len ×
vocab_size)` allocation entirely.

Benchmarked directly on `image_path=628`, one of the documents identified above
(2,969 ground-truth words, 1,629 substitutions — Polish, previously required the
256G residual pass):

| | Result |
|---|---|
| Legacy (old per-substitution batching) | **OOM after 207s**, tried to allocate 18.68 GiB on top of 21.96 GiB already in use |
| New (single-pass + incremental cache) | **41.6s, 12.29 GB peak GPU memory** |

Correctness verified two ways: (1) on the first 300 words of this same document
(103 substitutions, small enough for the legacy path to still complete), new vs.
legacy agree to `max_diff=0.000081` with 0 mismatches; (2) misnomer's existing test
suite passes 134/143, with all 9 remaining failures confirmed pre-existing
(identical on the unmodified baseline via `git stash` comparison) and unrelated to
this change. Packaged as a PR to `github.com/apjanco/misnomer`.

This doesn't retroactively rescue the 82 documents (74 Churro + residual Qwen)
dropped from the matched-pairs set above — that coverage decision is accepted as
final and the paper's numbers are already built on it — but it means future
scoring runs (revisions, new datasets) won't hit this ceiling.

---

## Expected Timeline

| Step | Wall time (estimated) | GPU needed |
|------|-----------------------|------------|
| 0 — Setup & model download | 1–2 h | No |
| 1 — Qwen inference (1,170 ex) | 2–4 h | 1× A100 |
| 1 — Churro inference (1,170 ex) | 2–4 h | 1× A100 |
| 2 — Semantic scoring (both), pre-v1.1 misnomer | 3–5 h | 1× GPU |
| 2 — Semantic scoring (both), misnomer v1.1 (actual, 2026-07-25) | ~35–38 GPU-hours serial-equivalent; ~9–10h wall-clock across 4 parallel shards | 4× A100 (sharded) |
| 3 — Analysis & figures | 30 min | No |

Steps 1 (Qwen) and 1 (Churro) can run in parallel if two GPU allocations are available. See "Known Limitations" above for why step 2's real cost diverged so far from the original estimate.

---

## Key Expected Findings (Hypotheses to Test)

- Fine-tuning substantially reduces CER/WER (established by the Churro paper: +14.5% printed, +27.2% handwritten). The question is whether semantic error improves proportionally, less, or more.
- Languages where zero-shot Qwen hallucinates heavily (Greek, Hebrew, Japanese, Chinese) should show disproportionately high semantic error counts alongside high CER.
- Some languages may show low CER improvement from fine-tuning but large semantic error improvement — or the reverse — revealing cases where the models are "wrong in different ways."
- Printed documents should show lower semantic error rates than handwritten ones for both models, consistent with the CER findings.

---

## References

- Churro paper: https://arxiv.org/html/2509.19768v1
- Churro dataset: https://huggingface.co/datasets/stanford-oval/churro-dataset
- Churro 3B model: https://huggingface.co/stanford-oval/churro-3B
- misnomer library: https://github.com/apjanco/misnomer
- Della cluster docs: https://researchcomputing.princeton.edu/systems/della
