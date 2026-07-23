"""
CER/WER + misnomer semantic scoring for Churro experiment.

Reads a predictions JSONL (output of infer.py) and appends CER, WER, and
misnomer semantic error metrics, writing a scored JSONL.

Usage:
    python src/score.py \
        --input  results/qwen_predictions.jsonl \
        --output results/qwen_scored.jsonl

    python src/score.py \
        --input  results/churro_predictions.jsonl \
        --output results/churro_scored.jsonl

Checkpointing: already-scored image_path values in --output are skipped, so
the job is safely restartable.

GPU required: misnomer runs in "full" mode (transformer LM + sentence-transformer)
which needs a CUDA device.  On a CPU-only node it will fall back to "text_only"
mode and skip semantic / obvious error classification.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import jiwer
import misnomer
from misnomer import ScorerConfig
from misnomer.models.lm import LMScorer
from misnomer.models.embedder import Embedder
from misnomer.scorer import _score_with_models

# ------------------------------------------------------------------
# misnomer scorer configuration
# use_multilingual_embedder=True selects paraphrase-multilingual-MiniLM-L12-v2
# which handles all 29 language clusters in the Churro test set.
# ------------------------------------------------------------------
SCORER_CONFIG = ScorerConfig(
    lm_model="Qwen/Qwen2.5-0.5B",
    use_multilingual_embedder=True,
    allow_download=True,
)


def extract_gt_text(xml_str: str) -> str:
    """Extract transcription lines from the HistoricalDocument XML ground truth.

    The dataset stores ground truth as structured XML with <Metadata> and <Page>
    sections; the actual transcription lives in <Line> elements.  The model
    predictions are plain text, so we must reduce the reference to plain text
    before computing CER/WER or semantic scores.
    """
    if not xml_str or not xml_str.strip().startswith("<"):
        return xml_str  # already plain text
    try:
        root = ET.fromstring(xml_str)
        lines = [
            elem.text.strip()
            for elem in root.iter()
            if (elem.tag == "Line" or elem.tag.endswith("}Line"))
            and elem.text and elem.text.strip()
        ]
        return "\n".join(lines)
    except ET.ParseError:
        # Fall back to stripping all tags
        return re.sub(r"<[^>]+>", " ", xml_str).strip()


def find_gt_coverage(pred: str, gt: str) -> tuple[str, float]:
    """Align a (possibly truncated) prediction to its best-matching GT region.

    For predictions that cover the full page, the full GT is returned.
    For short/truncated predictions, we find the GT window that best
    corresponds to what the model actually transcribed, so that CER is
    computed fairly rather than against the entire unseen document.

    Returns (aligned_gt_region, coverage_fraction).
    """
    import difflib

    if not pred or not gt:
        return gt, 0.0

    pred_len = len(pred)
    gt_len = len(gt)

    # If prediction covers ≥50% of GT length treat as full-page transcription.
    if pred_len >= gt_len * 0.5:
        return gt, min(1.0, pred_len / gt_len)

    # For short predictions search within the first portion of the GT
    # (predictions almost always correspond to the beginning of the page).
    window = min(int(pred_len * 3), gt_len)
    gt_window = gt[:window]

    matcher = difflib.SequenceMatcher(None, pred, gt_window, autojunk=False)
    blocks = matcher.get_matching_blocks()
    # Find the furthest GT position touched by matching blocks.
    last_gt_pos = max((b.b + b.size for b in blocks if b.size > 0), default=window)
    # Add a small buffer and cap at window length.
    aligned_gt = gt[:min(last_gt_pos + 50, window)]

    coverage = len(aligned_gt) / gt_len
    return aligned_gt, coverage


def compute_cer(pred: str, ref: str) -> float:
    """Character error rate as a percentage."""
    if not ref:
        return 0.0 if not pred else 100.0
    return jiwer.cer(ref, pred) * 100.0


def compute_wer(pred: str, ref: str) -> float:
    """Word error rate as a percentage."""
    if not ref:
        return 0.0 if not pred else 100.0
    return jiwer.wer(ref, pred) * 100.0


def score_semantic(pred: str, gt: str, lm: LMScorer, embedder: Embedder) -> dict:
    """
    Run misnomer scoring.  Returns a flat dict of semantic fields ready to
    merge into the output record.  On failure, returns an error field instead.
    """
    try:
        report = _score_with_models(
            predicted=pred,
            ground_truth=gt,
            lm=lm,
            embedder=embedder,
            cfg=SCORER_CONFIG,
        )
        return {
            "semantic_error_count": report.semantic_error_count,
            "obvious_error_count": report.obvious_error_count,
            "semantic_has_error": report.semantic_error_count > 0,
            "semantic_document_score": report.document_score,
            "semantic_document_error_type": getattr(report, "document_error_type", None),
            "semantic_document_embedding_similarity": getattr(
                report, "document_embedding_similarity", None
            ),
            "semantic_scorer_mode": report.scorer_mode,
            "semantic_scorer_version": getattr(report, "scorer_version", None),
            "semantic_lm_model": getattr(report, "lm_model", None),
            "semantic_embedder_model": getattr(report, "embedder_model", None),
        }
    except Exception as exc:
        return {
            "semantic_error_count": None,
            "obvious_error_count": None,
            "semantic_has_error": None,
            "semantic_document_score": None,
            "semantic_score_error": str(exc),
        }


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
    parser = argparse.ArgumentParser(description="Score predictions with CER/WER + misnomer.")
    parser.add_argument("--input",  required=True, help="Path to predictions JSONL")
    parser.add_argument("--output", required=True, help="Path to output scored JSONL")
    parser.add_argument(
        "--log_every", type=int, default=50,
        help="Print progress every N examples",
    )
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    done_ids = load_done_ids(out_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} examples already scored, skipping.")

    print("Loading misnomer models (LM + embedder)...")
    lm = LMScorer(SCORER_CONFIG)
    embedder = Embedder(SCORER_CONFIG)
    print("Models ready.")

    total = 0
    n_semantic_errors = 0

    with open(in_path) as fin, open(out_path, "a") as fout:
        for i, raw_line in enumerate(fin):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            rec = json.loads(raw_line)
            img_id = rec["image_path"]

            if img_id in done_ids:
                continue

            pred = rec.get("predicted_text") or ""
            gt   = extract_gt_text(rec.get("ground_truth") or "")

            # Align prediction to the GT region it actually covers.
            aligned_gt, coverage = find_gt_coverage(pred, gt)

            # --- standard metrics (aligned) ---
            rec["cer"]          = compute_cer(pred, aligned_gt)
            rec["cer_full"]     = compute_cer(pred, gt)       # against full GT
            rec["wer"]          = compute_wer(pred, aligned_gt)
            rec["coverage"]     = round(coverage, 4)

            # --- semantic scoring (aligned) ---
            if pred and aligned_gt:
                sem = score_semantic(pred, aligned_gt, lm, embedder)
            else:
                # empty prediction or ground truth → no semantic content to score
                sem = {
                    "semantic_error_count": 0,
                    "obvious_error_count": 0,
                    "semantic_has_error": False,
                    "semantic_document_score": None,
                }
            rec.update(sem)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            total += 1
            if rec.get("semantic_has_error"):
                n_semantic_errors += 1

            if (i + 1) % args.log_every == 0:
                rate = n_semantic_errors / total if total else 0.0
                print(
                    f"[{i + 1}] scored={total}  "
                    f"semantic_error_rate={rate:.1%}  "
                    f"last_cer={rec['cer']:.1f}"
                )

    print(f"\nFinished. {total} examples scored.")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
