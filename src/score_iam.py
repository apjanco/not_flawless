"""
misnomer semantic scoring for the IAM experiment.

The six IAM result files (one per model: ChatGPT-5, ChatGPT-4o, Claude,
Gemini, Qwen3-VL-8B-Instruct, Chandra) were produced on Adroit before this
project's misnomer integration existed. Each model instead computed its own
ad hoc, inconsistent "semantic_error" from raw logprobs/entropy, and two of
the six (chandra, claude-vision) have no semantic scoring at all.

This script leaves cer/wer/predicted_text untouched and only adds misnomer's
semantic fields, computed uniformly across all six files from the
ground_truth + predicted_text already present in each row.

IAM is English-only, so unlike score.py (Churro, 29 languages) this uses the
default English embedder rather than the multilingual one.

Two data issues confirmed present in the raw per-model files (per project
collaborator), handled here:

1. Checkpoint-resume duplicates: chatgpt5 (38 dupes) and gemini-portkey (85
   dupes) each have a handful of row ids appearing twice -- an empty row from
   an interrupted attempt followed by the real retried row. We dedupe by row
   id, last-write-wins, before scoring. All six files are exactly 10,373 rows
   after dedup.
2. Do NOT use full_eval_results.jsonl (in the misnomer repo's evaluation/
   dir) -- it was self-appended at some point and row id no longer uniquely
   identifies a row for 5 of 7 models there. Only the per-model *_results.jsonl
   files (copied into results/iam_raw/) are used.

Usage:
    python src/score_iam.py \
        --input-glob results/iam_raw/*_results.jsonl \
        --output-dir results/iam_scored

Checkpointing: resumes by row id already present in the output file, so the
job is safely restartable.
"""

import argparse
import json
from pathlib import Path

from misnomer import ScorerConfig
from misnomer.models.lm import LMScorer
from misnomer.models.embedder import Embedder
from misnomer.scorer import _build_dictionary, _score_with_models

ID_FIELDS = ("index", "image_idx", "image_path")


def detect_id_field(rec: dict) -> str:
    for field in ID_FIELDS:
        if field in rec:
            return field
    raise ValueError(f"No known id field ({ID_FIELDS}) in record: {list(rec.keys())}")


def load_deduped(in_path: Path) -> tuple[str, list[dict]]:
    """Read a result file and dedupe by row id, last-write-wins."""
    by_id: dict = {}
    id_field = None
    with open(in_path) as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            rec = json.loads(raw_line)
            if id_field is None:
                id_field = detect_id_field(rec)
            by_id[rec[id_field]] = rec  # last occurrence wins

    return id_field, list(by_id.values())


# scorer_version left unset -> defaults to misnomer's current version (1.1),
# which activates the v1.1 gates (NUMERIC/allograph/GARBLE/homoglyph/
# segmentation) requiring misnomer[gates] (pyspellchecker) to be installed.
# strict=True per the library's own guidance ("all paper/production runs
# should set strict=True") -- surfaces a hard error instead of silently
# degrading to the frequency-proxy LM or char-similarity embedder fallback.
# device="cuda": LMScorer reads cfg.device directly (default "cpu" for
# v1.0 bit-stable behavior); without this override it would ignore the GPU
# these jobs request. Embedder auto-detects CUDA on its own.
SCORER_CONFIG = ScorerConfig(
    lm_model="Qwen/Qwen2.5-0.5B",
    use_multilingual_embedder=False,
    allow_download=True,
    strict=True,
    device="cuda",
)


def score_semantic(pred: str, gt: str, lm: LMScorer, embedder: Embedder, dictionary) -> dict:
    try:
        report = _score_with_models(
            predicted=pred,
            ground_truth=gt,
            lm=lm,
            embedder=embedder,
            cfg=SCORER_CONFIG,
            dictionary=dictionary,
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
            # v1.1 gate-class counts and provenance
            "semantic_numeric_error_count": getattr(report, "numeric_error_count", None),
            "semantic_normalization_count": getattr(report, "normalization_count", None),
            "semantic_archaizing_count": getattr(report, "archaizing_count", None),
            "semantic_garble_count": getattr(report, "garble_count", None),
            "semantic_homoglyph_count": getattr(report, "homoglyph_count", None),
            "semantic_segmentation_count": getattr(report, "segmentation_count", None),
            "semantic_tradition": getattr(report, "tradition", None),
            "semantic_is_refusal": getattr(report, "is_refusal", None),
        }
    except Exception as exc:
        return {
            "semantic_error_count": None,
            "obvious_error_count": None,
            "semantic_has_error": None,
            "semantic_document_score": None,
            "semantic_score_error": str(exc),
        }


def score_file(in_path: Path, out_path: Path, lm: LMScorer, embedder: Embedder, dictionary, log_every: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    id_field, records = load_deduped(in_path)
    n_raw = sum(1 for line in open(in_path) if line.strip())
    n_dupes = n_raw - len(records)
    print(f"  {in_path.name}: id_field={id_field}, {n_raw} raw rows, {n_dupes} duplicate(s) removed, {len(records)} unique rows")

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)[id_field])
        if done_ids:
            print(f"  Resuming {in_path.name}: {len(done_ids)} rows already scored, skipping.")

    total = 0
    n_semantic_errors = 0

    with open(out_path, "a") as fout:
        for rec in records:
            if rec[id_field] in done_ids:
                continue

            pred = rec.get("predicted_text") or ""
            gt = rec.get("ground_truth") or ""

            # A model that returned nothing (e.g. GPT-5 on ~22% of IAM lines)
            # carries no semantic signal -- it is neither "correct" nor
            # scoreable, so leave the semantic fields null rather than 0/False.
            # Downstream analysis must compute semantic error rate over
            # has_prediction=True rows only, and report coverage separately.
            rec["has_prediction"] = bool(pred.strip()) and bool(gt.strip())
            if rec["has_prediction"]:
                sem = score_semantic(pred, gt, lm, embedder, dictionary)
            else:
                sem = {
                    "semantic_error_count": None,
                    "obvious_error_count": None,
                    "semantic_has_error": None,
                    "semantic_document_score": None,
                    "semantic_document_error_type": "no_prediction",
                }
            rec.update(sem)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            total += 1
            if rec.get("semantic_has_error"):
                n_semantic_errors += 1

            if log_every > 0 and total % log_every == 0:
                rate = n_semantic_errors / total if total else 0.0
                print(f"  [{in_path.name}] scored={total} semantic_error_rate={rate:.1%}")

    print(f"  Done {in_path.name}: {total} new rows scored ({n_semantic_errors} with semantic error).")


def main():
    parser = argparse.ArgumentParser(description="Add misnomer semantic scoring to IAM result files.")
    parser.add_argument("--input-glob", default="results/iam_raw/*_results.jsonl")
    parser.add_argument("--output-dir", default="results/iam_scored")
    parser.add_argument("--log_every", type=int, default=200)
    args = parser.parse_args()

    input_files = sorted(Path(".").glob(args.input_glob))
    if not input_files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(input_files)} IAM result files:")
    for p in input_files:
        print(f"  - {p}")

    print("\nLoading misnomer models (LM + English embedder) once for all files...")
    lm = LMScorer(SCORER_CONFIG)
    embedder = Embedder(SCORER_CONFIG)
    dictionary = _build_dictionary(SCORER_CONFIG)
    print(f"Resolved LM: {lm.resolved_model_name} (device={lm.resolved_device})")
    print(f"Resolved embedder: {embedder.resolved_model_name}")
    print(f"Dictionary available: {dictionary.available if dictionary else 'n/a (gates inactive)'}\n")

    for in_path in input_files:
        out_path = output_dir / in_path.name.replace("_results.jsonl", "_scored.jsonl")
        print(f"Scoring {in_path} -> {out_path}")
        score_file(in_path, out_path, lm, embedder, dictionary, args.log_every)


if __name__ == "__main__":
    main()
