"""Benchmark old vs new substitution_surprisals on a real pathological document.

Loads one Churro document (image_path=628, ~2986 ground-truth words) that
previously required the 256G residual pass to score without OOMing, builds
the same alignment/substitution inputs the scorer pipeline uses, and times
the legacy per-substitution implementation against the new single-pass +
incremental-cache implementation.
"""
import json
import time
import torch

from misnomer.aligner import align_words, tokenize_words
from misnomer.config import ScorerConfig
from misnomer.models.lm import LMScorer

DOC_PATH = "results/bench_doc_628.json"


def main():
    with open(DOC_PATH) as f:
        doc = json.load(f)

    ground_truth = doc["ground_truth"]
    predicted = doc["predicted_text"]

    cfg = ScorerConfig(scorer_version="1.1", strict=True, device="cuda")
    lm = LMScorer(cfg)

    alignment = align_words(predicted, ground_truth, tokenizer=lm.tokenizer)
    gt_words = tokenize_words(ground_truth, tokenizer=lm.tokenizer)

    substitution_words: dict[int, str] = {}
    gt_idx = -1
    for item in alignment:
        if item.alignment_type in {"MATCH", "SUBSTITUTION", "DELETION"}:
            gt_idx += 1
        if item.alignment_type == "SUBSTITUTION":
            substitution_words[gt_idx] = item.predicted_word

    print(f"gt_words: {len(gt_words)}")
    print(f"substitutions: {len(substitution_words)}")

    # --- Full document: this is the real pathological case that used to
    # require the 256G residual pass. Run NEW first (so we get its numbers
    # even if LEGACY OOMs), then attempt LEGACY and expect/report failure.
    torch.cuda.reset_peak_memory_stats()
    new_t0 = time.perf_counter()
    new_results = lm.substitution_surprisals(gt_words, substitution_words)
    new_t1 = time.perf_counter()
    new_peak = torch.cuda.max_memory_allocated() / 1e9
    new_time = new_t1 - new_t0
    print(f"NEW (full doc, {len(substitution_words)} substitutions): "
          f"{new_time:.2f}s, peak_gpu_mem={new_peak:.2f}GB")

    torch.cuda.reset_peak_memory_stats()
    legacy_t0 = time.perf_counter()
    try:
        legacy_full_results = lm._substitution_surprisals_legacy(gt_words, substitution_words)
        legacy_t1 = time.perf_counter()
        legacy_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"LEGACY (full doc): {legacy_t1 - legacy_t0:.2f}s, peak_gpu_mem={legacy_peak:.2f}GB")
        print(f"speedup: {(legacy_t1 - legacy_t0) / new_time:.1f}x")
    except torch.OutOfMemoryError as e:
        print(f"LEGACY (full doc): OOM after {time.perf_counter() - legacy_t0:.2f}s -- {e}")
        legacy_full_results = None
    torch.cuda.empty_cache()

    # --- Correctness check on a slice small enough for LEGACY to complete,
    # to confirm NEW's numbers agree with LEGACY where LEGACY can still run.
    slice_words = gt_words[:300]
    slice_substitutions = {i: w for i, w in substitution_words.items() if i < 300}
    legacy_slice = lm._substitution_surprisals_legacy(slice_words, slice_substitutions)
    new_slice = lm.substitution_surprisals(slice_words, slice_substitutions)

    max_diff = 0.0
    mismatches = 0
    for i in slice_substitutions:
        a = legacy_slice.get(i)
        b = new_slice.get(i)
        if a is None or b is None:
            if a != b:
                mismatches += 1
            continue
        d = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        max_diff = max(max_diff, d)
        if d > 1e-2:
            mismatches += 1
            print(f"  MISMATCH at {i}: legacy={a} new={b}")

    print(f"correctness check on first 300 words ({len(slice_substitutions)} substitutions): "
          f"max_diff={max_diff:.6f}, mismatches={mismatches}/{len(slice_substitutions)}")


if __name__ == "__main__":
    main()
