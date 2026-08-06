"""
Split qwen/churro predictions (and any already-scored progress) into N
shards by image_path % N, so scoring can run as N parallel single-GPU jobs
instead of one long serial job.

Churro/qwen documents are full pages (up to ~95K chars); misnomer's v1.1
per-word teacher-forced scoring is slow enough on these that a single serial
run couldn't finish qwen in an 8-hour job even with OOM fixed (173 rows in
8h). This mirrors the sharding approach already used for this exact
workload in the Adroit-origin project (hpc/submit_semantic_shard_jobs.sh).

Usage:
    python src/shard_prep.py --num-shards 4
"""

import argparse
import json
from pathlib import Path


def shard_file(input_path: Path, scored_path: Path, output_dir: Path, prefix: str, num_shards: int) -> None:
    shard_in = [open(output_dir / f"{prefix}_predictions_shard{i}of{num_shards}.jsonl", "w") for i in range(num_shards)]
    shard_out = [output_dir / f"{prefix}_scored_shard{i}of{num_shards}.jsonl" for i in range(num_shards)]

    n_total = 0
    with open(input_path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            shard_id = rec["image_path"] % num_shards
            shard_in[shard_id].write(line + "\n")
            n_total += 1
    for f in shard_in:
        f.close()

    # Pre-seed each shard's output file with already-valid scored rows, so
    # score.py's resume-by-image_path logic doesn't redo completed work.
    n_seeded = 0
    if scored_path.exists():
        seeded_lines = [[] for _ in range(num_shards)]
        with open(scored_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("semantic_score_error") or rec.get("semantic_error_count") is None:
                    continue
                shard_id = rec["image_path"] % num_shards
                seeded_lines[shard_id].append(line)
        for i, path in enumerate(shard_out):
            path.write_text("\n".join(seeded_lines[i]) + ("\n" if seeded_lines[i] else ""))
            n_seeded += len(seeded_lines[i])
    else:
        for path in shard_out:
            path.touch()

    print(f"{prefix}: {n_total} rows split into {num_shards} shards, {n_seeded} already-valid rows pre-seeded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output-dir", default="results/shards")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_file(Path("results/qwen_predictions.jsonl"), Path("results/qwen_scored.jsonl"), output_dir, "qwen", args.num_shards)
    shard_file(Path("results/churro_predictions.jsonl"), Path("results/churro_scored.jsonl"), output_dir, "churro", args.num_shards)


if __name__ == "__main__":
    main()
