"""
Analysis and visualisation for the Churro semantic error experiment.

Reads the scored JSONL files produced by score.py and generates:
  - A summary CSV table (language × model × metric)
  - A fine-tuning delta table (Churro vs Qwen per language)
  - Five figures saved to results/figures/

Usage:
    python src/analyze.py \
        --qwen   results/qwen_scored.jsonl \
        --churro results/churro_scored.jsonl \
        --outdir results/figures

Outputs:
    results/figures/01_semantic_error_by_language.png
    results/figures/02_cer_vs_semantic_score.png
    results/figures/03_semantic_error_heatmap_script.png
    results/figures/04_semantic_error_count_dist.png
    results/figures/05_finetuning_delta_by_language.png
    results/summary_table.csv
    results/finetuning_delta.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── aesthetics ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
QWEN_COLOR   = "#4C72B0"
CHURRO_COLOR = "#DD8452"
MODEL_PALETTE = {
    "Qwen 2.5 VL 3B": QWEN_COLOR,
    "Churro 3B": CHURRO_COLOR,
}
FIGURE_DPI = 150


# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path, model_label: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    df = pd.DataFrame(rows)
    df["model"] = model_label
    return df


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def word_count(text: str) -> int:
    return len(str(text).split()) if pd.notna(text) else 0


def build_combined(qwen_path: Path, churro_path: Path, matched_pairs_path: Path | None = None) -> pd.DataFrame:
    qwen   = load_jsonl(qwen_path,   "Qwen 2.5 VL 3B")
    churro = load_jsonl(churro_path, "Churro 3B")

    if matched_pairs_path is not None and matched_pairs_path.exists():
        matched_ids = set(json.loads(matched_pairs_path.read_text()))
        n_qwen_before, n_churro_before = len(qwen), len(churro)
        qwen = qwen[qwen["image_path"].isin(matched_ids)]
        churro = churro[churro["image_path"].isin(matched_ids)]
        print(
            f"  Restricted to {len(matched_ids)} matched pairs "
            f"(qwen {n_qwen_before}->{len(qwen)}, churro {n_churro_before}->{len(churro)}) "
            "so every comparison is over the same documents for both models."
        )

    df = pd.concat([qwen, churro], ignore_index=True)

    numeric_cols = [
        "cer", "cer_full", "wer", "coverage",
        "semantic_error_count", "obvious_error_count",
        "semantic_document_score", "semantic_document_embedding_similarity",
        "inference_time",
    ]
    df = safe_numeric(df, numeric_cols)

    # coverage defaults to 1 for records scored before the alignment patch
    if "coverage" not in df.columns:
        df["coverage"] = 1.0
    df["coverage"] = df["coverage"].fillna(1.0).clip(0, 1)

    # Normalised semantic error rate (errors per aligned-GT word)
    df["gt_word_count"] = df["ground_truth"].apply(word_count)
    df["semantic_error_rate"] = (
        df["semantic_error_count"] / df["gt_word_count"].clip(lower=1)
    )

    # Proportion of substitutions that are semantic (vs obvious)
    total_sub = df["semantic_error_count"].fillna(0) + df["obvious_error_count"].fillna(0)
    df["semantic_ratio"] = df["semantic_error_count"].fillna(0) / total_sub.clip(lower=1)

    # Flag: low CER but high semantic error count (missed by CER)
    df["missed_by_cer"] = (df["cer"] < 10) & (df["semantic_error_count"] > 5)

    return df


def agg_by_language(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "main_language"], dropna=False)
        .agg(
            cer=("cer", "mean"),
            cer_full=("cer_full", "mean"),
            wer=("wer", "mean"),
            coverage=("coverage", "mean"),
            semantic_error_count=("semantic_error_count", "mean"),
            semantic_error_rate=("semantic_error_rate", "mean"),
            semantic_document_score=("semantic_document_score", "mean"),
            semantic_has_error_pct=("semantic_has_error", "mean"),
            n=("image_path", "count"),
        )
        .reset_index()
    )


# ── helper: build bar chart ─────────────────────────────────────────────────

def _side_by_side_bars(
    agg: pd.DataFrame,
    metric: str,
    langs: list,
    ax,
    ylabel: str,
    sort_by_mean: bool = True,
) -> None:
    x = np.arange(len(langs))
    width = 0.35
    for offset, model in zip([-width / 2, width / 2], ["Qwen 2.5 VL 3B", "Churro 3B"]):
        vals = [
            agg.loc[
                (agg["model"] == model) & (agg["main_language"] == lang), metric
            ].values[0]
            if len(agg.loc[(agg["model"] == model) & (agg["main_language"] == lang)]) > 0
            else float("nan")
            for lang in langs
        ]
        ax.bar(x + offset, vals, width, label=model,
               color=MODEL_PALETTE[model], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=45, ha="right")
    ax.set_ylabel(ylabel)


# ── figure 1: coverage by language ──────────────────────────────────────────

def fig_coverage_by_language(agg: pd.DataFrame, outdir: Path) -> None:
    """Mean coverage (fraction of page transcribed) per language."""
    langs = (
        agg.groupby("main_language")["coverage"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    _side_by_side_bars(agg, "coverage", langs, ax,
                       ylabel="Mean coverage (fraction of GT page transcribed)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Page Coverage by Language (fraction of GT text the model transcribed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "01_coverage_by_language.png", dpi=FIGURE_DPI)
    plt.close(fig)
    print("  Saved 01_coverage_by_language.png")


# ── figure 2: aligned CER by language ───────────────────────────────────────

def fig_aligned_cer_by_language(agg: pd.DataFrame, outdir: Path) -> None:
    """Mean aligned CER (quality of transcribed text) per language."""
    langs = (
        agg.groupby("main_language")["cer"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    _side_by_side_bars(agg, "cer", langs, ax,
                       ylabel="Aligned CER (%) — quality of transcribed text")
    ax.set_title("Aligned Character Error Rate by Language\n"
                 "(CER measured against the GT region the model actually covered)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "02_aligned_cer_by_language.png", dpi=FIGURE_DPI)
    plt.close(fig)
    print("  Saved 02_aligned_cer_by_language.png")


# ── figure 3: semantic error rate by language ────────────────────────────────

def fig_semantic_by_language(df: pd.DataFrame, outdir: Path) -> None:
    agg = agg_by_language(df)
    langs = (
        agg.groupby("main_language")["semantic_error_rate"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    _side_by_side_bars(agg, "semantic_error_rate", langs, ax,
                       ylabel="Semantic errors per GT word")
    ax.set_title("Semantic Error Rate by Language (aligned scoring, test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "03_semantic_error_by_language.png", dpi=FIGURE_DPI)
    plt.close(fig)
    print("  Saved 03_semantic_error_by_language.png")


# ── figure 4: fine-tuning delta — two panels: coverage + semantic score ───────

def fig_finetuning_delta(agg: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    if "Churro 3B" not in agg["model"].values or "Qwen 2.5 VL 3B" not in agg["model"].values:
        print("  Skipping delta plot: one or both models missing.")
        return pd.DataFrame()

    piv_sem = agg.pivot(
        index="main_language", columns="model", values="semantic_document_score"
    ).dropna()
    piv_cov = agg.pivot(
        index="main_language", columns="model", values="coverage"
    ).dropna()

    # Sort by coverage delta to group languages together
    langs = piv_cov.index.intersection(piv_sem.index)
    delta_cov = (piv_cov.loc[langs, "Churro 3B"] - piv_cov.loc[langs, "Qwen 2.5 VL 3B"]
                 ).sort_values()
    delta_sem = (piv_sem.loc[delta_cov.index, "Churro 3B"]
                 - piv_sem.loc[delta_cov.index, "Qwen 2.5 VL 3B"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(5, len(langs) * 0.38)),
                                    sharey=True)

    # Panel 1 – coverage delta
    c1 = [CHURRO_COLOR if v >= 0 else QWEN_COLOR for v in delta_cov.values]
    ax1.barh(delta_cov.index, delta_cov.values, color=c1, alpha=0.85)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.set_xlabel("Δ coverage  (Churro − Qwen)")
    ax1.set_title("Coverage delta\n(positive = Churro covers more of the page)")

    # Panel 2 – semantic document score delta
    c2 = [CHURRO_COLOR if v >= 0 else QWEN_COLOR for v in delta_sem.values]
    ax2.barh(delta_sem.index, delta_sem.values, color=c2, alpha=0.85)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Δ semantic doc score  (Churro − Qwen)")
    ax2.set_title("Semantic document score delta\n(positive = Churro has fewer semantic errors)")

    fig.suptitle("Fine-Tuning Effect by Language (sorted by coverage delta)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "04_finetuning_delta_by_language.png",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 04_finetuning_delta_by_language.png")

    delta_df = pd.DataFrame({
        "main_language": langs,
        "delta_coverage": delta_cov.values,
        "delta_semantic_document_score": delta_sem.values,
    })
    return delta_df


# ── figure 5: per-example scatter — coverage vs aligned CER ──────────────────

def fig_coverage_vs_cer(df: pd.DataFrame, outdir: Path) -> None:
    """Scatter showing the quality/coverage tradeoff for each page."""
    sub = df.dropna(subset=["coverage", "cer"])
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, color in MODEL_PALETTE.items():
        mask = sub["model"] == model
        ax.scatter(
            sub.loc[mask, "coverage"],
            sub.loc[mask, "cer"],
            alpha=0.25, s=12, color=color, label=model,
        )
    ax.set_xlabel("Coverage (fraction of GT page transcribed)")
    ax.set_ylabel("Aligned CER (%) — quality of transcribed text")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Quality vs Coverage Trade-off per Page")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(outdir / "05_coverage_vs_aligned_cer.png", dpi=FIGURE_DPI)
    plt.close(fig)
    print("  Saved 05_coverage_vs_aligned_cer.png")


# ── figure 6: semantic_has_error heatmap by script × model ───────────────────

def fig_heatmap_script(df: pd.DataFrame, outdir: Path) -> None:
    if "main_script" not in df.columns:
        print("  Skipping heatmap: 'main_script' column not found.")
        return

    pivot = (
        df.groupby(["main_script", "model"])["semantic_has_error"]
        .mean()
        .unstack("model")
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(7, max(4, len(pivot) * 0.5)))
    sns.heatmap(
        pivot,
        annot=True, fmt=".0%", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Proportion of pages with ≥1 semantic error"},
    )
    ax.set_title("Semantic Error Prevalence by Script")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(outdir / "06_semantic_error_heatmap_script.png", dpi=FIGURE_DPI)
    plt.close(fig)
    print("  Saved 06_semantic_error_heatmap_script.png")


# ── figure 7: semantic error count distribution ───────────────────────────────

def fig_error_count_dist(df: pd.DataFrame, outdir: Path) -> None:
    sub = df.dropna(subset=["semantic_error_count"])
    fig, ax = plt.subplots(figsize=(8, 5))

    for model, color in MODEL_PALETTE.items():
        vals = sub.loc[sub["model"] == model, "semantic_error_count"]
        ax.hist(
            vals.clip(upper=vals.quantile(0.99)),
            bins=40, alpha=0.6, color=color, label=model, density=True,
        )

    ax.set_xlabel("Semantic error count per page (clipped at 99th percentile)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Semantic Error Counts (aligned scoring)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "07_semantic_error_count_dist.png", dpi=FIGURE_DPI)
    plt.close(fig)
    print("  Saved 07_semantic_error_count_dist.png")


# ── LaTeX report ─────────────────────────────────────────────────────────────

def write_latex_report(agg: pd.DataFrame, outdir: Path) -> None:
    """Generate results/report.tex with figures 01 & 03 and the semantic error table."""
    pivot = agg.pivot(
        index="main_language", columns="model", values="semantic_error_rate"
    ).dropna(subset=["Qwen 2.5 VL 3B", "Churro 3B"])

    pivot["delta"] = pivot["Churro 3B"] - pivot["Qwen 2.5 VL 3B"]
    mean_delta = pivot["delta"].mean()
    pivot["delta_vs_mean"] = pivot["delta"] - mean_delta
    pivot = pivot.sort_values("delta")

    def fmt(v: float, sign: bool = False) -> str:
        return f"{'%+.4f' % v if sign else '%.4f' % v}"

    rows = []
    for lang, row in pivot.iterrows():
        # Escape underscores / special chars in language names
        safe_lang = lang.replace("_", r"\_")
        rows.append(
            f"    {safe_lang} & {fmt(row['Qwen 2.5 VL 3B'])} & "
            f"{fmt(row['Churro 3B'])} & {fmt(row['delta'], sign=True)} & "
            f"{fmt(row['delta_vs_mean'], sign=True)} \\\\"
        )

    table_body = "\n".join(rows)
    fig_dir = outdir.name   # e.g. "figures" — relative to results/

    latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{caption}
\usepackage{float}
\geometry{margin=1in}

\begin{document}

\section*{Churro Fine-Tuning Evaluation: Coverage and Semantic Error Rate}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{""" + fig_dir + r"""/01_coverage_by_language.png}
  \caption{Page coverage by language---fraction of the GT page each model transcribed.
    Stanford OVAL's Churro fine-tuning introduces a truncation tendency, most pronounced for
    CJK scripts and several European languages.}
  \label{fig:coverage}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{""" + fig_dir + r"""/03_semantic_error_by_language.png}
  \caption{Semantic error rate by language (errors per GT word, aligned scoring).
    Metrics are computed against the GT region each model actually covered,
    so this reflects transcription quality independent of coverage.}
  \label{fig:semantic}
\end{figure}

\section*{Semantic Errors per GT Word: Qwen vs.\ Churro}

$\Delta = \text{Churro} - \text{Qwen}$ (negative = fine-tuning reduced errors).
Mean $\Delta = """ + fmt(mean_delta, sign=True) + r"""$; the final column shows
how each language deviates from that average shift.

\begin{table}[H]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
Language & Qwen & Churro & $\Delta$ & $\Delta - \bar{\Delta}$ \\
\midrule
""" + table_body + r"""
\midrule
    \textbf{Mean} & """ + fmt(pivot["Qwen 2.5 VL 3B"].mean()) + r""" & """ \
        + fmt(pivot["Churro 3B"].mean()) + r""" & """ \
        + fmt(mean_delta, sign=True) + r""" & --- \\
\bottomrule
\end{tabular}
\caption{Semantic error rate (errors/GT word) by language, sorted by $\Delta$.
  $\Delta < 0$ indicates fine-tuning reduced semantic errors;
  $\Delta - \bar{\Delta} < 0$ indicates the language improved more than average.}
\label{tab:delta}
\end{table}

\end{document}
"""

    out_path = Path("results/report.tex")
    out_path.write_text(latex)
    print(f"  Saved {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse and visualise Churro experiment results.")
    parser.add_argument("--qwen",   required=True, help="Path to qwen_scored.jsonl")
    parser.add_argument("--churro", required=True, help="Path to churro_scored.jsonl")
    parser.add_argument(
        "--outdir", default="results/figures",
        help="Directory for figure output (default: results/figures)",
    )
    parser.add_argument(
        "--matched-pairs", default="results/matched_pairs_image_paths.json",
        help=(
            "JSON list of image_path values valid for both models. Every figure "
            "here is a Qwen-vs-Churro comparison, and the two models have "
            "different (non-random) sets of documents that failed to score "
            "under misnomer v1.1 -- restricting to this set keeps every "
            "comparison over the same documents. Pass an empty string to "
            "disable and use each model's full valid set."
        ),
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    matched_pairs_path = Path(args.matched_pairs) if args.matched_pairs else None
    df = build_combined(Path(args.qwen), Path(args.churro), matched_pairs_path)
    print(f"  {len(df)} total records ({df['model'].value_counts().to_dict()})")

    # ── summary table ─────────────────────────────────────────────────────
    print("Computing summary table...")
    agg = agg_by_language(df)
    summary_path = Path("results/summary_table.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(summary_path, index=False)
    print(f"  Saved {summary_path}")
    print(agg.to_string(index=False))

    # ── figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    agg = agg_by_language(df)
    fig_coverage_by_language(agg, outdir)
    fig_aligned_cer_by_language(agg, outdir)
    fig_semantic_by_language(df, outdir)
    delta_df = fig_finetuning_delta(agg, outdir)
    if not delta_df.empty:
        delta_path = Path("results/finetuning_delta.csv")
        delta_df.to_csv(delta_path, index=False)
        print(f"  Saved {delta_path}")
    fig_coverage_vs_cer(df, outdir)
    fig_heatmap_script(df, outdir)
    fig_error_count_dist(df, outdir)

    # ── LaTeX report ──────────────────────────────────────────────────────
    print("\nGenerating LaTeX report...")
    write_latex_report(agg, outdir)

    # ── "missed by CER" summary ───────────────────────────────────────────
    print("\nExamples with low CER but high semantic error count ('missed by CER'):")
    missed = df[df["missed_by_cer"] == True]
    print(f"  {len(missed)} examples  ({len(missed) / len(df):.1%} of total)")
    if not missed.empty:
        print(
            missed.groupby("model")[["cer", "semantic_error_count"]]
            .agg(["mean", "count"])
            .to_string()
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
