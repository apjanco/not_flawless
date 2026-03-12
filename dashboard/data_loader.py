"""
dashboard/data_loader.py
Shared data loading, preprocessing, and error categorization.
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset

DATASET_ID = "apjanco/combined-data-mar-26"

REFUSAL_PATTERN = re.compile(
    r"(?:I'm sorry|I cannot|I can't|I am unable|sorry, I|can't assist|cannot transcribe|"
    r"unable to transcribe|can't transcribe|not able to)",
    re.IGNORECASE,
)


def categorize_error(row: pd.Series) -> str:
    """Assign an error category to a single prediction row."""
    pred = str(row.get("predicted", "") or "")
    cer = row.get("cer")

    if cer is None or pd.isna(cer):
        return "Error"
    if REFUSAL_PATTERN.search(pred):
        return "Refusal"
    if cer == 0:
        return "Perfect"
    if cer <= 5:
        return "Minor"
    if cer <= 25:
        return "Moderate"
    if cer <= 100:
        return "Major"
    # CER > 100 — predicted text is longer/wildly different
    return "Hallucination"


# Ordered for consistent coloring
CATEGORY_ORDER = ["Perfect", "Minor", "Moderate", "Major", "Hallucination", "Refusal", "Error"]
CATEGORY_COLORS = {
    "Perfect": "#2ecc71",
    "Minor": "#82e0aa",
    "Moderate": "#f4d03f",
    "Major": "#e67e22",
    "Hallucination": "#e74c3c",
    "Refusal": "#8e44ad",
    "Error": "#95a5a6",
}


@st.cache_data(show_spinner="Loading dataset from HuggingFace…")
def load_data() -> tuple[pd.DataFrame, list[dict]]:
    """
    Load the HF dataset and flatten into a DataFrame with one row per
    (sample, model) pair.  Also returns the raw dataset examples for
    image access.

    Returns
    -------
    df : pd.DataFrame
        Columns: sample_idx, ground_truth, gt_length, gt_word_count,
                 model, predicted, cer, wer, inference_time, confidence,
                 entropy, kl_divergence, mean_logprob, semantic_error,
                 error, category
    raw_examples : list[dict]
        Original dataset rows (kept in memory for PIL image access).
    """
    ds = load_dataset(DATASET_ID, split="train")

    raw_examples = list(ds)

    rows = []
    for i, ex in enumerate(raw_examples):
        for r in ex["model_results"]:
            rows.append(
                {
                    "sample_idx": i,
                    "ground_truth": ex["text"],
                    "gt_length": len(ex["text"]),
                    "gt_word_count": len(ex["text"].split()),
                    **r,
                }
            )

    df = pd.DataFrame(rows)

    # Derive error category
    df["category"] = df.apply(categorize_error, axis=1)

    # Ensure ordered categorical for consistent plotting
    df["category"] = pd.Categorical(df["category"], categories=CATEGORY_ORDER, ordered=True)

    return df, raw_examples


def get_models(df: pd.DataFrame) -> list[str]:
    """Sorted list of unique model names."""
    return sorted(df["model"].unique().tolist())


def filter_df(
    df: pd.DataFrame,
    models: list[str] | None = None,
    cer_range: tuple[float, float] | None = None,
    wer_range: tuple[float, float] | None = None,
    gt_length_range: tuple[int, int] | None = None,
    categories: list[str] | None = None,
    exclude_refusals: bool = False,
) -> pd.DataFrame:
    """Apply global sidebar filters."""
    mask = pd.Series(True, index=df.index)

    if models:
        mask &= df["model"].isin(models)
    if cer_range:
        mask &= df["cer"].between(*cer_range)
    if wer_range:
        mask &= df["wer"].between(*wer_range)
    if gt_length_range:
        mask &= df["gt_length"].between(*gt_length_range)
    if categories:
        mask &= df["category"].isin(categories)
    if exclude_refusals:
        mask &= df["category"] != "Refusal"

    return df.loc[mask].copy()
