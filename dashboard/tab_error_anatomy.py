"""
dashboard/tab_error_anatomy.py
Tab 2 — Error Anatomy: What Goes Wrong?
"""

import difflib
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import CATEGORY_COLORS, CATEGORY_ORDER


# ── helpers ────────────────────────────────────────────────────────────

def _classify_edit_ops(gt: str, pred: str) -> dict:
    """
    Use SequenceMatcher to classify character-level edits into
    substitution, insertion, deletion counts.
    """
    sm = difflib.SequenceMatcher(None, gt, pred)
    subs, ins, dels = 0, 0, 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
            subs += max(i2 - i1, j2 - j1)
        elif op == "insert":
            ins += j2 - j1
        elif op == "delete":
            dels += i2 - i1
    return {"substitutions": subs, "insertions": ins, "deletions": dels}


def _word_level_cer(gt_words: list[str], pred_words: list[str], position: int) -> float | None:
    """CER for a single word position, or None if position doesn't exist in GT."""
    if position >= len(gt_words):
        return None
    gt_w = gt_words[position]
    pred_w = pred_words[position] if position < len(pred_words) else ""
    if len(gt_w) == 0:
        return 100.0 if len(pred_w) > 0 else 0.0
    sm = difflib.SequenceMatcher(None, gt_w, pred_w)
    edits = sum(
        max(i2 - i1, j2 - j1) if op == "replace" else (j2 - j1 if op == "insert" else i2 - i1)
        for op, i1, i2, j1, j2 in sm.get_opcodes()
        if op != "equal"
    )
    return min(edits / len(gt_w) * 100, 200)  # cap at 200 for sanity


# ── main render ────────────────────────────────────────────────────────

def render(df: pd.DataFrame):
    st.header("🔬 Error Anatomy — What Goes Wrong?")

    if df.empty:
        st.warning("No data matches the current filters.")
        return

    models = sorted(df["model"].unique())

    # ── 1. Error Type Taxonomy ────────────────────────────────────────
    st.subheader("Error Type Taxonomy")
    st.caption("Character-level edit operations across all predictions (sampled for speed).")

    sample_n = min(2000, len(df))
    sample_df = df.dropna(subset=["predicted", "ground_truth"]).sample(n=sample_n, random_state=42)

    edit_rows = []
    for _, row in sample_df.iterrows():
        ops = _classify_edit_ops(str(row["ground_truth"]), str(row["predicted"]))
        ops["model"] = row["model"]
        ops["category"] = row["category"]
        edit_rows.append(ops)

    edit_df = pd.DataFrame(edit_rows)

    # Sunburst data
    sunburst_rows = []
    for model in models:
        mdf = edit_df[edit_df["model"] == model]
        total = mdf[["substitutions", "insertions", "deletions"]].sum().sum()
        if total == 0:
            continue
        for op in ["substitutions", "insertions", "deletions"]:
            val = mdf[op].sum()
            sunburst_rows.append({"model": model, "operation": op.title(), "count": int(val)})

    if sunburst_rows:
        sun_df = pd.DataFrame(sunburst_rows)
        fig = px.sunburst(
            sun_df,
            path=["model", "operation"],
            values="count",
            title="Edit Operations by Model",
            color="operation",
            color_discrete_map={"Substitutions": "#e74c3c", "Insertions": "#3498db", "Deletions": "#f39c12"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Bar chart for easier comparison
    if edit_rows:
        agg = edit_df.groupby("model")[["substitutions", "insertions", "deletions"]].mean().round(1)
        agg = agg.reset_index().melt(id_vars="model", var_name="operation", value_name="avg_chars")
        fig = px.bar(
            agg,
            x="model",
            y="avg_chars",
            color="operation",
            barmode="group",
            title="Mean Character-Level Edit Operations per Sample",
            labels={"avg_chars": "Avg chars", "model": "Model"},
            color_discrete_map={"substitutions": "#e74c3c", "insertions": "#3498db", "deletions": "#f39c12"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── 2. CER vs Text Length ─────────────────────────────────────────
    st.subheader("CER vs Ground Truth Length")
    st.caption("Do models struggle more with short or long text?")

    scatter_df = df.dropna(subset=["cer"]).copy()
    scatter_df["cer_capped"] = scatter_df["cer"].clip(upper=150)

    fig = px.scatter(
        scatter_df,
        x="gt_length",
        y="cer_capped",
        color="model",
        opacity=0.3,
        title="CER vs Ground Truth Character Length",
        labels={"gt_length": "Ground Truth Length (chars)", "cer_capped": "CER (capped at 150)"},
        hover_data=["ground_truth", "predicted"],
    )
    # Add LOESS-style trendlines
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

    # Binned view (cleaner)
    scatter_df["length_bin"] = pd.cut(scatter_df["gt_length"], bins=10)
    binned = scatter_df.groupby(["length_bin", "model"], observed=True)["cer"].median().reset_index()
    binned["length_bin_str"] = binned["length_bin"].astype(str)
    fig = px.line(
        binned,
        x="length_bin_str",
        y="cer",
        color="model",
        markers=True,
        title="Median CER by Text Length Bin",
        labels={"length_bin_str": "GT Length Bin", "cer": "Median CER"},
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # ── 3. Confidence vs Accuracy (GPT-4o) ────────────────────────────
    st.subheader("Confidence vs Accuracy")

    conf_df = df.dropna(subset=["confidence", "cer"]).copy()
    if not conf_df.empty:
        conf_df["cer_capped"] = conf_df["cer"].clip(upper=100)
        fig = px.scatter(
            conf_df,
            x="confidence",
            y="cer_capped",
            color="model",
            opacity=0.4,
            title="Model Confidence vs CER (models with logprobs)",
            labels={"confidence": "Confidence (%)", "cer_capped": "CER (capped at 100)"},
            hover_data=["ground_truth", "predicted"],
        )
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No confidence/logprobs data available in the current filter.")

    # ── 4. Entropy Distribution ───────────────────────────────────────
    st.subheader("Entropy Distribution")

    ent_df = df.dropna(subset=["entropy"]).copy()
    if not ent_df.empty:
        # Split into correct vs incorrect
        ent_df["outcome"] = np.where(ent_df["cer"] <= 5, "Correct (CER≤5)", "Incorrect (CER>5)")
        fig = px.histogram(
            ent_df,
            x="entropy",
            color="outcome",
            barmode="overlay",
            nbins=50,
            opacity=0.6,
            title="Entropy: Correct vs Incorrect Predictions",
            labels={"entropy": "Entropy", "outcome": "Outcome"},
            color_discrete_map={"Correct (CER≤5)": "#2ecc71", "Incorrect (CER>5)": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No entropy data available in the current filter.")

    # ── 5. Word-Position Error Heatmap ────────────────────────────────
    st.subheader("Error by Word Position")
    st.caption("CER at each word position — do models fail more at the start or end of lines?")

    max_pos = st.slider("Max word positions to show", 3, 15, 10, key="max_word_pos")

    pos_sample = df.dropna(subset=["predicted", "cer"]).copy()
    pos_sample = pos_sample[pos_sample["category"] != "Refusal"]  # exclude refusals
    pos_sample = pos_sample.head(5000)  # speed

    pos_rows = []
    for _, row in pos_sample.iterrows():
        gt_words = str(row["ground_truth"]).split()
        pred_words = str(row["predicted"]).split()
        for pos in range(min(max_pos, len(gt_words))):
            wcer = _word_level_cer(gt_words, pred_words, pos)
            if wcer is not None:
                pos_rows.append({"model": row["model"], "position": pos + 1, "word_cer": wcer})

    if pos_rows:
        pos_df = pd.DataFrame(pos_rows)
        heatmap_data = pos_df.groupby(["model", "position"])["word_cer"].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index="model", columns="position", values="word_cer")

        fig = px.imshow(
            heatmap_pivot,
            text_auto=".1f",
            color_continuous_scale="YlOrRd",
            title="Mean Word-Level CER by Position",
            labels={"color": "Mean CER", "x": "Word Position", "y": "Model"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for word-position analysis.")
