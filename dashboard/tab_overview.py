"""
dashboard/tab_overview.py
Tab 1 — Model Comparison Overview
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import CATEGORY_COLORS, CATEGORY_ORDER


def render(df: pd.DataFrame):
    st.header("📊 Model Comparison Overview")

    if df.empty:
        st.warning("No data matches the current filters.")
        return

    models = sorted(df["model"].unique())

    # ── 1. CER / WER Distributions ────────────────────────────────────
    st.subheader("CER & WER Distributions")
    col1, col2 = st.columns(2)

    # Cap for display (refusals / hallucinations create extreme outliers)
    cap = st.slider("Cap CER/WER at (for display)", 0, 200, 100, key="dist_cap")

    with col1:
        plot_df = df.copy()
        plot_df["cer_capped"] = plot_df["cer"].clip(upper=cap)
        fig = px.violin(
            plot_df,
            x="model",
            y="cer_capped",
            color="model",
            box=True,
            points=False,
            title="Character Error Rate (CER)",
            labels={"cer_capped": f"CER (capped at {cap})", "model": "Model"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        plot_df["wer_capped"] = plot_df["wer"].clip(upper=cap)
        fig = px.violin(
            plot_df,
            x="model",
            y="wer_capped",
            color="model",
            box=True,
            points=False,
            title="Word Error Rate (WER)",
            labels={"wer_capped": f"WER (capped at {cap})", "model": "Model"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── 2. Error Category Stacked Bars ────────────────────────────────
    st.subheader("Error Categories")

    cat_counts = (
        df.groupby(["model", "category"], observed=True)
        .size()
        .reset_index(name="count")
    )
    # Compute percentages
    totals = cat_counts.groupby("model")["count"].transform("sum")
    cat_counts["pct"] = (cat_counts["count"] / totals * 100).round(1)

    fig = px.bar(
        cat_counts,
        x="model",
        y="pct",
        color="category",
        color_discrete_map=CATEGORY_COLORS,
        category_orders={"category": CATEGORY_ORDER},
        title="Prediction Breakdown by Error Category",
        labels={"pct": "Percentage (%)", "model": "Model", "category": "Category"},
        text="pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
    fig.update_layout(barmode="stack", uniformtext_minsize=8, uniformtext_mode="hide")
    st.plotly_chart(fig, use_container_width=True)

    # Also show raw counts table
    with st.expander("Show raw counts"):
        pivot = cat_counts.pivot_table(index="model", columns="category", values="count", fill_value=0)
        pivot = pivot.reindex(columns=[c for c in CATEGORY_ORDER if c in pivot.columns])
        st.dataframe(pivot, use_container_width=True)

    # ── 3. Pairwise Agreement Heatmap ─────────────────────────────────
    st.subheader("Pairwise Agreement")
    st.caption("What fraction of samples do two models both get right (CER=0), both get wrong, or disagree on?")

    threshold = st.number_input("'Correct' CER threshold", value=5.0, min_value=0.0, max_value=50.0, step=1.0, key="pw_thresh")

    # Pivot to wide: one row per sample, columns = model CER
    wide = df.pivot_table(index="sample_idx", columns="model", values="cer")

    if len(models) >= 2:
        rows = []
        for i, m1 in enumerate(models):
            for m2 in models:
                if m1 == m2:
                    rows.append({"model_1": m1, "model_2": m2, "agreement": 100.0, "label": "—"})
                    continue
                both_ok = ((wide[m1] <= threshold) & (wide[m2] <= threshold)).mean() * 100
                both_bad = ((wide[m1] > threshold) & (wide[m2] > threshold)).mean() * 100
                agree = both_ok + both_bad
                rows.append({
                    "model_1": m1,
                    "model_2": m2,
                    "agreement": round(agree, 1),
                    "label": f"{both_ok:.0f}% both✓  {both_bad:.0f}% both✗",
                })

        pw_df = pd.DataFrame(rows)
        heatmap = pw_df.pivot(index="model_1", columns="model_2", values="agreement")

        fig = px.imshow(
            heatmap,
            text_auto=".1f",
            color_continuous_scale="Greens",
            title="Pairwise Agreement (%)",
            labels={"color": "Agreement %"},
        )
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # Detail table
        with st.expander("Agreement details"):
            labels_df = pw_df.pivot(index="model_1", columns="model_2", values="label")
            st.dataframe(labels_df, use_container_width=True)
    else:
        st.info("Need at least 2 models for pairwise comparison.")

    # ── 4. Inference Time ─────────────────────────────────────────────
    st.subheader("Inference Time")

    time_df = df.dropna(subset=["inference_time"])
    if not time_df.empty:
        fig = px.box(
            time_df,
            x="model",
            y="inference_time",
            color="model",
            title="Inference Time per Sample (seconds)",
            labels={"inference_time": "Time (s)", "model": "Model"},
            points=False,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        st.dataframe(
            time_df.groupby("model")["inference_time"]
            .describe()
            .round(3),
            use_container_width=True,
        )
    else:
        st.info("No inference time data available.")

    # ── 5. Summary Metrics Table ──────────────────────────────────────
    st.subheader("Summary Metrics")
    summary = (
        df.groupby("model")
        .agg(
            n=("cer", "size"),
            mean_cer=("cer", "mean"),
            median_cer=("cer", "median"),
            mean_wer=("wer", "mean"),
            median_wer=("wer", "median"),
            perfect_pct=("cer", lambda x: (x == 0).mean() * 100),
        )
        .round(2)
    )
    st.dataframe(summary, use_container_width=True)
