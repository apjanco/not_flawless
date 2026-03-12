"""
dashboard/tab_confidence.py
Tab 4 — Confidence & Calibration (models with logprobs data)
"""

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(df: pd.DataFrame):
    st.header("🎯 Confidence & Calibration")

    # Only look at rows that have logprobs data
    conf_df = df.dropna(subset=["confidence", "cer"]).copy()

    if conf_df.empty:
        st.info(
            "No confidence / logprobs data available for the current filter. "
            "This tab requires models that return logprobs (currently GPT-4o)."
        )
        return

    models_with_conf = sorted(conf_df["model"].unique())
    st.caption(f"Models with logprobs data: **{', '.join(models_with_conf)}**")

    # ── 1. Calibration Curve ──────────────────────────────────────────
    st.subheader("Calibration Curve")
    st.caption(
        "Binned by model confidence: does higher confidence actually mean lower error? "
        "A perfectly calibrated model would follow the diagonal."
    )

    n_bins = st.slider("Number of confidence bins", 5, 20, 10, key="cal_bins")

    conf_df["conf_bin"] = pd.cut(conf_df["confidence"], bins=n_bins)
    conf_df["is_correct"] = conf_df["cer"] <= 5  # threshold for "correct"

    cal_data = (
        conf_df.groupby(["model", "conf_bin"], observed=True)
        .agg(
            accuracy=("is_correct", "mean"),
            mean_conf=("confidence", "mean"),
            count=("is_correct", "size"),
        )
        .reset_index()
    )
    cal_data["accuracy_pct"] = cal_data["accuracy"] * 100

    fig = go.Figure()

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect calibration",
    ))

    for model in models_with_conf:
        mdata = cal_data[cal_data["model"] == model].sort_values("mean_conf")
        fig.add_trace(go.Scatter(
            x=mdata["mean_conf"],
            y=mdata["accuracy_pct"],
            mode="lines+markers",
            name=model,
            text=mdata["count"].apply(lambda c: f"n={c}"),
            hovertemplate="Confidence: %{x:.1f}%<br>Accuracy: %{y:.1f}%<br>%{text}",
            marker=dict(size=mdata["count"].clip(upper=100) / 5 + 4),
        ))

    fig.update_layout(
        title="Calibration: Confidence vs Actual Accuracy (CER≤5)",
        xaxis_title="Mean Confidence (%)",
        yaxis_title="Accuracy (% with CER ≤ 5)",
        xaxis=dict(range=[0, 105]),
        yaxis=dict(range=[0, 105]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 2. Confidence vs CER Scatter ──────────────────────────────────
    st.subheader("Confidence vs CER")

    scatter = conf_df.copy()
    scatter["cer_capped"] = scatter["cer"].clip(upper=100)

    fig = px.scatter(
        scatter,
        x="confidence",
        y="cer_capped",
        color="category",
        opacity=0.4,
        title="Confidence vs CER (per prediction)",
        labels={"confidence": "Confidence (%)", "cer_capped": "CER (capped 100)"},
        hover_data=["model", "ground_truth", "predicted"],
        category_orders={"category": ["Perfect", "Minor", "Moderate", "Major", "Hallucination", "Refusal"]},
    )
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

    # ── 3. KL Divergence ──────────────────────────────────────────────
    st.subheader("KL Divergence Distribution")
    st.caption("Higher KL divergence = model's token distribution is further from uniform (more 'opinionated').")

    kl_df = df.dropna(subset=["kl_divergence"]).copy()
    if not kl_df.empty:
        kl_df["outcome"] = np.where(kl_df["cer"] <= 5, "Correct (CER≤5)", "Incorrect (CER>5)")
        fig = px.histogram(
            kl_df,
            x="kl_divergence",
            color="outcome",
            barmode="overlay",
            nbins=50,
            opacity=0.6,
            title="KL Divergence: Correct vs Incorrect",
            labels={"kl_divergence": "KL Divergence", "outcome": ""},
            color_discrete_map={"Correct (CER≤5)": "#2ecc71", "Incorrect (CER>5)": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No KL divergence data available.")

    # ── 4. Confidence vs Semantic Error ───────────────────────────────
    st.subheader("Confidence vs Semantic Error")
    st.caption("Are models confidently wrong about *meaning*, not just characters?")

    sem_df = df.dropna(subset=["confidence", "semantic_error"]).copy()
    if not sem_df.empty:
        fig = px.scatter(
            sem_df,
            x="confidence",
            y="semantic_error",
            color="model",
            opacity=0.4,
            title="Confidence vs Semantic Error",
            labels={"confidence": "Confidence (%)", "semantic_error": "Semantic Error"},
            hover_data=["ground_truth", "predicted"],
        )
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)

        # Correlation
        for model in sem_df["model"].unique():
            mdf = sem_df[sem_df["model"] == model]
            corr = mdf[["confidence", "semantic_error"]].corr().iloc[0, 1]
            st.metric(f"{model} — Correlation(confidence, semantic_error)", f"{corr:.3f}")
    else:
        st.info("No semantic error + confidence data available.")

    # ── 5. Refusal Confidence ─────────────────────────────────────────
    st.subheader("Refusal Confidence")
    st.caption("When the model refuses, is it confident in its refusal or uncertain?")

    ref_df = conf_df.copy()
    ref_df["is_refusal"] = ref_df["category"] == "Refusal"

    refusal_with_conf = ref_df[ref_df["is_refusal"] & ref_df["confidence"].notna()]
    non_refusal_with_conf = ref_df[~ref_df["is_refusal"] & ref_df["confidence"].notna()]

    if len(refusal_with_conf) > 0:
        compare_df = pd.concat([
            refusal_with_conf.assign(group="Refusal"),
            non_refusal_with_conf.sample(n=min(2000, len(non_refusal_with_conf)), random_state=42).assign(group="Non-Refusal"),
        ])

        fig = px.box(
            compare_df,
            x="group",
            y="confidence",
            color="group",
            title="Confidence: Refusals vs Non-Refusals",
            labels={"confidence": "Confidence (%)", "group": ""},
            color_discrete_map={"Refusal": "#8e44ad", "Non-Refusal": "#3498db"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Refusal mean confidence", f"{refusal_with_conf['confidence'].mean():.1f}%")
            st.metric("Refusal count", len(refusal_with_conf))
        with col2:
            st.metric("Non-refusal mean confidence", f"{non_refusal_with_conf['confidence'].mean():.1f}%")
    else:
        st.info("No refusal predictions with confidence data found.")

    # ── 6. Mean Logprob Distribution ──────────────────────────────────
    st.subheader("Mean Log-Probability Distribution")

    lp_df = df.dropna(subset=["mean_logprob"]).copy()
    if not lp_df.empty:
        lp_df["outcome"] = np.where(lp_df["cer"] <= 5, "Correct (CER≤5)", "Incorrect (CER>5)")
        fig = px.histogram(
            lp_df,
            x="mean_logprob",
            color="outcome",
            barmode="overlay",
            nbins=50,
            opacity=0.6,
            title="Mean Log-Prob: Correct vs Incorrect",
            labels={"mean_logprob": "Mean Log Probability"},
            color_discrete_map={"Correct (CER≤5)": "#2ecc71", "Incorrect (CER>5)": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No mean_logprob data available.")
