"""
dashboard/app.py
Not Flawless — OCR vs VLM Error Visualization Dashboard
"""

import streamlit as st

st.set_page_config(
    page_title="Not Flawless — OCR vs VLM Errors",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from data_loader import (
    load_data,
    get_models,
    filter_df,
    CATEGORY_ORDER,
    CATEGORY_COLORS,
)
import tab_overview
import tab_error_anatomy
import tab_explorer
import tab_confidence


# ── Load data ─────────────────────────────────────────────────────────
df, raw_examples = load_data()
all_models = get_models(df)


# ── Sidebar — Global Filters ─────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Not Flawless")
    st.caption("OCR vs VLM Error Visualization")
    st.divider()

    st.subheader("Global Filters")

    selected_models = st.multiselect(
        "Models",
        all_models,
        default=all_models,
        key="global_models",
    )

    cer_min = float(df["cer"].min()) if df["cer"].notna().any() else 0.0
    cer_max = min(float(df["cer"].max()), 200.0)
    cer_range = st.slider(
        "CER range",
        min_value=0.0,
        max_value=200.0,
        value=(0.0, 200.0),
        step=1.0,
        key="global_cer",
    )

    gt_min = int(df["gt_length"].min())
    gt_max = int(df["gt_length"].max())
    gt_range = st.slider(
        "Ground truth length (chars)",
        min_value=gt_min,
        max_value=gt_max,
        value=(gt_min, gt_max),
        key="global_gt_len",
    )

    selected_categories = st.multiselect(
        "Error categories",
        CATEGORY_ORDER,
        default=CATEGORY_ORDER,
        key="global_cats",
    )

    exclude_refusals = st.checkbox("Exclude refusals", value=False, key="global_excl_ref")

    st.divider()

    # Stats
    filtered = filter_df(
        df,
        models=selected_models,
        cer_range=cer_range,
        gt_length_range=gt_range,
        categories=selected_categories,
        exclude_refusals=exclude_refusals,
    )
    n_samples = filtered["sample_idx"].nunique()
    n_rows = len(filtered)
    st.metric("Samples", f"{n_samples:,}")
    st.metric("Predictions", f"{n_rows:,}")


# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔬 Error Anatomy",
    "🔍 Sample Explorer",
    "🎯 Confidence",
])

with tab1:
    tab_overview.render(filtered)

with tab2:
    tab_error_anatomy.render(filtered)

with tab3:
    tab_explorer.render(filtered, raw_examples)

with tab4:
    tab_confidence.render(filtered)
