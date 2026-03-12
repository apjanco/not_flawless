"""
dashboard/tab_explorer.py
Tab 3 — Sample Explorer with character-level diffs
"""

import difflib
import html
import io
import base64

import pandas as pd
import streamlit as st
from PIL import Image

from data_loader import CATEGORY_ORDER


# ── diff rendering ─────────────────────────────────────────────────────

def _char_diff_html(ground_truth: str, predicted: str) -> str:
    """
    Produce an HTML string that highlights character-level differences
    between ground_truth and predicted.

    Green  = matching chars
    Red bg = substitution
    Blue bg + underline = insertion (in predicted, not in GT)
    Strikethrough gray = deletion (in GT, not in predicted)
    """
    gt = ground_truth or ""
    pred = predicted or ""

    sm = difflib.SequenceMatcher(None, gt, pred)

    gt_parts = []   # annotated ground truth
    pred_parts = []  # annotated prediction

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        gt_chunk = html.escape(gt[i1:i2])
        pred_chunk = html.escape(pred[j1:j2])

        if op == "equal":
            gt_parts.append(f'<span style="color:#2ecc71">{gt_chunk}</span>')
            pred_parts.append(f'<span style="color:#2ecc71">{pred_chunk}</span>')
        elif op == "replace":
            gt_parts.append(
                f'<span style="background:#fce4e4;color:#c0392b;text-decoration:line-through">{gt_chunk}</span>'
            )
            pred_parts.append(
                f'<span style="background:#fce4e4;color:#c0392b;font-weight:bold">{pred_chunk}</span>'
            )
        elif op == "delete":
            gt_parts.append(
                f'<span style="background:#f5f5f5;color:#999;text-decoration:line-through">{gt_chunk}</span>'
            )
            # nothing in pred
        elif op == "insert":
            # nothing in gt
            pred_parts.append(
                f'<span style="background:#d6eaf8;color:#2471a3;text-decoration:underline">{pred_chunk}</span>'
            )

    gt_html = "".join(gt_parts)
    pred_html = "".join(pred_parts)
    return gt_html, pred_html


def _pil_to_base64(img: Image.Image, max_width: int = 800) -> str:
    """Convert a PIL image to a base64-encoded PNG data URI."""
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── main render ────────────────────────────────────────────────────────

def render(df: pd.DataFrame, raw_examples: list[dict]):
    st.header("🔍 Sample Explorer")

    if df.empty:
        st.warning("No data matches the current filters.")
        return

    # ── Filters specific to explorer ──────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        sort_by = st.selectbox(
            "Sort samples by",
            ["Sample index", "Highest mean CER", "Lowest mean CER", "Most model disagreement"],
            key="explorer_sort",
        )
    with col_f2:
        filter_cat = st.multiselect(
            "Filter by category (any model)",
            CATEGORY_ORDER,
            default=[],
            key="explorer_cat",
        )
    with col_f3:
        only_disagreements = st.checkbox("Only show disagreements", key="explorer_disagree")

    # ── Identify available samples ────────────────────────────────────
    # Group by sample and collect per-model info
    sample_ids = df["sample_idx"].unique()

    sample_stats = []
    for sid in sample_ids:
        sdf = df[df["sample_idx"] == sid]
        cers = sdf["cer"].dropna()
        cats = sdf["category"].tolist()

        if filter_cat and not any(c in filter_cat for c in cats):
            continue

        if only_disagreements:
            # At least one model has CER=0 and one has CER>10
            if not (cers.min() <= 5 and cers.max() > 15):
                continue

        sample_stats.append({
            "sample_idx": sid,
            "mean_cer": cers.mean() if len(cers) > 0 else 999,
            "cer_range": cers.max() - cers.min() if len(cers) > 1 else 0,
        })

    if not sample_stats:
        st.info("No samples match the current explorer filters.")
        return

    stats_df = pd.DataFrame(sample_stats)

    if sort_by == "Highest mean CER":
        stats_df = stats_df.sort_values("mean_cer", ascending=False)
    elif sort_by == "Lowest mean CER":
        stats_df = stats_df.sort_values("mean_cer", ascending=True)
    elif sort_by == "Most model disagreement":
        stats_df = stats_df.sort_values("cer_range", ascending=False)
    else:
        stats_df = stats_df.sort_values("sample_idx")

    ordered_samples = stats_df["sample_idx"].tolist()
    total_samples = len(ordered_samples)

    # ── Pagination ────────────────────────────────────────────────────
    st.markdown(f"**{total_samples}** samples match filters")

    col_prev, col_idx, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("◀ Prev", key="prev"):
            if st.session_state.get("explorer_pos", 0) > 0:
                st.session_state["explorer_pos"] -= 1
    with col_next:
        if st.button("Next ▶", key="next"):
            if st.session_state.get("explorer_pos", 0) < total_samples - 1:
                st.session_state["explorer_pos"] += 1

    pos = st.session_state.get("explorer_pos", 0)
    pos = min(pos, total_samples - 1)

    with col_idx:
        new_pos = st.number_input(
            "Go to sample",
            min_value=1,
            max_value=total_samples,
            value=pos + 1,
            key="explorer_goto",
        )
        if new_pos - 1 != pos:
            pos = new_pos - 1
            st.session_state["explorer_pos"] = pos

    current_sample_idx = ordered_samples[pos]

    st.divider()

    # ── Display sample ────────────────────────────────────────────────
    example = raw_examples[current_sample_idx]
    sample_df = df[df["sample_idx"] == current_sample_idx].sort_values("model")

    # Image
    col_img, col_text = st.columns([2, 3])

    with col_img:
        pil_img = example["image"]
        img_b64 = _pil_to_base64(pil_img)
        st.markdown(
            f'<img src="data:image/png;base64,{img_b64}" style="width:100%;border:1px solid #ddd;border-radius:4px" />',
            unsafe_allow_html=True,
        )
        st.caption(f"Sample {pos + 1} / {total_samples}  (index {current_sample_idx})")

    with col_text:
        gt = example["text"]
        st.markdown(f"**Ground Truth:**")
        st.markdown(
            f'<div style="font-family:monospace;font-size:1.1em;padding:8px;background:#f8f9fa;border-radius:4px;margin-bottom:12px">{html.escape(gt)}</div>',
            unsafe_allow_html=True,
        )

        # Per-model predictions with diffs
        for _, row in sample_df.iterrows():
            model = row["model"]
            pred = str(row.get("predicted", "") or "")
            cer = row.get("cer")
            wer = row.get("wer")
            conf = row.get("confidence")
            cat = row.get("category", "")

            gt_html, pred_html = _char_diff_html(gt, pred)

            # Metrics line
            metrics_parts = []
            if cer is not None and not pd.isna(cer):
                metrics_parts.append(f"CER: {cer:.1f}%")
            if wer is not None and not pd.isna(wer):
                metrics_parts.append(f"WER: {wer:.1f}%")
            if conf is not None and not pd.isna(conf):
                metrics_parts.append(f"Conf: {conf:.1f}%")
            metrics_str = " &nbsp;|&nbsp; ".join(metrics_parts)

            # Category badge
            cat_colors = {
                "Perfect": "#2ecc71", "Minor": "#82e0aa", "Moderate": "#f4d03f",
                "Major": "#e67e22", "Hallucination": "#e74c3c", "Refusal": "#8e44ad", "Error": "#95a5a6",
            }
            badge_color = cat_colors.get(cat, "#95a5a6")
            badge = f'<span style="background:{badge_color};color:white;padding:2px 8px;border-radius:10px;font-size:0.8em">{cat}</span>'

            st.markdown(
                f"""
                <div style="border:1px solid #eee;border-radius:6px;padding:10px;margin-bottom:10px">
                    <div style="margin-bottom:6px"><strong>{html.escape(model)}</strong> {badge}</div>
                    <div style="font-family:monospace;font-size:1.05em;line-height:1.6;margin-bottom:6px">{pred_html}</div>
                    <div style="font-size:0.85em;color:#666">{metrics_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Legend
    with st.expander("Diff color legend"):
        st.markdown("""
        - <span style="color:#2ecc71">■</span> **Green** — Correct (matches ground truth)
        - <span style="color:#c0392b;font-weight:bold">■</span> **Red** — Substitution (wrong characters)
        - <span style="color:#2471a3;text-decoration:underline">■</span> **Blue underline** — Insertion (extra text not in GT)
        - <span style="color:#999;text-decoration:line-through">■</span> **Gray strikethrough** — Deletion (missing from prediction)
        """, unsafe_allow_html=True)
