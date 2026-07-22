#!/usr/bin/env python3
"""
Impact of each configuration axis on F1 Macro.

Reads results/final_run/score_table_consolidated.csv and writes two figures
to results/final_run/figures/:
  1. impact_tornado — per model, horizontal bars = F1 range per axis
  2. impact_heatmap — model × axis heatmap, color = F1 range

Impact is defined as: max(per-value mean F1) − min(per-value mean F1), where the
mean for each value is computed by marginalizing over all other axes.

For emb_mode and retrieval_mode, rows where the axis value is 'None' are excluded
(those correspond to zero-shot or random-retrieval configs where the axis is inactive).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    FINAL_RUN_FIGURES_DIR as FIGURES_DIR,
    MEASURE,
    SCORE_TABLE_CONSOLIDATED,
    load_consolidated_scores,
    plot_impact_heatmap,
    plot_impact_tornado,
)
from model_colors import sort_models

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AXES = ["prompt_mode", "demo_size", "emb_mode", "retrieval_mode"]

AXIS_LABELS = {
    "prompt_mode":    "Conditioning",
    "demo_size":      "0-/8-shot",
    "emb_mode":       "Embedding",
    "retrieval_mode": "Retrieval",
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_marginal_means(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Per axis: marginal mean F1 and SE per (model, axis_value), averaging over
    all other axes. 'None' values are excluded (axis inactive).
    """
    result = {}
    for axis in AXES:
        sub = df[df[axis].astype(str) != "None"].copy()
        grp = sub.groupby(["model", axis])[MEASURE]
        stats = grp.agg(["mean", "std", "count"]).reset_index()
        stats["se"] = stats["std"] / np.sqrt(stats["count"])
        result[axis] = stats
    return result


def compute_impact(marginal: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Per (model, axis): impact = max(mean) − min(mean) across axis values.
    Returns the [model, factor, impact] format the common plot helpers expect.
    """
    rows = []
    for axis, df in marginal.items():
        for model, grp in df.groupby("model"):
            impact = grp["mean"].max() - grp["mean"].min()
            rows.append({"model": model, "factor": axis, "impact": impact})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading scores from: {SCORE_TABLE_CONSOLIDATED}")
    df = load_consolidated_scores()
    print(f"  {len(df)} configurations across {df['model'].nunique()} models")

    models = sort_models(df["model"].unique())
    marginal = compute_marginal_means(df)
    impact_df = compute_impact(marginal)

    # The model itself is not an axis of the tornado/heatmap; report its
    # impact (range of per-model means) on the console for reference
    model_means = df.groupby("model")[MEASURE].mean()
    model_impact = model_means.max() - model_means.min()
    print(f"Model impact (F1 range across models): {model_impact:.4f}")

    plot_impact_tornado(impact_df, models, AXES, AXIS_LABELS, FIGURES_DIR, "impact_tornado")
    plot_impact_heatmap(impact_df, models, AXES, AXIS_LABELS, FIGURES_DIR, "impact_heatmap")


if __name__ == "__main__":
    main()
