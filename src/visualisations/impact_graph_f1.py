#!/usr/bin/env python3
"""
Impact of each configuration axis on F1 Macro.

Three figures:
  1. impact_tornado.pdf   — per model, horizontal bars = F1 range per axis
  2. impact_marginal.pdf  — per axis, per-value marginal mean F1 ± SE by model
  3. impact_heatmap.pdf   — model × axis heatmap, color = F1 range

Impact is defined as: max(per-value mean F1) − min(per-value mean F1), where the
mean for each value is computed by marginalizing over all folds and all other axes.

For emb_mode and retrieval_mode, rows where the axis value is 'None' are excluded
(those correspond to zero-shot or random-retrieval configs where the axis is inactive).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_graph_f1 import load_fold_scores
from common import (
    FINAL_RUN_FIGURES_DIR as FIGURES_DIR,
    MEASURE,
    SCORE_TABLE_FOLDS,
    save_figure,
)
from model_colors import get_model_colors

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AXES = ["prompt_mode", "demo_size", "emb_mode", "retrieval_mode"]

AXIS_LABELS = {
    "prompt_mode":    "Prompt mode",
    "demo_size":      "Demo size",
    "emb_mode":       "Embedding",
    "retrieval_mode": "Retrieval",
}

# Preferred display order for each axis; any unlisted values are appended sorted
VALUE_ORDER = {
    "prompt_mode":    ["title", "description", "implicit", "explicit"],
    "demo_size":      ["0", "8"],
    "emb_mode":       ["dense", "sparse", "fusion"],
    "retrieval_mode": ["random", "similarity", "diversity", "mmr"],
}


def _ordered_values(axis: str, present: set) -> list[str]:
    """Values in display order, restricted to what actually appears in the data."""
    base = [v for v in VALUE_ORDER.get(axis, []) if v in present]
    base += sorted(v for v in present if v not in base)
    return base


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_marginal_means(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Per axis: marginal mean F1 and SE per (model, axis_value), averaging over
    folds and all other axes. 'None' values are excluded (axis inactive).
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
    """
    rows = []
    for axis, df in marginal.items():
        for model, grp in df.groupby("model"):
            impact = grp["mean"].max() - grp["mean"].min()
            rows.append({"model": model, "axis": axis, "impact": impact})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1: Tornado chart
# ---------------------------------------------------------------------------

def plot_tornado(impact_df: pd.DataFrame, models: list[str], name: str) -> None:
    ncols = 2
    nrows = (len(models) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows * 3.2))
    axes_flat = np.array(axes).flatten()

    palette = get_model_colors(models)
    x_max = impact_df["impact"].max() * 1.1

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        sub = (
            impact_df[impact_df["model"] == model]
            .set_index("axis").reindex(AXES).reset_index()
        )
        ys = list(range(len(sub)))
        bars = ax.barh(ys, sub["impact"].values, color=palette[model],
                       edgecolor="white", height=0.55)
        ax.set_yticks(ys)
        if idx % ncols == 0:
            ax.set_yticklabels([AXIS_LABELS.get(a, a) for a in sub["axis"]], fontsize=9)
            ax.tick_params(axis="y", length=0)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        # if idx // ncols == nrows - 1:
        #     ax.set_xlabel("F1 range  (max mean − min mean)", fontsize=8)
        ax.set_title(model, fontsize=9, fontweight="bold")
        ax.set_xlim(0, x_max)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        for bar, val in zip(bars, sub["impact"].values):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=8)

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Sensitivity to each configuration axis\n"
                 "F1 Macro range across axis values (marginal means)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, FIGURES_DIR, name)
    plt.close(fig)



# ---------------------------------------------------------------------------
# Figure 2: Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(impact_df: pd.DataFrame, models: list[str], name: str) -> None:
    pivot = (
        impact_df
        .pivot(index="model", columns="axis", values="impact")
        .reindex(index=models, columns=AXES)
    )
    pivot.columns = [AXIS_LABELS[c] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(len(AXES) * 1.9 + 1.2, len(models) * 1.1 + 1.2))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "F1 range  (impact)", "shrink": 0.7},
        annot_kws={"fontsize": 10},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    ax.set_title(
        "Impact of each configuration axis on F1 Macro\n"
        "(range of per-value marginal means, per model)",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    save_figure(fig, FIGURES_DIR, name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading scores from: {SCORE_TABLE_FOLDS}")
    df = load_fold_scores()
    print(f"  {len(df)} fold entries across {df['model'].nunique()} models")

    models = sorted(df["model"].unique())
    marginal = compute_marginal_means(df)
    impact_df = compute_impact(marginal)

    plot_tornado(impact_df, models, "impact_tornado")
    plot_heatmap(impact_df, models, "impact_heatmap")


if __name__ == "__main__":
    main()
