"""Visualise exploration results: F1 macro by demonstration size (4, 8, 16, 32).

Reads results/exploration/score_table_consolidated.csv and writes
f1_by_demo_size.{pdf,svg} to results/exploration/figures/.
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_colors import get_model_colors, MODEL_LABELS
from common import (
    EXPLORATION_FIGURES_DIR as FIGURES_DIR,
    EXPLORATION_SCORE_TABLE,
    load_exploration_scores,
    save_figure,
)

DEMO_SIZES = [4, 8, 16, 32]

# Colour + marker per model — consistent across subplots
MODEL_STYLE = {
    model: {"color": color, "marker": "o"}
    for model, color in get_model_colors(MODEL_LABELS.keys()).items()
}


def plot_f1_by_demo_size(df: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)

    per_model_means = {}
    for model in MODEL_LABELS:
        model_data = df[df["model"] == model]
        if model_data.empty:
            continue
        style = MODEL_STYLE[model]

        means = (
            model_data.groupby("demo_size")["f1_macro"]
            .mean()
            .reindex(DEMO_SIZES)
            .dropna()
        )

        if means.empty:
            continue
        per_model_means[model] = means
        ax.plot(
            means.index,
            means.values,
            color=style["color"],
            marker=style["marker"],
            markersize=5,
            linewidth=1.6,
        )
        y_nudge = 0.015 if model == "gemma-4-26B-A4B-it" else -0.015
        ax.text(
            means.index[-1] + 0.5,
            means.values[-1] + y_nudge,
            MODEL_LABELS[model],
            color=style["color"],
            fontsize=10,
            va="center",
        )

    if per_model_means:
        mean_across_models = (
            pd.DataFrame(per_model_means).mean(axis=1).reindex(DEMO_SIZES).dropna()
        )
        ax.plot(
            mean_across_models.index,
            mean_across_models.values,
            color="black",
            marker="o",
            markersize=5,
            linewidth=1.6,
            linestyle="--",
        )
        ax.text(
            mean_across_models.index[-1] + 0.5,
            mean_across_models.values[-1],
            "Mean",
            color="black",
            fontsize=10,
            va="center",
        )

    ax.set_xlabel("Demonstration size (k)", fontsize=12)
    ax.set_ylabel("F1 Macro", fontsize=12)
    ax.set_xticks(DEMO_SIZES)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(labelsize=11)
    ax.set_xlim(2, 44)
    ax.set_ylim(0.30, 0.80)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    #ax.set_title("F1 Macro by Demonstration Size", fontsize=12)

    save_figure(fig, FIGURES_DIR, name)
    plt.close(fig)


if __name__ == "__main__":
    print(f"Loading scores from: {EXPLORATION_SCORE_TABLE}")
    df = load_exploration_scores()
    print(f"Loaded {len(df)} runs across {df['model'].nunique()} models")
    plot_f1_by_demo_size(df, "f1_by_demo_size")
