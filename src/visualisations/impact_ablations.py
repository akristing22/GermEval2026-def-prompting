#!/usr/bin/env python3
"""
Impact of demonstration ratio and ordering on F1 Macro.
Data: 2x3 factorial design (ratio x order) from results/ablations/balance_X_order/.

Five figures saved to results/ablations/balance_X_order/figures/:
  ablation_tornado                — per model, horizontal bars = marginal F1 range per factor
  ablation_heatmap                — model x factor heatmap, color = marginal F1 range
  ablation_cellmeans              — per model, ratio x order grid of F1 values
  ablation_interaction            — per model, line plot: order on x-axis, ratio as two lines
  ablation_interaction_combined   — all models in one panel, color=model, style=ratio

Marginal impact per factor:
  impact(ratio) = max(mean_over_order(F1)) - min(mean_over_order(F1))
  impact(order) = max(mean_over_ratio(F1)) - min(mean_over_ratio(F1))
  impact(interaction) = range of the interaction residuals from the additive model
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    ABLATION_FIGURES_DIR as FIGURES_DIR,
    ABLATION_SCORE_TABLE as SCORE_TABLE,
    MEASURE,
    plot_impact_heatmap,
    plot_impact_tornado,
    save_figure,
)
from model_colors import MODEL_LABELS, get_model_colors, sort_models

FACTORS = ["ratio", "order", "interaction"]
FACTOR_LABELS = {"ratio": "Ratio", "order": "Order", "interaction": "Interaction"}

RATIO_ORDER = ["balanced", "proportional"]
ORDER_ORDER = ["random", "true-first", "true-last"]

RATIO_LABELS = {"balanced": "Balanced", "proportional": "Proportional"}
ORDER_LABELS  = {"random": "Random", "true-first": "True first", "true-last": "True last"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_df() -> pd.DataFrame:
    print(f"Loading scores from: {SCORE_TABLE}")
    df = pd.read_csv(SCORE_TABLE)
    print(f"  {len(df)} rows | ratio: {sorted(df['ratio'].unique())} | order: {sorted(df['order'].unique())}")
    return df


def compute_marginal_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, factor): max(marginal mean F1) - min(marginal mean F1).
    For 'interaction': range of interaction residuals from the additive model
    (δ(r,o) = F1(r,o) − mean_ratio(r) − mean_order(o) + grand_mean).
    Returns the [model, factor, impact] format the common plot helpers expect."""
    rows = []
    for model, grp in df.groupby("model"):
        for factor in ("ratio", "order"):
            marginal = grp.groupby(factor)[MEASURE].mean()
            rows.append({"model": model, "factor": factor,
                         "impact": marginal.max() - marginal.min()})
        cell = grp.groupby(["ratio", "order"])[MEASURE].mean()
        grand = cell.mean()
        ratio_means = grp.groupby("ratio")[MEASURE].mean()
        order_means = grp.groupby("order")[MEASURE].mean()
        residuals = [
            f1 - ratio_means[r] - order_means[o] + grand
            for (r, o), f1 in cell.items()
        ]
        rows.append({"model": model, "factor": "interaction",
                     "impact": max(residuals) - min(residuals)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 3: Cell-means heatmap (ratio x order per model)
# ---------------------------------------------------------------------------

def plot_cellmeans(df: pd.DataFrame, models: list[str], name: str) -> None:
    ncols = 2
    nrows = (len(models) + 1) // 2
    # One shared color scale so the panels are comparable across models
    vmin, vmax = df[MEASURE].min(), df[MEASURE].max()

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 2.8))
    axes_flat = np.array(axes).flatten()

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        sub = df[df["model"] == model]
        pivot = (
            sub.pivot(index="ratio", columns="order", values=MEASURE)
            .reindex(index=RATIO_ORDER, columns=ORDER_ORDER)
        )
        pivot.index   = [RATIO_LABELS.get(r, r) for r in pivot.index]
        pivot.columns = [ORDER_LABELS.get(o, o)  for o in pivot.columns]

        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="Blues",
                    vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="white",
                    cbar=False, annot_kws={"fontsize": 10})
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8, rotation=0)

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    save_figure(fig, FIGURES_DIR, name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Interaction plot (order on x-axis, ratio as lines, per model)
# ---------------------------------------------------------------------------

def plot_interaction(df: pd.DataFrame, models: list[str], name: str) -> None:
    ncols = 2
    nrows = (len(models) + 1) // 2

    palette = get_model_colors(models)
    linestyles = {"balanced": "--", "proportional": "-"}
    markers    = {"balanced": "s",  "proportional": "o"}

    x_pos    = list(range(len(ORDER_ORDER)))
    x_labels = [ORDER_LABELS[o] for o in ORDER_ORDER]
    y_pad    = 0.05
    y_min    = df[MEASURE].min() - y_pad
    y_max    = df[MEASURE].max() + y_pad

    # The ratio legend is identical in every panel — show it only once
    legend_model = "gemma-4-26B-A4B-it"

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5, nrows * 2.1))
    axes_flat = np.array(axes).flatten()

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        sub = df[df["model"] == model]
        color = palette[model]

        for ratio in RATIO_ORDER:
            rsub = sub[sub["ratio"] == ratio].set_index("order").reindex(ORDER_ORDER)
            ax.plot(x_pos, rsub[MEASURE].values,
                    linestyle=linestyles[ratio], marker=markers[ratio],
                    color=color, label=RATIO_LABELS[ratio],
                    linewidth=1.8, markersize=5)

        is_bottom = idx >= (nrows - 1) * ncols
        is_left   = idx % ncols == 0

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels if is_bottom else [], fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=10, labelleft=is_left, left=is_left)
        if model == legend_model:
            ax.legend(fontsize=9, loc="lower right")

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.6)
    save_figure(fig, FIGURES_DIR, name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Interaction plot — all models collapsed into one panel
# ---------------------------------------------------------------------------

def plot_interaction_combined(df: pd.DataFrame, models: list[str], name: str) -> None:
    palette    = get_model_colors(models)
    linestyles = {"balanced": "--", "proportional": "-"}
    markers    = {"balanced": "s",  "proportional": "o"}

    x_pos    = list(range(len(ORDER_ORDER)))
    x_labels = [ORDER_LABELS[o] for o in ORDER_ORDER]
    y_pad    = 0.05
    y_min    = df[MEASURE].min() - y_pad
    y_max    = df[MEASURE].max() + y_pad

    fig, ax = plt.subplots(figsize=(3.8, 3.5))

    for model in models:
        sub   = df[df["model"] == model]
        color = palette[model]
        for ratio in RATIO_ORDER:
            rsub = sub[sub["ratio"] == ratio].set_index("order").reindex(ORDER_ORDER)
            ax.plot(x_pos, rsub[MEASURE].values,
                    linestyle=linestyles[ratio], marker=markers[ratio],
                    color=color, linewidth=1.8, markersize=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(y_min, y_max)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=10)

    # Two legend groups: model (color) and ratio (line style/marker)
    model_handles = [
        Line2D([0], [0], color=palette[m], linewidth=1.8,
               label=MODEL_LABELS.get(m, m))
        for m in models
    ]
    ratio_handles = [
        Line2D([0], [0], color="#444444", linestyle=linestyles[r], marker=markers[r],
               markersize=5, linewidth=1.8, label=RATIO_LABELS[r])
        for r in RATIO_ORDER
    ]
    ax.legend(handles=model_handles + ratio_handles,
              fontsize=8, loc="lower center", ncols=2,
              framealpha=0.9)

    plt.tight_layout(pad=0.5)
    save_figure(fig, FIGURES_DIR, name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_df()
    models = sort_models(df["model"].unique())
    impact_df = compute_marginal_impact(df)

    plot_impact_tornado(impact_df, models, FACTORS, FACTOR_LABELS, FIGURES_DIR, "ablation_tornado")
    plot_impact_heatmap(impact_df, models, FACTORS, FACTOR_LABELS, FIGURES_DIR, "ablation_heatmap")
    plot_cellmeans(df, models, "ablation_cellmeans")
    plot_interaction(df, models, "ablation_interaction")
    plot_interaction_combined(df, models, "ablation_interaction_combined")


if __name__ == "__main__":
    main()
