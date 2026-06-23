#!/usr/bin/env python3
"""
Impact of demonstration ratio and ordering on F1 Macro.
Data: 2x3 factorial design (ratio x order) from results/ablations/balance_X_order/.

Four figures saved to results/ablations/balance_X_order/figures/:
  ablation_tornado.pdf/.svg     — per model, horizontal bars = marginal F1 range per factor
  ablation_heatmap.pdf/.svg     — model x factor heatmap, color = marginal F1 range
  ablation_cellmeans.pdf/.svg   — per model, ratio x order grid of F1 values
  ablation_interaction.pdf/.svg — per model, line plot: order on x-axis, ratio as two lines

Marginal impact per factor:
  impact(ratio) = max(mean_over_order(F1)) - min(mean_over_order(F1))
  impact(order) = max(mean_over_ratio(F1)) - min(mean_over_ratio(F1))
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MEASURE, save_figure
from model_colors import MODEL_LABELS, get_model_colors

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABLATIONS_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "results", "ablations"))
RESULTS_DIR = os.path.join(_ABLATIONS_DIR, "balance_X_order")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SCORE_TABLE = os.path.join(RESULTS_DIR, "score_table_consolidated.csv")

FACTORS = ["ratio", "order"]
FACTOR_LABELS = {"ratio": "Balance", "order": "Order"}

RATIO_ORDER = ["balanced", "proportional"]
ORDER_ORDER = ["random", "true-first", "true-last"]

RATIO_LABELS = {"balanced": "Balanced", "proportional": "Proportional"}
ORDER_LABELS  = {"random": "Random", "true-first": "True first", "true-last": "True last"}

# Canonical model display order from MODEL_LABELS
_MODEL_KEY_ORDER = list(MODEL_LABELS.keys())


def _sort_models(models) -> list[str]:
    keyed = [m for m in _MODEL_KEY_ORDER if m in models]
    rest  = sorted(m for m in models if m not in _MODEL_KEY_ORDER)
    return keyed + rest


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_df() -> pd.DataFrame:
    print(f"Loading scores from: {SCORE_TABLE}")
    df = pd.read_csv(SCORE_TABLE)
    print(f"  {len(df)} rows | ratio: {sorted(df['ratio'].unique())} | order: {sorted(df['order'].unique())}")
    return df


def compute_marginal_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, factor): max(marginal mean F1) - min(marginal mean F1)."""
    rows = []
    for model, grp in df.groupby("model"):
        for factor in FACTORS:
            marginal = grp.groupby(factor)[MEASURE].mean()
            rows.append({"model": model, "ablation": factor,
                          "impact": marginal.max() - marginal.min()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1: Tornado (marginal impacts, one panel per model)
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
            .set_index("ablation").reindex(FACTORS).reset_index()
        )
        ys = list(range(len(sub)))
        bars = ax.barh(ys, sub["impact"].values, color=palette[model],
                       edgecolor="white", height=0.55)
        ax.set_yticks(ys)
        if idx % ncols == 0:
            ax.set_yticklabels([FACTOR_LABELS.get(a, a) for a in sub["ablation"]], fontsize=9)
            ax.tick_params(axis="y", length=0)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=9, fontweight="bold")
        ax.set_xlim(0, x_max)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        for bar, val in zip(bars, sub["impact"].values):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=8)

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, FIGURES_DIR, name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Heatmap (marginal impacts, model x factor)
# ---------------------------------------------------------------------------

def plot_heatmap(impact_df: pd.DataFrame, models: list[str], name: str) -> None:
    pivot = (
        impact_df
        .pivot(index="model", columns="ablation", values="impact")
        .reindex(index=models, columns=FACTORS)
    )
    pivot.columns = [FACTOR_LABELS[c] for c in pivot.columns]
    pivot.index   = [MODEL_LABELS.get(m, m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(len(FACTORS) * 1.9 + 1.2, len(models) * 1.1 + 1.2))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", cbar=False,
                annot_kws={"fontsize": 18})
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18, rotation=0)

    plt.tight_layout()
    save_figure(fig, FIGURES_DIR, name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Cell-means heatmap (ratio x order per model)
# ---------------------------------------------------------------------------

def plot_cellmeans(df: pd.DataFrame, models: list[str], name: str) -> None:
    ncols = 2
    nrows = (len(models) + 1) // 2
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

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.0))
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
                    linewidth=1.5, markersize=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylim(y_min, y_max)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=7, loc="lower right")

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    save_figure(fig, FIGURES_DIR, name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_df()
    models = _sort_models(df["model"].unique())
    impact_df = compute_marginal_impact(df)

    plot_tornado(impact_df, models, "ablation_tornado")
    plot_heatmap(impact_df, models, "ablation_heatmap")
    plot_cellmeans(df, models, "ablation_cellmeans")
    plot_interaction(df, models, "ablation_interaction")


if __name__ == "__main__":
    main()
