#!/usr/bin/env python3
"""
Specification curve — F1 Macro across prompting strategy configurations.

Uses consolidated per-config mean F1 (read from results/final_run/score_table_consolidated.csv).
Adapted from specification_graph.py (another project) for this research project.

Dimensions shown in spec grid: model, prompt_mode, demo_size, embedding_mode, retrieval_mode.
Color-coded by model.

Output: results/final_run/figures/spec_curve_f1_boxplot.{pdf,svg} (+ variants)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axisartist import axislines

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_colors import get_label_colors, MODEL_LABELS
from common import (
    FINAL_RUN_FIGURES_DIR as FIGURES_DIR,
    MEASURE,
    SCORE_TABLE_CONSOLIDATED,
    save_figure,
)
import common

ACCENT = "#2979a0"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROUP_COLS = ["model", "prompt_mode", "retrieval_config"]
COLOR_COL  = "model"

SPEC_ORDER = {
    #"model":         ["gemma-4-E4B-it", "Qwen3.5-9B", "gemma-4-26B-A4B-it", "EuroLLM-22B-Instruct-2512"],
    "prompt_mode":      ["Title", "Description", "Implicit", "Explicit"],
    "retrieval_config": ["Zero-Shot", "Random", "Similarity (Dense)", "Similarity (Sparse)",
                         "Similarity (Fusion)", "Diversity (Dense)", "MMR (Dense)"],
}
COL_LABEL = {
    #"model":         "Model",
    "prompt_mode":      "Prompt",
    "retrieval_config": "Retrieval",
}

_RETRIEVAL_CONFIG_MAP = {
    ("None",    "None"):       "Zero-Shot",
    ("None",    "random"):     "Random",
    ("dense",   "similarity"): "Similarity (Dense)",
    ("sparse",  "similarity"): "Similarity (Sparse)",
    ("fusion",  "similarity"): "Similarity (Fusion)",
    ("dense",   "diversity"):  "Diversity (Dense)",
    ("dense",   "mmr"):        "MMR (Dense)",
}

FIGURE_HEIGHT = 5.5
CURVE_TICK_FS  = 8
SPEC_LABEL_FS  = 8
SPEC_HEAD_FS   = 9
MARKER_SIZE    = 9
MARKER_EW      = 3
AXIS_LABEL_FS  = 9


# ---------------------------------------------------------------------------
# Data loading — one row per config (consolidated across folds)
# ---------------------------------------------------------------------------

def load_consolidated_scores(score_table_path: str = SCORE_TABLE_CONSOLIDATED) -> pd.DataFrame:
    """Load consolidated scores from score_table_consolidated.csv."""
    df = common.load_consolidated_scores(score_table_path)
    df["retrieval_config"] = (
        df[["emb_mode", "retrieval_mode"]]
        .apply(lambda r: _RETRIEVAL_CONFIG_MAP.get((r["emb_mode"], r["retrieval_mode"])), axis=1)
    )
    df["prompt_mode"] = df["prompt_mode"].str.capitalize()
    df["model"] = df["model"].map(lambda m: MODEL_LABELS.get(m, m))
    return df[GROUP_COLS + [MEASURE]]


# ---------------------------------------------------------------------------
# Plotting helpers (adapted from specification_graph.py)
# ---------------------------------------------------------------------------

def _format_axes(ax_curve, ax_specs):
    ax_curve.set_xticks([])
    ax_curve.axis["left"].major_ticks.set_tick_out(True)
    for spine in ("top", "right"):
        ax_specs.axis[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax_specs.axis[spine].line.set_visible(False)
        ax_specs.axis[spine].major_ticks.set_ticksize(0)
        ax_specs.axis[spine].minor_ticks.set_ticksize(0)
    for spine in ("top", "right", "bottom"):
        ax_curve.axis[spine].set_visible(False)


def _plot_column(ax_curve, ax_specs, agg, plot_df, method_colors, model_colors,
                 group_cols, color_col, model_lines=True):
    """Render one specification curve column: points + spec indicator grid."""

    # ── top panel ────────────────────────────────────────────────────────────
    best_idx = plot_df[MEASURE].idxmax()
    rng = np.random.default_rng(42)
    for j in range(len(agg)):
        mask = np.logical_and.reduce([plot_df[c] == agg.loc[j, c] for c in group_cols])
        sub = plot_df.loc[mask]
        xs = j + rng.uniform(-0.12, 0.12, size=len(sub))
        for k, (idx, row) in enumerate(sub.iterrows()):
            is_best = (idx == best_idx)
            ax_curve.scatter(xs[k], row[MEASURE],
                             color=model_colors[row["model"]],
                             s=60 if is_best else 18,
                             marker="*" if is_best else "o",
                             alpha=0.95 if is_best else 0.85,
                             zorder=4 if is_best else 3)

    if model_lines:
        for model in sorted(plot_df["model"].unique()):
            xs_line, ys_line = [], []
            for j in range(len(agg)):
                mask = np.logical_and.reduce([plot_df[c] == agg.loc[j, c] for c in group_cols])
                sub = plot_df.loc[mask & (plot_df["model"] == model)]
                if len(sub) > 0:
                    xs_line.append(j)
                    ys_line.append(sub[MEASURE].mean())
            if xs_line:
                ax_curve.plot(xs_line, ys_line, color=model_colors[model],
                              lw=1.0, alpha=0.6, zorder=2)

    ax_curve.set_xlim(-0.5, len(agg) - 0.5)

    # ── bottom panel: spec indicator grid ────────────────────────────────────
    y_pos = {}
    base_y, y_off = 1, 0.8
    minor_ys, minor_lbls = [], []
    major_ys, major_lbls = [], []

    for var in reversed(group_cols):
        cats = SPEC_ORDER.get(var, [str(v) for v in sorted(agg[var].unique())])
        for k, val in enumerate(cats):
            y_pos[(var, str(val))] = base_y + k * y_off
            minor_ys.append(base_y + k * y_off)
            minor_lbls.append(str(val))
        major_ys.append(base_y + (k + 0.7) * y_off)
        major_lbls.append(COL_LABEL.get(var, var))
        base_y += len(cats) + 1

    # draw markers
    for j in range(len(agg)):
        for var in reversed(group_cols):
            val = str(agg.loc[j, var])
            y = y_pos.get((var, val))
            if y is None:
                continue
            color = method_colors.get(val, "black") if var == color_col else "gray"
            ax_specs.plot(j, y, marker="|", color=color,
                          markersize=MARKER_SIZE, markeredgewidth=MARKER_EW)

    # horizontal guide lines
    for y in minor_ys:
        ax_specs.axhline(y, color="lightgray", linewidth=0.5, zorder=0)
    # vertical grid
    step = max(1, len(agg) // 8)
    for x in range(step, len(agg) - 1, step):
        for ax in (ax_specs, ax_curve):
            ax.axvline(x - 0.25, color="lightgray", linewidth=0.5, alpha=0.5, zorder=0)

    # y-axis labels
    ax_specs.set_yticks(minor_ys, labels=minor_lbls, minor=True)
    ax_specs.tick_params(axis="y", which="minor", labelsize=SPEC_LABEL_FS)
    ax_specs.set_yticks(major_ys, labels=major_lbls)
    ax_specs.axis["left"].major_ticklabels.set(
        ha="right", va="bottom",
        fontsize=SPEC_HEAD_FS, fontweight="bold",
    )
    ax_specs.axis["left"].minor_ticklabels.set(fontsize=SPEC_LABEL_FS)

    _format_axes(ax_curve, ax_specs)

    ymin, ymax = ax_specs.get_ylim()
    ax_specs.set_ylim(ymin, ymax + 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_figure(plot_df, group_cols, color_col, title, name, model_lines=True):
    """Build and save one specification curve figure."""
    agg = (
        plot_df
        .groupby(group_cols, dropna=False)[MEASURE]
        .mean()
        .reset_index()
        .sort_values(MEASURE, ascending=False)
        .reset_index(drop=True)
    )

    # scatter dot colors — always by model (fixed scheme from CLAUDE.md)
    models_sorted = sorted(plot_df["model"].unique())
    model_colors = get_label_colors(models_sorted)

    # spec-grid tick colors (for color_col row; gray if color_col not a spec dimension)
    cats_sorted = sorted(agg[color_col].unique()) if color_col in agg.columns else []
    if color_col == "model":
        method_colors = get_label_colors(cats_sorted)
    else:
        method_colors = dict(zip(cats_sorted, plt.cm.tab10.colors))

    n = len(agg)
    fig_w = max(10, n * 0.01 + 3)
    fig = plt.figure(figsize=(fig_w, FIGURE_HEIGHT))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.5, 2.2], hspace=0.03)
    ax_curve = fig.add_subplot(gs[0], axes_class=axislines.Axes)
    ax_specs = fig.add_subplot(gs[1], axes_class=axislines.Axes, sharex=ax_curve)

    _plot_column(ax_curve, ax_specs, agg, plot_df, method_colors, model_colors,
                 group_cols, color_col, model_lines=model_lines)

    ax_curve.set_ylabel("F1 Macro", fontsize=AXIS_LABEL_FS)
    ax_curve.tick_params(axis="y", labelsize=CURVE_TICK_FS)
    ax_curve.set_ylim(0, 0.8)
    baseline = ax_curve.axhline(0.64, color="grey", ls="--", lw=0.8, alpha=0.7, zorder=0)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=model_colors[m], markersize=8, label=m)
        for m in models_sorted
    ]
    handles.append(baseline)
    handles[-1].set_label("KNN baseline (0.64)")
    ax_curve.legend(handles=handles, 
                    #title="Model", 
                    loc="lower left",
                    fontsize=7, title_fontsize=7, framealpha=0.8)

    fig.suptitle(title, fontsize=10, fontweight="bold", y=0.99)
    plt.subplots_adjust(top=0.95)
    save_figure(fig, FIGURES_DIR, name)
    plt.close(fig)


def main():
    print(f"Loading scores from: {SCORE_TABLE_CONSOLIDATED}")
    plot_df = load_consolidated_scores()
    print(f"  {len(plot_df)} configs")

    # Figure 1: model included as a spec dimension, colored by model
    group_cols_full = ["model", "prompt_mode", "retrieval_config"]
    print(f"  Figure 1: {plot_df.groupby(group_cols_full).ngroups} specs (with model)")
    _build_figure(
        plot_df,
        group_cols=group_cols_full,
        color_col="model",
        title="Specification Curve — F1 Macro (mean across folds)",
        name="spec_curve_f1_boxplot",
    )

    # Figure 2: model excluded from spec dimensions — dots per model, colored by model
    group_cols_no_model = ["prompt_mode", "retrieval_config"]
    print(f"  Figure 2: {plot_df.groupby(group_cols_no_model).ngroups} specs (model pooled)")
    _build_figure(
        plot_df,
        group_cols=group_cols_no_model,
        color_col="model",
        title="Specification Curve — F1 Macro (all models, mean across folds)",
        name="spec_curve_f1_boxplot_nomodel",
        model_lines=False,
    )

    # Figures 3–6: one per model, no model dimension
    for model_name in sorted(plot_df["model"].unique()):
        model_df = plot_df[plot_df["model"] == model_name].copy()
        n_specs = model_df.groupby(group_cols_no_model).ngroups
        safe_name = model_name.replace("/", "-")
        print(f"  Figure (per-model, {model_name}): {n_specs} specs")
        _build_figure(
            model_df,
            group_cols=group_cols_no_model,
            color_col="model",
            title=f"Specification Curve — {model_name} — F1 Macro (mean across folds)",
            name=f"spec_curve_f1_boxplot_{safe_name}",
        )


if __name__ == "__main__":
    main()
