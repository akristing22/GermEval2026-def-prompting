#!/usr/bin/env python3
"""
Specification curve (boxplot variant) — F1 Macro across prompting strategy configurations.

Keeps per-fold F1 values (read from results/final_run/score_table_folds.csv)
so each box shows fold-level variance.
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
from model_colors import get_model_colors
from common import (
    FINAL_RUN_FIGURES_DIR as FIGURES_DIR,
    MEASURE,
    SCORE_TABLE_FOLDS,
    save_figure,
)
import common

ACCENT = "#2979a0"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROUP_COLS = ["model", "prompt_mode", "demo_size", "emb_mode", "retrieval_mode"]
COLOR_COL  = "model"

SPEC_ORDER = {
    #"model":         ["gemma-4-E4B-it", "Qwen3.5-9B", "gemma-4-26B-A4B-it", "EuroLLM-22B-Instruct-2512"],
    "prompt_mode":   ["title", "description", "implicit", "explicit"],
    "demo_size":     ["0", "8"],
    "emb_mode":      ["None", "dense", "sparse", "fusion"],
    "retrieval_mode":["None", "random", "similarity", "diversity", "mmr"],
}
COL_LABEL = {
    #"model":         "Model",
    "prompt_mode":   "Prompt mode",
    "demo_size":     "Demo size",
    "emb_mode":      "Embedding",
    "retrieval_mode":"Retrieval",
}

FIGURE_HEIGHT = 5.5
CURVE_TICK_FS  = 8
SPEC_LABEL_FS  = 8
SPEC_HEAD_FS   = 9
MARKER_SIZE    = 7
MARKER_EW      = 2
AXIS_LABEL_FS  = 9


# ---------------------------------------------------------------------------
# Data loading — keep one row per (spec, fold)
# ---------------------------------------------------------------------------

def load_fold_scores(score_table_path: str = SCORE_TABLE_FOLDS) -> pd.DataFrame:
    """Load per-fold scores from score_table_folds.csv; demo_size as string
    so it lines up with the categorical SPEC_ORDER grid."""
    df = common.load_fold_scores(score_table_path)
    df["demo_size"] = df["demo_size"].astype(str)
    return df[GROUP_COLS + ["fold", MEASURE]]


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
                 group_cols, color_col, scatter_only=False, model_lines=True):
    """Render one specification curve column: violin (or points+mean) + spec indicator grid."""

    # ── top panel ────────────────────────────────────────────────────────────
    data_per_spec = [
        plot_df.loc[
            np.logical_and.reduce([plot_df[c] == agg.loc[j, c] for c in group_cols]),
            MEASURE,
        ].values
        for j in range(len(agg))
    ]

    if not scatter_only:
        vp = ax_curve.violinplot(
            data_per_spec,
            positions=list(range(len(agg))),
            widths=0.6,
            showmedians=True,
            showextrema=False,
        )
        for body in vp["bodies"]:
            body.set_facecolor(ACCENT)
            body.set_edgecolor(ACCENT)
            body.set_alpha(0.25)
            body.set_linewidth(1.2)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)

    rng = np.random.default_rng(42)
    for j in range(len(agg)):
        mask = np.logical_and.reduce([plot_df[c] == agg.loc[j, c] for c in group_cols])
        sub = plot_df.loc[mask]
        xs = j + rng.uniform(-0.12, 0.12, size=len(sub))
        for k, (_, row) in enumerate(sub.iterrows()):
            ax_curve.scatter(xs[k], row[MEASURE],
                             color=model_colors[row["model"]], s=18, alpha=0.85, zorder=3)

    if scatter_only and model_lines:
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

def _build_figure(plot_df, group_cols, color_col, title, name, scatter_only=False, model_lines=True):
    """Build and save one specification curve figure."""
    agg = (
        plot_df
        .groupby(group_cols, dropna=False)[MEASURE]
        .agg(["median", "std"])
        .reset_index()
        .sort_values("median", ascending=False)
        .reset_index(drop=True)
    )

    # scatter dot colors — always by model (fixed scheme from CLAUDE.md)
    models_sorted = sorted(plot_df["model"].unique())
    model_colors = get_model_colors(models_sorted)

    # spec-grid tick colors (for color_col row; gray if color_col not a spec dimension)
    cats_sorted = sorted(agg[color_col].unique()) if color_col in agg.columns else []
    if color_col == "model":
        method_colors = get_model_colors(cats_sorted)
    else:
        method_colors = dict(zip(cats_sorted, plt.cm.tab10.colors))

    n = len(agg)
    fig_w = max(10, n * 0.14 + 3)
    fig = plt.figure(figsize=(fig_w, FIGURE_HEIGHT))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.5, 2.2], hspace=0.03)
    ax_curve = fig.add_subplot(gs[0], axes_class=axislines.Axes)
    ax_specs = fig.add_subplot(gs[1], axes_class=axislines.Axes, sharex=ax_curve)

    _plot_column(ax_curve, ax_specs, agg, plot_df, method_colors, model_colors,
                 group_cols, color_col, scatter_only=scatter_only, model_lines=model_lines)

    ax_curve.set_ylabel("F1 Macro", fontsize=AXIS_LABEL_FS)
    ax_curve.tick_params(axis="y", labelsize=CURVE_TICK_FS)
    ax_curve.set_ylim(0.2, 0.8)
    baseline = ax_curve.axhline(0.64, color="grey", ls="--", lw=0.8, alpha=0.7, zorder=0)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=model_colors[m], markersize=8, label=m)
        for m in models_sorted
    ]
    handles.append(baseline)
    handles[-1].set_label("KNN baseline (0.64)")
    ax_curve.legend(handles=handles, title="Model", loc="lower right",
                    fontsize=7, title_fontsize=7, framealpha=0.8)

    fig.suptitle(title, fontsize=10, fontweight="bold", y=0.99)
    plt.subplots_adjust(top=0.95)
    save_figure(fig, FIGURES_DIR, name)
    plt.close(fig)


def main():
    print(f"Loading scores from: {SCORE_TABLE_FOLDS}")
    plot_df = load_fold_scores()
    print(f"  {len(plot_df)} fold entries")

    # Figure 1: model included as a spec dimension, colored by model
    group_cols_full = ["model", "prompt_mode", "demo_size", "emb_mode", "retrieval_mode"]
    print(f"  Figure 1: {plot_df.groupby(group_cols_full).ngroups} specs (with model)")
    _build_figure(
        plot_df,
        group_cols=group_cols_full,
        color_col="model",
        title="Specification Curve — F1 Macro (violins = cross-validation folds)",
        name="spec_curve_f1_boxplot",
    )

    # Figure 2: model excluded from spec dimensions — violins span all models × folds,
    # scatter dots colored by model
    group_cols_no_model = ["prompt_mode", "demo_size", "emb_mode", "retrieval_mode"]
    print(f"  Figure 2: {plot_df.groupby(group_cols_no_model).ngroups} specs (model pooled)")
    _build_figure(
        plot_df,
        group_cols=group_cols_no_model,
        color_col="model",
        title="Specification Curve — F1 Macro (points + mean, all models pooled)",
        name="spec_curve_f1_boxplot_nomodel",
        scatter_only=True,
        model_lines=False,
    )

    # Figures 3–6: one per model — violins = cross-validation folds, no model dimension
    for model_name in sorted(plot_df["model"].unique()):
        model_df = plot_df[plot_df["model"] == model_name].copy()
        n_specs = model_df.groupby(group_cols_no_model).ngroups
        safe_name = model_name.replace("/", "-")
        print(f"  Figure (per-model, {model_name}): {n_specs} specs")
        _build_figure(
            model_df,
            group_cols=group_cols_no_model,
            color_col="model",
            title=f"Specification Curve — {model_name} — F1 Macro (points + mean)",
            name=f"spec_curve_f1_boxplot_{safe_name}",
            scatter_only=True,
        )


if __name__ == "__main__":
    main()
