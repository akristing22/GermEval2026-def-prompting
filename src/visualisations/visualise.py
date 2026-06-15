#!/usr/bin/env python3
"""
Visualisations for prompting strategy comparison.

Reads results/final_run/score_table_consolidated.csv and generates three
figures (each as .pdf and .svg) into results/final_run/figures/:
  1. specification_curve  — sorted F1 Macro + spec-choice grid
  2. heatmap_model_prompt — mean F1 Macro per model × prompt_mode
  3. violin_retrieval     — F1 Macro distribution by retrieval_mode

Note: specification_curve v0.3.9 runs OLS regressions and does not accept
pre-computed metrics; the curve below follows the same Simonsohn et al. (2020)
visual convention, implemented directly with matplotlib.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_colors import get_model_colors, MODEL_LABELS
from common import (
    FINAL_RUN_FIGURES_DIR as FIGURES_DIR,
    SCORE_TABLE_CONSOLIDATED as SCORE_TABLE,
    load_consolidated_scores,
    save_figure,
)

# ---------------------------------------------------------------------------
# Spec dimensions (order matches the filename format in CLAUDE.md)
# ---------------------------------------------------------------------------
SPEC_DIMS = [
    "model", "prompt_mode", "demo_size",
    "emb_mode", "retrieval_mode"
]
DIM_LABELS = [
    "Model", "Prompt mode", "Demo. size",
    "Embed. mode", "Retrieval"
]

MODELS = MODEL_LABELS

ACCENT = "#2979a0"
GREY   = "#bbbbbb"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(score_table_path: str = SCORE_TABLE) -> pd.DataFrame:
    """Load pre-computed scores from score_table_consolidated.csv."""
    scores = load_consolidated_scores(score_table_path)
    return scores[SPEC_DIMS + ["f1_macro"]]


# ---------------------------------------------------------------------------
# 1. Specification curve
# ---------------------------------------------------------------------------

def plot_specification_curve(scores: pd.DataFrame) -> None:
    """
    Specification curve (Simonsohn et al. 2020 convention):
    - Top panel  : F1 Macro values sorted ascending, one dot per configuration.
    - Bottom panel: dot grid — filled dot = that choice was active for each spec.
    """
    df = scores.sort_values("f1_macro").reset_index(drop=True)
    n = len(df)

    def _order(vals: list) -> list:
        """Sort categories; push 'None' and 'False' to the end."""
        vals = [str(v) for v in vals]
        tail = [v for v in ("False", "None") if v in vals]
        rest = sorted(v for v in vals if v not in tail)
        return rest + tail

    dim_cats = {d: _order(df[d].unique()) for d in SPEC_DIMS}
    n_rows   = sum(len(v) for v in dim_cats.values())

    ROW_H  = 0.30
    fig_h  = 4.0 + n_rows * ROW_H
    fig_w  = max(10.0, n * 0.35)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[4.0, n_rows * ROW_H], hspace=0.04)
    ax_t = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_t)

    # ── top panel ─────────────────────────────────────────────────────────────
    xs = np.arange(n)
    ax_t.plot(xs, df["f1_macro"], color=ACCENT, lw=1.5, zorder=2)
    ax_t.scatter(xs, df["f1_macro"], color=ACCENT, s=22, zorder=3)
    ax_t.axhline(0.64, color="grey", ls="--", lw=0.8, alpha=0.7, label="KNN baseline (0.64)")
    ax_t.set_ylabel("F1 Macro", fontsize=11)
    ax_t.set_title("Specification Curve — F1 Macro across prompting strategy configurations",
                   fontsize=12, pad=8)
    ax_t.set_ylim(0, 1.0)
    ax_t.legend(fontsize=8, loc="upper left", frameon=False)
    ax_t.tick_params(bottom=False, labelbottom=False)
    ax_t.spines[["top", "right"]].set_visible(False)

    # ── bottom panel ──────────────────────────────────────────────────────────
    ax_b.set_xlim(-0.5, n - 0.5)

    y          = 0
    ytick_pos  = []
    ytick_lbls = []
    group_mids = []         # (mid data-y, dim label)
    separators = []         # data-y positions of group boundaries

    for dim, dim_label in zip(SPEC_DIMS, DIM_LABELS):
        cats    = dim_cats[dim]
        y_start = y
        for cat in cats:
            active = df[dim].astype(str) == cat
            # inactive outline dots
            ax_b.scatter(xs, [y] * n,
                         color="none", edgecolors=GREY, s=14, lw=0.6, zorder=1)
            # filled active dots
            ax_b.scatter(xs[active], [y] * int(active.sum()),
                         color=ACCENT, s=14, zorder=2)
            ytick_pos.append(y)
            ytick_lbls.append(f"  {cat}")
            y += 1
        group_mids.append(((y_start + y - 1) / 2, dim_label))
        separators.append(y - 0.5)

    ax_b.set_yticks(ytick_pos)
    ax_b.set_yticklabels(ytick_lbls, fontsize=7)
    ax_b.set_ylim(-0.5, y - 0.5)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Specification (sorted by F1 Macro)", fontsize=10)
    ax_b.tick_params(left=True, bottom=False, labelbottom=False, length=0)
    ax_b.spines[["top", "right", "bottom", "left"]].set_visible(False)

    for sep in separators[:-1]:
        ax_b.axhline(sep, color="#e0e0e0", lw=0.8)

    # dimension group labels on the right edge
    # with invert_yaxis: axes frac 1 = data y=-0.5, frac 0 = data y=y-0.5
    total = y
    for mid_y, dlabel in group_mids:
        frac_y = 1.0 - (mid_y + 0.5) / total
        ax_b.annotate(
            dlabel,
            xy=(1.0, frac_y), xycoords="axes fraction",
            xytext=(5, 0), textcoords="offset points",
            fontsize=8, fontweight="bold", va="center", ha="left",
        )

    save_figure(fig, FIGURES_DIR, "specification_curve")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Heatmap — model × prompt_mode
# ---------------------------------------------------------------------------

def plot_heatmap(scores: pd.DataFrame) -> None:
    """Mean F1 Macro per model × prompt_mode combination."""
    pivot = scores.pivot_table(
        values="f1_macro", index="model", columns="prompt_mode", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.8), max(4, len(pivot) * 1.2)))
    sns.heatmap(
        pivot, ax=ax,
        annot=True, fmt=".2f",
        cmap="Blues", vmin=0.0, vmax=1.0,
        linewidths=0.5, linecolor="#eeeeee",
        cbar_kws={"label": "F1 Macro", "shrink": 0.8},
    )
    ax.set_title("Mean F1 Macro — Model × Prompt mode", fontsize=12, pad=10)
    ax.set_xlabel("Prompt mode", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    save_figure(fig, FIGURES_DIR, "heatmap_model_prompt")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Violin — F1 Macro by retrieval_mode
# ---------------------------------------------------------------------------

def plot_violin_by_retrieval(scores: pd.DataFrame) -> None:
    """
    F1 Macro distribution per retrieval_mode, sorted by group mean.
    Falls back to a strip plot when any group has fewer than 3 observations
    (too few points for a meaningful violin shape).
    Points are coloured by model.
    """
    df = scores.copy()
    df["retrieval_mode"] = df["retrieval_mode"].astype(str)

    # sort violins by mean F1 ascending
    means = df.groupby("retrieval_mode")["f1_macro"].mean().sort_values()
    order = means.index.tolist()

    min_group = df.groupby("retrieval_mode")["f1_macro"].count().min()

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 1.8), 5))

    if min_group >= 3:
        sns.violinplot(
            data=df, x="retrieval_mode", y="f1_macro", hue="retrieval_mode",
            order=order, ax=ax,
            palette="Blues", inner=None, linewidth=1.2, cut=0, legend=False,
        )
        for poly in ax.collections:
            poly.set_facecolor((*matplotlib.colors.to_rgb(ACCENT), 0.15))
            poly.set_edgecolor(ACCENT)
            poly.set_linewidth(1.2)
    else:
        sns.boxplot(
            data=df, x="retrieval_mode", y="f1_macro",
            order=order, ax=ax,
            palette="Blues", linewidth=1.2,
        )

    models = sorted(df["model"].unique())
    model_palette = get_model_colors(models)

    sns.stripplot(
        data=df, x="retrieval_mode", y="f1_macro", hue="model",
        order=order, hue_order=models, ax=ax,
        palette=model_palette, size=5, alpha=0.85, jitter=True, dodge=False,
    )

    chance_line = ax.axhline(0.64, color="grey", ls="--", lw=0.8, alpha=0.7)

    # legend: model colours + chance line
    handles, labels = ax.get_legend_handles_labels()
    labels = [MODELS[label] for label in labels]
    handles.append(chance_line)
    labels.append("KNN baseline (0.64)")
    ax.legend(handles, labels, fontsize=8, loc="lower right", frameon=False,
              title="Model", title_fontsize=8)
    
    ax.set_xticklabels(["zero-shot" if lbl.get_text() == "None" else lbl.get_text()
                        for lbl in ax.get_xticklabels()])
    ax.set_title("F1 Macro by Retrieval Mode", fontsize=12, pad=8)
    ax.set_xlabel("Retrieval mode", fontsize=10)
    ax.set_ylabel("F1 Macro", fontsize=10)
    ax.set_ylim(0.1, 0.9)
    ax.spines[["top", "right"]].set_visible(False)

    save_figure(fig, FIGURES_DIR, "violin_retrieval")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Violin — F1 Macro by prompt_mode
# ---------------------------------------------------------------------------

def plot_violin_by_prompt(scores: pd.DataFrame) -> None:
    """
    F1 Macro distribution per prompt_mode, sorted by group mean.
    Falls back to a strip plot when any group has fewer than 3 observations
    (too few points for a meaningful violin shape).
    Points are coloured by model.
    """
    df = scores.copy()
    df["prompt_mode"] = df["prompt_mode"].astype(str)

    # sort violins by mean F1 ascending
    means = df.groupby("prompt_mode")["f1_macro"].mean().sort_values()
    order = means.index.tolist()

    min_group = df.groupby("prompt_mode")["f1_macro"].count().min()

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 1.8), 5))

    if min_group >= 3:
        sns.violinplot(
            data=df, x="prompt_mode", y="f1_macro", hue="prompt_mode",
            order=order, ax=ax,
            palette="Blues", inner=None, linewidth=1.2, cut=0, legend=False,
        )
        for poly in ax.collections:
            poly.set_facecolor((*matplotlib.colors.to_rgb(ACCENT), 0.15))
            poly.set_edgecolor(ACCENT)
            poly.set_linewidth(1.2)
    else:
        sns.boxplot(
            data=df, x="prompt_mode", y="f1_macro",
            order=order, ax=ax,
            palette="Blues", linewidth=1.2,
        )

    models = sorted(df["model"].unique())
    model_palette = get_model_colors(models)

    sns.stripplot(
        data=df, x="prompt_mode", y="f1_macro", hue="model",
        order=order, hue_order=models, ax=ax,
        palette=model_palette, size=5, alpha=0.85, jitter=True, dodge=False,
    )

    chance_line = ax.axhline(0.64, color="grey", ls="--", lw=0.8, alpha=0.7)

    # legend: model colours + chance line
    handles, labels = ax.get_legend_handles_labels()
    labels = [MODELS[label] for label in labels]
    handles.append(chance_line)
    labels.append("KNN baseline (0.64)")
    ax.legend(handles, labels, fontsize=8, loc="lower right", frameon=False,
              title="Model", title_fontsize=8)
    
    ax.set_xticklabels(["zero-shot" if lbl.get_text() == "None" else lbl.get_text()
                        for lbl in ax.get_xticklabels()])
    ax.set_title("F1 Macro by Prompt Mode", fontsize=12, pad=8)
    ax.set_xlabel("Prompt mode", fontsize=10)
    ax.set_ylabel("F1 Macro", fontsize=10)
    ax.set_ylim(0.1, 0.9)
    ax.spines[["top", "right"]].set_visible(False)

    save_figure(fig, FIGURES_DIR, "violin_prompt")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(score_table_path: str = SCORE_TABLE) -> None:
    print(f"Loading scores from: {score_table_path}")
    scores = load_scores(score_table_path)
    print(f"  {len(scores)} unique configurations")
    print(scores[SPEC_DIMS + ["f1_macro"]].to_string(index=False))
    print()

    plot_specification_curve(scores)
    plot_heatmap(scores)
    plot_violin_by_retrieval(scores)
    plot_violin_by_prompt(scores)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
