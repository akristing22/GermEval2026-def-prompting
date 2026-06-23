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

_RETRIEVAL_CONFIG_MAP = {
    ("None",   "None"):       "Zero-Shot",
    ("None",   "random"):     "Random",
    ("dense",  "similarity"): "Similarity (Dense)",
    ("sparse", "similarity"): "Similarity (Sparse)",
    ("fusion", "similarity"): "Similarity (Fusion)",
    ("dense",  "diversity"):  "Diversity (Dense)",
    ("dense",  "mmr"):        "MMR (Dense)",
}

PROMPT_MARKERS = {
    "Title":       "o",
    "Description": "s",
    "Implicit":    "^",
    "Explicit":    "D",
}

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
    "Model", "Conditioning", "Demo. size",
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
    ax.set_title("Mean F1 Macro — Model × Conditioning", fontsize=12, pad=10)
    ax.set_xlabel("Conditioning", fontsize=10)
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
    F1 Macro distribution per retrieval configuration (embedding mode + retrieval mode
    combined), sorted by group mean. Points are coloured by model and shaped by
    prompt_mode. Falls back to a box plot when any group has fewer than 3 observations.
    """
    df = scores.copy()
    df["retrieval_config"] = df.apply(
        lambda r: _RETRIEVAL_CONFIG_MAP.get(
            (r["emb_mode"], r["retrieval_mode"]),
            f"{r['retrieval_mode']} ({r['emb_mode']})",
        ),
        axis=1,
    )
    df["prompt_mode"] = df["prompt_mode"].str.capitalize()

    means = df.groupby("retrieval_config")["f1_macro"].mean().sort_values()
    order = means.index.tolist()
    x_idx = {v: i for i, v in enumerate(order)}

    min_group = df.groupby("retrieval_config")["f1_macro"].count().min()

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.8), 5))

    if min_group >= 3:
        sns.violinplot(
            data=df, x="retrieval_config", y="f1_macro", hue="retrieval_config",
            order=order, ax=ax,
            palette="Blues", inner=None, linewidth=1.2, cut=0, legend=False,
        )
        for poly in ax.collections:
            poly.set_facecolor((*matplotlib.colors.to_rgb(ACCENT), 0.15))
            poly.set_edgecolor(ACCENT)
            poly.set_linewidth(1.2)
    else:
        sns.boxplot(
            data=df, x="retrieval_config", y="f1_macro",
            order=order, ax=ax,
            palette="Blues", linewidth=1.2,
        )

    models = sorted(df["model"].unique())
    model_palette = get_model_colors(models)
    prompt_modes_present = [p for p in PROMPT_MARKERS if p in df["prompt_mode"].unique()]

    rng = np.random.default_rng(42)
    for prompt in prompt_modes_present:
        for model in models:
            sub = df[(df["prompt_mode"] == prompt) & (df["model"] == model)]
            if sub.empty:
                continue
            xs = sub["retrieval_config"].map(x_idx).values.astype(float)
            xs += rng.uniform(-0.15, 0.15, size=len(sub))
            ax.scatter(
                xs, sub["f1_macro"].values,
                color=model_palette[model],
                marker=PROMPT_MARKERS[prompt],
                s=25, alpha=0.85, zorder=3,
            )

    chance_line = ax.axhline(0.64, color="grey", ls="--", lw=0.8, alpha=0.7)

    model_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=model_palette[m], markersize=7,
                   label=MODELS.get(m, m))
        for m in models
    ]
    shape_handles = [
        plt.Line2D([0], [0], marker=PROMPT_MARKERS[p], color="w",
                   markerfacecolor="#555555", markeredgecolor="#555555",
                   markersize=7, label=p)
        for p in prompt_modes_present
    ]
    chance_handle = plt.Line2D([0], [0], color="grey", ls="--", lw=0.8,
                               label="KNN baseline (0.64)")
    ax.legend(
        handles=model_handles + shape_handles + [chance_handle],
        fontsize=8, loc="lower right", frameon=False,
        title="Model / Conditioning", title_fontsize=8,
    )

    ax.tick_params(axis="x", rotation=15)
    ax.set_title("F1 Macro by Retrieval Configuration", fontsize=12, pad=8)
    ax.set_xlabel("Retrieval configuration", fontsize=10)
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
    ax.set_title("F1 Macro by Conditioning", fontsize=12, pad=8)
    ax.set_xlabel("Conditioning", fontsize=10)
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

    plot_heatmap(scores)
    plot_violin_by_retrieval(scores)
    plot_violin_by_prompt(scores)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
