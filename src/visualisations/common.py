"""Shared paths, loaders, and plotting helpers for the visualisation scripts.

Conventions enforced here:
  - Figures are written next to their score tables: final-run figures to
    results/final_run/figures/, exploration figures to
    results/exploration/new/figures/, ablation figures to
    results/ablations/balance_X_order/figures/.
  - Plots are built only from the score tables (score_table_folds.csv /
    score_table_consolidated.csv), never from individual result CSVs.
  - Every figure is saved as both .pdf and .svg via save_figure().
  - Model colors come from model_colors.py — never define ad-hoc palettes.
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
from model_colors import MODEL_LABELS, get_model_colors

MEASURE = "f1_macro"

# 1-NN TF-IDF baseline: mean F1 Macro across the four folds (see
# src/knn_baseline.py); drawn as a reference line in several figures
KNN_BASELINE_F1 = 0.64

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "results"))

FINAL_RUN_DIR = os.path.join(_RESULTS_DIR, "final_run")
# "new" holds the second (current) exploration round
EXPLORATION_DIR = os.path.join(_RESULTS_DIR, "exploration", "new")
ABLATION_DIR = os.path.join(_RESULTS_DIR, "ablations", "balance_X_order")

FINAL_RUN_FIGURES_DIR = os.path.join(FINAL_RUN_DIR, "figures")
FINAL_RUN_TABLES_DIR = os.path.join(FINAL_RUN_DIR, "tables")
EXPLORATION_FIGURES_DIR = os.path.join(EXPLORATION_DIR, "figures")
EXPLORATION_TABLES_DIR = os.path.join(EXPLORATION_DIR, "tables")
ABLATION_FIGURES_DIR = os.path.join(ABLATION_DIR, "figures")
ABLATION_TABLES_DIR = os.path.join(ABLATION_DIR, "tables")

SCORE_TABLE_FOLDS = os.path.join(FINAL_RUN_DIR, "score_table_folds.csv")
SCORE_TABLE_CONSOLIDATED = os.path.join(FINAL_RUN_DIR, "score_table_consolidated.csv")
EXPLORATION_SCORE_TABLE = os.path.join(EXPLORATION_DIR, "score_table_consolidated.csv")
ABLATION_SCORE_TABLE = os.path.join(ABLATION_DIR, "score_table_consolidated.csv")

# Human-readable label for each valid (emb_mode, retrieval_mode) combination
RETRIEVAL_CONFIG_MAP = {
    ("None",   "None"):       "Zero-Shot",
    ("None",   "random"):     "Random",
    ("dense",  "similarity"): "Similarity (Dense)",
    ("sparse", "similarity"): "Similarity (Sparse)",
    ("fusion", "similarity"): "Similarity (Fusion)",
    ("dense",  "diversity"):  "Diversity (Dense)",
    ("dense",  "mmr"):        "MMR (Dense)",
}


def _load_score_table(path: str) -> pd.DataFrame:
    """Load a score table and normalize it for plotting: embedding_mode is
    renamed to emb_mode, demo_size becomes int, and missing config values
    (pandas reads the literal 'None' as NaN) become the string 'None'."""
    df = pd.read_csv(path)
    df = df.rename(columns={"embedding_mode": "emb_mode"})
    df["demo_size"] = df["demo_size"].fillna(0).astype(int)
    for col in ("demo_mode", "emb_mode", "retrieval_mode"):
        df[col] = df[col].fillna("None").astype(str)
    return df


def load_fold_scores(path: str = SCORE_TABLE_FOLDS) -> pd.DataFrame:
    """Final run: one row per config and fold."""
    return _load_score_table(path)


def load_consolidated_scores(path: str = SCORE_TABLE_CONSOLIDATED) -> pd.DataFrame:
    """Final run: one row per config, scored on the concatenated folds."""
    return _load_score_table(path)


def load_exploration_scores(path: str = EXPLORATION_SCORE_TABLE) -> pd.DataFrame:
    """Exploration: one row per config (single 70/30 runs, no folds)."""
    return _load_score_table(path)


def add_retrieval_config_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a human-readable 'retrieval_config' column combining
    emb_mode and retrieval_mode (see RETRIEVAL_CONFIG_MAP)."""
    df = df.copy()
    df["retrieval_config"] = [
        RETRIEVAL_CONFIG_MAP.get((emb, ret), f"{ret} ({emb})")
        for emb, ret in zip(df["emb_mode"], df["retrieval_mode"])
    ]
    return df


def save_figure(fig, figures_dir: str, name: str) -> str:
    """Save a figure as both .pdf and .svg into figures_dir; returns the .pdf path."""
    os.makedirs(figures_dir, exist_ok=True)
    pdf_path = os.path.join(figures_dir, f"{name}.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, f"{name}.svg"), bbox_inches="tight")
    print(f"Saved: {pdf_path} (+ .svg)")
    return pdf_path


# ---------------------------------------------------------------------------
# Shared impact figures (used for the config axes and for the ablation factors)
#
# Both take an impact_df with columns [model, factor, impact], where impact is
# the range of marginal mean F1 across the factor's values.
# ---------------------------------------------------------------------------

def plot_impact_tornado(impact_df: pd.DataFrame, models: list[str],
                        factor_order: list[str], factor_labels: dict[str, str],
                        figures_dir: str, name: str) -> None:
    """Tornado chart: one panel per model, horizontal bars = F1 range per factor."""
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
            .set_index("factor").reindex(factor_order).reset_index()
        )
        ys = list(range(len(sub)))
        bars = ax.barh(ys, sub["impact"].values, color=palette[model],
                       edgecolor="white", height=0.55)
        ax.set_yticks(ys)
        # Factor names only on the left column; the right column shares them
        if idx % ncols == 0:
            ax.set_yticklabels([factor_labels.get(f, f) for f in sub["factor"]], fontsize=9)
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
    save_figure(fig, figures_dir, name)
    plt.close(fig)


def plot_impact_heatmap(impact_df: pd.DataFrame, models: list[str],
                        factor_order: list[str], factor_labels: dict[str, str],
                        figures_dir: str, name: str) -> None:
    """Heatmap: model x factor, cell value/color = F1 range for that factor."""
    pivot = (
        impact_df
        .pivot(index="model", columns="factor", values="impact")
        .reindex(index=models, columns=factor_order)
    )
    pivot.columns = [factor_labels.get(c, c) for c in pivot.columns]
    pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(len(factor_order) * 1.9 + 1.2, len(models) * 1.1 + 1.2))
    sns.heatmap(
        pivot, ax=ax,
        annot=True, fmt=".3f",
        cmap="BuPu",
        linewidths=0.5, linecolor="white",
        cbar=False,
        annot_kws={"fontsize": 18},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18, rotation=0)
    plt.tight_layout()
    save_figure(fig, figures_dir, name)
    plt.close(fig)
