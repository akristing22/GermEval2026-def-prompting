"""Shared paths, score-table loaders, and figure saving for the visualisation scripts.

Conventions enforced here:
  - Final-run figures go to results/final_run/figures/,
    exploration figures to results/exploration/figures/.
  - Plots are built only from the score tables (score_table_folds.csv /
    score_table_consolidated.csv), never from individual result CSVs.
  - Every figure is saved as both .pdf and .svg via save_figure().
"""
import os

import pandas as pd

MEASURE = "f1_macro"

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "results"))

FINAL_RUN_DIR = os.path.join(_RESULTS_DIR, "final_run")
EXPLORATION_DIR = os.path.join(_RESULTS_DIR, "exploration")

FINAL_RUN_FIGURES_DIR = os.path.join(FINAL_RUN_DIR, "figures")
EXPLORATION_FIGURES_DIR = os.path.join(EXPLORATION_DIR, "figures")

SCORE_TABLE_FOLDS = os.path.join(FINAL_RUN_DIR, "score_table_folds.csv")
SCORE_TABLE_CONSOLIDATED = os.path.join(FINAL_RUN_DIR, "score_table_consolidated.csv")
EXPLORATION_SCORE_TABLE = os.path.join(EXPLORATION_DIR, "score_table_consolidated.csv")


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


def save_figure(fig, figures_dir: str, name: str) -> str:
    """Save a figure as both .pdf and .svg into figures_dir; returns the .pdf path."""
    os.makedirs(figures_dir, exist_ok=True)
    pdf_path = os.path.join(figures_dir, f"{name}.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, f"{name}.svg"), bbox_inches="tight")
    print(f"Saved: {pdf_path} (+ .svg)")
    return pdf_path
