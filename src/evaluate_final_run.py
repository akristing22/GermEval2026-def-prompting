"""Compute score tables for the final experiment run.

Reads the result CSVs of the full experiment and writes two score tables:

  - {RESULTS_DIR}/score_table_folds.csv         one row per config and fold,
    scored on that fold's test split
  - {RESULTS_DIR}/score_table_consolidated.csv  one row per config, scored on
    the consolidated full-dataset results (all four folds concatenated by
    consolidate_folds.py)

Config parameters are parsed from the result filenames
({model}_{prompt}_{demo_mode}_{demo_size}_{embedding}_{retrieval}[_fold-N][_thinking].csv).
Rows where the model abstained (predicted_label is NaN) are excluded from
scoring; per-row abstention counts are included in the tables. Metrics follow
the project conventions: F1 Macro as the primary metric, reported alongside
accuracy and per-class precision/recall.

Run from src/. Standalone script; does not import util.py.
"""

import os
import re

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

RESULTS_DIR = "../results/ablations/static"
CONSOLIDATED_DIR = os.path.join(RESULTS_DIR, "consolidated")

CONFIG_COLUMNS = ["model", "prompt_mode", "demo_mode", "demo_size",
                  "embedding_mode", "retrieval_mode", "thinking_mode"]
METRIC_COLUMNS = ["n_scored", "n_abstained", "accuracy", "f1_macro", "cohen_kappa",
                  "precision_true", "recall_true", "precision_false", "recall_false"]


def parse_filename(filename: str) -> dict | None:
    """
    Parse the experiment config out of a result filename.

    Returns a dict with the CONFIG_COLUMNS plus "fold" (int or None for
    consolidated files), or None when the filename does not follow the
    result naming scheme (e.g. score tables, baseline files).
    """
    spec = filename.removesuffix(".csv")

    thinking = spec.endswith("_thinking")
    spec = spec.removesuffix("_thinking")

    fold = None
    fold_match = re.search(r"_fold-(\d+)$", spec)
    if fold_match:
        fold = int(fold_match.group(1))
        spec = spec[:fold_match.start()]

    # Split from the right: the last five fields are fixed, everything before
    # belongs to the model name (which may itself contain underscores)
    parts = spec.rsplit("_", 5)
    if len(parts) != 6:
        return None

    return {
        "model": parts[0],
        "prompt_mode": parts[1],
        "demo_mode": parts[2],
        "demo_size": parts[3],
        "embedding_mode": parts[4],
        "retrieval_mode": parts[5],
        "thinking_mode": thinking,
        "fold": fold,
    }


def score_file(path: str) -> dict:
    """
    Compute all metrics for one result CSV.

    Abstentions (NaN predicted_label) are excluded from scoring but counted.
    Precision/recall are reported per class (True = prosecutable).
    """
    df = pd.read_csv(path)

    mask = pd.notna(df["predicted_label"])
    y_true = df.loc[mask, "DEF"].astype(bool)
    y_pred = df.loc[mask, "predicted_label"].astype(bool)

    return {
        "n_scored": int(mask.sum()),
        "n_abstained": int((~mask).sum()),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        "precision_true": round(precision_score(y_true, y_pred, pos_label=True, zero_division=0), 4),
        "recall_true": round(recall_score(y_true, y_pred, pos_label=True, zero_division=0), 4),
        "precision_false": round(precision_score(y_true, y_pred, pos_label=False, zero_division=0), 4),
        "recall_false": round(recall_score(y_true, y_pred, pos_label=False, zero_division=0), 4),
    }


def build_score_table(directory: str, with_fold: bool) -> pd.DataFrame:
    """
    Score every parseable result CSV in a directory.

    Args:
        directory: Directory containing result CSVs (not searched recursively).
        with_fold: True for per-fold files (keeps the fold column and requires
                   a fold suffix), False for consolidated files.

    Returns:
        One row per scored file, sorted by F1 Macro (descending).
    """
    rows = []

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".csv") or filename.startswith("score_table"):
            continue

        config = parse_filename(filename)
        if config is None or (config["fold"] is None) == with_fold:
            print(f"Skipping {filename} (does not match expected naming scheme)")
            continue

        scores = score_file(os.path.join(directory, filename))
        if scores["n_abstained"] > 0:
            print(f"{filename}: scored on {scores['n_scored']} samples "
                  f"({scores['n_abstained']} abstained)")

        rows.append({**config, **scores})

    table = pd.DataFrame(rows, columns=CONFIG_COLUMNS + ["fold"] + METRIC_COLUMNS)
    if not with_fold:
        table = table.drop(columns="fold")

    return table.sort_values(by="f1_macro", ascending=False, ignore_index=True)


def main():
    #print(f"Scoring per-fold results in {RESULTS_DIR}...")
    #folds_table = build_score_table(RESULTS_DIR, with_fold=True)
    #folds_path = os.path.join(RESULTS_DIR, "score_table_folds.csv")
    #folds_table.to_csv(folds_path, index=False)
    #print(f"{len(folds_table)} rows written to {folds_path}\n")

    print(f"Scoring consolidated results in {CONSOLIDATED_DIR}...")
    consolidated_table = build_score_table(CONSOLIDATED_DIR, with_fold=False)
    consolidated_path = os.path.join(RESULTS_DIR, "score_table_consolidated.csv")
    consolidated_table.to_csv(consolidated_path, index=False)
    print(f"{len(consolidated_table)} rows written to {consolidated_path}\n")

    print("Top configurations by consolidated F1 Macro:")
    print(consolidated_table[CONFIG_COLUMNS + ["f1_macro"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
