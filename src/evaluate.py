"""Compute score tables from experiment result CSVs.

Usage (from src/):
    python evaluate.py final_run
    python evaluate.py balance_X_order
    python evaluate.py static
    python evaluate.py final_run --results-dir ../results/exploration/new

The positional argument selects the filename schema:
  - final_run       standard result filenames
                    ({model}_{prompt}_{demo_mode}_{demo_size}_{embedding}_{retrieval}
                    [_fold-N][_thinking].csv); also used for exploration and
                    competition-style result folders via --results-dir
  - balance_X_order ablation filenames ({model}_{ratio}_{order}[_fold-N].csv)
  - static          static demonstration ablation (standard filenames,
                    results in results/ablations/static/)

One score table is written into the results directory:
  - score_table_consolidated.csv  one row per config, scored on the merged
                                  full-dataset results that consolidate_folds.py
                                  writes to the consolidated/ subfolder
Per-fold result files (_fold-N suffix) are ignored.

Rows where the model abstained (predicted_label is NaN) are excluded from
scoring; per-row abstention counts are included in the tables. Metrics follow
the project conventions: F1 Macro as the primary metric, reported alongside
accuracy, Cohen's kappa, and per-class precision/recall
(True = prosecutable under §§185-187 StGB).

Standalone script; does not import util.py.
"""

import argparse
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

DEFAULT_RESULTS_DIRS = {
    "final_run": "../results/final_run",
    "balance_X_order": "../results/ablations/balance_X_order",
    "static": "../results/ablations/static",
}

CONFIG_COLUMNS = {
    "final_run": ["model", "prompt_mode", "demo_mode", "demo_size",
                  "embedding_mode", "retrieval_mode", "thinking_mode"],
    "balance_X_order": ["model", "ratio", "order"],
}
# The static demonstration ablation uses the standard result filename format,
# so it shares the final_run config columns and parser
CONFIG_COLUMNS["static"] = CONFIG_COLUMNS["final_run"]

METRIC_COLUMNS = ["n_scored", "n_abstained", "accuracy", "f1_macro", "cohen_kappa",
                  "precision_true", "recall_true", "precision_false", "recall_false"]

# Ablation filename fields ({model}_{ratio}_{order}); the model name may itself
# contain "_", so ratio/order are matched against their known vocabulary
_ABLATION_PATTERN = re.compile(
    r"^(?P<model>.+)_(?P<ratio>balanced|proportional)_(?P<order>random|true_first|true_last)$"
)


def _split_fold_suffix(spec: str) -> tuple[str, int | None]:
    """Cut a "_fold-N" suffix off an extension-less filename spec.

    Returns (spec, fold); fold is None for consolidated files.
    """
    fold_match = re.search(r"_fold-(\d+)$", spec)
    if fold_match:
        return spec[:fold_match.start()], int(fold_match.group(1))
    return spec, None


def parse_filename_final_run(filename: str) -> dict | None:
    """Parse a standard-format result filename into its config fields.

    Returns a dict with the final_run CONFIG_COLUMNS plus "fold" (int, or None
    for consolidated files), or None when the filename does not follow the
    naming scheme (e.g. baseline files).
    """
    spec = filename.removesuffix(".csv")

    thinking = spec.endswith("_thinking")
    spec = spec.removesuffix("_thinking")

    spec, fold = _split_fold_suffix(spec)

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


def parse_filename_balance_x_order(filename: str) -> dict | None:
    """Parse an ablation result filename ({model}_{ratio}_{order}[_fold-N].csv).

    The order value is normalised from the filename's underscore form
    (true_first) to the hyphenated form used in the score tables (true-first).
    """
    spec, fold = _split_fold_suffix(filename.removesuffix(".csv"))
    match = _ABLATION_PATTERN.match(spec)
    if match is None:
        return None

    return {
        "model": match["model"],
        "ratio": match["ratio"],
        "order": match["order"].replace("_", "-"),
        "fold": fold,
    }


PARSERS = {
    "final_run": parse_filename_final_run,
    "balance_X_order": parse_filename_balance_x_order,
    "static": parse_filename_final_run,
}


def score_file(path: str) -> dict:
    """Compute all metrics for one result CSV.

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


def build_score_table(directory: str, schema: str) -> pd.DataFrame:
    """Score every parseable consolidated result CSV in a directory.

    Per-fold files (_fold-N suffix) are skipped silently.

    Args:
        directory: Directory containing result CSVs (not searched recursively).
        schema:    Filename schema key (see PARSERS).

    Returns:
        One row per scored file, sorted by F1 Macro (descending).
    """
    parse = PARSERS[schema]
    rows = []

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".csv") or filename.startswith("score_table"):
            continue

        config = parse(filename)
        if config is None:
            print(f"Skipping {filename} (does not match expected naming scheme)")
            continue
        if config["fold"] is not None:
            continue

        scores = score_file(os.path.join(directory, filename))
        if scores["n_abstained"] > 0:
            print(f"{filename}: scored on {scores['n_scored']} samples "
                  f"({scores['n_abstained']} abstained)")

        rows.append({**config, **scores})

    table = pd.DataFrame(rows, columns=CONFIG_COLUMNS[schema] + METRIC_COLUMNS)
    return table.sort_values(by="f1_macro", ascending=False, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("schema", choices=sorted(PARSERS),
                        help="filename schema of the result CSVs")
    parser.add_argument("--results-dir", default=None,
                        help="results directory (default: the schema's standard directory)")
    args = parser.parse_args()

    results_dir = args.results_dir or DEFAULT_RESULTS_DIRS[args.schema]
    consolidated_dir = os.path.join(results_dir, "consolidated")

    # Without a consolidated/ subfolder the fold-less result CSVs live in the
    # results directory itself (e.g. the exploration single runs)
    if not os.path.isdir(consolidated_dir):
        print(f"No consolidated/ subfolder in {results_dir} — "
              "scoring fold-less result files in the directory itself.")
        consolidated_dir = results_dir

    print(f"Scoring consolidated results in {consolidated_dir}...")
    consolidated_table = build_score_table(consolidated_dir, args.schema)
    consolidated_path = os.path.join(results_dir, "score_table_consolidated.csv")
    consolidated_table.to_csv(consolidated_path, index=False)
    print(f"{len(consolidated_table)} rows written to {consolidated_path}\n")

    print("Top configurations by consolidated F1 Macro:")
    print(consolidated_table[CONFIG_COLUMNS[args.schema] + ["f1_macro"]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
