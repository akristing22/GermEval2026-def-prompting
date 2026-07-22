"""Merge per-fold result CSVs into one CSV per experiment configuration.

The experiment scripts write one result file per config and fold
({config}_fold-N[_thinking].csv, N = 0..3). For evaluation across the whole
dataset the folds of each config are concatenated here into a single CSV
(every post appears exactly once, since each fold's test split is disjoint)
and written to {results_dir}/consolidated/{config}.csv.

Usage (from src/):
    python consolidate_folds.py                       # ../results/final_run
    python consolidate_folds.py ../results/ablations/balance_X_order

The consolidated/ subfolder is created automatically if it does not exist.
"""

import argparse
import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

DEFAULT_RESULTS_DIR = "../results/ablations/static"

# "_fold-N" sits before an optional "_thinking" suffix, so it is removed
# wherever it appears rather than only at the end of the name
_FOLD_PATTERN = re.compile(r"_fold-(\d+)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR,
                        help=f"directory with per-fold result CSVs (default: {DEFAULT_RESULTS_DIR})")
    args = parser.parse_args()

    consolidated_dir = os.path.join(args.results_dir, "consolidated")
    os.makedirs(consolidated_dir, exist_ok=True)

    # Group the fold files by their config name (the filename without the
    # fold suffix); non-fold CSVs (score tables, already-consolidated files)
    # are left alone
    fold_files = defaultdict(list)
    for filename in sorted(os.listdir(args.results_dir)):
        if not filename.endswith(".csv") or "score_table" in filename:
            continue
        spec, n_subs = _FOLD_PATTERN.subn("", filename.removesuffix(".csv"), count=1)
        if n_subs == 1:
            fold_files[spec].append(filename)

    for spec, filenames in tqdm(fold_files.items()):
        if len(filenames) != 4:
            print(f"Warning: {spec} has {len(filenames)} fold files (expected 4)")

        # Concatenate the disjoint test splits into one full-dataset result
        df = pd.concat(
            (pd.read_csv(os.path.join(args.results_dir, f)) for f in filenames),
            ignore_index=True,
        )
        df.to_csv(os.path.join(consolidated_dir, spec + ".csv"), index=False)

    print(f"{len(fold_files)} configs consolidated into {consolidated_dir}")


if __name__ == "__main__":
    main()
