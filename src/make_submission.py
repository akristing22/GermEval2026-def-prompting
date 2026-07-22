"""Format competition result CSVs as submission files.

Reads every result CSV from the results directory, keeps the id and
predicted_label columns, renames predicted_label to DEF, and writes a
semicolon-delimited submission file ("MUCnoHARM_def_" prefix — the team
name) next to each input. Already-converted files are skipped.

Usage (from src/):
    python make_submission.py                     # ../results/competition/balance_ratio
    python make_submission.py ../results/competition
"""

import argparse
import os

import pandas as pd

DEFAULT_RESULTS_DIR = "../results/competition/balance_ratio"
SUBMISSION_PREFIX = "MUCnoHARM_def_"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR,
                        help=f"directory with competition result CSVs (default: {DEFAULT_RESULTS_DIR})")
    args = parser.parse_args()

    for filename in sorted(os.listdir(args.results_dir)):
        # Skip non-CSVs and the submission files produced by earlier runs
        if not filename.endswith(".csv") or filename.startswith(SUBMISSION_PREFIX):
            continue

        df = pd.read_csv(os.path.join(args.results_dir, filename))
        submission = df[["id", "predicted_label"]].rename(columns={"predicted_label": "DEF"})

        out_path = os.path.join(args.results_dir, SUBMISSION_PREFIX + filename)
        submission.to_csv(out_path, index=False, sep=";")
        print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
