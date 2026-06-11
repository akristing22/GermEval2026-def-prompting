"""Merge per-fold result CSVs into one CSV per experiment configuration.

run_all.py writes one result file per config and fold
({config}_fold-N.csv, N = 0..3). For evaluation across the whole dataset the
four folds of each config are concatenated here into a single CSV (every post
appears exactly once, since each fold's test split is disjoint) and written to
{RESULTS_DIR}/consolidated/{config}.csv.

Run from src/; expects the fold files in ../results/final_run/ and the
subdirectory "consolidated" to exist.
"""

import pandas as pd
import os
from tqdm import tqdm

RESULTS_DIR = "../results/final_run/"

def main():

    file_names = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
    # Derive the config name by cutting ".csv" and the fold suffix off each
    # filename. Note: str.strip() removes *characters* (not substrings) from
    # both ends, so this relies on the config name not ending in one of the
    # stripped characters; one spec per file means each config appears four
    # times in the list (once per fold).
    specs = [s.strip(".csv").strip("_fold-0").strip("_fold-1").strip("_fold-2").strip("_fold-3") for s in file_names]

    for spec in tqdm(specs):
        # Collect all fold files belonging to this config via prefix match
        spec_files = [f for f in file_names if f.startswith(spec)]
        dfs = []
        for f in spec_files:
            dfs.append(pd.read_csv(os.path.join(RESULTS_DIR,f)))

        # Concatenate the disjoint test splits into one full-dataset result
        df = pd.concat(dfs,ignore_index=True)

        df.to_csv(os.path.join(RESULTS_DIR,"consolidated",spec+".csv"),index=False)

if __name__ == "__main__":
    main()
