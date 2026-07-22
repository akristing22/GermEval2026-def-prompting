"""Ablation study: 2x3 factorial design over demonstration balance and order.

Crosses two factors that are orthogonal to the main prompting-strategy grid:
  - balance: class balance of the k demonstrations
      balanced     — k/2 positive and k/2 negative examples
      proportional — per-class counts mirror the train split's class distribution
  - order: position of the demonstrations in the prompt
      random / true_first / true_last

All six cells run under the same stratified 4-fold CV as run_all.py
(random_state=42), with the remaining strategy parameters taken from
config.yaml. Results are written to
{results_path}/ablations/balance_X_order/{model}_{balance}_{order}_fold-N.csv.

Requires the HF_TOKEN environment variable for HuggingFace Hub access.
"""

import gc
import os
import time

import pandas as pd
import torch
import yaml
from huggingface_hub import login
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import util

N_SPLITS = 4

BALANCE_CONDITIONS = ["balanced", "proportional"]
ORDER_CONDITIONS = ["random", "true_first", "true_last"]


def result_filename(cfg: dict, split: int) -> str:
    """Ablation filename schema: {model}_{balance}_{order}_fold-N.csv
    (the strategy axes are fixed in this ablation, so they are omitted)."""
    model_short = cfg["model_name"].split("/")[-1]
    return f"{model_short}_{cfg['balance']}_{cfg['order']}_fold-{split}.csv"


def run_config(cfg, test, lm, kb, split):
    """Run one (balance, order) cell and save its result CSV.

    Skips when the output already exists (resume support). Returns
    (out_path, elapsed_seconds); elapsed is None for skipped runs.
    """
    fname = result_filename(cfg, split)
    out_path = os.path.join(cfg["results_path"], fname)

    if os.path.exists(out_path):
        print(f"  skip (exists): {fname}")
        return out_path, None

    print(f"  running: {fname}")
    t_start = time.time()

    # The ordering is applied at prompt-construction time and the ratio at
    # retrieval time, so the knowledge base can be shared across all cells.
    pc = util.PromptConstructor(kb, cfg)
    results = util.single_step_generation(
        test, lm, pc, cfg, order=cfg["order"], ratio=cfg["ratio"]
    )
    results.to_csv(out_path, index=False)

    elapsed = round(time.time() - t_start, 1)
    print(f"  done in {elapsed}s: {fname}")
    return out_path, elapsed


def main():
    with open("config.yaml") as stream:
        cfg = yaml.safe_load(stream)
    cfg["results_path"] = cfg["results_path"] + "/ablations/balance_X_order"
    os.makedirs(cfg["results_path"], exist_ok=True)

    login(token=os.environ["HF_TOKEN"])

    print("Loading data...")
    df = pd.read_csv(cfg["data_path"] + "/def_train.csv", sep=";")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = {
        i: (train_index, test_index)
        for i, (train_index, test_index) in enumerate(skf.split(df["description"], df["DEF"]))
    }

    model_name = cfg["model_name"]
    # The largest model only fits alongside the embedding model when quantised
    quantisation = model_name == "google/gemma-4-26B-A4B-it"

    for split in tqdm(range(N_SPLITS), desc="Folds"):
        print(f"\nStarting fold {split}")

        train_idx, test_idx = folds[split]
        train = df.loc[train_idx]
        test = df.loc[test_idx]

        # Proportional per-class counts are derived from this fold's train split
        demo_ratio_train = util.proportional_demo_ratio(train["DEF"], cfg["demonstration_size"])

        torch.cuda.empty_cache()
        gc.collect()

        # One knowledge base per fold, shared by all six (balance, order) cells
        print("Building Knowledge Base...")
        kb = (
            util.KnowledgeBase(train["description"], train["DEF"], cfg)
            if cfg["demonstration_size"] > 0
            else None
        )

        print("Loading model...")
        lm = util.LM(model_name, quantisation)

        for demo_order in ORDER_CONDITIONS:
            cfg["order"] = demo_order
            for balance_label in BALANCE_CONDITIONS:
                cfg["balance"] = balance_label
                cfg["ratio"] = demo_ratio_train if balance_label == "proportional" else None

                try:
                    run_config(cfg, test, lm, kb, split)
                except Exception as e:
                    print(f"  ERROR in {result_filename(cfg, split)}: {e}")

        # Free the model before the next fold's knowledge base is embedded
        del lm
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
