"""Ablation study: does the ordering of few-shot demonstrations affect classification?

Tests two orderings of the labelled demonstrations in the prompt:
  - true_first : all True-labelled examples before all False-labelled ones
  - true_last  : all False-labelled examples before all True-labelled ones

Each ordering is written to its own subfolder under results/ablations/demo_order/
so result filenames stay identical across conditions.

Runs under stratified 4-fold CV (random_state=42).

Requires the HF_TOKEN environment variable for HuggingFace Hub access.
"""

import gc
import json
import os
import time
from datetime import datetime

import pandas as pd
import yaml
import torch
from huggingface_hub import login
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import util

N_SPLITS = 4

def result_filename(cfg: dict, split: int) -> str:
    """Build the result filename:
    {model}_{prompt}_{demo_mode}_{demo_size}_{embedding}_{retrieval}_fold-N[_thinking].csv

    None values appear literally as "None".
    """
    model_short = cfg["model_name"].split("/")[-1]

    return (
        f"{model_short}_"
        f"{cfg["balance"]}_"
        f"{cfg["order"]}_"
        f"fold-{split}"
        ".csv"
    )


def build_kb(cfg, train):
    """Build the KnowledgeBase for the given config and train split.

    Called once per (cfg, fold) and reused across both demo orders, since
    ordering is applied at prompt-construction time, not retrieval time.
    Returns None when demonstration_size is 0.
    """
    if cfg["demonstration_size"] == 0:
        return None
    return util.KnowledgeBase(train["description"], train["DEF"], cfg)


def run_config(cfg, test, lm, kb, split):
    fname    = result_filename(cfg, split)
    out_path = os.path.join(cfg["results_path"], fname)

    if os.path.exists(out_path):
        print(f"  skip (exists): {fname}")
        return out_path, None

    print(f"  running: {fname}")
    t_start = time.time()

    pc = util.PromptConstructor(kb, cfg)

    if cfg["embedding_mode"] is not None:
        print(f"Retrieving demonstrations {datetime.now()}...")
    prompts = [pc.construct(text, ratio=cfg["ratio"],order=cfg["order"]) for text in tqdm(test["description"])]

    print(f"Classifying {datetime.now()}...")
    generated_answers = lm.generate(
        prompts,
        max_tokens=cfg["max_tokens"],
        thinking_mode=cfg["thinking_mode"],
    )

    y_pred = [util.read_labels_from_answer(a) for a in generated_answers]

    pd.DataFrame({
        "id":              test["id"],
        "description":     test["description"],
        "DEF":             test["DEF"],
        "predicted_label": y_pred,
        "reply":           generated_answers,
        "prompt":          [json.dumps(p, ensure_ascii=False) for p in prompts],
    }).to_csv(out_path, index=False)

    elapsed = round(time.time() - t_start, 1)
    print(f"  done in {elapsed}s: {fname}")
    return out_path, elapsed


def main():
    with open("config.yaml") as stream:
        cfg = yaml.safe_load(stream)
    base_results_path = cfg["results_path"] + "/ablations/balance_X_order"
    paths = {
        "data_path":     cfg["data_path"],
        "results_path":  base_results_path,
        "template_path": cfg["template_path"],
    }

    os.makedirs(base_results_path, exist_ok=True)

    login(token=os.environ["HF_TOKEN"])
    cfg["results_path"] = base_results_path



    print("Loading data...")
    df          = pd.read_csv(paths["data_path"] + "/def_train.csv", sep=";")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = {
        i: (train_index, test_index)
        for i, (train_index, test_index) in enumerate(skf.split(df["description"], df["DEF"]))
    }

    timings = {}


    for split in tqdm(range(N_SPLITS), desc="Folds"):
        print(f"\nStarting fold {split}")

        train_idx, test_idx = folds[split]
        train = df.loc[train_idx]
        test  = df.loc[test_idx]

        num_neg = int(max(1,round(cfg["demonstration_size"]*len(train[train["DEF"]==False])/len(train),0)))
        num_pos = int(max(1,round(cfg["demonstration_size"]*len(train[train["DEF"]==True])/len(train),0)))

        diff = cfg["demonstration_size"] - (num_neg+num_pos)
        if diff == 1:
            num_pos += 1 if min(num_pos,num_neg) == num_pos else 0
            num_neg += 1 if min(num_pos,num_neg) == num_neg else 0
        if diff == -1:
            num_pos -=1 if max(num_pos,num_neg) == num_pos else 0
            num_neg -=1 if max(num_neg,num_pos) == num_neg else 0

        demo_ratio_train = {"neg":num_neg,"pos":num_pos}

        assert demo_ratio_train["neg"] + demo_ratio_train["pos"] == cfg["demonstration_size"], "Demo ratio does not add up to set demonstration size"

        model_name = cfg["model_name"]

        torch.cuda.empty_cache()
        gc.collect()
        quantisation = True if model_name == "google/gemma-4-26B-A4B-it" else False


        print("Building Knowledge Base...")
        kb = build_kb(cfg, train)

        print("Loading model...")
        lm = util.LM(model_name,quantisation)

        for demo_order in ["random","true_first","true_last"]:
            cfg["order"] = demo_order
            #for demo_ratio, balance_label in [(None, "balanced"), (demo_ratio_train, "proportional")]:
            for demo_ratio, balance_label in [(demo_ratio_train, "proportional")]:
                cfg["balance"] = balance_label
                cfg["ratio"] = demo_ratio
        
        
                filename = result_filename(cfg,split)

                if filename in os.listdir(cfg["results_path"]):
                    print(f"{filename} already exists, skipping config")
                    continue

                try:
                    _, elapsed = run_config(cfg, test, lm, kb, split)
                    timings[filename] = elapsed
                except Exception as e:
                    print(f"  ERROR in {filename}: {e}")

        del lm
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
