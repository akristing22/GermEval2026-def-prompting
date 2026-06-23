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
    thinking_suffix = "_thinking" if cfg["thinking_mode"] else ""

    return (
        f"{model_short}_"
        f"{cfg['prompt_mode']}_"
        f"{cfg['demonstration_mode']}_"
        f"{cfg['demonstration_size']}_"
        f"{cfg['embedding_mode']}_"
        f"{cfg['retrieval_mode']}_"
        f"fold-{split}"
        f"{thinking_suffix}.csv"
    )


def build_kb(cfg, train, annotations):
    """Build the KnowledgeBase for the given config and train split.

    Called once per (cfg, fold) and reused across both demo orders, since
    ordering is applied at prompt-construction time, not retrieval time.
    Returns None when demonstration_size is 0.
    """
    if cfg["demonstration_size"] == 0:
        return None
    if cfg["prompt_mode"] == "explicit":
        return util.MultiStepKnowledgeBase(annotations, train["id"], cfg)
    return util.KnowledgeBase(train["description"], train["DEF"], cfg)


def run_config(cfg, train, test, lm, kb, annotations, split):
    fname    = result_filename(cfg, split)
    out_path = os.path.join(cfg["results_path"], fname)

    if os.path.exists(out_path):
        print(f"  skip (exists): {fname}")
        return out_path, None

    print(f"  running: {fname}")
    t_start = time.time()

    pc = util.PromptConstructor(kb, cfg)

    if cfg["prompt_mode"] == "explicit":
        all_replies = util.multi_step_generation(test, lm, pc, cfg, cfg["order"])

        all_replies["prompt"] = all_replies["prompt"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if x is not None else None
        )
        all_replies.to_csv(out_path, index=False)

    else:
        if cfg["embedding_mode"] is not None:
            print(f"Retrieving demonstrations {datetime.now()}...")
        prompts = [pc.construct(text, ratio=cfg["demo_ratio"]) for text in tqdm(test["description"])]

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
    base_results_path = cfg["results_path"] + "/ablations/demo_balance"
    paths = {
        "data_path":     cfg["data_path"],
        "results_path":  base_results_path,
        "template_path": cfg["template_path"],
    }

    login(token=os.environ["HF_TOKEN"])


    print("Loading data...")
    df          = pd.read_csv(paths["data_path"] + "/def_train.csv", sep=";")
    annotations = pd.read_csv(paths["data_path"] + "/single_step_annotation.csv")

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

        demo_ratio = {"neg":len(train[train["DEF"]==False])/len(train),"pos":len(train[train["DEF"]==True])/len(train)}
        assert demo_ratio["neg"] + demo_ratio["pos"] == 1, "Demo ratio does not add up to one!"

        model_name = cfg["model_name"]

        torch.cuda.empty_cache()
        gc.collect()
        quantisation = True if model_name == "google/gemma-4-26B-A4B-it" else False


        print("Building Knowledge Base...")
        kb = build_kb(cfg, train, annotations)

        print("Loading model...")
        lm = util.LM(model_name,quantisation)

        cfg["demo_ratio"]   = demo_ratio
        cfg["results_path"] = base_results_path

        filename = result_filename(cfg,split)

        if filename in os.listdir(cfg["results_path"]):
            print(f"{filename} already exists, skipping config")

        try:
            _, elapsed = run_config(cfg, train, test, lm, kb, annotations, split)
            timings[filename] = elapsed
        except Exception as e:
            print(f"  ERROR in {filename}: {e}")

        del lm
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
