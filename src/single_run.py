"""Single experiment run using the settings in config.yaml.

Runs the config.yaml prompting-strategy configuration (prompt_mode,
demonstration_mode/size, embedding_mode, retrieval_mode, thinking_mode) under
stratified 4-fold cross-validation (random_state=42) and writes one result CSV
per fold to {results_path} using the standard result filename format with a
_fold-N suffix.

For the full grid over all valid combinations under 4-fold CV, use
run_all.py instead.

Requires the HF_TOKEN environment variable for HuggingFace Hub access.
"""

import json
import os
from datetime import datetime

import pandas as pd
import yaml
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


def main():
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)
    # Normalize incompatible parameter combinations (see validate_config()).
    config = util.validate_config(config)

    login(token=os.environ["HF_TOKEN"])

    print("Loading data...")
    df = pd.read_csv(config["data_path"] + "/def_train.csv", sep=";")
    annotations = pd.read_csv(config["data_path"] + "/single_step_annotation.csv")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = {
        i: (train_index, test_index)
        for i, (train_index, test_index) in enumerate(skf.split(df["description"], df["DEF"]))
    }

    print("Loading model...")
    lm = util.LM(config["model_name"])

    for split in tqdm(range(N_SPLITS), desc="Folds"):
        print(f"\nStarting fold {split}")

        output_filename = result_filename(config, split)
        output_path = os.path.join(config["results_path"], output_filename)
        if os.path.exists(output_path):
            print(f"Results for fold {split} already exist, skipping...")
            continue

        train_idx, test_idx = folds[split]
        train = df.loc[train_idx]
        test = df.loc[test_idx]

        # Build the demonstration pool for few-shot prompting; zero-shot runs
        # (demonstration_size == 0) need no knowledge base.
        if config["demonstration_size"] > 0:
            print("Building Knowledge Base...")
            if config["prompt_mode"] == "explicit":
                # The explicit pipeline retrieves per step, using the per-step
                # annotations instead of the final DEF label.
                kb = util.MultiStepKnowledgeBase(annotations, train["id"], config)
            else:
                kb = util.KnowledgeBase(train["description"], train["DEF"], config)
        else:
            kb = None

        pc = util.PromptConstructor(kb, config)

        if config["prompt_mode"] == "explicit":
            # Multi-step pipeline: one inference call per legal decision step,
            # with early stopping per explicit_decisions.yaml. Returns a finished
            # results frame (id, text, labels, reply, ...).
            all_replies = util.multi_step_generation(test, lm, pc, config)

            # Serialize the chat-format prompts so the CSV stays one row per
            # test instance.
            all_replies["prompt"] = all_replies["prompt"].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if x is not None else None
            )
            all_replies.to_csv(output_path, index=False)

        else:
            # Single-step modes: build one prompt per test instance (includes
            # demonstration retrieval if configured).
            if config["embedding_mode"] is not None:
                print(f"Retrieving demonstrations {datetime.now()}...")
            prompts = [pc.construct(text) for text in tqdm(test["description"])]

            # could be changed to - syncs to util.py/line 1130 then:
            # prompts = [
                # pc.construct(text, system_prompt=True)
                # for text in tqdm(test["description"])
            # ]

            print(f"Classifying {datetime.now()}...")
            generated_answers = lm.generate(
                prompts,
                max_tokens=config["max_tokens"],
                thinking_mode=config["thinking_mode"],
            )
            # Parse True/False from each reply; None marks an abstention
            # (excluded later by evaluate.py).
            y_pred = [util.read_labels_from_answer(a) for a in generated_answers]

            pd.DataFrame({
                "id": test["id"],
                "description": test["description"],
                "DEF": test["DEF"],
                "predicted_label": y_pred,
                "reply": generated_answers,
                "prompt": [json.dumps(p, ensure_ascii=False) for p in prompts],
            }).to_csv(output_path, index=False)

        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
