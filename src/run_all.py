"""Full experiment grid over all valid prompting-strategy combinations.

Runs every combination of prompt_mode x demonstration_mode x demonstration_size
x embedding_mode x retrieval_mode under stratified 4-fold cross-validation
(random_state=42). Invalid combinations (see validate_config() in util.py) and
combinations whose result file already exists are skipped, so the script can be
interrupted and resumed.

Base settings (model, paths, max_tokens, thinking_mode) are read from
config.yaml; the grid below overrides the strategy axes per run. Results are
written to {results_path}/final_run/ using the standard result filename format.

Requires the HF_TOKEN environment variable for HuggingFace Hub access.
"""

import gc
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from huggingface_hub import login
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import util

# --- Experiment grid -------------------------------------------------------
# The cross product of these lists is filtered through validate_config(), so
# redundant combinations (e.g. retrieval_mode set while embedding_mode is
# sparse) collapse to a single canonical config and run only once.

EMBEDDING_MODES = [
    "sparse",
    "dense",
    "fusion",
    None,
]

RETRIEVAL_MODES = [
    "random",
    "diversity",
    "similarity",
    "mmr",
    None,
]

PROMPT_MODES = [
    # "title",
    # "description",
    # "implicit",
    "explicit",
]

DEMONSTRATION_MODES = [
    "dynamic",
    None,
]

DEMONSTRATION_SIZES = [
    # 0,  # zero-shot
    8,
]

N_SPLITS = 4


def get_data_splits(fold: tuple[np.ndarray, np.ndarray], df: pd.DataFrame):
    """Split df into (train, test) using the row indices of one CV fold."""
    return df.loc[fold[0]], df.loc[fold[1]]


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
    # Free any GPU memory left over from a previous run in the same process.
    torch.cuda.empty_cache()
    gc.collect()

    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)
    config = util.validate_config(config)

    login(token=os.environ["HF_TOKEN"])

    # Main dataset (semicolon-delimited) and the per-step annotations used by
    # the explicit multi-step pipeline (comma-delimited subset of def_train).
    df = pd.read_csv(config["data_path"] + "/def_train.csv", sep=";")
    annotations = pd.read_csv(config["data_path"] + "/single_step_annotation.csv")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = {
        i: [train_index, test_index]
        for i, (train_index, test_index) in enumerate(skf.split(df["description"], df["DEF"]))
    }

    lm = None  # loaded lazily after the first fold's knowledge bases are built

    for split in tqdm(range(N_SPLITS), desc="Splits"):
        print(f"Starting with split {split}")
        train, test = get_data_splits(folds[split], df)

        # Retrieval stores are built once per fold from the train split only,
        # then reconfigured per grid combination via update_config().
        know_base = util.KnowledgeBase(train["description"], train["DEF"], config)
        # Per-step knowledge base for explicit mode, restricted to the current
        # train split; reuses the embedding model to avoid loading it twice.
        step_know_base = util.MultiStepKnowledgeBase(
            annotations, train["id"], config, embedding_model=know_base.embedding_model
        )

        if lm is None:
            print(f"Loading model ({config['model_name']})...")
            lm = util.LM(config["model_name"])

        for demo_size in DEMONSTRATION_SIZES:
            config["demonstration_size"] = demo_size

            for demo_mode in DEMONSTRATION_MODES:
                config["demonstration_mode"] = demo_mode

                for embedding_mode in EMBEDDING_MODES:
                    config["embedding_mode"] = embedding_mode

                    for retrieval_mode in RETRIEVAL_MODES:
                        config["retrieval_mode"] = retrieval_mode

                        for prompt_mode in PROMPT_MODES:
                            config["prompt_mode"] = prompt_mode

                            if demo_size != 0:
                                # Explicit mode retrieves demonstrations per
                                # step from its own knowledge base.
                                kb = step_know_base if prompt_mode == "explicit" else know_base
                            else:
                                kb = None  # zero-shot: no demonstrations needed

                            # Skip combinations that validate_config() would
                            # rewrite: only the canonical form of each config
                            # is executed, which deduplicates the grid.
                            if config != util.validate_config(config.copy(), check_model=False):
                                continue

                            if kb is not None:
                                kb.update_config(config)

                            # Resume support: skip configs that already have a
                            # result file from a previous (partial) run.
                            output_filename = result_filename(config, split)
                            output_path = os.path.join(
                                config["results_path"], "final_run", output_filename
                            )
                            if os.path.exists(output_path):
                                print(f"Results for config {output_filename} already exist, skipping...")
                                continue

                            print(f"Running config: {output_filename.removesuffix('.csv')}")

                            pc = util.PromptConstructor(kb, config)

                            if config["prompt_mode"] == "explicit":
                                # Multi-step pipeline: one inference call per
                                # legal decision step, with early stopping per
                                # explicit_decisions.yaml. Returns a finished
                                # results frame (id, text, labels, reply, ...).
                                all_replies = util.multi_step_generation(test, lm, pc, config)

                                # Serialize the chat-format prompts so the CSV
                                # stays one row per test instance.
                                all_replies["prompt"] = all_replies["prompt"].apply(
                                    lambda x: json.dumps(x, ensure_ascii=False) if x is not None else None
                                )
                                all_replies.to_csv(output_path, index=False)
                                print(f"Results saved to {output_path}")

                            else:
                                # Single-step modes: build one prompt per test
                                # instance (includes retrieval if configured).
                                if config["embedding_mode"] is not None:
                                    print(f"Retrieving demonstrations {datetime.now()}...")
                                prompts = [pc.construct(text) for text in tqdm(test["description"])]

                                # Re-prime the KV cache for the shared prompt
                                # prefix before batched generation.
                                lm.update_kv_cache()

                                print(f"Generating answers {datetime.now()}...")
                                generated_answers = lm.generate(
                                    prompts,
                                    max_tokens=config["max_tokens"],
                                    thinking_mode=config["thinking_mode"],
                                )
                                # Parse True/False from each reply; None marks
                                # an abstention (excluded later by evaluate.py).
                                y_pred = [util.read_labels_from_answer(a) for a in generated_answers]

                                pd.DataFrame({
                                    "id": test["id"],
                                    "text": test["description"],
                                    "true_label": test["DEF"],
                                    "predicted_label": y_pred,
                                    "reply": generated_answers,
                                    "prompt": [json.dumps(p, ensure_ascii=False) for p in prompts],
                                }).to_csv(output_path, index=False)
                                print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
