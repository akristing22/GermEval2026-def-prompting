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
import os
from itertools import product

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
    "title",
    "description",
    "implicit",
    "explicit",
]

DEMONSTRATION_MODES = [
    "dynamic",
    None,
]

DEMONSTRATION_SIZES = [
    0,  # zero-shot
    8,
]

N_SPLITS = 4


def get_data_splits(fold: tuple[np.ndarray, np.ndarray], df: pd.DataFrame):
    """Split df into (train, test) using the row indices of one CV fold."""
    return df.loc[fold[0]], df.loc[fold[1]]


def main():
    # Free any GPU memory left over from a previous run in the same process.
    torch.cuda.empty_cache()
    gc.collect()

    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)
    config = util.validate_config(config)

    # The grid loops below mutate `config` in place, so the per-fold knowledge
    # bases are built from a pristine copy — forced to dynamic mode and sized
    # for the largest grid demonstration size so all index structures exist
    # regardless of which combination ran last.
    kb_config = {
        **config,
        "demonstration_mode": "dynamic",
        "demonstration_size": max(DEMONSTRATION_SIZES),
    }

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
        if max(DEMONSTRATION_SIZES) > 0:
            if any(mode in PROMPT_MODES for mode in ("title", "description", "implicit")):
                know_base = util.KnowledgeBase(train["description"], train["DEF"], kb_config)
            else:
                know_base = None
            if "explicit" in PROMPT_MODES:
                # Per-step knowledge base for explicit mode, restricted to the
                # current train split; reuses the embedding model (if already
                # loaded) to avoid loading it twice.
                embedding_model = know_base.embedding_model if know_base is not None else None
                step_know_base = util.MultiStepKnowledgeBase(
                    annotations, train["id"], kb_config, embedding_model=embedding_model
                )
            else:
                step_know_base = None
        else:
            know_base = step_know_base = None  # zero-shot-only grid

        if lm is None:
            print(f"Loading model ({config['model_name']})...")
            lm = util.LM(config["model_name"])

        for demo_size, demo_mode, embedding_mode, retrieval_mode, prompt_mode in product(
            DEMONSTRATION_SIZES, DEMONSTRATION_MODES, EMBEDDING_MODES,
            RETRIEVAL_MODES, PROMPT_MODES,
        ):
            config["demonstration_size"] = demo_size
            config["demonstration_mode"] = demo_mode
            config["embedding_mode"] = embedding_mode
            config["retrieval_mode"] = retrieval_mode
            config["prompt_mode"] = prompt_mode

            if demo_size != 0:
                # Explicit mode retrieves demonstrations per step from its
                # own knowledge base.
                kb = step_know_base if prompt_mode == "explicit" else know_base
            else:
                kb = None  # zero-shot: no demonstrations needed

            # Skip combinations that validate_config() would rewrite: only the
            # canonical form of each config is executed, which deduplicates
            # the grid.
            if config != util.validate_config(config.copy(), check_model=False):
                continue

            if kb is not None:
                kb.update_config(config)

            # Resume support: skip configs that already have a result file
            # from a previous (partial) run.
            output_filename = util.result_filename(config, split)
            output_path = os.path.join(config["results_path"], "final_run", output_filename)
            if os.path.exists(output_path):
                print(f"Results for config {output_filename} already exist, skipping...")
                continue

            print(f"Running config: {output_filename.removesuffix('.csv')}")

            pc = util.PromptConstructor(kb, config)

            if config["prompt_mode"] == "explicit":
                # Multi-step pipeline: one inference call per legal decision
                # step, with early stopping per explicit_decisions.yaml.
                results = util.multi_step_generation(test, lm, pc, config)
            else:
                results = util.single_step_generation(test, lm, pc, config)

            results.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")

        # Release the model and knowledge bases before the next fold's index
        # structures are built, so their embedding work has the GPU to itself.
        del know_base
        del step_know_base
        del lm
        gc.collect()
        lm = None


if __name__ == "__main__":
    main()
