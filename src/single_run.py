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

import os

import pandas as pd
import yaml
from huggingface_hub import login
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import util

N_SPLITS = 4

# Static demonstration files per prompt_mode (same files KnowledgeBase.build()
# loads); static mode is only implemented for these prompt modes.
STATIC_DEMONSTRATION_FILES = {
    "title": "static_title.yaml",
    "implicit": "static_implicit.yaml",
}


def normalize_text(text: str) -> str:
    """Reduce a post to its lowercase alphanumeric characters.

    The static demonstration YAMLs differ from their def_train.csv source rows
    in punctuation (dropped commas) and newline encoding (literal '\\r\\n'
    instead of real newlines), so texts are compared in this normalized form.
    """
    text = text.replace("\\r\\n", " ")
    return "".join(ch for ch in text.lower() if ch.isalnum())


def static_demonstration_ids(df: pd.DataFrame, config: dict) -> set:
    """Ids of the def_train.csv rows used as static demonstrations.

    Matches each demonstration text from the prompt_mode's static YAML back to
    its dataset row via normalize_text(). Raises if a demonstration cannot be
    matched, so a silent train/test leak is impossible.
    """
    filename = STATIC_DEMONSTRATION_FILES[config["prompt_mode"]]
    with open(os.path.join(config["template_path"], filename)) as stream:
        demonstrations = yaml.safe_load(stream)

    normalized = df["description"].map(normalize_text)
    ids = set()
    for text in demonstrations:
        matches = df.loc[normalized == normalize_text(text), "id"]
        if matches.empty:
            raise ValueError(
                f"Static demonstration not found in def_train.csv: {text[:60]!r}"
            )
        ids.update(matches)
    return ids


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

    # Static demonstrations are fixed posts from def_train.csv shown in every
    # prompt, so they must never be scored as test instances. Their rows are
    # removed from both splits only after the folds were computed on the full
    # dataset, keeping the fold assignment identical to all other runs.
    if config["demonstration_mode"] == "static":
        demonstration_ids = static_demonstration_ids(df, config)
        print(f"Excluding {len(demonstration_ids)} static demonstration posts from the CV splits")
    else:
        demonstration_ids = set()

    print("Loading model...")
    lm = util.LM(config["model_name"])

    for split in tqdm(range(N_SPLITS), desc="Folds"):
        print(f"\nStarting fold {split}")

        output_path = os.path.join(config["results_path"], util.result_filename(config, split))
        if os.path.exists(output_path):
            print(f"Results for fold {split} already exist, skipping...")
            continue

        train_idx, test_idx = folds[split]
        train = df.loc[train_idx]
        test = df.loc[test_idx]

        if demonstration_ids:
            train = train[~train["id"].isin(demonstration_ids)]
            test = test[~test["id"].isin(demonstration_ids)]

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
            # with early stopping per explicit_decisions.yaml.
            results = util.multi_step_generation(test, lm, pc, config)
        else:
            results = util.single_step_generation(test, lm, pc, config)

        results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
