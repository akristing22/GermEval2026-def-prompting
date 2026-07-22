"""Ablation study: static few-shot exemplar selection via random search.

Replaces the Promptolution-based selection (ablation_promptolution.py):
draws N_TRIALS random class-balanced candidate sets of demonstration_size
exemplars from def_train.csv and scores each set by classifying a fixed
stratified evaluation subset of the training data (TRAIN_SIZE posts,
disjoint from the demonstration pool). The best-scoring set is kept.

The search runs once per prompt mode ('title' and 'implicit') using the
project's prompt templates via PromptConstructor; the same candidate sets
are evaluated under both prompts. Selection metric: F1 Macro (abstained
predictions are excluded from scoring but counted; ties on F1 are broken
by fewer abstentions).

Outputs, per prompt mode, under {results_path}/random_search/:
  - static_{prompt_mode}.yaml — the winning demonstration set, in the same
    format as templates/static_{prompt_mode}.yaml (drop-in replacement)
  - trials_{prompt_mode}.csv  — one row per candidate set: F1 Macro,
    abstention count, runtime, and the demonstrations (JSON-serialised)

Model, demonstration_size and paths come from config.yaml; the number of
candidate sets (N_TRIALS) and the evaluation-subset size (TRAIN_SIZE) are
constants below. Requires the HF_TOKEN environment variable.
"""

import gc
import json
import os
import random
import time

import pandas as pd
import torch
import yaml
from huggingface_hub import login
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import util

PROMPT_MODES = ["title", "implicit"]
SEED = 42
TRAIN_SIZE = 300
N_TRIALS = 30


def sample_candidate_sets(
        pool: pd.DataFrame,
        k: int,
        n_trials: int,
        rng: random.Random,
        ) -> list[dict[str, bool]]:
    """Draw n_trials class-balanced demonstration sets (k/2 True, k/2 False) from the pool."""
    texts_true = pool.loc[pool["DEF"] == True, "description"].tolist()
    texts_false = pool.loc[pool["DEF"] == False, "description"].tolist()

    trial_sets = []
    for _ in range(n_trials):
        demos = {text: True for text in rng.sample(texts_true, k // 2)}
        demos |= {text: False for text in rng.sample(texts_false, k // 2)}
        trial_sets.append(demos)
    return trial_sets


def score_predictions(y_true: list[bool], y_pred: list[bool | None]) -> tuple[float, int]:
    """F1 Macro over the non-abstained predictions, plus the abstention count."""
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    abstentions = len(y_pred) - len(pairs)
    if not pairs:
        return 0.0, abstentions
    true_part, pred_part = zip(*pairs)
    f1 = f1_score(true_part, pred_part, labels=[False, True], average="macro")
    return f1, abstentions


def main():
    torch.cuda.empty_cache()
    # create_message() shuffles the demonstration order via the global RNG
    random.seed(SEED)

    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)

    login(token=os.environ["HF_TOKEN"])

    output_dir = os.path.join(config["results_path"], "random_search")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(config["data_path"] + "/def_train.csv", sep=";")

    # Fixed stratified evaluation subset, shared by all candidate sets so
    # their scores are comparable; the demonstration pool is disjoint from it
    eval_df, pool = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        stratify=df["DEF"],
        random_state=SEED,
    )
    y_true = eval_df["DEF"].tolist()

    # The same candidate sets are evaluated under both prompt modes, so the
    # two searches are directly comparable trial by trial
    trial_sets = sample_candidate_sets(
        pool, config["demonstration_size"], N_TRIALS, random.Random(SEED)
    )

    print("Loading model...")
    lm = util.LM(config["model_name"])

    for prompt_mode in PROMPT_MODES:
        trials_path = os.path.join(output_dir, f"trials_{prompt_mode}.csv")
        if os.path.exists(trials_path):
            print(f"skip (exists): {trials_path}")
            continue

        # The PromptConstructor only loads the template and assembles chat
        # messages here — demonstrations are passed in directly, so no
        # knowledge base is needed
        pc = util.PromptConstructor(None, {**config, "prompt_mode": prompt_mode})

        records = []
        best = None  # (f1, abstentions, trial index)

        for trial, demonstrations in enumerate(trial_sets):
            t_start = time.time()

            prompts = [
                pc.create_message(text, pc.template, demonstrations=demonstrations)
                for text in eval_df["description"]
            ]
            lm.update_kv_cache()
            answers = lm.generate(
                prompts,
                max_tokens=config["max_tokens"],
                thinking_mode=config["thinking_mode"],
            )
            y_pred = [util.read_labels_from_answer(a) for a in answers]

            f1, abstentions = score_predictions(y_true, y_pred)
            elapsed = round(time.time() - t_start, 1)
            print(f"[{prompt_mode}] trial {trial}: F1 Macro {f1:.4f}, {abstentions} abstentions ({elapsed}s)")

            records.append({
                "trial": trial,
                "f1_macro": f1,
                "abstentions": abstentions,
                "seconds": elapsed,
                "demonstrations": json.dumps(demonstrations, ensure_ascii=False),
            })
            if best is None or (f1, -abstentions) > (best[0], -best[1]):
                best = (f1, abstentions, trial)

        pd.DataFrame(records).to_csv(trials_path, index=False)

        best_f1, best_abstentions, best_trial = best
        yaml_path = os.path.join(output_dir, f"static_{prompt_mode}.yaml")
        with open(yaml_path, "w") as f:
            yaml.safe_dump(trial_sets[best_trial], f, allow_unicode=True, sort_keys=False)

        print(f"[{prompt_mode}] best: trial {best_trial} (F1 Macro {best_f1:.4f}, {best_abstentions} abstentions)")
        print(f"[{prompt_mode}] demonstrations written to {yaml_path}, trial scores to {trials_path}")

    del lm
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
