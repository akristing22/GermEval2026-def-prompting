"""Competition run using the settings in config.yaml.

Variant of run_all.py for the final competition submission:
  - trains on the full data/def_train.csv (no cross-validation),
  - classifies the unlabelled data/def_test.csv (no DEF column, so the
    result CSV has no ground-truth column either),
  - additionally respects the optional 'ratio' (balanced/proportional class
    balance of demonstrations) and 'order' (demonstration ordering) config
    parameters; both default to the standard behaviour (balanced, random)
    when not set.

The result CSV is written to {results_path} using the standard filename
format (without a fold suffix); make_submission.py converts it into the
required submission format.

Requires the HF_TOKEN environment variable for HuggingFace Hub access.
"""

import os

import pandas as pd
import yaml
from huggingface_hub import login

import util


def main():
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)
    # Normalize incompatible parameter combinations (see validate_config()).
    config = util.validate_config(config)
    model_name = config["model_name"]

    login(token=os.environ["HF_TOKEN"])

    print("Loading data...")
    train = pd.read_csv(config["data_path"] + "/def_train.csv", sep=";")
    annotations = pd.read_csv(config["data_path"] + "/single_step_annotation.csv")
    test = pd.read_csv(config["data_path"] + "/def_test.csv", sep=";")

    # The largest model only fits alongside the embedding model when quantised
    quantisation = model_name == "google/gemma-4-26B-A4B-it"

    output_path = os.path.join(config["results_path"], util.result_filename(config))

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

    lm = util.LM(model_name, quantisation)

    # Optional ablation parameters; absent keys mean the standard behaviour
    order = config.get("order") or "random"
    if config.get("ratio") == "proportional":
        demo_ratio = util.proportional_demo_ratio(train["DEF"], config["demonstration_size"])
    else:
        demo_ratio = None  # balanced k/2 : k/2 split

    if config["prompt_mode"] == "explicit":
        # Multi-step pipeline: one inference call per legal decision step,
        # with early stopping per explicit_decisions.yaml. The per-step
        # retrieval has no ratio support, so only the ordering is passed on.
        results = util.multi_step_generation(test, lm, pc, config, order=order)
    else:
        results = util.single_step_generation(test, lm, pc, config, order=order, ratio=demo_ratio)

    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
