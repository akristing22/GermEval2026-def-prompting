"""Single competition run using the settings in config.yaml.

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
from tqdm import tqdm

import util



def result_filename(cfg: dict) -> str:
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
        f"{thinking_suffix}.csv"
    )


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


    quantisation = True if model_name == "google/gemma-4-26B-A4B-it" else False


    output_filename = result_filename(config)
    output_path = os.path.join(config["results_path"], output_filename)


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

    lm = util.LM(model_name,quantisation)

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

        if config["ratio"] == "proportional":

            num_neg = int(max(1,round(config["demonstration_size"]*len(train[train["DEF"]==False])/len(train),0)))
            num_pos = int(max(1,round(config["demonstration_size"]*len(train[train["DEF"]==True])/len(train),0)))

            diff = config["demonstration_size"] - (num_neg+num_pos)
            if diff == 1:
                num_pos += 1 if min(num_pos,num_neg) == num_pos else 0
                num_neg += 1 if min(num_pos,num_neg) == num_neg else 0
            if diff == -1:
                num_pos -=1 if max(num_pos,num_neg) == num_pos else 0
                num_neg -=1 if max(num_neg,num_pos) == num_neg else 0

            demo_ratio_train = {"neg":num_neg,"pos":num_pos}

            assert demo_ratio_train["neg"] + demo_ratio_train["pos"] == config["demonstration_size"], "Demo ratio does not add up to set demonstration size"
        else:
            demo_ratio_train = None
        

        prompts = [pc.construct(text,order = config["order"],ratio=demo_ratio_train) for text in tqdm(test["description"])]

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
            #"DEF": test["DEF"],
            "predicted_label": y_pred,
            "reply": generated_answers,
            "prompt": [json.dumps(p, ensure_ascii=False) for p in prompts],
        }).to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
