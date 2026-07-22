"""Hyperparameter exploration of the few-shot demonstration size.

Runs the configurations listed in EXPLORATION_CONFIGS once per demonstration
size in DEMO_SIZES on a single 70/30 train/test split (util.load_data,
random_state=42) — no cross-validation, since this is a coarse pre-study to
pick the demonstration size for the main experiment. Runs whose result file
already exists are skipped, so the script can be interrupted and resumed.

Base settings (model, paths, max_tokens, ...) come from config.yaml; each
entry in EXPLORATION_CONFIGS overrides the strategy axes. Result CSVs go to
{results_path} in the standard filename format. Scoring is done separately
(evaluate.py); figures come from visualisations/visualise_exploration.py.

Requires the HF_TOKEN environment variable for HuggingFace Hub access.
"""

import gc
import os
import time

import torch
import yaml
from huggingface_hub import login

import util

# ─── Exploration grid ─────────────────────────────────────────────────────────

DEMO_SIZES = [4, 8, 16, 32]  # few-shot counts to compare — edit as needed

# Strategy combinations to explore; missing keys are filled from config.yaml
EXPLORATION_CONFIGS = [
    {
        "prompt_mode": "title",
        "demonstration_mode": "dynamic",
        "embedding_mode": "dense",
        "retrieval_mode": "similarity",
    },
    {
        "prompt_mode": "description",
        "demonstration_mode": "dynamic",
        "embedding_mode": None,
        "retrieval_mode": "random",
    },
]


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_config(cfg, test, lm, kb):
    """Run one classification config and save results. Skips if output already exists.

    Returns:
        (out_path, elapsed_seconds) — elapsed is None for skipped runs.
    """
    fname = util.result_filename(cfg)
    out_path = os.path.join(cfg["results_path"], fname)

    if os.path.exists(out_path):
        print(f"  skip (exists): {fname}")
        return out_path, None

    print(f"  running: {fname}")
    t_start = time.time()

    # The shared knowledge base only adopts the changed demonstration settings;
    # its index structures are reused across all runs.
    kb.update_config(cfg)

    pc = util.PromptConstructor(kb, cfg)
    results = util.single_step_generation(test, lm, pc, cfg)
    results.to_csv(out_path, index=False)

    elapsed = round(time.time() - t_start, 1)
    print(f"  done in {elapsed}s: {fname}")
    return out_path, elapsed


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    login(token=os.environ["HF_TOKEN"])

    with open("config.yaml") as f:
        base_cfg = yaml.safe_load(f)

    # Merge each exploration config over the base config and validate it the
    # same way as the other runners (rewrites inconsistent combinations).
    selected = []
    for overrides in EXPLORATION_CONFIGS:
        cfg = {**base_cfg, **overrides}
        selected.append(util.validate_config(cfg, check_model=False))

    train, test = util.load_data(base_cfg["data_path"] + "/def_train.csv")

    # One knowledge base for all runs, sized for the largest demonstration
    # count so its KMeans clusters can serve every grid size (the cluster
    # count is fixed at build time, see KnowledgeBase.build).
    kb_config = {
        **base_cfg,
        "demonstration_mode": "dynamic",
        "demonstration_size": max(DEMO_SIZES),
    }
    kb = util.KnowledgeBase(train["description"], train["DEF"], kb_config)

    torch.cuda.empty_cache()
    gc.collect()
    lm = util.LM(base_cfg["model_name"])

    for cfg in selected:
        for demo_size in DEMO_SIZES:
            cfg["demonstration_size"] = demo_size
            try:
                run_config(cfg, test, lm, kb)
            except Exception as e:
                print(f"  ERROR in {util.result_filename(cfg)}: {e}")

    del lm
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
