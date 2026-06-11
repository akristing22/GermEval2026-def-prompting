# GermEval 2026 — Prompting Strategies for Legal Hate Speech Classification

Research code for benchmarking prompting strategies on a binary classification task: given a German social media post, decide whether it is **prosecutable under §§185–187 StGB** (Beleidigung, üble Nachrede, Verleumdung). The project compares zero-shot and dynamic few-shot prompting (with several retrieval modes) across multiple open-weight LLMs, evaluated with stratified 4-fold cross-validation.

## Task and Legal Framework

The label is binary: `True` = prosecutable under at least one of §§185–187 StGB, `False` = not prosecutable.

- **§185 StGB — Beleidigung (insult):** direct attack on a person's honor via expression of contempt or disrespect
- **§186 StGB — Üble Nachrede (defamation):** asserting or spreading facts about a person that are not demonstrably true and make them contemptible
- **§187 StGB — Verleumdung (calumny):** knowingly asserting false facts that damage another person's reputation

The annotation schema follows Zufall et al. (2019), [*From legal to technical concept: Towards an automated classification of German political Twitter postings as criminal offenses*](https://aclanthology.org/N19-1135/), which operationalizes the legal assessment as a 6-step decision schema:

1. Is there a defamatory object (individual, distinguishable group, or collective entity)?
2. Is there a disparaging statement directed at that object?
3. Is it primarily a value judgement (not a factual claim)?
4. Is it an abusive insult? → if yes: **prosecutable**
5. Is it of public interest? → if yes: **not prosecutable**
6. Is it abusive criticism? → if yes: **prosecutable**, otherwise **not prosecutable**

## Repository Structure

```
├── config.yaml                    # Config for a single experiment run
├── data/
│   ├── def_train.csv              # Main dataset (semicolon-delimited; columns: description, DEF, ...)
│   └── single_step_annotation.csv # Subset with per-step annotations for the 6 decision criteria (comma-delimited)
├── templates/                     # Prompt templates, one file per prompt_mode
│   ├── title
│   ├── description
│   ├── implicit
│   ├── explicit
│   └── explicit_decisions.yaml    # Maps each step's True/False output → "continue" or final label
├── src/
│   ├── util.py                    # All pipeline code: LM, KnowledgeBase, PromptConstructor, helpers
│   ├── single_run.py              # One configuration on a 70/30 split (config.yaml)
│   ├── run_all.py                 # Full grid over all valid configurations, 4-fold CV
│   ├── knn_baseline.py            # Non-LLM baseline: 1-NN over TF-IDF vectors, same CV folds
│   ├── consolidate_folds.py       # Merges per-fold result CSVs into one CSV per configuration
│   ├── evaluate_final_run.py      # Computes score tables (per-fold and consolidated)
│   └── visualisations/            # Figure scripts (specification curve, axis impact, ...)
└── results/
    ├── final_runs/                # Score tables, figures, and kNN baseline of the main experiment
    └── exploration/               # Hyperparameter exploration (demonstration size 4/8/16/32)
```

## Dataset

- `data/def_train.csv` — semicolon-delimited; key columns are `description` (post text) and `DEF` (`True`/`False` label).
- `data/single_step_annotation.csv` — a comma-delimited subset of `def_train.csv` with one extra column per step of the 6-step schema. It is used to evaluate the intermediate steps of the `explicit` pipeline and as the demonstration pool for few-shot retrieval in `explicit` mode.
- **Do not modify the dataset files.** All preprocessing happens in code at load time.

Splits: `single_run.py` uses a 70/30 train-test split; `run_all.py` and `knn_baseline.py` use stratified 4-fold cross-validation. All splits use `random_state=42`, so folds are identical across experiments.

## Prompting Strategies

All behavior is controlled via config parameters along five independent axes:

### `prompt_mode` — how the task is described to the model

| Value | Description |
|---|---|
| `title` | Minimal prompt: just the label names and a True/False answer format |
| `description` | Adds a short description of the classification criteria |
| `implicit` | All 6 legal decision steps inline in a single prompt; one inference call |
| `explicit` | Multi-step chain-of-thought: each step is a separate inference call; `templates/explicit_decisions.yaml` controls whether a step's answer yields an early final label or continues |

Templates live in `templates/` and use XML-like tags: `<system>` for the system prompt, `<task>` for single-step modes, `<step1>`–`<step6>` for `explicit` mode.

In `explicit` mode, each step gets its own retrieval store built from the per-step labels in `single_step_annotation.csv` (restricted to the current train split), and each step runs as an independent conversation — previous steps' prompts and replies are not included. Per-step demonstrations stay class-balanced; a step where one class has no annotated examples runs zero-shot.

### `demonstration_mode` — whether examples are included

| Value | Description |
|---|---|
| `dynamic` | Examples retrieved per test instance via the retrieval pipeline |
| `static` | A fixed example set reused for all test instances (**not yet implemented**) |

Zero-shot is expressed as `demonstration_size: 0` (all demonstration/retrieval settings are then ignored).

### `demonstration_size` — number of few-shot examples (k)

Always split evenly between classes (k/2 per class). The exploration phase compared 4/8/16/32; the main experiment uses `8`.

### `embedding_mode` — retrieval index type (dynamic only)

| Value | Description |
|---|---|
| `dense` | Embedding-based vector search (LangChain `InMemoryVectorStore`) |
| `sparse` | BM25 retrieval via separate per-class indices |
| `fusion` | Interleaves dense and sparse results |

### `retrieval_mode` — example selection strategy (dynamic only)

| Value | Description |
|---|---|
| `similarity` | Top-k by cosine similarity (dense) or BM25 score (sparse) |
| `diversity` | KMeans clustering of the train embeddings, one random example per cluster (dense only) |
| `mmr` | Maximal Marginal Relevance — balances relevance and diversity (dense only) |
| `random` | Uniform random sampling from the training pool (no embedding needed) |

### Config compatibility rules

`validate_config()` in [src/util.py](src/util.py) normalizes invalid combinations (with a warning):

- `demonstration_size == 0` → `embedding_mode`, `demonstration_mode`, `retrieval_mode` all become `None`
- `demonstration_mode == 'static'` → `embedding_mode` and `retrieval_mode` become `None`
- `retrieval_mode == 'random'` → `embedding_mode` becomes `None`
- `embedding_mode in ['sparse', 'fusion']` → `retrieval_mode` is forced to `'similarity'`
- `demonstration_mode == 'dynamic'` with no embedding/retrieval mode → defaults to `retrieval_mode = 'random'`

`run_all.py` filters its grid through these rules, so redundant combinations collapse and run only once.

## Models

Inference models (loaded from the HuggingFace Hub via `AutoModelForCausalLM`, `bfloat16`, `device_map='auto'`):

- `google/gemma-4-26B-A4B-it`
- `google/gemma-4-E4B-it`
- `Qwen/Qwen3.5-9B`
- `utter-project/EuroLLM-22B-Instruct-2512`

Embedding model for dense retrieval: `codefuse-ai/F2LLM-v2-1.7B`.

The `LM` class in `util.py` handles automatic batch sizing based on available GPU memory and recovers from OOM by halving the batch.

## Running Experiments

Requirements: a CUDA GPU and the `HF_TOKEN` environment variable (HuggingFace Hub access). Key dependencies: `transformers`/`torch`, `langchain`/`langchain-community`/`langchain-huggingface`, `scikit-learn`, `pandas`, `jinja2`, `pyyaml`, `tqdm`.

**Note on paths:** `config.yaml` ships with container-internal absolute paths (`/data`, `/results`, `/templates`) used on the HPC cluster — adapt `data_path`, `results_path`, and `template_path` to your environment. The evaluation and visualisation scripts use paths relative to `src/`, so run them from there.

```bash
cd src

# One configuration (as set in config.yaml), 70/30 split:
python single_run.py

# Full grid over all valid configurations, 4-fold CV.
# Skips combinations whose result file already exists, so it can be
# interrupted and resumed:
python run_all.py

# Non-LLM baseline (1-NN over TF-IDF, same folds):
python knn_baseline.py
```

Result CSVs contain the columns `id`, `text`, `true_label`, `predicted_label`, `reply` and are named

```
{model}_{prompt_mode}_{demonstration_mode}_{demonstration_size}_{embedding_mode}_{retrieval_mode}[_fold-N][_thinking].csv
```

with `None` values appearing literally as `None`, e.g. `gemma-4-26B-A4B-it_implicit_dynamic_8_dense_similarity_fold-0.csv`.

## Evaluation

The **primary metric is F1 Macro** (unweighted average of per-class F1), reported alongside accuracy and per-class precision/recall. Rows where the model abstained (`predicted_label` is NaN) are excluded from scoring; abstention counts are reported.

```bash
cd src

# Merge the four per-fold CSVs of each config into one full-dataset CSV:
python consolidate_folds.py

# Compute score tables (parses configs from the result filenames):
python evaluate_final_run.py
```

This produces `score_table_folds.csv` (one row per config and fold) and `score_table_consolidated.csv` (one row per config, all folds concatenated — every post appears exactly once since the test splits are disjoint). Both tables for the main experiment are checked in under [results/final_runs/](results/final_runs/).

## Visualisations

The scripts in [src/visualisations/](src/visualisations/) build all figures from the score tables (never from individual result CSVs) and save each figure as both `.pdf` and `.svg`:

- `spec_graph_f1.py` — specification curve of F1 Macro across all configurations, with fold-level variance as boxplots
- `impact_graph_f1.py` — impact of each configuration axis on F1 Macro (tornado, marginal-means, and heatmap views)
- `visualise_exploration.py` — F1 Macro by demonstration size from the exploration phase

All figures use a fixed, colorblind-safe model color scheme (Okabe–Ito palette) defined in [src/visualisations/model_colors.py](src/visualisations/model_colors.py) — import from there instead of defining ad-hoc palettes. Generated figures are checked in under [results/final_runs/figures/](results/final_runs/figures/) and [results/exploration/figures/](results/exploration/figures/).
