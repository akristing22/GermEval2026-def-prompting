# ToDo 

### Supervised Fine-Tuning (PEFT): completed

#### Gemma 4 QLoRA run

- Model: `google/gemma-4-E4B-it`
- Fine-tuning: PEFT/LoRA with 4-bit NF4 quantisation, double quantisation, and
  `paged_adamw_8bit`
- Evaluation setup: stratified 4-fold CV (`random_state=42`)
- Prompting: implicit decision-schema template, zero-shot inference
- Demonstrations/retrieval: disabled (`demonstration_size: 0`)
- Training: 3 epochs, batch size 1, gradient accumulation 32, learning rate
  `2e-5`, gradient checkpointing enabled, maximum sequence length 1024
- Coverage: all 3,263 instances scored exactly once; no duplicate IDs, missing
  IDs, or abstentions

Consolidated results:

- F1 Macro: **0.7405**
- Accuracy: **0.9044**
- Positive-class F1: **0.5343**
- Positive precision: **0.7020**
- Positive recall: **0.4313**
- Confusion matrix: TN 2,772; FP 76; FN 236; TP 179
- Fold F1 Macro: 0.7575, 0.7525, 0.7756, 0.6662
- Fold mean +/- standard deviation: **0.7379 +/- 0.0488**

Artifacts:

- Fold predictions and evaluation files: `results/gemma_run/`
- Consolidated predictions:
  `results/gemma_run/gemma-4-E4B-it-qlora_implicit_None_0_None_None_consolidated.csv`
- Evaluation summary: `results/gemma_run/evaluation_summary.json`
- Fold metrics: `results/gemma_run/score_table_folds.csv`
- Misclassifications: `results/gemma_run/errors.csv`

Implementation and Colab notes:

- `qlora_max_length: 512` truncated the completion labels for the implicit
  prompt and produced zero loss/gradients; 1024 was used for the successful run.
- Gemma 4 vision/audio modules must be excluded from LoRA injection because
  their `Gemma4ClippableLinear` wrappers are unsupported by PEFT.
- Models, retrieval embeddings, and CUDA cache must be released between folds
  to avoid cumulative GPU OOM errors.
- Existing `fold-N/adapter/adapter_config.json` files are reused automatically,
  allowing interrupted CV runs to resume.
- Colab outputs must be written to Google Drive, for example
  `/content/drive/MyDrive/germeval/qlora_gemma4_e4b`, because `/content` is
  deleted when the runtime ends.

### API Inference: completed

- Implemented OpenAI Responses API inference through `LM_API`
- Completed stratified 4-fold CV for both configurations

Report: 
- unedited did not work:
  - had to switch a demonstration mode, 
  - embedding mode,
  - retrieval mode and 
  - allow for at least 16 tokens (OpenAI minimum)
- ran config 1 + config 2 as documented in the config.yaml

Artifacts:

- Config 1 fold predictions and scores: `results/config1_run/`
- Config 2 fold predictions: `results/config2_run/`
- Run configurations: `results/gpt_configs_run1_run2.yaml`

## TODO Inference: 
- run all gemma adapters w. majority vote on the test data split

## Paper 
- Sektionen: 4.3.1.-4.3.3
- 