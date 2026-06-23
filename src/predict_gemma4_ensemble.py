"""Run four QLoRA fold adapters on the GermEval DEF test set.

The base Gemma model is loaded once in 4-bit NF4. The fold adapters are then
loaded into that model and activated one at a time. Each adapter scores the
forced-choice completions ``True`` and ``False``; the final prediction is the
majority vote across adapters. A 2-2 tie is resolved by the mean score margin.

The submission CSV contains exactly the required semicolon-separated columns:

    id;DEF
    1234;TRUE
    4321;FALSE

Example:

    python src/predict_gemma4_ensemble.py \
        --adapter-root outputs/qlora_gemma4_e4b \
        --adapter-pattern "fold-{fold}/adapter" \
        --test-file data/def_test.csv \
        --output-dir outputs
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import util  # noqa: E402

ID_COL = "id"
TEXT_COL = "description"
LABEL_COL = "DEF"
POS_LABEL = "TRUE"
NEG_LABEL = "FALSE"
POS_COMPLETION = " True"
NEG_COMPLETION = " False"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble four Gemma-4 QLoRA fold adapters on def_test.csv."
    )
    parser.add_argument(
        "--adapter-root",
        type=Path,
        required=True,
        help="Folder containing the fold adapter directories.",
    )
    parser.add_argument(
        "--adapter-pattern",
        default="fold-{fold}/adapter",
        help="Path below --adapter-root; {fold} is replaced by each fold number.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Fold adapters to ensemble (default: 0 1 2 3).",
    )
    parser.add_argument(
        "--model-id",
        default="google/gemma-4-E4B-it",
        help="Hugging Face base model ID.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/def_test.csv"),
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=Path("templates"),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["title", "description", "implicit"],
        default="implicit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--output-name",
        default="gemma4_def_submission.csv",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of test posts scored together (start with 1 on small GPUs).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum prompt-plus-label length used during scoring.",
    )
    parser.add_argument(
        "--tie-break",
        choices=["margin", "true", "false"],
        default="margin",
        help="How to resolve an even vote split (default: mean score margin).",
    )
    parser.add_argument(
        "--diagnostics-name",
        default=None,
        help="Optional filename for per-fold votes and score margins.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the base model with NF4 quantization (default: true).",
    )
    return parser.parse_args()


def sniff_separator(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
    except csv.Error:
        return ";"


def read_test_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Test file does not exist: {path}")

    frame = pd.read_csv(path, sep=sniff_separator(path), encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    columns = {column.lower(): column for column in frame.columns}
    missing = {ID_COL, TEXT_COL} - set(columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    frame = frame.rename(
        columns={
            columns[ID_COL]: ID_COL,
            columns[TEXT_COL]: TEXT_COL,
        }
    )[[ID_COL, TEXT_COL]].copy()
    if frame[ID_COL].isna().any():
        raise ValueError("Test data contains missing IDs.")
    if frame[ID_COL].duplicated().any():
        raise ValueError("Test data contains duplicate IDs.")
    frame[TEXT_COL] = frame[TEXT_COL].fillna("").astype(str)
    return frame


def resolve_adapter_paths(args: argparse.Namespace) -> list[tuple[int, Path]]:
    adapters = []
    for fold in args.folds:
        relative_path = args.adapter_pattern.format(fold=fold, split=fold)
        adapter_path = args.adapter_root / relative_path
        config_path = adapter_path / "adapter_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Fold {fold} adapter not found: expected {config_path}"
            )
        adapters.append((fold, adapter_path))
    return adapters


def make_prompt_texts(
    frame: pd.DataFrame,
    tokenizer,
    template_path: Path,
    prompt_mode: str,
) -> list[str]:
    prompt_config = {
        "prompt_mode": prompt_mode,
        "template_path": str(template_path),
        "demonstration_size": 0,
        "demonstration_mode": None,
        "embedding_mode": None,
        "retrieval_mode": None,
    }
    constructor = util.PromptConstructor(None, prompt_config)
    prompts = []
    for text in tqdm(frame[TEXT_COL], desc="Building prompts"):
        messages = constructor.construct(text, system_prompt=True)
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                "\n".join(
                    f"{message['role']}: {message['content']}"
                    for message in messages
                )
                + "\nassistant:"
            )
        prompts.append(prompt)
    return prompts


def completion_token_ids(tokenizer, completion: str) -> list[int]:
    token_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError(f"Completion produced no tokens: {completion!r}")
    return token_ids


@torch.inference_mode()
def score_prompt_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_length: int,
) -> tuple[list[float], list[float]]:
    """Return average log-probability scores for True and False per prompt."""

    candidates = [
        (POS_COMPLETION, completion_token_ids(tokenizer, POS_COMPLETION)),
        (NEG_COMPLETION, completion_token_ids(tokenizer, NEG_COMPLETION)),
    ]
    sequences: list[list[int]] = []
    completion_ranges: list[tuple[int, int]] = []

    for prompt in prompts:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for completion, completion_ids in candidates:
            if len(prompt_ids) + len(completion_ids) > max_length:
                raise ValueError(
                    "Prompt plus completion exceeds --max-length "
                    f"({len(prompt_ids)} + {len(completion_ids)} > {max_length}). "
                    "Increase --max-length; truncating would invalidate scoring."
                )
            input_ids = prompt_ids + completion_ids
            start = len(prompt_ids)
            if start == 0:
                raise ValueError(f"Prompt produced no tokens before {completion!r}.")
            sequences.append(input_ids)
            completion_ranges.append((start, len(input_ids)))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither a padding token nor an EOS token.")

    longest = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), longest),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, :length] = 1

    input_device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(input_device)
    attention_mask = attention_mask.to(input_device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    scores = []
    for row, (start, end) in enumerate(completion_ranges):
        target_ids = input_ids[row, start:end]
        token_log_probs = log_probs[row, start - 1 : end - 1].gather(
            1, target_ids.unsqueeze(1)
        )
        scores.append(float(token_log_probs.mean().cpu()))

    true_scores = scores[0::2]
    false_scores = scores[1::2]
    return true_scores, false_scores


def score_adapter(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_length: int,
    fold: int,
) -> tuple[list[bool], list[float]]:
    votes: list[bool] = []
    margins: list[float] = []
    batches = range(0, len(prompts), batch_size)
    for start in tqdm(
        batches,
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc=f"Fold {fold}",
    ):
        true_scores, false_scores = score_prompt_batch(
            model,
            tokenizer,
            prompts[start : start + batch_size],
            max_length=max_length,
        )
        batch_margins = [
            true_score - false_score
            for true_score, false_score in zip(true_scores, false_scores)
        ]
        margins.extend(batch_margins)
        votes.extend(margin > 0 for margin in batch_margins)
    return votes, margins


def load_model_and_adapters(
    model_id: str,
    adapters: list[tuple[int, Path]],
    load_in_4bit: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": "auto",
    }
    if load_in_4bit:
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    first_fold, first_path = adapters[0]
    print(f"Loading base model and fold {first_fold} adapter from {first_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model = PeftModel.from_pretrained(
        base_model,
        first_path,
        adapter_name=f"fold_{first_fold}",
        is_trainable=False,
    )

    for fold, adapter_path in adapters[1:]:
        print(f"Loading fold {fold} adapter from {adapter_path}...")
        model.load_adapter(
            adapter_path,
            adapter_name=f"fold_{fold}",
            is_trainable=False,
        )
    model.eval()
    return model, tokenizer


def aggregate_votes(
    fold_votes: dict[int, list[bool]],
    fold_margins: dict[int, list[float]],
    tie_break: str,
) -> tuple[list[str], list[int], list[float]]:
    labels: list[str] = []
    true_vote_counts: list[int] = []
    mean_margins: list[float] = []
    folds = list(fold_votes)

    for row in range(len(next(iter(fold_votes.values())))):
        true_votes = sum(fold_votes[fold][row] for fold in folds)
        mean_margin = sum(fold_margins[fold][row] for fold in folds) / len(folds)

        if true_votes * 2 > len(folds):
            prediction = True
        elif true_votes * 2 < len(folds):
            prediction = False
        elif tie_break == "margin":
            prediction = mean_margin > 0
        else:
            prediction = tie_break == "true"

        labels.append(POS_LABEL if prediction else NEG_LABEL)
        true_vote_counts.append(true_votes)
        mean_margins.append(mean_margin)

    return labels, true_vote_counts, mean_margins


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.max_length < 1:
        raise ValueError("--max-length must be positive.")
    if not args.template_path.is_dir():
        raise FileNotFoundError(f"Template folder does not exist: {args.template_path}")

    test = read_test_file(args.test_file)
    adapters = resolve_adapter_paths(args)
    print(f"Loaded {len(test)} test rows and found {len(adapters)} fold adapters.")

    model, tokenizer = load_model_and_adapters(
        args.model_id,
        adapters,
        load_in_4bit=args.load_in_4bit,
    )
    prompts = make_prompt_texts(
        test,
        tokenizer,
        template_path=args.template_path,
        prompt_mode=args.prompt_mode,
    )

    fold_votes: dict[int, list[bool]] = {}
    fold_margins: dict[int, list[float]] = {}
    for fold, _ in adapters:
        adapter_name = f"fold_{fold}"
        print(f"Activating {adapter_name}...")
        model.set_adapter(adapter_name)
        votes, margins = score_adapter(
            model,
            tokenizer,
            prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            fold=fold,
        )
        fold_votes[fold] = votes
        fold_margins[fold] = margins

    labels, true_vote_counts, mean_margins = aggregate_votes(
        fold_votes,
        fold_margins,
        tie_break=args.tie_break,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    submission = pd.DataFrame({ID_COL: test[ID_COL], LABEL_COL: labels})
    submission.to_csv(
        output_path,
        sep=";",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    if list(submission.columns) != [ID_COL, LABEL_COL]:
        raise AssertionError("Submission columns do not match the required contract.")
    if not submission[LABEL_COL].isin([POS_LABEL, NEG_LABEL]).all():
        raise AssertionError("Submission contains labels outside TRUE/FALSE.")

    if args.diagnostics_name:
        diagnostics = test[[ID_COL]].copy()
        for fold, _ in adapters:
            diagnostics[f"fold_{fold}_vote"] = [
                POS_LABEL if vote else NEG_LABEL for vote in fold_votes[fold]
            ]
            diagnostics[f"fold_{fold}_margin"] = fold_margins[fold]
        diagnostics["true_votes"] = true_vote_counts
        diagnostics["mean_margin"] = mean_margins
        diagnostics[LABEL_COL] = labels
        diagnostics.to_csv(
            args.output_dir / args.diagnostics_name,
            sep=";",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

    print(f"Wrote {len(submission)} predictions to {output_path}")
    print(submission[LABEL_COL].value_counts().to_string())

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
