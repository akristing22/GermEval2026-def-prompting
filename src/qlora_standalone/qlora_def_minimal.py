"""
Minimal notebook-style QLoRA/PEFT fine-tuning script for GermEval 2026 DEF.

What it does:
1. Reads the DEF CSV.
2. Builds a small instruction dataset.
3. Creates a stratified train/holdout split.
4. Fine-tunes a causal LM with 4-bit QLoRA.
5. Evaluates on the holdout split with competition-relevant metrics.
6. Writes predictions, metrics, and TP/FP/FN/TN tables.

Run:
    python qlora_def_minimal.py

Optional:
    python qlora_def_minimal.py --model_id LSX-UniWue/LLaMmlein_1B --epochs 3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import util

# Config

DEFAULT_DATA_PATH = Path("data/def_train.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/qlora_def_minimal")

ID_COL = "id"
TEXT_COL = "description"
LABEL_COL = "DEF"
POS_LABEL = "TRUE"
NEG_LABEL = "FALSE"

POS_COMPLETION = " True"
NEG_COMPLETION = " False"

DEFAULT_PROMPT_CONFIG = {
    "prompt_mode": "description",
    "template_path": "templates",
    "demonstration_size": 0,
    "demonstration_mode": None,
    "embedding_mode": None,
    "retrieval_mode": None,
}


# Small utilities


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sniff_sep(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
    except csv.Error:
        return ";"


def normalize_def_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(col).strip() for col in frame.columns]

    lower_to_col = {col.lower(): col for col in frame.columns}
    required = {ID_COL.lower(), TEXT_COL.lower(), LABEL_COL.lower()}
    missing = required - set(lower_to_col)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in input frame")

    frame = frame.rename(
        columns={
            lower_to_col[ID_COL.lower()]: ID_COL,
            lower_to_col[TEXT_COL.lower()]: TEXT_COL,
            lower_to_col[LABEL_COL.lower()]: LABEL_COL,
        }
    )
    frame = frame[[ID_COL, TEXT_COL, LABEL_COL]].copy()
    frame[TEXT_COL] = frame[TEXT_COL].fillna("").astype(str)

    if frame[LABEL_COL].dtype == bool:
        frame[LABEL_COL] = np.where(frame[LABEL_COL], POS_LABEL, NEG_LABEL)
    else:
        frame[LABEL_COL] = frame[LABEL_COL].astype(str).str.strip().str.upper()

    frame = frame[frame[LABEL_COL].isin([NEG_LABEL, POS_LABEL])].reset_index(drop=True)
    frame["y"] = (frame[LABEL_COL] == POS_LABEL).astype(int)
    return frame


def read_def_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=sniff_sep(path), encoding="utf-8-sig")
    return normalize_def_frame(frame)


def messages_to_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) + "\nassistant:"


def make_prompt_config(config: dict | None) -> dict:
    prompt_config = {**DEFAULT_PROMPT_CONFIG, **(config or {})}
    prompt_config["demonstration_size"] = 0
    prompt_config["demonstration_mode"] = None
    prompt_config["embedding_mode"] = None
    prompt_config["retrieval_mode"] = None
    prompt_config["prompt_mode"] = prompt_config.get(
        "qlora_prompt_mode", prompt_config["prompt_mode"]
    )
    if prompt_config["prompt_mode"] == "explicit":
        raise ValueError(
            "QLoRA DEF fine-tuning supports title/description/implicit prompts. "
            "Explicit mode needs per-step labels and should be trained separately."
        )
    return prompt_config


def make_instruction_frame(
    frame: pd.DataFrame, tokenizer=None, config: dict | None = None
) -> pd.DataFrame:
    out = frame.copy()
    if tokenizer is None:
        prompt_config = DEFAULT_PROMPT_CONFIG
        pc = util.PromptConstructor(None, prompt_config)
        out["prompt"] = out[TEXT_COL].map(
            lambda text: "\n".join(
                msg["content"] for msg in pc.construct(text, system_prompt=True)
            )
        )
    else:
        prompt_config = make_prompt_config(config)
        pc = util.PromptConstructor(None, prompt_config)
        out["prompt"] = out[TEXT_COL].map(
            lambda text: messages_to_prompt(
                tokenizer, pc.construct(text, system_prompt=True)
            )
        )
    out["completion"] = np.where(out["y"].eq(1), POS_COMPLETION, NEG_COMPLETION)
    return out


def make_splits(
    frame: pd.DataFrame, holdout_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, holdout = train_test_split(
        frame,
        test_size=holdout_size,
        random_state=seed,
        stratify=frame["y"],
    )
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


# Tokenization for supervised fine-tuning


def build_tokenize_fn(tokenizer, max_length: int):
    eos = tokenizer.eos_token or ""

    def tokenize(row: dict) -> dict:
        prompt = row["prompt"]
        completion = row["completion"] + eos
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = tokenizer(
            prompt + completion,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )
        labels = full["input_ids"].copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        full["labels"] = labels
        return full

    return tokenize


def to_hf_dataset(frame: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(frame[["prompt", "completion"]], preserve_index=False)


# Model setup


def load_qlora_model(model_id: str, gradient_checkpointing: bool):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype="auto",
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)
    lora_kwargs = {}
    if getattr(model.config, "model_type", None) == "gemma4":
        # Gemma 4's vision/audio projections are wrapped in
        # Gemma4ClippableLinear, which PEFT cannot replace. The language-model
        # projections use supported linear layers and are the only ones needed
        # for this text classification task.
        lora_kwargs["exclude_modules"] = r".*\.(vision_tower|audio_tower)\..*"
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        **lora_kwargs,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


class QLoRALM:
    """4-bit PEFT adapter inference wrapper with the same generate() shape as util.LM."""

    def __init__(self, model_id: str, adapter_path: str | Path):
        self.model_name = model_id
        self.adapter_path = str(adapter_path)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype="auto",
        )
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()

        chat_template = getattr(self.tokenizer, "chat_template", None) or ""
        self.thinking_capable = "enable_thinking" in chat_template

    def update_kv_cache(self) -> None:
        return None

    def generate(
        self,
        prompts: list[list[dict[str, str]]],
        max_tokens: int = 10,
        thinking_mode: bool = False,
    ) -> list[str]:
        outputs = []
        for prompt in tqdm(
            prompts, total=len(prompts), mininterval=1.0, dynamic_ncols=True
        ):
            if getattr(self.tokenizer, "chat_template", None):
                template_kwargs = {
                    "tokenize": True,
                    "add_generation_prompt": True,
                    "return_tensors": "pt",
                    "return_dict": True,
                }
                if self.thinking_capable:
                    template_kwargs["enable_thinking"] = thinking_mode

                inputs = self.tokenizer.apply_chat_template(
                    prompt,
                    **template_kwargs,
                ).to(self.model.device)
            else:
                prompt_text = messages_to_prompt(self.tokenizer, prompt)
                inputs = self.tokenizer(
                    prompt_text, return_tensors="pt"
                ).to(self.model.device)
            input_length = inputs["input_ids"].shape[1]
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = generated_ids[:, input_length:]
            outputs.append(
                self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            )
        return outputs


def qlora_adapter_path(config: dict, split: int) -> str | None:
    adapter_path = config.get("qlora_adapter_path")
    if adapter_path is None:
        return None
    return adapter_path.format(split=split, fold=split)


def train_qlora_adapter(
    train_frame: pd.DataFrame,
    config: dict,
    output_dir: str | Path,
    seed: int = 42,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    train_frame = normalize_def_frame(train_frame)

    model_id = config["model_name"]
    model, tokenizer = load_qlora_model(
        model_id,
        gradient_checkpointing=config.get("qlora_gradient_checkpointing", True),
    )
    instructions = make_instruction_frame(train_frame, tokenizer=tokenizer, config=config)
    instructions.to_csv(output_dir / "train_instructions.csv", sep=";", index=False)

    tokenize = build_tokenize_fn(
        tokenizer, max_length=config.get("qlora_max_length", 512)
    )
    train_ds = to_hf_dataset(instructions).map(
        tokenize, remove_columns=["prompt", "completion"]
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=config.get("qlora_batch_size", 1),
        gradient_accumulation_steps=config.get("qlora_grad_accum_steps", 32),
        num_train_epochs=config.get("qlora_epochs", 3),
        learning_rate=config.get("qlora_lr", 2e-5),
        warmup_ratio=config.get("qlora_warmup_ratio", 0.06),
        weight_decay=config.get("qlora_weight_decay", 0.01),
        max_grad_norm=0.3,
        logging_steps=config.get("qlora_logging_steps", 10),
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        bf16=False,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adapter_dir


def resolve_or_train_adapter(
    train_frame: pd.DataFrame,
    config: dict,
    split: int,
) -> str:
    adapter_path = qlora_adapter_path(config, split)
    if adapter_path is not None:
        return adapter_path

    output_dir = Path(config.get("qlora_output_dir", "outputs/qlora")) / f"fold-{split}"
    adapter_dir = output_dir / "adapter"
    if (adapter_dir / "adapter_config.json").exists():
        return str(adapter_dir)

    return str(
        train_qlora_adapter(
            train_frame,
            config,
            output_dir,
            seed=config.get("qlora_seed", 42),
        )
    )


def load_finetuned_lm(config: dict, train_frame: pd.DataFrame, split: int) -> QLoRALM:
    adapter_path = resolve_or_train_adapter(train_frame, config, split)
    print(f"Loading QLoRA adapter from {adapter_path}...")
    return QLoRALM(config["model_name"], adapter_path)


# Forced-choice evaluation


@torch.inference_mode()
def score_completion(
    model, tokenizer, prompt: str, completion: str, max_length: int
) -> float:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        prompt + completion,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    input_ids = full["input_ids"]
    if input_ids.shape[1] <= len(prompt_ids):
        return -1e9

    logits = model(**full).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    start = max(len(prompt_ids) - 1, 0)
    completion_log_probs = token_log_probs[:, start:]
    return float(completion_log_probs.mean().detach().cpu())


def evaluate_holdout(
    model, tokenizer, holdout: pd.DataFrame, max_length: int
) -> pd.DataFrame:
    model.eval()
    rows = []
    for row in holdout.to_dict("records"):
        prompt = row["prompt"]
        pos_score = score_completion(
            model, tokenizer, prompt, POS_COMPLETION, max_length=max_length
        )
        neg_score = score_completion(
            model, tokenizer, prompt, NEG_COMPLETION, max_length=max_length
        )
        pred = int(pos_score > neg_score)
        rows.append(
            {
                ID_COL: row[ID_COL],
                TEXT_COL: row[TEXT_COL],
                "gold": int(row["y"]),
                "gold_label": POS_LABEL if row["y"] else NEG_LABEL,
                "pred": pred,
                "pred_label": POS_LABEL if pred else NEG_LABEL,
                "score_true": pos_score,
                "score_false": neg_score,
                "score_margin_true_minus_false": pos_score - neg_score,
            }
        )
    return pd.DataFrame(rows)


def compute_metrics(predictions: pd.DataFrame) -> dict:
    y_true = predictions["gold"].to_numpy()
    y_pred = predictions["pred"].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "n": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "positive_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "positive_precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "positive_recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "confusion_matrix_labels": [NEG_LABEL, POS_LABEL],
        "confusion_matrix": cm.tolist(),
        "tp_fp_fn_tn": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[NEG_LABEL, POS_LABEL],
            output_dict=True,
            zero_division=0,
        ),
    }


def add_error_bucket(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    conditions = [
        out["gold"].eq(1) & out["pred"].eq(1),
        out["gold"].eq(0) & out["pred"].eq(1),
        out["gold"].eq(1) & out["pred"].eq(0),
        out["gold"].eq(0) & out["pred"].eq(0),
    ]
    out["bucket"] = np.select(conditions, ["TP", "FP", "FN", "TN"], default="UNK")
    return out


# Main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_id", default="google/gemma-4-E4B-it")
    parser.add_argument("--template_path", default="templates")
    parser.add_argument(
        "--prompt_mode",
        default="description",
        choices=["title", "description", "implicit"],
    )
    parser.add_argument("--holdout_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV -> instruction rows -> stratified holdout split.
    raw = read_def_csv(args.data_path)
    train_raw, holdout_raw = make_splits(raw, holdout_size=args.holdout_size, seed=args.seed)

    print("Label distribution:")
    print(raw[LABEL_COL].value_counts())
    print(f"Train rows: {len(train_raw)} | Holdout rows: {len(holdout_raw)}")

    # QLoRA model and prompt-masked SFT dataset.
    model, tokenizer = load_qlora_model(
        args.model_id, gradient_checkpointing=args.gradient_checkpointing
    )
    prompt_config = {
        "prompt_mode": args.prompt_mode,
        "template_path": args.template_path,
    }
    train_frame = make_instruction_frame(
        train_raw, tokenizer=tokenizer, config=prompt_config
    )
    holdout_frame = make_instruction_frame(
        holdout_raw, tokenizer=tokenizer, config=prompt_config
    )
    train_frame.to_csv(args.output_dir / "train_instructions.csv", sep=";", index=False)
    holdout_frame.to_csv(
        args.output_dir / "holdout_instructions.csv", sep=";", index=False
    )
    tokenize = build_tokenize_fn(tokenizer, max_length=args.max_length)
    train_ds = to_hf_dataset(train_frame).map(
        tokenize, remove_columns=["prompt", "completion"]
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=0.3,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        bf16=False,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Holdout evaluation via forced choice: compare logprob(" JA") vs logprob(" NEIN").
    predictions = evaluate_holdout(
        trainer.model, tokenizer, holdout_frame, max_length=args.max_length
    )
    predictions = add_error_bucket(predictions)
    metrics = compute_metrics(predictions)

    predictions.to_csv(
        args.output_dir / "holdout_predictions.csv", sep=";", index=False
    )
    for bucket, bucket_frame in predictions.groupby("bucket"):
        bucket_frame.to_csv(
            args.output_dir / f"holdout_{bucket.lower()}.csv", sep=";", index=False
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nMain metrics:")
    main_metrics = {
        k: metrics[k]
        for k in [
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "positive_f1",
            "positive_precision",
            "positive_recall",
            "tp_fp_fn_tn",
            "confusion_matrix",
        ]
    }
    print(json.dumps(main_metrics, indent=2))
    print(f"\nWrote artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
