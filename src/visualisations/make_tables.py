#!/usr/bin/env python3
"""Generate LaTeX result tables for the main experiment (final_run).

Tables produced
---------------
table_a1_model_prompt.tex   – model × prompt mode: ZS | best k=8 | worst k=8
table_a2_model_shot.tex     – model × zero-shot vs few-shot: ZS | best | worst | Δ(best-ZS)
table_a3_model_retrieval.tex– model × retrieval config (mean F1 over prompt modes)
table_b_best_fewshot.tex    – best k=8 per (model, prompt_mode) with retrieval subscript
table_d_full.tex            – full appendix table, all 112 configurations
table_e_best_per_model.tex  – best config per model with full metrics
"""

import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model_colors import MODEL_LABELS

DATA_PATH = os.path.join(os.path.dirname(__file__),
                         "../../results/final_run/score_table_consolidated.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__),
                       "../../results/final_run/tables/")

PROMPT_ORDER = ["title", "description", "implicit", "explicit"]
PROMPT_LABELS = {
    "title":       "Title",
    "description": "Desc.",
    "implicit":    "Implicit",
    "explicit":    "Explicit",
}

# Order matches CLAUDE.md retrieval_mode description order
RETRIEVAL_ORDER = [
    ("dense",  "similarity"),
    ("dense",  "mmr"),
    ("dense",  "diversity"),
    ("fusion", "similarity"),
    ("sparse", "similarity"),
    (None,     "random"),
]
RETRIEVAL_LABELS = {
    ("dense",  "similarity"): "D-Sim",
    ("dense",  "mmr"):        "D-MMR",
    ("dense",  "diversity"):  "D-Div",
    ("fusion", "similarity"): "F-Sim",
    ("sparse", "similarity"): "S-Sim",
    (None,     "random"):     "Rand",
}

# Ordered by best F1 macro descending (from data)
MODEL_ORDER = [
    "gemma-4-26B-A4B-it",
    "gemma-4-E4B-it",
    "Qwen3.5-9B",
    "EuroLLM-22B-Instruct-2512",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(v, bold=False, italic=False):
    """Format float as .3f; optionally bold or italic."""
    if pd.isna(v):
        return "---"
    s = f"{v:.3f}"
    if bold:
        s = r"\textbf{" + s + r"}"
    if italic:
        s = r"\textit{" + s + r"}"
    return s


def _norm_key(emb, ret):
    """Convert (embedding_mode, retrieval_mode) pandas values to a canonical dict key."""
    e = None if (not isinstance(emb, str) and pd.isna(emb)) else emb
    r = None if (not isinstance(ret, str) and pd.isna(ret)) else ret
    return (e, r)


def _add_ret_label(df):
    df = df.copy()
    df["ret_label"] = [
        RETRIEVAL_LABELS.get(_norm_key(e, r), f"{e}/{r}")
        for e, r in zip(df["embedding_mode"], df["retrieval_mode"])
    ]
    df["ret_key"] = [_norm_key(e, r) for e, r in zip(df["embedding_mode"], df["retrieval_mode"])]
    return df


def save(latex: str, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        f.write(latex)
    print(f"Written: {path}")


def load():
    df = pd.read_csv(DATA_PATH)
    df = _add_ret_label(df)
    return df


# ---------------------------------------------------------------------------
# Table A1 – Model × Prompt Mode
# ---------------------------------------------------------------------------

def make_a1_model_prompt(df):
    """ZS | best k=8 | worst k=8 for each (model, prompt_mode) cell."""
    zs = df[df["demo_size"] == 0].groupby(["model", "prompt_mode"])["f1_macro"].first()
    fs_best  = df[df["demo_size"] == 8].groupby(["model", "prompt_mode"])["f1_macro"].max()
    fs_worst = df[df["demo_size"] == 8].groupby(["model", "prompt_mode"])["f1_macro"].min()

    # Per-column (prompt_mode × sub-column) max for bolding
    col_max = {}
    for pm in PROMPT_ORDER:
        col_max[(pm, "zs")]   = max(zs.xs(pm, level="prompt_mode").max(), 0)
        col_max[(pm, "best")] = max(fs_best.xs(pm, level="prompt_mode").max(), 0)

    n_pm = len(PROMPT_ORDER)
    col_spec = "l" + "rrr" * n_pm
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{F1 Macro by model and prompt mode. "
        r"Each group shows zero-shot (ZS), best few-shot, and worst few-shot (k=8).}",
        r"\label{tab:a1-model-prompt}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Header row 1: prompt-mode group spans
    h1 = "Model"
    for pm in PROMPT_ORDER:
        h1 += r" & \multicolumn{3}{c}{" + PROMPT_LABELS[pm] + "}"
    lines.append(h1 + r" \\")

    # Cmidrules under each group
    cmidrules = []
    for i in range(n_pm):
        start = 2 + i * 3
        cmidrules.append(r"\cmidrule(lr){" + f"{start}-{start+2}" + "}")
    lines.append(" ".join(cmidrules))

    # Header row 2: sub-columns
    h2 = ""
    for _ in PROMPT_ORDER:
        h2 += r" & ZS & Best & Worst"
    lines.append(h2 + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        row = MODEL_LABELS[model]
        for pm in PROMPT_ORDER:
            zs_v  = zs.get((model, pm), float("nan"))
            best_v = fs_best.get((model, pm), float("nan"))
            wrst_v = fs_worst.get((model, pm), float("nan"))
            bold_zs   = not pd.isna(zs_v)  and abs(zs_v  - col_max[(pm, "zs")])   < 1e-6
            bold_best = not pd.isna(best_v) and abs(best_v - col_max[(pm, "best")]) < 1e-6
            row += f" & {fmt(zs_v, bold=bold_zs)} & {fmt(best_v, bold=bold_best)} & {fmt(wrst_v)}"
        lines.append(row + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table A2 – Model × Zero-shot vs Few-shot
# ---------------------------------------------------------------------------

def make_a2_model_shot(df):
    """Best ZS and best/worst k=8 per model (aggregated over all prompt modes)."""
    zs_best  = df[df["demo_size"] == 0].groupby("model")["f1_macro"].max()
    fs_best  = df[df["demo_size"] == 8].groupby("model")["f1_macro"].max()
    fs_worst = df[df["demo_size"] == 8].groupby("model")["f1_macro"].min()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Best zero-shot vs.\ best and worst few-shot (k=8) F1 Macro per model "
        r"(best over all prompt and retrieval settings). "
        r"$\Delta$ is the gain from zero-shot to best few-shot.}",
        r"\label{tab:a2-model-shot}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & ZS & Best $k$=8 & Worst $k$=8 & $\Delta$ \\",
        r"\midrule",
    ]

    for model in MODEL_ORDER:
        zs_v  = zs_best[model]
        best_v = fs_best[model]
        wrst_v = fs_worst[model]
        delta  = best_v - zs_v
        lines.append(
            f"{MODEL_LABELS[model]}"
            f" & {fmt(zs_v)}"
            f" & {fmt(best_v)}"
            f" & {fmt(wrst_v)}"
            f" & {fmt(delta)}"
            r" \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table A3 – Model × Retrieval Config
# ---------------------------------------------------------------------------

def make_a3_model_retrieval(df):
    """F1 Macro averaged over prompt modes for each (model, retrieval_config).

    Best value per row is bold; worst is italic.
    """
    fs = df[df["demo_size"] == 8].copy()
    # Use string label as groupby key to avoid pandas MultiIndex expansion of tuples
    pivot = (fs.groupby(["model", "ret_label"])["f1_macro"]
               .mean()
               .unstack("ret_label"))

    ret_col_order = [RETRIEVAL_LABELS[c] for c in RETRIEVAL_ORDER
                     if RETRIEVAL_LABELS[c] in pivot.columns]
    col_spec = "l" + "r" * len(ret_col_order) + "r"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{F1 Macro by model and retrieval configuration (k=8, "
        r"mean over all prompt modes). "
        r"\textbf{Bold}: best retrieval config per model; "
        r"\textit{italic}: worst. "
        r"$\Delta$ = best$-$worst.}",
        r"\label{tab:a3-model-retrieval}",
        r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    header = "Model" + "".join([f" & {c}" for c in ret_col_order]) + r" & $\Delta$"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        vals = [pivot.loc[model, c] if c in pivot.columns else float("nan")
                for c in ret_col_order]
        finite = [v for v in vals if not pd.isna(v)]
        best_v = max(finite) if finite else None
        wrst_v = min(finite) if finite else None
        delta = (best_v - wrst_v) if (best_v is not None and wrst_v is not None) else float("nan")
        row = MODEL_LABELS[model]
        for v in vals:
            is_best  = best_v is not None and not pd.isna(v) and abs(v - best_v) < 1e-6
            is_worst = wrst_v is not None and not pd.isna(v) and abs(v - wrst_v) < 1e-6
            row += f" & {fmt(v, bold=is_best, italic=(is_worst and not is_best))}"
        row += f" & {fmt(delta)}"
        lines.append(row + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table B – Best few-shot per (model, prompt_mode)
# ---------------------------------------------------------------------------

def make_table_b(df):
    """Best k=8 F1 Macro per cell, annotated with retrieval config as subscript.

    Column-best values are bold.
    """
    fs = df[df["demo_size"] == 8]
    idx = fs.groupby(["model", "prompt_mode"])["f1_macro"].idxmax()
    best_rows = fs.loc[idx]

    best_f1  = best_rows.pivot(index="model", columns="prompt_mode", values="f1_macro")
    best_ret = best_rows.pivot(index="model", columns="prompt_mode", values="ret_label")

    col_bests = {pm: best_f1[pm].max() for pm in PROMPT_ORDER if pm in best_f1.columns}

    col_spec = "l" + "r" * len(PROMPT_ORDER)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Best few-shot (k=8) F1 Macro per model and prompt mode. "
        r"Subscripts indicate the retrieval configuration that achieved the result. "
        r"\textbf{Bold}: column maximum.}",
        r"\label{tab:b-best-fewshot}",
        r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    header = "Model" + "".join([f" & {PROMPT_LABELS[pm]}" for pm in PROMPT_ORDER])
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        row = MODEL_LABELS[model]
        for pm in PROMPT_ORDER:
            v   = best_f1.loc[model, pm]   if pm in best_f1.columns   else float("nan")
            ret = best_ret.loc[model, pm]   if pm in best_ret.columns  else ""
            is_best = not pd.isna(v) and abs(v - col_bests.get(pm, -1)) < 1e-6
            cell = fmt(v, bold=is_best)
            if isinstance(ret, str) and ret:
                cell += r"$_{\text{" + ret + r"}}$"
            row += f" & {cell}"
        lines.append(row + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table E – Best configuration per model with full metrics
# ---------------------------------------------------------------------------

def make_table_e(df):
    """One row per model: best config (any prompt/retrieval) with all metrics."""
    idx = df.groupby("model")["f1_macro"].idxmax()
    best = df.loc[idx].set_index("model")

    # For bolding: best value per metric column
    metric_cols = ["f1_macro", "accuracy", "cohen_kappa",
                   "precision_true", "recall_true", "precision_false", "recall_false"]
    col_max = {c: best[c].max() for c in metric_cols}

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Best configuration per model (highest F1 Macro over all prompt modes "
        r"and retrieval settings). "
        r"P$_\mathsf{T}$/R$_\mathsf{T}$: precision/recall for the prosecutable class "
        r"(\emph{True}); "
        r"P$_\mathsf{F}$/R$_\mathsf{F}$: precision/recall for the non-prosecutable class "
        r"(\emph{False}).}",
        r"\label{tab:e-best-per-model}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Model & Config & F1 & Acc. & $\kappa$ "
        r"& P$_\mathsf{T}$ & R$_\mathsf{T}$ & P$_\mathsf{F}$ & R$_\mathsf{F}$ \\",
        r"\midrule",
    ]

    for model in MODEL_ORDER:
        r = best.loc[model]
        pm  = PROMPT_LABELS[r["prompt_mode"]]
        k   = int(r["demo_size"])
        ret = r["ret_label"]
        if k == 0:
            config = f"{pm}, ZS"
        else:
            config = f"{pm}, {ret}"

        def b(col):
            return fmt(r[col], bold=(abs(r[col] - col_max[col]) < 1e-6))

        lines.append(
            f"{MODEL_LABELS[model]} & {config}"
            f" & {b('f1_macro')}"
            f" & {b('accuracy')}"
            f" & {b('cohen_kappa')}"
            f" & {b('precision_true')}"
            f" & {b('recall_true')}"
            f" & {b('precision_false')}"
            f" & {b('recall_false')}"
            r" \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table D – Full appendix table
# ---------------------------------------------------------------------------

def make_table_d(df):
    """All 112 configurations with all metrics."""
    sort_model = {m: i for i, m in enumerate(MODEL_ORDER)}
    sort_prompt = {p: i for i, p in enumerate(PROMPT_ORDER)}

    df_sorted = (df.assign(
                    _mo=df["model"].map(sort_model),
                    _po=df["prompt_mode"].map(sort_prompt),
                 )
                 .sort_values(["_mo", "_po", "demo_size",
                               "embedding_mode", "retrieval_mode"]))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full results for all 112 configurations. "
        r"P$_\mathsf{T}$/R$_\mathsf{T}$: precision/recall for the prosecutable class; "
        r"P$_\mathsf{F}$/R$_\mathsf{F}$: precision/recall for the non-prosecutable class.}",
        r"\label{tab:d-full}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lllrrrrrrrr}",
        r"\toprule",
        r"Model & Prompt & Retrieval"
        r" & F1 & Acc. & $\kappa$"
        r" & P$_\mathsf{T}$ & R$_\mathsf{T}$ & P$_\mathsf{F}$ & R$_\mathsf{F}$ \\",
        r"\midrule",
    ]

    prev_model = None
    for _, row in df_sorted.iterrows():
        model = row["model"]
        if model != prev_model and prev_model is not None:
            lines.append(r"\midrule")
        prev_model = model

        pm  = PROMPT_LABELS[row["prompt_mode"]]
        k   = int(row["demo_size"])
        ret = row["ret_label"]
        if k == 0:
            ret_cell = "ZS"
        else:
            ret_cell = f"k=8, {ret}"

        lines.append(
            f"{MODEL_LABELS[model]} & {pm} & {ret_cell}"
            f" & {fmt(row['f1_macro'])}"
            f" & {fmt(row['accuracy'])}"
            f" & {fmt(row['cohen_kappa'])}"
            f" & {fmt(row['precision_true'])}"
            f" & {fmt(row['recall_true'])}"
            f" & {fmt(row['precision_false'])}"
            f" & {fmt(row['recall_false'])}"
            r" \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load()

    save(make_a1_model_prompt(df),   "table_a1_model_prompt.tex")
    save(make_a2_model_shot(df),     "table_a2_model_shot.tex")
    save(make_a3_model_retrieval(df), "table_a3_model_retrieval.tex")
    save(make_table_b(df),           "table_b_best_fewshot.tex")
    save(make_table_e(df),           "table_e_best_per_model.tex")
    save(make_table_d(df),           "table_d_full.tex")


if __name__ == "__main__":
    main()
