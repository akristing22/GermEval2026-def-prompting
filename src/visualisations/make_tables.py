#!/usr/bin/env python3
"""Generate LaTeX result tables for the main experiment (final_run).

Reads results/final_run/score_table_consolidated.csv and writes the tables
to results/final_run/tables/. The ablation table additionally reads
results/ablations/balance_X_order/score_table_consolidated.csv and is
written to results/ablations/balance_X_order/tables/.

Tables produced
---------------
table_a1_model_prompt.tex   – model × prompt mode: ZS | best k=8 | worst k=8
table_a2_model_shot.tex     – model × zero-shot vs few-shot: mean ZS | mean k=8
table_a3_model_retrieval.tex– model × retrieval config (mean F1 over prompt modes)
table_b_best_fewshot.tex    – mean F1 per (model, prompt_mode) over all configs
table_d_full.tex            – full appendix table, all configurations
table_e_best_per_model.tex  – best config per model with full metrics
table_abstentions.tex       – abstention counts per model
table_ablation_balance_order.tex – model × (ratio × order), balance_X_order ablation
table_ablation_pr_marginal.tex – model × factor level, true-class P/R marginalised
                                 over the other factor (balance_X_order ablation)
table_ablation_pr_grid.tex   – model × ratio rows × order, true-class P/R for every
                                 ratio × order combination (balance_X_order ablation)
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_colors import MODEL_LABELS, MODEL_ORDER
from common import (
    ABLATION_SCORE_TABLE,
    ABLATION_TABLES_DIR,
    FINAL_RUN_TABLES_DIR as OUT_DIR,
    SCORE_TABLE_CONSOLIDATED as DATA_PATH,
)

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

# balance_X_order ablation factor levels and labels (match impact_ablations.py)
RATIO_ORDER = ["balanced", "proportional"]
ORDER_ORDER = ["random", "true-first", "true-last"]
RATIO_LABELS = {"balanced": "Balanced", "proportional": "Proportional"}
ORDER_LABELS = {"random": "Random", "true-first": "True first", "true-last": "True last"}

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


def save(latex: str, name: str, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
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
        r"\caption{$F1_{macro}$ by model and prompt mode. "
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
    """Mean ZS and mean k=8 per model (aggregated over all prompt modes)."""
    zs_mean = df[df["demo_size"] == 0].groupby("model")["f1_macro"].mean()
    fs_mean = df[df["demo_size"] == 8].groupby("model")["f1_macro"].mean()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean zero-shot vs.\ mean few-shot (k=8) $F1_{macro}$ per model "
        r"(mean over all prompt and retrieval settings).}"
        r"\label{tab:a2-model-shot}",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Model & Mean ZS & Mean FS\\",
        r"\midrule",
    ]

    for model in MODEL_ORDER:
        zs_v   = zs_mean[model]
        fs_v   = fs_mean[model]
        #delta  = fs_v - zs_v
        lines.append(
            f"{MODEL_LABELS[model]}"
            f" & {fmt(zs_v)}"
            f" & {fmt(fs_v)}"
            #f" & {fmt(delta)}"
            r" \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table A3 – Model × Retrieval Config
# ---------------------------------------------------------------------------

def make_a3_model_retrieval(df):
    """$F1_{macro}$ averaged over prompt modes for each (model, retrieval_config).

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
        r"\caption{$F1_{macro}$ by model and retrieval configuration (k=8, "
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
    """Mean $F1_{macro}$ over all configurations per (model, prompt_mode) cell.

    Row-best values (best prompt mode per model) are bold.
    """
    mean_f1 = df.groupby(["model", "prompt_mode"])["f1_macro"].mean().unstack("prompt_mode")

    row_bests = {
        model: mean_f1.loc[model].max()
        for model in MODEL_ORDER if model in mean_f1.index
    }

    col_spec = "l" + "r" * len(PROMPT_ORDER)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Mean $F1_{macro}$ per model and prompt mode (over all configurations). "
        r"\textbf{Bold}: best prompt mode per model.}",
        r"\label{tab:b-best-fewshot}",
        r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    header = "Model" + "".join([f" & {PROMPT_LABELS[pm]}" for pm in PROMPT_ORDER])
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        row_best = row_bests.get(model, float("nan"))
        row = MODEL_LABELS[model]
        for pm in PROMPT_ORDER:
            v = mean_f1.loc[model, pm] if (pm in mean_f1.columns and model in mean_f1.index) else float("nan")
            is_best = not pd.isna(v) and not pd.isna(row_best) and abs(v - row_best) < 1e-6
            row += f" & {fmt(v, bold=is_best)}"
        lines.append(row + r" \\")

    # Mean row
    col_means = {
        pm: mean_f1[pm].mean() if pm in mean_f1.columns else float("nan")
        for pm in PROMPT_ORDER
    }
    best_mean = max(v for v in col_means.values() if not pd.isna(v))
    mean_row = r"\textit{Mean}"
    for pm in PROMPT_ORDER:
        v = col_means[pm]
        is_best = not pd.isna(v) and abs(v - best_mean) < 1e-6
        mean_row += f" & {fmt(v, bold=is_best, italic=True)}"
    lines.append(r"\midrule")
    lines.append(mean_row + r" \\")

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
    metric_cols = ["f1_macro", "accuracy", "precision_true", "recall_true"]
    col_max = {c: best[c].max() for c in metric_cols}

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Best configuration per model (highest $F1_{macro}$ over all prompt modes "
        r"and retrieval settings). "
        r"P$_\mathsf{T}$/R$_\mathsf{T}$: precision/recall for the prosecutable class "
        r"(\emph{True}).}",
        r"\label{tab:e-best-per-model}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Config & F1 & Acc. & P$_\mathsf{T}$ & R$_\mathsf{T}$ \\",
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
            f" & {b('precision_true')}"
            f" & {b('recall_true')}"
            r" \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table – Abstentions per model
# ---------------------------------------------------------------------------

def make_table_abstentions(df):
    """Total, mean, and median abstentions per model (across all configurations)."""
    stats = (df.groupby("model")["n_abstained"]
               .agg(total="sum", mean="mean", median="median"))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Abstentions per model across all configurations "
        r"(total, mean per config, and median per config).}",
        r"\label{tab:abstentions}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Total & Mean & Median \\",
        r"\midrule",
    ]

    for model in MODEL_ORDER:
        row_stats = stats.loc[model]
        lines.append(
            f"{MODEL_LABELS[model]}"
            f" & {int(row_stats['total'])}"
            f" & {row_stats['mean']:.1f}"
            f" & {row_stats['median']:.1f}"
            r" \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table – balance_X_order ablation: model × (ratio × order)
# ---------------------------------------------------------------------------

def make_table_ablation(df):
    """$F1_{macro}$ for every (model, ratio, order) cell of the ablation.

    Models in rows; ratio × order as grouped columns.
    Best value per row is bold; worst is italic.
    """
    f1 = df.set_index(["model", "ratio", "order"])["f1_macro"]

    n_ord = len(ORDER_ORDER)
    col_spec = "l" + "r" * n_ord * len(RATIO_ORDER)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{$F1_{macro}$ by demonstration class ratio and ordering "
        r"(balance $\times$ order ablation). "
        r"\textbf{Bold}: best combination per model; "
        r"\textit{italic}: worst.}",
        r"\label{tab:ablation-balance-order}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Header row 1: ratio group spans
    h1 = "Model"
    for ratio in RATIO_ORDER:
        h1 += r" & \multicolumn{" + str(n_ord) + r"}{c}{" + RATIO_LABELS[ratio] + "}"
    lines.append(h1 + r" \\")

    # Cmidrules under each group
    cmidrules = []
    for i in range(len(RATIO_ORDER)):
        start = 2 + i * n_ord
        cmidrules.append(r"\cmidrule(lr){" + f"{start}-{start + n_ord - 1}" + "}")
    lines.append(" ".join(cmidrules))

    # Header row 2: order sub-columns
    h2 = ""
    for _ in RATIO_ORDER:
        h2 += "".join([f" & {ORDER_LABELS[o]}" for o in ORDER_ORDER])
    lines.append(h2 + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        vals = [f1.get((model, ratio, order), float("nan"))
                for ratio in RATIO_ORDER for order in ORDER_ORDER]
        finite = [v for v in vals if not pd.isna(v)]
        best_v = max(finite) if finite else None
        wrst_v = min(finite) if finite else None
        row = MODEL_LABELS[model]
        for v in vals:
            is_best  = best_v is not None and not pd.isna(v) and abs(v - best_v) < 1e-6
            is_worst = wrst_v is not None and not pd.isna(v) and abs(v - wrst_v) < 1e-6
            row += f" & {fmt(v, bold=is_best, italic=(is_worst and not is_best))}"
        lines.append(row + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table – balance_X_order ablation: true-class P/R marginalised per factor
# ---------------------------------------------------------------------------

def make_table_ablation_pr_marginal(df):
    """True-class precision/recall marginalised over each ablation factor.

    Ratio levels average over the order levels; order levels average over the
    ratio levels (mean of the per-cell precision_true/recall_true, matching the
    marginal means in impact_ablations.py). Within each factor block the best
    P_T and best R_T per model are bold.
    """
    ratio_p = df.groupby(["model", "ratio"])["precision_true"].mean()
    ratio_r = df.groupby(["model", "ratio"])["recall_true"].mean()
    order_p = df.groupby(["model", "order"])["precision_true"].mean()
    order_r = df.groupby(["model", "order"])["recall_true"].mean()

    n_r, n_o = len(RATIO_ORDER), len(ORDER_ORDER)
    col_spec = "l" + "rr" * (n_r + n_o)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{True-class precision (P$_\mathsf{T}$) and recall (R$_\mathsf{T}$) "
        r"marginalised over each factor of the balance $\times$ order ablation "
        r"(ratio levels averaged over order; order levels averaged over ratio). "
        r"\textbf{Bold}: best value per model within each factor block.}",
        r"\label{tab:ablation-pr-marginal}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Header row 1: Ratio / Order super-groups
    h1 = ("Model"
          + r" & \multicolumn{" + str(2 * n_r) + r"}{c}{Ratio}"
          + r" & \multicolumn{" + str(2 * n_o) + r"}{c}{Order}")
    lines.append(h1 + r" \\")
    end_ratio = 1 + 2 * n_r
    lines.append(r"\cmidrule(lr){2-" + str(end_ratio) + "}"
                 + r" \cmidrule(lr){" + str(end_ratio + 1) + "-" + str(end_ratio + 2 * n_o) + "}")

    # Header row 2: factor-level names, each spanning P/R
    levels = [("ratio", lvl) for lvl in RATIO_ORDER] + [("order", lvl) for lvl in ORDER_ORDER]
    h2 = "Model"
    for fac, lvl in levels:
        lab = RATIO_LABELS[lvl] if fac == "ratio" else ORDER_LABELS[lvl]
        h2 += r" & \multicolumn{2}{c}{" + lab + "}"
    lines.append(h2 + r" \\")
    cmid = [r"\cmidrule(lr){" + f"{2 + 2 * i}-{3 + 2 * i}" + "}" for i in range(n_r + n_o)]
    lines.append(" ".join(cmid))

    # Header row 3: P/R sub-columns
    h3 = "".join([r" & P$_\mathsf{T}$ & R$_\mathsf{T}$" for _ in levels])
    lines.append(h3 + r" \\")
    lines.append(r"\midrule")

    def _best(d):
        vals = [v for v in d.values() if not pd.isna(v)]
        return max(vals) if vals else None

    for model in MODEL_ORDER:
        rp = {lvl: ratio_p.get((model, lvl), float("nan")) for lvl in RATIO_ORDER}
        rr = {lvl: ratio_r.get((model, lvl), float("nan")) for lvl in RATIO_ORDER}
        op = {lvl: order_p.get((model, lvl), float("nan")) for lvl in ORDER_ORDER}
        orr = {lvl: order_r.get((model, lvl), float("nan")) for lvl in ORDER_ORDER}
        best = {"rp": _best(rp), "rr": _best(rr), "op": _best(op), "orr": _best(orr)}

        def cell(v, target):
            is_best = target is not None and not pd.isna(v) and abs(v - target) < 1e-6
            return fmt(v, bold=is_best)

        row = MODEL_LABELS[model]
        for lvl in RATIO_ORDER:
            row += f" & {cell(rp[lvl], best['rp'])} & {cell(rr[lvl], best['rr'])}"
        for lvl in ORDER_ORDER:
            row += f" & {cell(op[lvl], best['op'])} & {cell(orr[lvl], best['orr'])}"
        lines.append(row + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table – balance_X_order ablation: true-class P/R for every ratio × order cell
# ---------------------------------------------------------------------------

def make_table_ablation_pr_grid(df):
    """True-class precision/recall for every (model, ratio, order) combination.

    Models group rows (ratio as a sub-row); the three order levels are column
    groups, each split into P_T and R_T. The best P_T and best R_T per model
    (across all six of its combinations) are bold.
    """
    p = df.set_index(["model", "ratio", "order"])["precision_true"]
    r = df.set_index(["model", "ratio", "order"])["recall_true"]

    n_o = len(ORDER_ORDER)
    col_spec = "ll" + "rr" * n_o
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{True-class precision (P$_\mathsf{T}$) and recall (R$_\mathsf{T}$) "
        r"for every demonstration ratio $\times$ order combination "
        r"(balance $\times$ order ablation). "
        r"\textbf{Bold}: best value per model across all combinations.}",
        r"\label{tab:ablation-pr-grid}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Header row 1: order super-groups
    h1 = "Model & Ratio"
    for order in ORDER_ORDER:
        h1 += r" & \multicolumn{2}{c}{" + ORDER_LABELS[order] + "}"
    lines.append(h1 + r" \\")
    cmid = [r"\cmidrule(lr){" + f"{3 + 2 * i}-{4 + 2 * i}" + "}" for i in range(n_o)]
    lines.append(" ".join(cmid))

    # Header row 2: P/R sub-columns
    h2 = " & " + "".join([r" & P$_\mathsf{T}$ & R$_\mathsf{T}$" for _ in ORDER_ORDER])
    lines.append(h2 + r" \\")
    lines.append(r"\midrule")

    for m_idx, model in enumerate(MODEL_ORDER):
        all_p = [p.get((model, ratio, order), float("nan"))
                 for ratio in RATIO_ORDER for order in ORDER_ORDER]
        all_r = [r.get((model, ratio, order), float("nan"))
                 for ratio in RATIO_ORDER for order in ORDER_ORDER]
        best_p = max([v for v in all_p if not pd.isna(v)], default=None)
        best_r = max([v for v in all_r if not pd.isna(v)], default=None)

        for r_idx, ratio in enumerate(RATIO_ORDER):
            label = MODEL_LABELS[model] if r_idx == 0 else ""
            row = f"{label} & {RATIO_LABELS[ratio]}"
            for order in ORDER_ORDER:
                pv = p.get((model, ratio, order), float("nan"))
                rv = r.get((model, ratio, order), float("nan"))
                bp = best_p is not None and not pd.isna(pv) and abs(pv - best_p) < 1e-6
                br = best_r is not None and not pd.isna(rv) and abs(rv - best_r) < 1e-6
                row += f" & {fmt(pv, bold=bp)} & {fmt(rv, bold=br)}"
            lines.append(row + r" \\")
        if m_idx != len(MODEL_ORDER) - 1:
            lines.append(r"\addlinespace")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table D – Full appendix table
# ---------------------------------------------------------------------------

def make_table_d(df):
    """All 112 configurations with all metrics, sorted by F1 Macro descending."""
    df_sorted = df.sort_values("f1_macro", ascending=False)

    lines = [
        r"\begin{longtable}{lllrrrrr}",
        r"\caption{Full results for all 112 configurations, sorted by $F1_{macro}$ (descending). "
        r"P$_\mathsf{T}$/R$_\mathsf{T}$: precision/recall for the prosecutable class.}"
        r"\label{tab:d-full}\\",
        r"\toprule",
        r"Model & Prompt & Retrieval"
        r" & F1 & Acc."
        r" & P$_\mathsf{T}$ & R$_\mathsf{T}$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{l}{\small\textit{(continued)}} \\",
        r"\toprule",
        r"Model & Prompt & Retrieval"
        r" & F1 & Acc."
        r" & P$_\mathsf{T}$ & R$_\mathsf{T}$ \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{7}{r}{\small\textit{(continued on next page)}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for _, row in df_sorted.iterrows():
        model = row["model"]
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
            f" & {fmt(row['precision_true'])}"
            f" & {fmt(row['recall_true'])}"
            r" \\"
        )

    lines.append(r"\end{longtable}")
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
    save(make_table_abstentions(df), "table_abstentions.tex")

    df_abl = pd.read_csv(ABLATION_SCORE_TABLE)
    save(make_table_ablation(df_abl), "table_ablation_balance_order.tex",
         out_dir=ABLATION_TABLES_DIR)
    save(make_table_ablation_pr_marginal(df_abl), "table_ablation_pr_marginal.tex",
         out_dir=ABLATION_TABLES_DIR)
    save(make_table_ablation_pr_grid(df_abl), "table_ablation_pr_grid.tex",
         out_dir=ABLATION_TABLES_DIR)


if __name__ == "__main__":
    main()
