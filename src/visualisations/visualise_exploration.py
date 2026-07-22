"""Visualise exploration results: F1 macro by demonstration size (4, 8, 16, 32).

Reads results/exploration/new/score_table_consolidated.csv and writes
  - f1_by_demo_size.{pdf,svg} to results/exploration/new/figures/
  - table_exploration_demo_size.tex to results/exploration/new/tables/
plus a plain-text version of the table on the console.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_colors import get_model_colors, MODEL_LABELS
from common import (
    EXPLORATION_FIGURES_DIR as FIGURES_DIR,
    EXPLORATION_SCORE_TABLE,
    EXPLORATION_TABLES_DIR as TABLES_DIR,
    load_exploration_scores,
    save_figure,
)

DEMO_SIZES = [4, 8, 16, 32]

# Colour + marker per model — consistent across subplots
MODEL_STYLE = {
    model: {"color": color, "marker": "o"}
    for model, color in get_model_colors(MODEL_LABELS.keys()).items()
}


# ---------------------------------------------------------------------------
# Shared table/plot inputs
# ---------------------------------------------------------------------------

def mean_f1_per_model(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Mean F1 Macro per demonstration size for each model present in df,
    indexed by DEMO_SIZES (NaN where a size was not run)."""
    rows = {}
    for model in MODEL_LABELS:
        model_data = df[df["model"] == model]
        if model_data.empty:
            continue
        rows[model] = (
            model_data.groupby("demo_size")["f1_macro"]
            .mean()
            .reindex(DEMO_SIZES)
        )
    return rows


def value_delta_cells(means: pd.Series, fmt_value, fmt_delta, missing: str) -> list[str]:
    """Format one table row: each cell shows the value and, from the second
    size onward, the delta to the previous demonstration size in parentheses."""
    cells, prev = [], None
    for k in DEMO_SIZES:
        val = means[k]
        if pd.isna(val):
            cells.append(missing)
        elif prev is None or pd.isna(prev):
            cells.append(fmt_value(val))
        else:
            cells.append(f"{fmt_value(val)} ({fmt_delta(val - prev)})")
        prev = val
    return cells


def save_tex(latex: str, name: str) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, name)
    with open(path, "w") as f:
        f.write(latex)
    print(f"Written: {path}")


# ---------------------------------------------------------------------------
# Figure: F1 Macro by demonstration size
# ---------------------------------------------------------------------------

def plot_f1_by_demo_size(df: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)

    per_model_means = {
        model: means.dropna() for model, means in mean_f1_per_model(df).items()
        if not means.dropna().empty
    }

    for model, means in per_model_means.items():
        style = MODEL_STYLE[model]
        ax.plot(
            means.index,
            means.values,
            color=style["color"],
            marker=style["marker"],
            markersize=5,
            linewidth=1.6,
        )
        # Direct line labels instead of a legend; nudge to avoid overlap
        y_nudge = 0.015 if model == "gemma-4-26B-A4B-it" else -0.015
        ax.text(
            means.index[-1] + 0.5,
            means.values[-1] + y_nudge,
            MODEL_LABELS[model],
            color=style["color"],
            fontsize=10,
            va="center",
        )

    if per_model_means:
        # Dashed black line: mean across all models
        mean_across_models = (
            pd.DataFrame(per_model_means).mean(axis=1).reindex(DEMO_SIZES).dropna()
        )
        ax.plot(
            mean_across_models.index,
            mean_across_models.values,
            color="black",
            marker="o",
            markersize=5,
            linewidth=1.6,
            linestyle="--",
        )

        print("\nF1 Macro differences between consecutive demonstration sizes:")
        sizes = mean_across_models.index.tolist()
        for a, b in zip(sizes, sizes[1:]):
            diff = mean_across_models[b] - mean_across_models[a]
            print(f"  {a} → {b}: {diff:+.4f} (mean across models)")
        print()
        for model, means in per_model_means.items():
            label = MODEL_LABELS.get(model, model)
            parts = []
            for a, b in zip(sizes, sizes[1:]):
                if a in means.index and b in means.index:
                    parts.append(f"{a}→{b}: {means[b] - means[a]:+.4f}")
            if parts:
                print(f"  {label}: {', '.join(parts)}")

    ax.set_xlabel("Demonstration size (k)", fontsize=12)
    ax.set_ylabel("F1 Macro", fontsize=12)
    ax.set_xticks(DEMO_SIZES)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(labelsize=11)
    ax.set_xlim(2, 44)  # extra right margin for the direct line labels
    ax.set_ylim(0.30, 0.80)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    save_figure(fig, FIGURES_DIR, name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables: console and LaTeX versions of the same numbers
# ---------------------------------------------------------------------------

def print_f1_table(df: pd.DataFrame) -> None:
    col_w = 18
    header = f"{'Model':<22}" + "".join(f"k={k}".center(col_w) for k in DEMO_SIZES)
    print("\n" + header)
    print("-" * len(header))

    fmt_value = lambda v: f"{v:.3f}"
    fmt_delta = lambda d: f"{d:+.3f}"

    rows = mean_f1_per_model(df)
    for model, means in rows.items():
        label = MODEL_LABELS.get(model, model)
        cells = value_delta_cells(means, fmt_value, fmt_delta, "—")
        print(f"{label:<22}" + "".join(c.center(col_w) for c in cells))

    if rows:
        mean_across = pd.DataFrame(rows).mean(axis=1).reindex(DEMO_SIZES)
        cells = value_delta_cells(mean_across, fmt_value, fmt_delta, "—")
        print("-" * len(header))
        print(f"{'Mean':<22}" + "".join(c.center(col_w) for c in cells))
    print()


def make_tex_table(df: pd.DataFrame) -> str:
    def fmt(v, bold=False):
        s = f"{v:.3f}"
        return r"\textbf{" + s + r"}" if bold else s

    fmt_delta = lambda d: f"{d:+.3f}"

    rows = mean_f1_per_model(df)

    n_cols = len(DEMO_SIZES)
    col_spec = "l" + "r" * n_cols
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean F1 Macro per model and demonstration size. "
        r"Values in parentheses show the gain/loss relative to the previous $k$.}",
        r"\label{tab:exploration-demo-size}",
        r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Model" + "".join(f" & $k$={k}" for k in DEMO_SIZES) + r" \\",
        r"\midrule",
    ]

    for model, means in rows.items():
        label = MODEL_LABELS.get(model, model)
        cells = value_delta_cells(means, fmt, fmt_delta, "---")
        lines.append(label + "".join(f" & {c}" for c in cells) + r" \\")

    if rows:
        mean_across = pd.DataFrame(rows).mean(axis=1).reindex(DEMO_SIZES)
        cells = value_delta_cells(mean_across, lambda v: fmt(v, bold=True), fmt_delta, "---")
        lines.append(r"\midrule")
        lines.append(r"Mean" + "".join(f" & {c}" for c in cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Loading scores from: {EXPLORATION_SCORE_TABLE}")
    df = load_exploration_scores()
    print(f"Loaded {len(df)} runs across {df['model'].nunique()} models")
    plot_f1_by_demo_size(df, "f1_by_demo_size")
    print_f1_table(df)
    save_tex(make_tex_table(df), "table_exploration_demo_size.tex")
