"""Shared model color scheme for all visualisations.

Single source of truth for the model → color mapping defined in CLAUDE.md
("Model Color Scheme"). Colors are taken from the Okabe–Ito colorblind-safe
palette so figures remain readable under protanopia, deuteranopia, and
tritanopia.
"""

# Okabe–Ito hues, one fixed color per model
MODEL_COLORS = {
    "gemma-4-26B-A4B-it":        "#0072B2",  # blue
    "gemma-4-E4B-it":            "#E69F00",  # orange
    "Qwen3.5-9B":                "#009E73",  # bluish green
    "EuroLLM-22B-Instruct-2512": "#CC79A7",  # reddish purple
}

MODEL_LABELS = {
    "gemma-4-26B-A4B-it":        "Gemma-4 26B",
    "gemma-4-E4B-it":            "Gemma-4 E4B",
    "Qwen3.5-9B":                "Qwen3.5 9B",
    "EuroLLM-22B-Instruct-2512": "EuroLLM 22B",
}

FALLBACK_COLOR = "#999999"  # gray, for models not in MODEL_COLORS


def get_model_colors(models) -> dict[str, str]:
    """Color mapping for the given models, falling back to gray for unknown ones."""
    return {m: MODEL_COLORS.get(m, FALLBACK_COLOR) for m in models}


_LABEL_COLORS = {label: MODEL_COLORS[key] for key, label in MODEL_LABELS.items() if key in MODEL_COLORS}


def get_label_colors(labels) -> dict[str, str]:
    """Color mapping for display labels (from MODEL_LABELS), falling back to gray."""
    return {lb: _LABEL_COLORS.get(lb, FALLBACK_COLOR) for lb in labels}
