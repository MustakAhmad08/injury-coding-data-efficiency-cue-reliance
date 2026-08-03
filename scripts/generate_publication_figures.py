"""Generate publication figures directly from the saved evaluation outputs.

Exports vector SVG/PDF and 600-DPI PNG versions at manuscript-ready sizes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
SUMMARY = ROOT / "results" / "primary" / "model_summary.csv"
BOOTSTRAP = ROOT / "results" / "primary" / "bootstrap_95ci.csv"
MASKING = ROOT / "results" / "cue_masking" / "cue_reliance_summary.csv"
CONTRASTS = ROOT / "results" / "contrasts" / "contrast_metrics.csv"

# Okabe-Ito colors: distinguishable under common forms of color-vision deficiency.
COLORS = {
    "Qwen": "#0072B2",
    "Mistral": "#D55E00",
    "GPT-5.4": "#CC79A7",
    "Hosted Llama": "#009E73",
    "TF-IDF": "#4D4D4D",
}

LABELS = {
    "mistral_0shot": "Mistral zero-shot",
    "mistral_10shot": "Mistral 10-shot",
    "qwen_0shot": "Qwen zero-shot",
    "qwen_10shot": "Qwen 10-shot",
    "gpt54_0shot": "GPT-5.4 zero-shot",
    "gpt54_10shot": "GPT-5.4 10-shot",
    "hosted_llama_0shot": "Hosted Llama zero-shot",
    "hosted_llama_10shot": "Hosted Llama 10-shot",
    "mistral_sft_1000": "Mistral SFT 1K",
    "mistral_sft_3000": "Mistral SFT 3K",
    "mistral_sft_6000": "Mistral SFT 6K",
    "qwen_sft_1000": "Qwen SFT 1K",
    "qwen_sft_3000": "Qwen SFT 3K",
    "qwen_sft_6000": "Qwen SFT 6K",
    "tfidf_logreg_6000": "TF-IDF + logistic regression",
}


def family(condition: str) -> str:
    if condition.startswith("qwen"):
        return "Qwen"
    if condition.startswith("mistral"):
        return "Mistral"
    if condition.startswith("gpt54"):
        return "GPT-5.4"
    if condition.startswith("hosted_llama"):
        return "Hosted Llama"
    return "TF-IDF"


def public_condition(condition: str) -> str:
    """Replace an institution-specific hosted-Llama prefix with a public label."""
    return re.sub(r"^[^_]+_llama_", "hosted_llama_", condition)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def export(fig: mpl.figure.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.035, **kwargs)
    plt.close(fig)


def _experimental_design_column_legacy() -> None:
    """Retained one-column layout for comparison; not used for publication."""
    fig, ax = plt.subplots(figsize=(3.35, 4.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#202A35"
    blue = "#356E95"
    amber = "#9B6A12"
    teal = "#34786E"
    gray = "#66727E"
    pale = "#F4F6F8"
    pale_blue = "#EDF4F8"
    pale_amber = "#FBF5E8"
    pale_teal = "#EDF6F3"

    def panel_label(letter, title, y):
        ax.text(0.025, y, letter, ha="left", va="center", fontsize=7.2,
                fontweight="bold", color=ink)
        ax.text(0.065, y, title, ha="left", va="center", fontsize=7.2,
                fontweight="bold", color=ink)

    def box(x, y, w, h, title, body="", fc="white", ec=gray, lw=0.75,
            title_size=7.0, body_size=6.1, title_color=ink,
            title_pos=0.66, body_pos=0.28):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                               linewidth=lw, joinstyle="miter"))
        ax.text(x + w / 2, y + h * (title_pos if body else 0.50), title,
                ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=title_color, linespacing=1.0)
        if body:
            ax.text(x + w / 2, y + h * body_pos, body, ha="center", va="center",
                    fontsize=body_size, color="#39434D", linespacing=1.05)

    def arrow(x1, y1, x2, y2, color=ink, dashed=False, lw=0.75):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
            linewidth=lw, color=color,
            linestyle=(0, (3, 2)) if dashed else "solid",
            shrinkA=1.0, shrinkB=1.0))

    # (a) Data partitions and their distinct roles.
    panel_label("a", "Temporal partitions (class-balanced; 30 diagnoses)", 0.974)
    split_y, split_h = 0.858, 0.082
    box(0.035, split_y, 0.285, split_h, "TRAIN",
        "2021–2023  |  n=6,000", pale_teal, teal)
    box(0.3575, split_y, 0.285, split_h, "DEVELOPMENT",
        "2024  |  n=300", pale_amber, amber, title_size=6.7)
    box(0.68, split_y, 0.285, split_h, "LOCKED TEST",
        "2025  |  n=1,500", pale_blue, blue)

    # (b) The three compared modeling strategies.
    panel_label("b", "Model development", 0.786)
    model_y, model_h = 0.635, 0.115
    box(0.035, model_y, 0.285, model_h, "LEXICAL",
        "TF–IDF + LR\n6K", pale, gray, body_size=6.0)
    box(0.3575, model_y, 0.285, model_h, "PROMPTING",
        "zero-shot / 10-shot\nfour models", pale, gray, body_size=5.9)
    box(0.68, model_y, 0.285, model_h, "QLoRA SFT",
        "Mistral / Qwen\n1K / 3K / 6K", pale, gray, body_size=5.9)
    arrow(0.50, 0.857, 0.50, 0.753, amber)
    # Training also supplies the fixed demonstrations and SFT subsets.
    ax.plot([0.1775, 0.335, 0.335, 0.1775, 0.8225],
            [0.857, 0.825, 0.770, 0.770, 0.770],
            color=teal, linewidth=0.65)
    arrow(0.1775, 0.770, 0.1775, 0.753, teal, lw=0.65)
    arrow(0.8225, 0.770, 0.8225, 0.753, teal, lw=0.65)

    box(0.19, 0.548, 0.62, 0.055, "CONFIGURATION FROZEN",
        "checkpoint selection: 2024 development only", "white", amber,
        title_size=6.3, body_size=5.45, title_color=ink)
    arrow(0.50, model_y, 0.50, 0.606, ink)

    # Explicit barrier between development and the one-time locked evaluation.
    ax.plot([0.035, 0.965], [0.515, 0.515], color=blue, linewidth=0.8,
            linestyle=(0, (4, 2)))
    ax.text(0.50, 0.522, "LOCKED TEST — no model selection",
            ha="center", va="bottom", fontsize=6.1, fontweight="bold", color=blue)

    # (c) Primary, prespecified outcome and inference.
    panel_label("c", "Primary locked-test analysis", 0.477)
    box(0.10, 0.365, 0.80, 0.083, "OUTCOME",
        "exact primary-diagnosis-code prediction (30 classes)", pale_blue, blue,
        title_size=6.7, body_size=6.0)
    arrow(0.50, 0.546, 0.50, 0.451, ink)
    # Locked test is routed directly to the primary analysis only after freezing.
    ax.plot([0.965, 0.982, 0.982], [0.899, 0.899, 0.407],
            color=blue, linewidth=0.75)
    arrow(0.982, 0.407, 0.904, 0.407, blue)

    box(0.10, 0.254, 0.80, 0.077, "METRICS AND TESTS",
        "macro F1 + bootstrap CI  |  category metrics\n"
        "exact McNemar tests + Holm correction",
        "white", blue, title_size=6.1, body_size=5.75,
        title_pos=0.76, body_pos=0.32)
    arrow(0.50, 0.363, 0.50, 0.334, blue)

    # (d) Clearly labeled diagnostic analyses; these are not confirmatory tests.
    panel_label("d", "Diagnostic analyses", 0.213)
    ax.add_patch(Rectangle((0.10, 0.087), 0.80, 0.095, facecolor="#FAFAFA",
                           edgecolor=gray, linewidth=0.75, linestyle=(0, (3, 2))))
    ax.text(0.50, 0.148, "DEVELOPMENT: prompt-component ablation",
            ha="center", va="center", fontsize=5.8, fontweight="bold", color=amber)
    ax.text(0.50, 0.115,
            "POST HOC: cue masking  |  natural contrasts  |  secondary diagnosis",
            ha="center", va="center", fontsize=5.3, color="#39434D")
    arrow(0.50, 0.252, 0.50, 0.184, gray, dashed=True)

    ax.plot([0.105, 0.165], [0.052, 0.052], color=ink, linewidth=0.8)
    ax.text(0.175, 0.052, "prespecified flow", ha="left", va="center",
            fontsize=5.25, color=gray)
    ax.plot([0.515, 0.575], [0.052, 0.052], color=gray, linewidth=0.8,
            linestyle=(0, (3, 2)))
    ax.text(0.585, 0.052, "diagnostic flow", ha="left", va="center",
            fontsize=5.25, color=gray)
    export(fig, "experimental_design_schematic")


def experimental_design() -> None:
    """Draw the actual study design as a full-width scientific schematic."""
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#24313C"
    blue = "#3274A1"
    amber = "#B17612"
    teal = "#2A7F6E"
    gray = "#68737D"
    pale = "#F5F6F7"
    pale_blue = "#EDF4F8"
    pale_amber = "#FBF4E5"
    pale_teal = "#EDF6F3"

    def box(x, y, w, h, title, body="", fc="white", ec=gray, lw=0.8,
            title_size=8.1, body_size=6.8, dashed=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            linestyle=(0, (4, 2.5)) if dashed else "solid"))
        ax.text(x + w / 2, y + h * (0.68 if body else 0.50), title,
                ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=ink, linespacing=1.0)
        if body:
            ax.text(x + w / 2, y + h * 0.29, body, ha="center", va="center",
                    fontsize=body_size, color="#3F4952", linespacing=1.08)

    def arrow(x1, y1, x2, y2, color=ink, dashed=False, lw=0.85):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
            linewidth=lw, color=color,
            linestyle=(0, (4, 2.5)) if dashed else "solid",
            shrinkA=1.5, shrinkB=1.5))

    # (a) Temporally ordered partitions and their distinct roles.
    ax.text(0.02, 0.96, "a", fontsize=9.0, fontweight="bold", color=ink, va="center")
    ax.text(0.045, 0.96,
            "Temporal partitions (30 diagnosis classes; balanced within split)",
            fontsize=8.6, fontweight="bold", color=ink, va="center")
    split_y, split_h = 0.79, 0.115
    box(0.035, split_y, 0.255, split_h, "TRAIN",
        "2021-2023  |  200/class  |  n=6,000", pale_teal, teal)
    box(0.3725, split_y, 0.255, split_h, "DEVELOPMENT",
        "2024  |  10/class  |  n=300", pale_amber, amber)
    box(0.71, split_y, 0.255, split_h, "LOCKED TEST",
        "2025  |  50/class  |  n=1,500", pale_blue, blue)
    ax.text(0.035, 0.755, "past", ha="left", va="center", fontsize=6.1, color=gray)
    ax.text(0.965, 0.755, "future", ha="right", va="center", fontsize=6.1, color=gray)
    ax.plot([0.07, 0.93], [0.755, 0.755], color="#B7C0C8", lw=0.65)
    arrow(0.93, 0.755, 0.96, 0.755, color=gray, lw=0.65)

    # (b) The three model-development regimes actually compared.
    ax.text(0.02, 0.705, "b", fontsize=9.0, fontweight="bold", color=ink, va="center")
    ax.text(0.045, 0.705, "Model development and selection", fontsize=8.6,
            fontweight="bold", color=ink, va="center")
    model_y, model_h = 0.47, 0.15
    box(0.035, model_y, 0.255, model_h, "TF-IDF + LOGISTIC REGRESSION",
        "6K training cases\ndev grid: n-grams and C", pale, gray,
        title_size=6.0, body_size=5.8)
    box(0.3725, model_y, 0.255, model_h, "ZERO-/10-SHOT PROMPTING",
        "four model families\nfixed codebook, rules, and 10 demonstrations",
        pale, gray, title_size=6.0, body_size=5.7)
    box(0.71, model_y, 0.255, model_h, "QLoRA SUPERVISED ADAPTATION",
        "Mistral-7B and Qwen2.5-7B\nnested 1K / 3K / 6K training sets",
        pale, gray, title_size=5.8, body_size=5.7)

    # Training supplies supervised examples/demonstrations; development selects settings.
    ax.plot([0.1625, 0.8375], [0.645, 0.645], color=teal, lw=0.75)
    arrow(0.1625, split_y, 0.1625, 0.647, teal, lw=0.75)
    for center in (0.1625, 0.50, 0.8375):
        arrow(center, 0.645, center, model_y + model_h, teal, lw=0.75)
    ax.text(0.30, 0.651, "training cases / demonstrations (as applicable)", ha="center",
            va="bottom", fontsize=5.6, color=teal)

    # Development selection is routed through the gap between model families.
    ax.plot([0.50, 0.655, 0.655], [split_y, 0.655, 0.425],
            color=amber, lw=0.85)
    arrow(0.655, 0.425, 0.64, 0.402, amber, lw=0.85)

    box(0.255, 0.34, 0.49, 0.06,
        "CONFIGURATIONS FROZEN AFTER DEVELOPMENT SELECTION",
        fc="white", ec=amber, title_size=6.1)
    ax.plot([0.1625, 0.8375], [0.43, 0.43], color=ink, lw=0.7)
    for center in (0.1625, 0.50, 0.8375):
        arrow(center, model_y, center, 0.432, ink, lw=0.7)
    arrow(0.50, 0.43, 0.50, 0.403, ink, lw=0.75)

    # Explicit boundary prevents the incorrect impression of test-set tuning.
    ax.plot([0.025, 0.975], [0.305, 0.305], color=blue, lw=1.0,
            linestyle=(0, (5, 2.5)))
    ax.text(0.50, 0.305,
            "LOCKED TEST: one-time evaluation; no model selection",
            ha="center", va="center", fontsize=5.7, fontweight="bold", color=blue,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

    # (c) Primary endpoint and inferential summary.
    ax.text(0.02, 0.255, "c", fontsize=9.0, fontweight="bold", color=ink, va="center")
    ax.text(0.045, 0.255, "Prespecified primary analysis", fontsize=8.2,
            fontweight="bold", color=ink, va="center")
    box(0.035, 0.105, 0.39, 0.105, "PRIMARY OUTCOME",
        "exact Diagnosis_1 code (1 of 30)", pale_blue, blue,
        title_size=6.4, body_size=5.9)
    box(0.455, 0.105, 0.51, 0.105, "LOCKED-TEST SUMMARY",
        "macro F1 + 95% stratified-bootstrap CI  |  category metrics\n"
        "exact paired McNemar tests + Holm correction",
        pale_blue, blue, title_size=6.4, body_size=5.7)
    arrow(0.40, 0.339, 0.40, 0.213, blue)
    arrow(0.425, 0.1575, 0.452, 0.1575, blue)
    ax.plot([0.965, 0.978, 0.978], [split_y + split_h / 2, split_y + split_h / 2, 0.24],
            color=blue, lw=0.8)
    arrow(0.978, 0.24, 0.94, 0.213, blue, lw=0.8)

    # Diagnostic work is visually and inferentially separate from the primary analysis.
    box(0.035, 0.02, 0.93, 0.05,
        "DIAGNOSTIC ANALYSES (dashed):  2024 prompt ablation  |  post-hoc cue masking  |  natural contrasts  |  secondary diagnosis",
        fc="#FAFAFA", ec=gray, title_size=5.5, dashed=True)
    arrow(0.71, 0.105, 0.71, 0.071, gray, dashed=True, lw=0.75)
    export(fig, "experimental_design_schematic")


def model_forest() -> None:
    summary = pd.read_csv(SUMMARY)[["condition", "macro_f1"]]
    ci = pd.read_csv(BOOTSTRAP)
    ci = ci.loc[ci["metric"].eq("macro_f1"), ["condition", "ci_low", "ci_high"]]
    summary["condition"] = summary["condition"].map(public_condition)
    ci["condition"] = ci["condition"].map(public_condition)
    data = summary.merge(ci, on="condition", validate="one_to_one").sort_values("macro_f1")
    data["label"] = data["condition"].map(LABELS)

    fig, ax = plt.subplots(figsize=(3.35, 5.15), constrained_layout=True)
    y = np.arange(len(data))
    for i, row in data.reset_index(drop=True).iterrows():
        fam = family(row.condition)
        marker = "D" if "sft" in row.condition else ("s" if row.condition.startswith("tfidf") else "o")
        ax.errorbar(
            row.macro_f1,
            i,
            xerr=[[row.macro_f1 - row.ci_low], [row.ci_high - row.macro_f1]],
            fmt=marker,
            color=COLORS[fam],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.6,
            elinewidth=1.05,
            capsize=1.8,
            zorder=3,
        )
    ax.set_yticks(y, data["label"])
    ax.set_xlabel("Macro F1 (95% bootstrap CI)")
    ax.set_xlim(0.14, 0.94)
    ax.set_xticks(np.arange(0.2, 1.0, 0.1))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for tick, cond in zip(ax.get_yticklabels(), data["condition"]):
        tick.set_color(COLORS[family(cond)])
    fig.align_ylabels()
    export(fig, "model_comparison_forest")


def masking_effects() -> None:
    data = pd.read_csv(MASKING)
    order = ["qwen_sft_6000", "mistral_sft_3000", "tfidf_logreg_6000"]
    short = {"qwen_sft_6000": "Qwen SFT 6K", "mistral_sft_3000": "Mistral SFT 3K", "tfidf_logreg_6000": "TF-IDF + LR"}
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.6), constrained_layout=True)
    specs = [
        ("marker_only", "(a) Formatting-marker removal", (-0.021, 0.012), [-0.02, -0.01, 0.0, 0.01]),
        ("diagnosis_tail", "(b) Diagnosis-tail removal", (-0.405, -0.245), [-0.40, -0.35, -0.30, -0.25]),
    ]
    for ax, (variant, title, limits, ticks) in zip(axes, specs):
        subset = data.loc[data["variant"].eq(variant)].set_index("condition").loc[order]
        y = np.arange(len(order))[::-1]
        for yy, (cond, row) in zip(y, subset.iterrows()):
            estimate = -row.macro_f1_drop
            low = -row.macro_f1_drop_ci_high
            high = -row.macro_f1_drop_ci_low
            ax.errorbar(
                estimate,
                yy,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                color=COLORS[family(cond)],
                markeredgecolor="white",
                markeredgewidth=0.45,
                markersize=5.0,
                elinewidth=1.15,
                capsize=2.2,
                zorder=3,
            )
        ax.axvline(0, color="#333333", linewidth=0.65, zorder=1)
        ax.set_yticks(y, [short[c] for c in order])
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
        ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold", pad=2)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[-1].set_xlabel("Change in macro F1 (masked − original)")
    export(fig, "cue_masking_effects")


def contrast_heatmap() -> None:
    data = pd.read_csv(CONTRASTS)
    models = ["qwen_sft_6000", "mistral_sft_3000", "tfidf_logreg_6000", "gpt54_10shot"]
    contrasts = [
        ("scald_vs_thermal_burn", "Scald / thermal burn"),
        ("electrical_burn_vs_shock", "Electrical burn / shock"),
        ("concussion_vs_internal_injury", "Concussion / internal head"),
        ("laceration_vs_avulsion", "Laceration / avulsion"),
        ("ingestion_vs_aspiration", "Ingestion / aspiration"),
    ]
    matrix = (
        data.loc[data["model_key"].isin(models)]
        .pivot(index="contrast_id", columns="model_key", values="accuracy")
        .loc[[c[0] for c in contrasts], models]
    )
    values = np.vstack([matrix.to_numpy(), matrix.mean(axis=0).to_numpy()])
    row_labels = [c[1] for c in contrasts] + ["Mean"]
    col_labels = ["Qwen\n6K", "Mistral\n3K", "TF-IDF", "GPT-5.4\n10-shot"]

    fig, ax = plt.subplots(figsize=(3.35, 3.15), constrained_layout=True)
    image = ax.imshow(values, cmap="cividis", vmin=0.60, vmax=1.00, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.axhline(4.5, color="white", linewidth=1.5)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] < 0.78 else "#111111"
            ax.text(j, i, f"{values[i, j]:.3f}"[1:], ha="center", va="center", color=color, fontsize=7.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.07, pad=0.08, aspect=25)
    cbar.set_label("Subset accuracy", fontsize=8)
    cbar.set_ticks([0.6, 0.7, 0.8, 0.9, 1.0])
    cbar.outline.set_linewidth(0.5)
    export(fig, "natural_contrast_heatmap")


def learning_curve() -> None:
    """Plot single-seed SFT performance without implying a fitted scaling law."""
    conditions = {
        "Mistral-7B": ["mistral_sft_1000", "mistral_sft_3000", "mistral_sft_6000"],
        "Qwen2.5-7B": ["qwen_sft_1000", "qwen_sft_3000", "qwen_sft_6000"],
    }
    sizes = np.array([1000, 3000, 6000])
    ci = pd.read_csv(BOOTSTRAP)
    ci = ci.loc[ci["metric"].eq("macro_f1")].set_index("condition")

    styles = {
        "Mistral-7B": {"color": COLORS["Mistral"], "marker": "o"},
        "Qwen2.5-7B": {"color": COLORS["Qwen"], "marker": "s"},
    }
    fig, ax = plt.subplots(figsize=(3.35, 2.55), constrained_layout=True)

    for model, keys in conditions.items():
        rows = ci.loc[keys]
        values = rows["estimate"].to_numpy()
        lower = values - rows["ci_low"].to_numpy()
        upper = rows["ci_high"].to_numpy() - values
        style = styles[model]
        ax.plot(
            sizes,
            values,
            color=style["color"],
            linewidth=1.45,
            marker=style["marker"],
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.25,
            zorder=3,
        )
        ax.errorbar(
            sizes,
            values,
            yerr=[lower, upper],
            fmt="none",
            ecolor=style["color"],
            elinewidth=0.95,
            capsize=2.4,
            capthick=0.95,
            zorder=2,
        )
        label_y = values[-1] + (0.010 if model.startswith("Qwen") else -0.012)
        ax.text(
            6200,
            label_y,
            model,
            color=style["color"],
            fontsize=7.2,
            fontweight="bold",
            ha="left",
            va="center",
        )

    ax.set_xlabel("SFT training examples")
    ax.set_ylabel("Macro F1")
    ax.set_xticks(sizes, ["1K", "3K", "6K"])
    ax.set_xlim(700, 7200)
    ax.set_ylim(0.47, 0.93)
    ax.set_yticks(np.arange(0.5, 1.0, 0.1))
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", length=3.0, width=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.7)
    ax.text(
        0.015,
        0.985,
        "Points: single training runs; bars: 95% test-set bootstrap CI",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color="#58636E",
    )
    export(fig, "learning_curve_macro_f1")


def write_manifest() -> None:
    manifest = {
        "inputs": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (SUMMARY, BOOTSTRAP, MASKING, CONTRASTS)
        },
        "outputs": [
            f"figures/{stem}.{suffix}"
            for stem in (
                "experimental_design_schematic",
                "model_comparison_forest",
                "cue_masking_effects",
                "natural_contrast_heatmap",
                "learning_curve_macro_f1",
            )
            for suffix in ("svg", "pdf", "png")
        ],
        "png_dpi": 600,
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    experimental_design()
    model_forest()
    masking_effects()
    contrast_heatmap()
    learning_curve()
    write_manifest()


if __name__ == "__main__":
    main()
