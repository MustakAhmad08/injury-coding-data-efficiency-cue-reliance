"""Create one reproducible Jupyter notebook per publication figure."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "publication_figures"

FIGURES = [
    (
        "01_experimental_design_schematic.ipynb",
        "Experimental Design Schematic",
        "experimental_design",
        "experimental_design_schematic.png",
        "Scientific overview of the temporal partitions, model-development conditions, frozen configuration, locked-test analysis, and diagnostic analyses.",
    ),
    (
        "02_model_comparison_forest.ipynb",
        "Model Comparison Forest Plot",
        "model_forest",
        "model_comparison_forest.png",
        "Macro-F1 point estimates and diagnosis-stratified 95% bootstrap confidence intervals for all evaluated conditions.",
    ),
    (
        "03_cue_masking_effects.ipynb",
        "Cue-Masking Effects",
        "masking_effects",
        "cue_masking_effects.png",
        "Paired macro-F1 changes after formatting-marker removal and diagnosis-tail removal.",
    ),
    (
        "04_natural_contrast_heatmap.ipynb",
        "Natural-Contrast Heatmap",
        "contrast_heatmap",
        "natural_contrast_heatmap.png",
        "Accuracy on five post-hoc natural diagnosis contrasts, including the across-contrast mean.",
    ),
    (
        "05_sft_learning_curve.ipynb",
        "SFT Training-Size Comparison",
        "learning_curve",
        "learning_curve_macro_f1.png",
        "Single-seed macro-F1 estimates at 1K, 3K, and 6K training examples with test-set bootstrap uncertainty.",
    ),
]


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(title: str, function_name: str, output_name: str, description: str) -> dict:
    setup = '''from pathlib import Path
import sys

from IPython.display import Image, display


def find_project_root() -> Path:
    """Find the repository root whether Jupyter starts here or above it."""
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "scripts" / "generate_publication_figures.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate scripts/generate_publication_figures.py")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_publication_figures import configure_style, FUNCTION_NAME

OUTPUT = ROOT / "figures" / "paper" / "OUTPUT_NAME"
print(f"Project root: {ROOT}")
print(f"Output: {OUTPUT}")
'''.replace("FUNCTION_NAME", function_name).replace("OUTPUT_NAME", output_name)

    generate = f'''configure_style()
{function_name}()

assert OUTPUT.exists(), f"Figure was not created: {{OUTPUT}}"
display(Image(filename=str(OUTPUT)))
'''

    verify = '''for suffix in ("svg", "pdf", "png"):
    artifact = OUTPUT.with_suffix(f".{suffix}")
    assert artifact.exists() and artifact.stat().st_size > 0, artifact
    print(f"{artifact.name}: {artifact.stat().st_size:,} bytes")
'''

    return {
        "cells": [
            markdown(
                f"# {title}\n\n{description}\n\n"
                "This notebook regenerates the figure from the repository's saved evaluation outputs. "
                "It calls the canonical plotting function so notebook and manuscript figures cannot silently diverge. "
                "Running it exports editable SVG, vector PDF, and 600-DPI PNG files to `figures/`.\n"
            ),
            markdown("## Setup and project paths\n"),
            code(setup),
            markdown("## Generate and preview\n"),
            code(generate),
            markdown("## Verify exported artifacts\n"),
            code(verify),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, title, function_name, output_name, description in FIGURES:
        path = OUT / filename
        path.write_text(
            json.dumps(notebook(title, function_name, output_name, description), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
