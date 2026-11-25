# Publication figures and notebooks

Render the camera-ready benchmark and behavioral figures from included aggregate tables,
including the primary RoBERTa-base run and its masking results.

Run commands from the repository root.

| Entry point | Purpose |
|---|---|
| [generate_publication_figures.py](generate_publication_figures.py) | Generate SVG/PDF/PNG figures and their input manifest. |
| [build_figure_notebooks.py](build_figure_notebooks.py) | Generate five notebooks that call the shared plotting functions. |

Figure outputs go to `figures/`; notebooks go to `notebooks/publication_figures/`. The notebook builder needs only standard-library Python; rendering requires plotting dependencies. The forest plot, masking plot, diagnosis-pair heatmap, and training-size plot match the camera-ready manuscript. The experimental-design schematic is retained as a historical supplement describing the original benchmark; it does not depict every later extension.

The figure manifest records hashes of aggregate inputs, the generator, and all output
files. Regenerating PDFs may change metadata and file hashes even when their rendered
content is identical. Run the full generator to refresh the manifest after rendering.

[Published outputs / reference](../../figures) ? [Reproduction commands](../../docs/REPRODUCIBILITY.md) ? [Authoritative result map](../../docs/RESULT_MAP.md)
