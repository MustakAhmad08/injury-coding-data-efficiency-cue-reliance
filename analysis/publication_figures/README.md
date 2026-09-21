# Publication figures and notebooks

Render the original benchmark and behavior figures from included aggregate tables.

Run commands from the repository root.

| Entry point | Purpose |
|---|---|
| [generate_publication_figures.py](generate_publication_figures.py) | Generate SVG/PDF/PNG figures and their input manifest. |
| [build_figure_notebooks.py](build_figure_notebooks.py) | Generate five notebooks that call the shared plotting functions. |

Figure outputs go to `figures/`; notebooks go to `notebooks/publication_figures/`. The notebook builder needs only standard-library Python; rendering requires plotting dependencies. These figures cover the original benchmark/behavior analyses, not all v2 additions.

[Published outputs / reference](../../figures) ? [Reproduction commands](../../docs/REPRODUCIBILITY.md) ? [Authoritative result map](../../docs/RESULT_MAP.md)
