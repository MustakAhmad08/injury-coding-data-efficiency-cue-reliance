# Publication-figure notebooks

Each notebook independently regenerates one figure from
`analysis/publication_figures/generate_publication_figures.py` using the saved evaluation outputs.
Run the notebooks from top to bottom. They export SVG, PDF, and 600-DPI PNG
artifacts to `figures/` and verify that all three files were created.

1. `01_experimental_design_schematic.ipynb`
2. `02_model_comparison_forest.ipynb`
3. `03_cue_masking_effects.ipynb`
4. `04_natural_contrast_heatmap.ipynb`
5. `05_sft_learning_curve.ipynb`

The forest, masking, and diagnosis-pair figures include the primary RoBERTa-base
run (seed 20260718). The schematic describes the original benchmark and is retained
as a historical supplement. After regenerating individual figures, run the full
generator to refresh `figures/figure_manifest.json`.
