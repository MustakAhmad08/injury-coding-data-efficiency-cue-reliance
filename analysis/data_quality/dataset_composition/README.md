# Dataset composition

[All analyses](../../README.md) / [Data quality](../README.md)

`results/` contains exact counts for all 30 categories across the 1K/3K/6K training subsets, development, and balanced test. Training subsets are nested. The full training, development, and test sets contain 200/10/50 cases per category; 1K has 33 or 34 and 3K has 100. Training years are 2021–2023; development 2024; test 2025.

Run `python analysis/data_quality/dataset_composition/build_counts.py` after reconstructing the data to check counts, nesting, and separation. This also emits a table and textual summary. These class-balanced splits support category comparison, not source prevalence. See [source distribution](../source_distribution/README.md) and [proportional evaluation](../../robustness/proportional_test/README.md).
