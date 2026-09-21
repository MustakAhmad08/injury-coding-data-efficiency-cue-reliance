# Source diagnosis distribution

[All analyses](../../README.md) / [Data quality](../README.md)

The eligible 2025 source pool has 410,201 records. `results/original_2025_distribution_and_recall.csv` reports all class frequencies alongside balanced-test recalls; its RoBERTa recall mean/SD is a supplementary three-seed summary, not the primary seed-20260718 estimate. Frequency estimates are unweighted and are not population-level injury estimates.

```bash
python analysis/data_quality/source_distribution/build_distribution.py --source-file /path/to/NEISS_2025.XLSX
python analysis/data_quality/source_distribution/test_distribution.py
```

See `config.json` and recorded provenance for eligibility and source fingerprints. This descriptive analysis performs no new inference. Frozen-model evaluation under source proportions is documented separately in [the proportional test](../../robustness/proportional_test/README.md).
