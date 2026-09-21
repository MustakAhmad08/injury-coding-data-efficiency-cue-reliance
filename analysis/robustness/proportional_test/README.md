# Proportional-distribution test

[All analyses](../../README.md) / [Robustness](../README.md)

This separate 1,500-case sample uses largest-remainder quotas from 410,201 eligible 2025 source records (sampling seed 20260918). It excludes overlap with training/development and the original test, plus exact duplicate narratives. Quotas reflect original counts before exclusions. Records and within-class composition therefore differ from the balanced test; score changes cannot be attributed solely to prevalence.

All model weights and prompts were frozen. Primary reporting uses TF-IDF 6K, Qwen 6K, Mistral 3K, and RoBERTa seed 20260718. The other two encoder seeds are supplementary. `results/metrics.csv` contains all six runs; `results/per_class.csv` is the full-precision per-category source. [Category tables](PER_CATEGORY_RESULTS.md) are generated from it. Accuracy, weighted F1, and supported-label macro-F1 are summarized in [Results](../../../docs/RESULTS.md).

Only 28 categories appear: codes 46 and 47 have zero cases. Recall/F1 are unavailable for zero-support classes; precision is unavailable with no predictions. Supported-label macro-F1 averages 28 labels; fixed-30 macro-F1 assigns zero to absent labels. Rare-class estimates can be unstable. These are unweighted source proportions, not national injury estimates.

```bash
python analysis/robustness/proportional_test/experiment.py prepare --source-file /path/to/NEISS_2025.XLSX
python analysis/robustness/proportional_test/experiment.py infer --condition qwen_sft_6000
python analysis/robustness/proportional_test/analyze.py
python analysis/robustness/proportional_test/verify.py
```

Run inference for all six conditions listed in `experiment.py` before the full analysis. Adapters default to `methods_runs/{condition}`; set `NEISS_ADAPTER_ROOT` for another root. Reconstructed data, the TF-IDF pipeline, selected encoder checkpoints, and original balanced predictions are required. Original frozen weights/predictions are not bundled. Checksums distinguish the recorded experiment from newly trained replications. Preparation refuses to overwrite a frozen local sample.

For aggregate-only tables, run `python analysis/robustness/proportional_test/report_per_category.py`; this does not require model weights. Original 1,000-resample bootstrap intervals are preserved. New numeric runs should use a separate working copy and keep their own provenance.
