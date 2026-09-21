# Supplementary analyses

See the [authoritative result map](../docs/RESULT_MAP.md) for source tables, commands, and required inputs.

Start with the question you want to answer. Each study keeps its scripts, documentation, and published aggregate results together.

```text
analysis/
  data_quality/                 # What data and labels are being evaluated?
    dataset_composition/        # Counts, training subsets, split separation
    source_distribution/        # Eligible 2025 source-record frequencies
    label_audit/                # Two-reviewer label agreement
  robustness/                   # How do results change under other conditions?
    proportional_test/          # Separate test with source-record proportions
    diagnosis_masking/          # Evaluation after masking diagnosis terms
  error_analysis/               # Where do predictions succeed or fail?
    secondary_diagnoses/        # Distinct primary/secondary diagnosis subgroup
    qualitative_errors/         # Selected burn examples and confusion counts
```

## Find a study

| Group | Study and instructions | Main published output |
|---|---|---|
| Data quality | [Dataset composition](data_quality/dataset_composition/README.md) | [Counts by class and split](data_quality/dataset_composition/results/per_class_split_counts.csv) |
| Data quality | [Source distribution](data_quality/source_distribution/README.md) | [Source frequencies and balanced-test recall](data_quality/source_distribution/results/original_2025_distribution_and_recall.csv) |
| Data quality | [Label audit](data_quality/label_audit/README.md) | [Reviewer agreement summary](data_quality/label_audit/results/summary.json) |
| Robustness | [Proportional test](robustness/proportional_test/README.md) | [Metrics](robustness/proportional_test/results/metrics.csv), [category report](robustness/proportional_test/PER_CATEGORY_RESULTS.md) |
| Robustness | [Diagnosis masking](robustness/diagnosis_masking/README.md) | [Results report](robustness/diagnosis_masking/RESULTS.md) |
| Error analysis | [Secondary diagnoses](error_analysis/secondary_diagnoses/README.md) | [254-case metrics](error_analysis/secondary_diagnoses/results/distinct_254_metrics.csv) |
| Error analysis | [Qualitative errors](error_analysis/qualitative_errors/README.md) | [Examples and interpretation](error_analysis/qualitative_errors/README.md), [thermal-burn counts](error_analysis/qualitative_errors/results/thermal_burn_confusions.csv) |

## File conventions

- `README.md`: study question, interpretation, inputs, and reproduction instructions.
- `*.py`: preparation, evaluation, reporting, or verification scripts. Run documented commands from the repository root.
- `test_*.py`: checks using synthetic or aggregate inputs; these do not run model inference.
- `results/`: authoritative published aggregate tables and recorded provenance. The label-audit snapshot is also stored here.
- `RESULTS.md` or `PER_CATEGORY_RESULTS.md`, where present: readable reports alongside the study README.
- `local/`, created during reproduction: ignored private inputs and case-level outputs; absent from the release.

The repository-level [results directory](../results/README.md) holds the main benchmark and earlier aggregate analyses. The study-specific `results/` folders here hold these supplementary analyses. The [overall results guide](../docs/RESULTS.md) brings both together. RoBERTa training and evaluation code lives in [methods/roberta](../methods/roberta/README.md).

## Validate or reproduce

From the repository root, check the release using published aggregates only:

```bash
python tests/verify_v2.py
```

For individual experiments, follow the linked study README and the [reproduction guide](../docs/REPRODUCIBILITY.md). Most reproduction scripts require reconstructed data, private predictions, or trained models; browsing published results does not.

These analyses supplement the balanced benchmark. None selected generative model weights or prompts using the new test results. Recorded provenance and numeric outputs are preserved: their hashes refer to the original experiment files and script versions, not necessarily the reorganized source files.

## Changes to the v2 layout

The seven study names are unchanged. Their former locations directly under `analysis/` now sit under the three groups shown above. `label_audit/summary.json` is now `data_quality/label_audit/results/summary.json`. Commands, imports, and documentation links use the new locations.
