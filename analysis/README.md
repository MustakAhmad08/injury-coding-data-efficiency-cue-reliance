# Analysis and reporting

See the [authoritative result map](../docs/RESULT_MAP.md) for source tables, commands, and required inputs.

Start with the question you want to answer. This directory contains benchmark aggregation, diagnostic studies, and figure generation. Each README identifies its authoritative published outputs and generated local files.

```text
analysis/
  benchmark/                    # Aggregate the original balanced benchmark
  prompt_ablation/              # Development-only prompt comparisons
  publication_figures/          # Plotting code and notebook builder
  data_quality/                 # What data and labels are being evaluated?
    dataset_composition/        # Counts, training subsets, split separation
    source_distribution/        # Eligible 2025 source-record frequencies
    label_audit/                # Two-reviewer label agreement
  robustness/                   # How do results change under other conditions?
    proportional_test/          # Separate test with source-record proportions
    diagnosis_masking/          # Evaluation after masking diagnosis terms
    cue_masking/                # Formatting-marker and diagnosis-tail interventions
  error_analysis/               # Where do predictions succeed or fail?
    secondary_diagnoses/        # Distinct primary/secondary diagnosis subgroup
    qualitative_errors/         # Selected burn examples and confusion counts
    contrasts/                  # Natural difficult-diagnosis contrasts
    secondary_all_cases/        # All 332 cases with secondary diagnoses
    error_pairs/                # Aggregate confusions and review preparation
```

## Find a study

| Group | Study and instructions | Main published output |
|---|---|---|
| Reporting | [Balanced benchmark aggregation](benchmark/README.md) | [Published outputs](../results/primary/) |
| Diagnostic study | [Natural diagnosis contrasts](error_analysis/contrasts/README.md) | [Published outputs](../results/contrasts/) |
| Diagnostic study | [All secondary-diagnosis cases](error_analysis/secondary_all_cases/README.md) | [Published outputs](../results/secondary_all_cases/) |
| Diagnostic study | [Error pairs and manual-review preparation](error_analysis/error_pairs/README.md) | [Published outputs](../results/error_pairs/) |
| Diagnostic study | [Marker and diagnosis-tail cue masking](robustness/cue_masking/README.md) | [Published outputs](../results/cue_masking/) |
| Diagnostic study | [Development prompt ablation](prompt_ablation/README.md) | [Published outputs](../results/prompt_ablation/) |
| Reporting | [Publication figures and notebooks](publication_figures/README.md) | [Published outputs](../figures/) |
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
- `results/`, where present: authoritative aggregate tables and recorded provenance for that study. Other analysis READMEs link to the existing top-level `results/` or `figures/` outputs; files are not duplicated.
- `RESULTS.md` or `PER_CATEGORY_RESULTS.md`, where present: readable reports alongside the study README.
- `local/`, created during reproduction: ignored private inputs and case-level outputs; absent from the release.

The repository-level [results directory](../results/README.md) holds the main benchmark and earlier aggregate analyses. The study-specific `results/` folders here hold these supplementary analyses. The [overall results guide](../docs/RESULTS.md) brings both together. RoBERTa training and evaluation code lives in [methods/roberta](../methods/roberta/README.md).

## Validate or reproduce

From the repository root, check the release using published aggregates only:

```bash
python tests/verify_v2.py
```

For individual experiments, follow the linked study README and the [reproduction guide](../docs/REPRODUCIBILITY.md). Most reproduction scripts require reconstructed data, private predictions, or trained models; browsing published results does not.

The diagnostic studies supplement the balanced benchmark. None selected generative model weights or prompts using the new test results. Recorded provenance and numeric outputs are preserved: their hashes refer to the original experiment files and script versions, not necessarily the reorganized source files.

## Changes to the v2 layout

The seven study names are unchanged. Their former locations directly under `analysis/` now sit under the three groups shown above. `label_audit/summary.json` is now `data_quality/label_audit/results/summary.json`. Commands, imports, and documentation links use the new locations.

The former flat script collection is now distributed between [methods](../methods/README.md) and this directory. Published result files and private run-output locations retain their existing paths.
