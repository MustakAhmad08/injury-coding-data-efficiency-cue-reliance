# Authoritative results and reproduction entry points

Use the linked numeric files as the published sources. Read [Results](RESULTS.md) for interpretation and [Reproduction](REPRODUCIBILITY.md) for setup. All commands below run from the repository root in a separate reproduction copy; many regenerate files or need unpublished inputs. A reporting command alone does not reproduce training or inference.

## Main benchmark and earlier analyses

| Experiment | Authoritative published output | Reproduction / reporting command | Required inputs |
|---|---|---|---|
| Balanced benchmark, 15 original conditions | [Overall metrics](../results/primary/model_summary.csv), [per-category metrics](../results/primary/per_category_metrics.csv), [intervals](../results/primary/bootstrap_95ci.csv), [paired tests](../results/primary/mcnemar_all_pairs.csv) | `python analysis/benchmark/evaluate_methods.py --prediction NAME=CSV --output-dir methods_runs/combined_evaluation` | Repeat `--prediction NAME=CSV` for all 15 original condition prediction files |
| RoBERTa baseline | [Per-seed summary](../methods/roberta/results/test_summary.json), [per-category metrics](../methods/roberta/results/per_category_metrics.csv), [paired comparisons](../methods/roberta/results/paired_comparisons.csv) | `python methods/roberta/analyze_results.py` | Selected encoder predictions plus reference predictions; [training protocol](../methods/roberta/README.md) |
| Development prompt ablation | [Summary](../results/prompt_ablation/prompt_ablation_summary.csv) | `python analysis/prompt_ablation/run_prompt_component_ablation.py --stage report` | Completed development ablation runs |
| Marker/tail cue masking | [Summary](../results/cue_masking/cue_reliance_summary.csv), [per-category results](../results/cue_masking/cue_reliance_per_category.csv) | `python analysis/robustness/cue_masking/run_cue_masking_analysis.py --stage report` | Completed original and cue-masked runs |
| Natural contrasts | [Subset metrics](../results/contrasts/contrast_metrics.csv) | `python analysis/error_analysis/contrasts/analyze_difficult_contrasts.py` | Reconstructed narratives and predictions |
| All secondary-diagnosis cases (332) | [Summary](../results/secondary_all_cases/secondary_diagnosis_summary.csv) | `python analysis/error_analysis/secondary_all_cases/analyze_secondary_diagnosis.py` | Reconstructed test data and predictions |
| Aggregate error pairs | [Counts](../results/error_pairs/error_pair_counts.csv) | `python analysis/error_analysis/error_pairs/prepare_error_analysis.py --prediction NAME=CSV --test-file data/neiss_temporal_2021_2025/locked_test.csv --output-dir methods_runs/error_analysis` | Repeat prediction arguments for compared models; also writes private review rows, so use ignored output storage |

`NAME=CSV` is a placeholder for a condition identifier and its prediction CSV path. Preserve the published condition set when comparing regenerated tables. Several scripts write into `methods_runs/`; published aggregate snapshots have been collected under `results/`.

## Supplementary studies

| Study | Authoritative published output | Reproduction / reporting command | Required inputs |
|---|---|---|---|
| Dataset composition | [Split counts](../analysis/data_quality/dataset_composition/results/per_class_split_counts.csv) | `python analysis/data_quality/dataset_composition/build_counts.py` | Reconstructed nested training, development, and test splits |
| Source distribution | [Frequencies and recall](../analysis/data_quality/source_distribution/results/original_2025_distribution_and_recall.csv) | `python analysis/data_quality/source_distribution/build_distribution.py --source-file /path/to/NEISS_2025.XLSX` | Source workbook and reference metrics |
| Label audit | [Agreement snapshot](../analysis/data_quality/label_audit/results/summary.json) | `python analysis/data_quality/label_audit/analyze_audit.py /path/to/human_review_100.csv` | Private paired-review worksheet; command prints aggregates |
| Proportional test | [Metrics](../analysis/robustness/proportional_test/results/metrics.csv), [per-category results](../analysis/robustness/proportional_test/results/per_class.csv) | `python analysis/robustness/proportional_test/analyze.py` | All six prediction runs and prepared sample; [prepare/infer protocol](../analysis/robustness/proportional_test/README.md) |
| Diagnosis-term masking | [Metrics](../analysis/robustness/diagnosis_masking/results/metrics.csv), [per-category results](../analysis/robustness/diagnosis_masking/results/per_class.csv) | `python analysis/robustness/diagnosis_masking/run_test.py report` | Original and masked predictions; [prepare/infer protocol](../analysis/robustness/diagnosis_masking/README.md) |
| Distinct secondary diagnoses (254) | [Subgroup metrics](../analysis/error_analysis/secondary_diagnoses/results/distinct_254_metrics.csv) | `python analysis/error_analysis/secondary_diagnoses/analyze_distinct.py` | Reconstructed test data and frozen predictions |
| Qualitative errors | [Examples and interpretation](../analysis/error_analysis/qualitative_errors/README.md), [thermal-burn confusion counts](../analysis/error_analysis/qualitative_errors/results/thermal_burn_confusions.csv) | `python analysis/error_analysis/qualitative_errors/verify_examples.py` | Private selected-case record, test data, and predictions; verifier covers thermal-burn examples |

The 332-case and 254-case secondary analyses have different denominators. Cue masking and diagnosis-term masking also use different interventions. Their outputs are separate sources, not interchangeable copies.

## Reports from included aggregates

| Artifact | Command | Inputs |
|---|---|---|
| [Proportional category report](../analysis/robustness/proportional_test/PER_CATEGORY_RESULTS.md) | `python analysis/robustness/proportional_test/report_per_category.py` | Included proportional tables; requires pandas and can rewrite the per-class CSV |
| [Publication figures](../figures/) | `python analysis/publication_figures/generate_publication_figures.py` | Included aggregate tables and plotting dependencies; covers original baseline/behavior figures |

Use `python tests/verify_v2.py` to check release integrity without installing scientific packages or regenerating outputs.
