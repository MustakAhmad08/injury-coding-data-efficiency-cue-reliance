# NEISS Diagnosis Classification

Code and aggregate results for **What Do Fine-Tuned Models Learn About Injury Coding?
Data Efficiency and Cue Reliance in Narrative Classification** (EACL 2027).

Mustak Ahmad, Rishab Chakravarty, and Gaurav Nanda — Purdue University, West Lafayette.
See [citation metadata](CITATION.cff), the [camera-ready release notes](CHANGELOG.md),
and the [GitHub repository](https://github.com/MustakAhmad08/injury-coding-data-efficiency-cue-reliance).

A temporal benchmark for classifying short injury-surveillance narratives into 30 primary-diagnosis categories. The repository compares TF-IDF logistic regression, fine-tuned RoBERTa, prompted language models, and QLoRA-adapted Qwen and Mistral.

**Start here:** [results](docs/RESULTS.md) · [reproduction](docs/REPRODUCIBILITY.md) · [experimental settings](docs/EXPERIMENT_SETTINGS.md) · [data access](data/README.md).

See also: [release changes](CHANGELOG.md), [authoritative result map](docs/RESULT_MAP.md), and [recorded environments](docs/ENVIRONMENT.md).

## Study design

Training uses 2021–2023 records (6,000); development uses 2024 (300); the balanced 2025 test contains 1,500 records, 50 per category. Training subsets of 1K, 3K, and 6K are nested. A separate, disjoint 1,500-case proportional test approximates the unweighted class distribution among 410,201 eligible 2025 source records. Neither test establishes national injury prevalence or prospective deployment performance.

The primary model seed is **20260718**, including RoBERTa. Two additional RoBERTa seeds are supplementary. Its learning rate was selected using three-seed development means; the primary reporting seed was aligned after evaluation, not selected to maximize test performance. Qwen and Mistral training-size comparisons remain single-seed observations.

## Repository map

| Directory | Contents |
|---|---|
| `methods/roberta/` | Encoder training, checkpoint selection, evaluation, and aggregate results |
| [methods/sft/](methods/sft/README.md) | QLoRA training and shared local generation/evaluation functions |
| [methods/](methods/README.md) | Dataset construction, experiment configuration, model training, and prediction generation |
| [analysis/benchmark/](analysis/benchmark/README.md) | Balanced benchmark metrics, intervals, and paired tests |
| [analysis/prompt_ablation/](analysis/prompt_ablation/README.md) | Development prompt-component comparisons |
| [analysis/publication_figures/](analysis/publication_figures/README.md) | Figure-generation code and notebook builder |
| [analysis/data_quality/](analysis/data_quality/README.md) | Dataset composition, source distribution, and label audit |
| [analysis/robustness/](analysis/robustness/README.md) | Proportional test, diagnosis-term masking, and [three-seed RoBERTa masking](analysis/robustness/roberta_masking/README.md) |
| [analysis/error_analysis/](analysis/error_analysis/README.md) | Secondary-diagnosis subgroup and qualitative error studies |
| `configs/`, `artifacts/prompts/` | Recorded settings and exact prompt artifacts |
| `results/` | Baseline comparison, paired tests, prompt/cue analyses, and other aggregate outputs |
| `docs/`, `data/` | Methods, reproduction instructions, source access, and reference fingerprints |
| `figures/`, `notebooks/` | Publication figures and figure-generation entry points |
| `tests/` | Repository integrity and analysis checks |

The [method guide](methods/README.md) locates data/model execution code; the [analysis guide](analysis/README.md) locates scoring and reporting code and links to authoritative outputs.

## Quick validation

```bash
python tests/verify_v2.py
python -B tests/test_entrypoints.py
```

The [release workflow](.github/workflows/validate-release.yml) runs this check on every push and pull request. This uses only published aggregate files and Python's standard library. Training and inference require additional dependencies and data; see the reproduction guide. Static validation is not an independent numerical replication.

## Data and model availability

Raw workbooks, narrative-level splits, case predictions, private audit worksheets, credentials, and trained weights are not distributed. The ten fixed de-identified demonstrations are included to reconstruct prompting. Obtain source data from CPSC, reconstruct the splits, and train/download models as documented. Local data and outputs belong in ignored directories. Hosted evaluations require provider access; the mutable Llama alias cannot guarantee identical future weights.

## Scope

Macro-F1 supports equal-weighted category comparison; proportional accuracy and weighted F1 describe the source-record class mix. Per-category precision, recall, F1, and support clarify uneven performance. Masking and selected examples describe output behavior, not proof of contextual reasoning. The intended application is human-assisted surveillance coding, not clinical triage; labor savings, cost effectiveness, and injury prevention were not measured.

Code is MIT licensed. NEISS data and third-party models retain their own terms. Repository version: **2.1.0**.
