# NEISS Diagnosis Classification

A temporal benchmark for classifying short injury-surveillance narratives into 30 primary-diagnosis categories. The repository compares TF-IDF logistic regression, fine-tuned RoBERTa, prompted language models, and QLoRA-adapted Qwen and Mistral.

**Start here:** [results](docs/RESULTS.md) · [reproduction](docs/REPRODUCIBILITY.md) · [experimental settings](docs/EXPERIMENT_SETTINGS.md) · [data access](data/README.md).

## Study design

Training uses 2021–2023 records (6,000); development uses 2024 (300); the balanced 2025 test contains 1,500 records, 50 per category. Training subsets of 1K, 3K, and 6K are nested. A separate, disjoint 1,500-case proportional test approximates the unweighted class distribution among 410,201 eligible 2025 source records. Neither test establishes national injury prevalence or prospective deployment performance.

The primary model seed is **20260718**, including RoBERTa. Two additional RoBERTa seeds are supplementary. Its learning rate was selected using three-seed development means; the primary reporting seed was aligned after evaluation, not selected to maximize test performance. Qwen and Mistral training-size comparisons remain single-seed observations.

## Repository map

| Directory | Contents |
|---|---|
| `methods/roberta/` | Encoder training, checkpoint selection, evaluation, and aggregate results |
| `Injury_LLM_SFT.py`, `scripts/` | QLoRA, TF-IDF, prompted/hosted evaluation, dataset preparation, statistics, and figures |
| [analysis/data_quality/](analysis/data_quality/README.md) | Dataset composition, source distribution, and label audit |
| [analysis/robustness/](analysis/robustness/README.md) | Proportional-test evaluation and diagnosis-term masking |
| [analysis/error_analysis/](analysis/error_analysis/README.md) | Secondary-diagnosis subgroup and qualitative error studies |
| `configs/`, `artifacts/prompts/` | Recorded settings and exact prompt artifacts |
| `results/` | Baseline comparison, paired tests, prompt/cue analyses, and other aggregate outputs |
| `docs/`, `data/` | Methods, reproduction instructions, source access, and reference fingerprints |
| `figures/`, `notebooks/` | Publication figures and figure-generation entry points |
| `tests/` | Repository integrity and analysis checks |

The [analysis guide](analysis/README.md) links each study to its published outputs and explains the file conventions. Each study keeps its code and `results/` together.

## Quick validation

```bash
python tests/verify_v2.py
```

This uses only published aggregate files and Python's standard library. Training and inference require additional dependencies and data; see the reproduction guide. Static validation is not an independent numerical replication.

## Data and model availability

Raw workbooks, narrative-level splits, case predictions, private audit worksheets, credentials, and trained weights are not distributed. The ten fixed de-identified demonstrations are included to reconstruct prompting. Obtain source data from CPSC, reconstruct the splits, and train/download models as documented. Local data and outputs belong in ignored directories. Hosted evaluations require provider access; the mutable Llama alias cannot guarantee identical future weights.

## Scope

Macro-F1 supports equal-weighted category comparison; proportional accuracy and weighted F1 describe the source-record class mix. Per-category precision, recall, F1, and support clarify uneven performance. Masking and selected examples describe output behavior, not proof of contextual reasoning. The intended application is human-assisted surveillance coding, not clinical triage; labor savings, cost effectiveness, and injury prevention were not measured.

Code is MIT licensed. NEISS data and third-party models retain their own terms. Repository version: **2.0.0**.
