# NEISS Injury Diagnosis Classification

Anonymous reproducibility package for a 30-class primary-diagnosis coding study
using National Electronic Injury Surveillance System (NEISS) emergency-
department narratives.

The benchmark uses a prespecified temporal design: 2021--2023 for training,
2024 for development, and a locked, class-balanced 2025 test set. It compares
TF--IDF logistic regression, zero- and 10-shot prompting, and QLoRA supervised
fine-tuning of Mistral-7B-Instruct-v0.3 and Qwen2.5-7B-Instruct at 1K, 3K, and
6K training examples.

## What is included

- training, inference, prompting, baseline, and evaluation code;
- frozen local-model experiment configuration and random seed;
- exact prompt-condition artifacts and ordered demonstrations;
- aggregate, paired-test, and per-diagnosis results;
- post-hoc validation-analysis outputs;
- scripts and notebooks for the publication figures.

Raw NEISS workbooks, derived narrative-level splits, case-level model outputs,
API responses, credentials, checkpoints, and adapter weights are intentionally
excluded. The only narrative-level exception is the fixed set of ten
de-identified demonstrations required to reconstruct the prompting condition.
See [data/README.md](data/README.md).

## Repository structure

```text
.
|-- Injury_LLM_SFT.py       # Local QLoRA training and inference
|-- configs/                # Frozen experiment matrix
|-- scripts/                # Data, model, evaluation, and figure CLIs
|-- artifacts/prompts/      # Exact prompt-ablation artifacts
|-- data/                   # Data-access and reconstruction instructions
|-- results/                # Aggregate and per-code results (no narratives)
|-- figures/                # Vector publication figures
|-- notebooks/              # One notebook per publication figure
|-- docs/                   # Diagnosis inventory and reproducibility guide
`-- tests/                  # Static release-integrity check
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Local QLoRA runs require a CUDA-capable GPU and a compatible PyTorch build.
Hosted evaluations require the corresponding provider credentials, supplied
only through environment variables.

## Quick checks

```bash
python tests/verify_release.py
python scripts/run_methods_condition.py qwen_sft_6000 --stage train --dry-run
python scripts/generate_publication_figures.py
```

Dataset reconstruction and the complete execution order are described in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The locked test set is
protected by an explicit `--confirm-locked-test` gate in evaluation commands.

## Key reported result

Qwen SFT with 6K examples achieved the strongest observed locked-test macro F1:
0.905 (95% diagnosis-stratified bootstrap CI 0.892--0.918). Training-size runs
use one seed per condition and should not be interpreted as replicated scaling
estimates.

## Human label-consistency audit

A blinded-independent adjudication study was not conducted. In the finalized
100-case, single-reviewer consistency check, 94 assignments agreed with the
narrative and six did not. The worksheet is omitted because it contains injury
narratives; the audit protocol and limitations are documented in the paper.

## Anonymity and citation

This review repository intentionally contains no author or institutional
identity. During anonymous review, cite it as the anonymous code repository
linked from the manuscript. Author metadata, a permanent repository URL, and a
versioned archival DOI should be added only after the review policy permits
de-anonymization.

## License

Code is released under the MIT License. NEISS source data and third-party model
artifacts remain subject to their respective terms.
