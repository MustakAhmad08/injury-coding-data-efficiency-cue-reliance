# Changelog

## 2.0.0

### Added

- Fine-tuned RoBERTa-base baseline with development-selected checkpoints, primary seed 20260718, and two supplementary seeds.
- A separate proportional-distribution test, diagnosis-term masking, dataset/source-distribution summaries, a distinct secondary-diagnosis subgroup analysis, qualitative examples, and a two-reviewer label audit.
- Recorded SFT settings, dataset fingerprints, and expanded reproduction instructions.
- Automated release integrity checks on pushes and pull requests, an [authoritative result map](docs/RESULT_MAP.md), and a [recorded environment reference](docs/ENVIRONMENT.md).

### Organization and validation

- The SFT entry point now lives under `methods/sft/train.py`, alongside the encoder method. Experiment runners and shared-function imports use this location.
- Main-benchmark error-pair outputs and the all-cases secondary-diagnosis outputs use `results/error_pairs/` and `results/secondary_all_cases/`; the distinct subgroup remains under `analysis/error_analysis/secondary_diagnoses/`.

- Supplementary studies are grouped under `analysis/data_quality/`, `analysis/robustness/`, and `analysis/error_analysis/`, with code and aggregate results together.
- Label-audit statistics are stored in `analysis/data_quality/label_audit/results/summary.json`.
- Release validation rejects obsolete analysis folders and checks links, prompt hashes, and aggregate result consistency.
- Validation instructions distinguish dependency-free integrity checks, CPU analysis tests, and full data/model reproduction.

The original benchmark outputs and historical experiment fingerprints are retained. New supplementary evaluations do not replace the balanced test or establish independent end-to-end replication.

## 1.0.0

Initial reproducibility release: balanced temporal benchmark, TF-IDF and prompted/SFT model comparisons, prompt artifacts, aggregate statistical and behavioral analyses, and publication figures.
