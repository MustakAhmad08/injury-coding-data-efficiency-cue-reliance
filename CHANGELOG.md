# Changelog

## 2.1.0 — 2026-11-25

- Added post-hoc RoBERTa marker, diagnosis-tail, and diagnosis-term masking for three development-selected seeds, including distinct-254 primary accuracy and archived inference checks.
- Preserved completed-experiment outputs and original script provenance; added a portable reproduction entry point that writes separately from published snapshots.
- Added encoder diagnosis-pair accuracy derived from full-precision per-category results; updated forest, masking, and diagnosis-pair figures to include the primary encoder run.
- Updated plotting code, notebook descriptions, figure hashes, and camera-ready result documentation. Original 15-condition benchmark outputs remain unchanged.
- Added author attribution and citation metadata, corrected README navigation, and replaced anonymity-only checks with public-release integrity checks.
- Added validation of all nine masking rows, subgroup results, provenance hashes, and figure inputs/outputs.
- Renamed the GitHub repository to `injury-coding-data-efficiency-cue-reliance` and updated its citation and README links.

This release packages existing experiments; it does not claim a new training/inference replication.

## 2.0.0

### Added

- Fine-tuned RoBERTa-base baseline with development-selected checkpoints, primary seed 20260718, and two supplementary seeds.
- A separate proportional-distribution test, diagnosis-term masking, dataset/source-distribution summaries, a distinct secondary-diagnosis subgroup analysis, qualitative examples, and a two-reviewer label audit.
- Recorded SFT settings, dataset fingerprints, and expanded reproduction instructions.
- Automated release integrity checks on pushes and pull requests, an [authoritative result map](docs/RESULT_MAP.md), and a [recorded environment reference](docs/ENVIRONMENT.md).

### Organization and validation

- Distributed the 17 utility scripts into method-specific and analysis-specific directories, with updated commands, imports, and navigation.
- Cue masking now honors `NEISS_ADAPTER_ROOT`; figure notebooks locate the plotting module and outputs in their current locations.

- The SFT entry point now lives under `methods/sft/train.py`, alongside the encoder method. Experiment runners and shared-function imports use this location.
- Main-benchmark error-pair outputs and the all-cases secondary-diagnosis outputs use `results/error_pairs/` and `results/secondary_all_cases/`; the distinct subgroup remains under `analysis/error_analysis/secondary_diagnoses/`.

- Supplementary studies are grouped under `analysis/data_quality/`, `analysis/robustness/`, and `analysis/error_analysis/`, with code and aggregate results together.
- Label-audit statistics are stored in `analysis/data_quality/label_audit/results/summary.json`.
- Release validation rejects obsolete analysis folders and checks links, prompt hashes, and aggregate result consistency.
- Validation instructions distinguish dependency-free integrity checks, CPU analysis tests, and full data/model reproduction.

The original benchmark outputs and historical experiment fingerprints are retained. New supplementary evaluations do not replace the balanced test or establish independent end-to-end replication.

## 1.0.0

Initial reproducibility release: balanced temporal benchmark, TF-IDF and prompted/SFT model comparisons, prompt artifacts, aggregate statistical and behavioral analyses, and publication figures.
