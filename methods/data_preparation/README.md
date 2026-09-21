# Data preparation

Construct and verify the temporal benchmark inputs.

Run commands from the repository root.

| Entry point | Purpose |
|---|---|
| [prepare_neiss_temporal_dataset.py](prepare_neiss_temporal_dataset.py) | Build temporal splits, nested training subsets, and local manifests from annual source workbooks. |
| [verify_dataset_fingerprints.py](verify_dataset_fingerprints.py) | Compare local workbook or split hashes with the published reference manifest. |

Generated splits stay in ignored `data/neiss_temporal_2021_2025/`. Verification prints checks; it does not generate a scientific result table.

[Published outputs / reference](../../data/README.md) ? [Reproduction commands](../../docs/REPRODUCIBILITY.md) ? [Authoritative result map](../../docs/RESULT_MAP.md)
