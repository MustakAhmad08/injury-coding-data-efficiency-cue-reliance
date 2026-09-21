# Data access and reconstruction

This repository does not redistribute narrative-level NEISS records. Obtain the
2021--2025 annual NEISS workbooks from the U.S. Consumer Product Safety
Commission and place them in a local directory as `NEISS_2021.XLSX` through
`NEISS_2025.XLSX`.

Reconstruct the deterministic temporal benchmark with:

```bash
python methods/data_preparation/prepare_neiss_temporal_dataset.py \
  --source-dir /path/to/annual/workbooks \
  --output-dir data/neiss_temporal_2021_2025
```

The frozen seed is `20260718`. The script creates balanced training (6,000),
development (300), and locked-test (1,500) splits and nested 1K/3K/6K training
subsets. It checks year separation, case-ID separation, exact normalized-
narrative separation, category balance, ablation nesting, and JSON validity.

Do not commit the generated splits. Although NEISS is public and de-identified,
injury narratives may contain sensitive details. Apart from the ten fixed
demonstrations needed to validate prompt construction, the repository therefore
publishes only aggregate and per-category results.

The 30 evaluated labels are listed in
[docs/DIAGNOSIS_CODES.md](../docs/DIAGNOSIS_CODES.md). Code 70 (`Other`) is
defined by NEISS but was not represented in the benchmark because no eligible
records occurred in the supplied annual workbooks.

## Official source and recorded fingerprints

Obtain annual files through the [CPSC NEISS data page](https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data). This release does not host those files. Source files may change after download; use `reference_dataset_manifest.json` to check the exact workbook versions used in the study.

```bash
python methods/data_preparation/verify_dataset_fingerprints.py --source-dir /path/to/annual/workbooks
python methods/data_preparation/prepare_neiss_temporal_dataset.py --source-dir /path/to/annual/workbooks --output-dir data/neiss_temporal_2021_2025
python methods/data_preparation/verify_dataset_fingerprints.py --data-dir data/neiss_temporal_2021_2025
```

The reference manifest contains aggregate split metadata and file hashes, not narratives or case identifiers. Hash mismatches can arise from changed source workbooks or serialization/environment differences; investigate rather than replacing the reference hashes. Exact byte equality is stricter than equivalent record membership. The benchmark uses unweighted records; source frequencies and the proportional test are not survey-weighted national estimates.
