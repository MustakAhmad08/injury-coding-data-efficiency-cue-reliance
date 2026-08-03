# Data access and reconstruction

This repository does not redistribute narrative-level NEISS records. Obtain the
2021--2025 annual NEISS workbooks from the U.S. Consumer Product Safety
Commission and place them in a local directory as `NEISS_2021.XLSX` through
`NEISS_2025.XLSX`.

Reconstruct the deterministic temporal benchmark with:

```bash
python scripts/prepare_neiss_temporal_dataset.py \
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
