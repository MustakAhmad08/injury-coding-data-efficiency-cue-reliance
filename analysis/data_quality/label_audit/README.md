# Label-appropriateness audit

[All analyses](../../README.md) / [Data quality](../README.md)

Two reviewers assessed the same 100 narrative–code pairs. Reviewer independence is confirmed by the authors. The recorded decisions accept 87 labels jointly, reject six jointly, and disagree on seven. Observed agreement is **93%**; Cohen's kappa is **0.593023**. This measures binary label-appropriateness judgments, not independent diagnosis-code assignment or overall dataset accuracy.

The sample has not been enlarged. Reviewer qualifications and whether the edited worksheet preserves original pre-adjudication judgments are not documented. The stored agreement column has 13 inconsistent entries; the reported statistics are computed directly from the reviewer columns. The worksheet is private and unchanged; `results/summary.json` records its fingerprint and counts. No model was retrained or gold label changed.

```bash
python analysis/data_quality/label_audit/analyze_audit.py /path/to/human_review_100.csv
```

The script prints aggregate statistics only and rejects incomplete paired decisions. Preserve independent decisions separately from consensus/adjudication.
