# All secondary-diagnosis cases

Evaluate all 332 balanced-test cases with secondary diagnoses, including repeated primary/secondary codes.

Run commands from the repository root.

| Entry point | Purpose |
|---|---|
| [analyze_secondary_diagnosis.py](analyze_secondary_diagnosis.py) | Align saved predictions and report primary/secondary accuracy, error types, and narrative-marker summaries. |

Generated reports go to `methods_runs/secondary_diagnosis_analysis/`. The separate 254-case study with distinct codes remains in `analysis/error_analysis/secondary_diagnoses/`; the two denominators are not interchangeable.

[Published outputs / reference](../../../results/secondary_all_cases) ? [Reproduction commands](../../../docs/REPRODUCIBILITY.md) ? [Authoritative result map](../../../docs/RESULT_MAP.md)
