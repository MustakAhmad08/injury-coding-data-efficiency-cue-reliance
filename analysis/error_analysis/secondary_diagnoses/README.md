# Distinct primary and secondary diagnoses

[All analyses](../../README.md) / [Error analysis](../README.md)

Among 1,500 balanced-test cases, 332 have secondary diagnoses. Of those, 78 repeat the primary code and 254 have distinct codes. `analyze_distinct.py` evaluates the latter using frozen original predictions, reporting primary, secondary, both-ordered accuracy, exact swaps, and Wilson intervals. Single-label classifiers supply primary accuracy only. These narratives retain diagnostic wording; subgroup performance is not proof of reasoning.

```bash
python analysis/error_analysis/secondary_diagnoses/analyze_distinct.py
python analysis/error_analysis/secondary_diagnoses/verify_analysis.py
```

Requires reconstructed test data, original model predictions under `methods_runs/`, and the three encoder prediction files under `methods/roberta/local/test_predictions/`. Results are in `results/`; masked subgroup comparisons are under `../../robustness/diagnosis_masking/`. The full 332-case analysis is separately retained in the root `results/secondary/` directory.
