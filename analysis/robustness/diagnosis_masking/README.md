# Diagnosis-term masking

[All analyses](../../README.md) / [Robustness](../README.md)

The lexicon in `run_test.py` replaces diagnosis terms and common abbreviations with `[MASK]`, preserving characters outside matched spans. On the 1,500-case balanced test, 1,565 spans in 1,091 narratives change; 409 narratives remain unchanged. All three models were frozen. [Results](RESULTS.md) report full-test and changed-only scores plus the distinct 254-case subgroup.

```bash
python analysis/robustness/diagnosis_masking/test_masking.py
python analysis/robustness/diagnosis_masking/run_test.py prepare
python analysis/robustness/diagnosis_masking/run_test.py infer --condition tfidf_logreg_6000
python analysis/robustness/diagnosis_masking/run_test.py infer --condition qwen_sft_6000
python analysis/robustness/diagnosis_masking/run_test.py infer --condition mistral_sft_3000
python analysis/robustness/diagnosis_masking/run_test.py report
```

Adapters default to `methods_runs/{condition}`; `NEISS_ADAPTER_ROOT` can override that root. See the reproduction guide for original-data/model requirements. Predictions and span audits remain private. This automated lexicon is not exhaustive or clinician-validated; incidental/negated uses may be masked, unlisted expressions may remain, and masking creates distribution shift. Neither score drops nor residual performance prove contextual reasoning. This term intervention is distinct from the marker/tail analyses under `results/cue_masking/`.
