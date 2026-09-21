# TF-IDF logistic regression

Train a development-selected classical baseline and evaluate its frozen pipeline.

Run commands from the repository root.

| Entry point | Purpose |
|---|---|
| [run_tfidf_logreg_baseline.py](run_tfidf_logreg_baseline.py) | Use `--stage tune` for model selection and `--stage final-eval --confirm-locked-test` for final evaluation. |

The pipeline, predictions, and run records default to `methods_runs/tfidf_logreg_6000/`. Published aggregate metrics are in the linked result directory.

[Published outputs / reference](../../results/primary) ? [Reproduction commands](../../docs/REPRODUCIBILITY.md) ? [Authoritative result map](../../docs/RESULT_MAP.md)
