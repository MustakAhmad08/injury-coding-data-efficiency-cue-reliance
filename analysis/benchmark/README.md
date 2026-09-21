# Balanced benchmark aggregation

Score saved predictions for the original 15-condition balanced benchmark.

Run commands from the repository root.

| Entry point | Purpose |
|---|---|
| [evaluate_methods.py](evaluate_methods.py) | Compute overall/per-category metrics, bootstrap confidence intervals, and paired McNemar tests. |

Supply `--prediction NAME=CSV` once per condition and an `--output-dir`, such as `methods_runs/combined_evaluation`. This reports saved predictions; it does not run inference. Encoder comparisons retain their own implementation under `methods/roberta/`.

[Published outputs / reference](../../results/primary) ? [Reproduction commands](../../docs/REPRODUCIBILITY.md) ? [Authoritative result map](../../docs/RESULT_MAP.md)
