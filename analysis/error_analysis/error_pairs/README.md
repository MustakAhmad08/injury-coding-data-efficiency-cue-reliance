# Error pairs and manual-review preparation

Summarize diagnosis confusions and construct a reproducible review sample.

Run commands from the repository root.

| Entry point | Purpose |
|---|---|
| [prepare_error_analysis.py](prepare_error_analysis.py) | Write aggregate error-pair counts, a review protocol, and a private failure-review worksheet. |

Supply prediction arguments, `--test-file`, and an ignored `--output-dir` such as `methods_runs/error_analysis`. Keep generated review rows private; only the aggregate counts and protocol are published.

[Published outputs / reference](../../../results/error_pairs) ? [Reproduction commands](../../../docs/REPRODUCIBILITY.md) ? [Authoritative result map](../../../docs/RESULT_MAP.md)
