# Experiment configuration and execution

Configure and run the local Mistral/Qwen conditions.

Run commands from the repository root.

| Entry point | Purpose |
|---|---|
| [prepare_methods_experiment.py](prepare_methods_experiment.py) | Select fixed few-shot demonstrations and regenerate the experiment matrix. |
| [run_methods_condition.py](run_methods_condition.py) | Invoke the shared SFT/generation entry point for one configured training or evaluation condition; supports `--dry-run`. |

Runs write to ignored `methods_runs/{condition}/`. Preparation rewrites `configs/experiment_matrix.json`; preserve the published matrix before regenerating it.

[Published outputs / reference](../../results/primary) ? [Reproduction commands](../../docs/REPRODUCIBILITY.md) ? [Authoritative result map](../../docs/RESULT_MAP.md)
