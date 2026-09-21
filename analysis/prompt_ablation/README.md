# Development prompt ablation

Compare nested prompt components on the 2024 development set.

Run commands from the repository root.

| Entry point | Purpose |
|---|---|
| [run_prompt_component_ablation.py](run_prompt_component_ablation.py) | Use `--stage run` to evaluate a model and `--stage report` to summarize prompt-component comparisons. |

Runs write to `methods_runs/prompt_component_ablation/`; exact published prompt artifacts remain under `artifacts/prompts/`. This experiment does not read the locked test set.

[Published outputs / reference](../../results/prompt_ablation) ? [Reproduction commands](../../docs/REPRODUCIBILITY.md) ? [Authoritative result map](../../docs/RESULT_MAP.md)
