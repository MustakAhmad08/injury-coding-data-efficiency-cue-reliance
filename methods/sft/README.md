# Supervised fine-tuning and local generation

`train.py` contains the QLoRA training pipeline and shared prompt, parsing, loading, and evaluation functions used by the experiment scripts. Training and inference require the [research environment](../../docs/ENVIRONMENT.md), reconstructed data, and model access.

Use the configured runner from the repository root:

```bash
python methods/experiments/run_methods_condition.py qwen_sft_6000 --stage train --dry-run
python methods/experiments/run_methods_condition.py qwen_sft_6000 --stage train
python methods/experiments/run_methods_condition.py qwen_sft_6000 --stage final-eval --confirm-locked-test
```

The runner invokes `methods/sft/train.py` with the model, split paths, seed, and output directory from the experiment matrix. The final-evaluation flag must only be supplied after model choices are frozen. For direct CLI options in an installed research environment, run `python methods/sft/train.py --help`.

Default training, validation, and few-shot input paths resolve against the repository root. Run outputs from the configured runner remain under ignored `methods_runs/` directories. Shared functions are imported as `methods.sft.train`.

See [experimental settings](../../docs/EXPERIMENT_SETTINGS.md), [reproduction](../../docs/REPRODUCIBILITY.md), and the [result map](../../docs/RESULT_MAP.md). The main benchmark outputs live under `results/`; self-contained supplementary studies live under `analysis/`.
