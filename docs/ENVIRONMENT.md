# Recorded environments and validation requirements

## Choose the environment for your task

| Task | Environment | Data or model access |
|---|---|---|
| Release integrity (`tests/verify_v2.py`) | Python standard library; CI targets Python 3.12 on Linux and Windows | Included aggregate files only |
| Proportional-allocation, masking, and source-distribution unit tests | [CPU analysis dependencies](../configs/requirements-analysis.txt) | Synthetic inputs; no model downloads |
| Encoder unit tests | Scientific packages plus PyTorch and Transformers from the [full requirements](../requirements.txt) | Synthetic inputs; imported model libraries required, no training |
| Training, inference, and data-dependent verification | Full requirements and compatible hardware; use the recorded versions below as a reference | Reconstructed data, model access, and original predictions/checkpoints where specified |

Python 3.12 is the release-validation target, not a claim about the original training interpreter. The packaged records do not establish a complete operating-system, driver, interpreter, and transitive-dependency lock.

## Original experiment records

The following versions come from [the SFT settings](../configs/recorded_sft_settings.json) and the recorded [masking](../analysis/robustness/diagnosis_masking/results/environment.json) and [proportional-test](../analysis/robustness/proportional_test/results/environment.json) environments.

| Package | SFT records | Masking / proportional records |
|---|---|---|
| torch | 2.11.0+cu128 | 2.11.0+cu128 |
| transformers | 4.52.4 | 4.52.4 |
| peft | 0.13.2 | 0.13.2 |
| trl | 0.11.4 | Not recorded in these snapshots |
| bitsandbytes | 0.48.2 | 0.48.2 |
| datasets | 2.21.0 | Not recorded in these snapshots |
| scikit-learn | Not recorded in these snapshots | 1.9.0 |
| pandas | Not recorded in these snapshots | 2.3.3 |
| numpy | Not recorded in these snapshots | 1.26.4 |

The [experimental settings](EXPERIMENT_SETTINGS.md) report RTX 5090 hardware and CUDA 12.8 for SFT. These are historical experiment records, not a newly tested installation recipe or a guarantee of identical results. Missing entries do not mean that a package was absent.

## Reproduction practice

Use an isolated environment and a CUDA-compatible PyTorch build for GPU experiments. The full requirements and CPU analysis requirements specify installation ranges; neither is an exact lockfile. Follow the [reproduction guide](REPRODUCIBILITY.md) for the task-specific commands.

For a new run, save `python --version`, `python -m pip freeze`, and the GPU/driver report from `nvidia-smi` alongside your local run records. Keep those new records separate from published historical provenance. Fixed seeds do not eliminate differences caused by hardware, libraries, source-data changes, or model revisions.

## Automated validation

The [release workflow](../.github/workflows/validate-release.yml) runs only the standard-library integrity check. It does not install the training stack, access private data, or invoke model/provider APIs. Its checkout and Python setup follow the [official GitHub Actions documentation](https://github.com/actions/setup-python).
