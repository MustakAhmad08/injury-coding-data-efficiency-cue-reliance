# Reproduce the study

Run commands from the repository root. Use a separate working copy for new experiments: published result files record the completed study, and several analysis scripts write their outputs to the adjacent `results/` directory.

## 1. Choose a validation or reproduction path

| Goal | Dependencies | Private data / GPU |
|---|---|---|
| Check the published release | Standard-library Python only | Neither |
| Run the three CPU analysis unit-test suites | `configs/requirements-analysis.txt` | Neither |
| Run encoder unit tests | Scientific packages, PyTorch, and Transformers | Neither; model libraries must be importable |
| Reproduce training and inference | Full `requirements.txt` and task-specific setup | Reconstructed data and model/provider access; GPU for the recorded neural runs |

### Release integrity

Run this immediately after obtaining the repository; no package installation is needed:

```bash
python -B tests/verify_v2.py
python -B tests/test_entrypoints.py
```

The [GitHub Actions workflow](../.github/workflows/validate-release.yml) runs the same check on pushes, pull requests, and manual dispatch, using Python 3.12 on Linux and Windows. Passing it validates packaged artifacts, not a numerical replication.

### CPU analysis unit tests

Create an isolated environment with `python -m venv .venv`, then activate it with `.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Unix.

```bash
python -m pip install -r configs/requirements-analysis.txt
python analysis/robustness/proportional_test/test_protocol.py
python analysis/robustness/diagnosis_masking/test_masking.py
python analysis/data_quality/source_distribution/test_distribution.py
```

These tests use synthetic inputs and do not train models. Encoder unit tests additionally import PyTorch and Transformers; run `python methods/roberta/test_encoder.py` in the full research environment.

### Full reproduction environment

In an activated research environment, install a compatible CUDA PyTorch build and the full dependencies:

```bash
python -m pip install -r requirements.txt
```

Consult the [recorded environment reference](ENVIRONMENT.md) and [experimental settings](EXPERIMENT_SETTINGS.md). Installation ranges are not a complete environment lock. Continue below for data and model preparation.

## 2. Obtain and reconstruct data

Follow [data access](../data/README.md) to download CPSC annual workbooks for 2021–2025. Check source hashes, reconstruct the temporal splits, and check generated hashes:

```bash
python methods/data_preparation/verify_dataset_fingerprints.py --source-dir /path/to/workbooks
python methods/data_preparation/prepare_neiss_temporal_dataset.py --source-dir /path/to/workbooks --output-dir data/neiss_temporal_2021_2025
python methods/data_preparation/verify_dataset_fingerprints.py --data-dir data/neiss_temporal_2021_2025
```

Source updates and serialization differences may cause mismatches; investigate before claiming exact reconstruction. Do not redistribute generated narratives, audit forms, or case predictions.

## 3. Prepare local models and prompts

Download the models/tokenizers at the commits recorded in `configs/recorded_sft_settings.json`. Hugging Face model access and any applicable license acceptance are prerequisites. The model runner enables offline mode, so an empty cache will fail. It loads model IDs from cache rather than enforcing a commit argument; verify the loaded commit in its run manifest before interpreting results as a replication of the recorded model. Do not silently use a different cached model version.

RoBERTa has a pinned download helper:

```bash
python methods/roberta/prepare_model.py
```

For local 10-shot experiments, `python methods/experiments/prepare_methods_experiment.py` creates the demonstration file after data preparation. It rewrites the experiment matrix, so retain the supplied matrix first. Compare demonstration order and contents with `artifacts/prompts/`; exact prompt artifacts are the reference, not newly sampled examples.

## 4. Train and evaluate

```bash
python methods/tfidf/run_tfidf_logreg_baseline.py --stage tune
python methods/tfidf/run_tfidf_logreg_baseline.py --stage final-eval --confirm-locked-test
python methods/experiments/run_methods_condition.py qwen_sft_6000 --stage train
python methods/experiments/run_methods_condition.py qwen_sft_6000 --stage final-eval --confirm-locked-test
python methods/experiments/run_methods_condition.py mistral_sft_3000 --stage train
python methods/experiments/run_methods_condition.py mistral_sft_3000 --stage final-eval --confirm-locked-test
python methods/roberta/run_encoder.py train
python methods/roberta/run_encoder.py final-eval --confirm-locked-test
python methods/roberta/analyze_results.py
```

Repeat the local runner for the other condition IDs in `configs/experiment_matrix.json`; prompting conditions have only a final-evaluation stage. `--dry-run` prints a command without executing it. Freeze choices before granting the explicit locked-test gate. Model selection uses development data; test-based seed/checkpoint selection changes the protocol.

Local generative outputs and adapters use `methods_runs/{condition}`. Diagnosis-term masking, marker/tail cue masking, and proportional inference use this adapter root by default; set `NEISS_ADAPTER_ROOT` to use a separate directory. They also need the saved TF-IDF pipeline and RoBERTa checkpoints. Newly trained weights may differ from archived fingerprints; retain their new provenance and distinguish fresh replication from exact frozen-artifact verification.

Hosted-model scripts are `methods/gpt54/run_gpt54_evaluation.py` and `methods/hosted_llama/run_hosted_llama_evaluation.py`. Inspect `--help` for their arguments. Supply credentials only through environment variables. Hosted evaluations transmit narrative inputs to the provider and require appropriate data-access authorization. The Llama endpoint used a mutable alias; identical future weights cannot be guaranteed.

## 5. Analyses and reporting

The [authoritative result map](RESULT_MAP.md) links published tables to reporting commands and required inputs. The [analysis index](../analysis/README.md) organizes the supplementary studies. Run `analysis/benchmark/evaluate_methods.py --help` for baseline aggregation. Proportional analysis requires all six model prediction files, including supplementary encoder seeds. Diagnosis masking uses TF-IDF, Qwen, and Mistral. Distinct secondary-diagnosis analysis additionally needs the GPT 10-shot predictions.

Published aggregate results can be inspected without source data or model access. To regenerate readable proportional category tables from the included counts:

```bash
python analysis/robustness/proportional_test/report_per_category.py
```

The generator checks private predictions when they are available; absent local predictions are not treated as newly verified. Original provenance hashes refer to the scripts/artifacts used in the experiment. Reorganized source files naturally have different hashes; do not rewrite historical experiment fingerprints to hide that difference.

```bash
python analysis/publication_figures/generate_publication_figures.py
```

This recreates figures from the published aggregate tables. It covers the original baseline/behavior figures; see `docs/RESULTS.md` for the additional encoder and proportional comparisons.

## 6. Checks and limitations

Use the validation path in section 1 that matches your environment. Dataset/model-dependent verification commands are documented in each analysis and cannot run from aggregate files alone. Static checks validate structure, file-content anonymity markers, prompt hashes, and result consistency; they do not validate hosting-account anonymity or independently reproduce the experiments. Hardware, library versions, data updates, model availability, and unbundled trained artifacts remain material requirements.
