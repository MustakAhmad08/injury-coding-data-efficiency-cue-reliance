# Reproducibility guide

## 1. Reconstruct the benchmark

Follow [data/README.md](../data/README.md). Generated row-level data remain
local and are ignored by Git.

## 2. Inspect the frozen design

`configs/experiment_matrix.json` records the ten local Mistral/Qwen conditions,
their model IDs, dataset paths, training sizes, and the single study seed. The
TF--IDF and hosted conditions use their dedicated scripts. Preview a local
condition without starting training:

```bash
python scripts/run_methods_condition.py qwen_sft_6000 --stage train --dry-run
```

## 3. Run baselines and SFT conditions

- `Injury_LLM_SFT.py`: local QLoRA training and inference.
- `scripts/run_methods_condition.py`: configuration-driven local runner.
- `scripts/run_tfidf_logreg_baseline.py`: TF--IDF logistic-regression baseline.
- `scripts/run_gpt54_evaluation.py`: hosted GPT evaluation.
- `scripts/run_hosted_llama_evaluation.py`: generic OpenAI-compatible hosted
  Llama runner; the exact provider alias used in the paper was mutable.

The hosted-Llama script reads `HOSTED_LLM_ENDPOINT` and
`HOSTED_LLM_API_KEY` by default. No provider URL or credential is stored here.

## 4. Evaluate frozen predictions

`scripts/evaluate_methods.py` computes accuracy, macro precision/recall/F1,
diagnosis-stratified bootstrap intervals, and pairwise exact McNemar tests with
Holm correction. The test set may be accessed only after explicitly passing
`--confirm-locked-test`.

## 5. Validation analyses

- `run_prompt_component_ablation.py`: development-only prompt ablation.
- `run_cue_masking_analysis.py`: post-hoc cue masking.
- `analyze_difficult_contrasts.py`: predefined natural contrast subsets.
- `analyze_secondary_diagnosis.py`: exploratory secondary-diagnosis analysis.

These analyses are not part of the primary locked-test comparison. Exact prompt
artifacts are under `artifacts/prompts/`; aggregate outputs are under `results/`.

## 6. Recreate figures

```bash
python scripts/generate_publication_figures.py
```

The script reads only the released aggregate tables and writes SVG, PDF, and
600-DPI PNG outputs. The notebooks call the same figure functions so that the
script remains the single source of plotting logic.

## Scope of reproducibility

The repository permits exact inspection of code, settings, prompts, reported
aggregate metrics, paired tests, and per-code results. Full numerical reruns of
hosted systems remain subject to provider access and model availability. The
institutionally hosted Llama condition used a mutable alias, so its future
weights cannot be guaranteed to match the evaluated endpoint.
