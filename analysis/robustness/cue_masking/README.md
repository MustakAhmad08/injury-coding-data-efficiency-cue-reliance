# Marker and diagnosis-tail cue masking

Measure score changes after removing formatting markers or diagnosis tails.

Run commands from the repository root.

| Entry point | Purpose |
|---|---|
| [run_cue_masking_analysis.py](run_cue_masking_analysis.py) | Run `--stage prepare`, `tfidf`, `local`, or `report` to construct masked inputs, evaluate frozen models, or summarize predictions. |

Generated inputs, predictions, and summaries stay in `methods_runs/cue_masking/`. Adapters default to `methods_runs/{condition}`; set `NEISS_ADAPTER_ROOT` to another adapter root. This intervention differs from lexicon-based diagnosis-term masking in `analysis/robustness/diagnosis_masking/`.

[Published outputs / reference](../../../results/cue_masking) ? [Reproduction commands](../../../docs/REPRODUCIBILITY.md) ? [Authoritative result map](../../../docs/RESULT_MAP.md)

The camera-ready [RoBERTa masking extension](../roberta_masking/README.md) reports all three selected seeds and is kept separate from the original three-model snapshots.
