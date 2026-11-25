# Fine-tuned RoBERTa-base

All 124,668,702 parameters are fine-tuned with a 30-class head on the shared 6K training split. `config.json` contains the model commit, grid, seeds, and optimization settings. Checkpoints maximize development macro-F1; the learning rate maximizes the mean across three development-selected runs. See [settings](../../docs/EXPERIMENT_SETTINGS.md).

Primary reporting uses seed 20260718: balanced macro-F1 **0.898196**, accuracy **0.898000**. Seeds 20260719/20 are supplementary; all remain in the result files. Selection used learning rate 5e-5 and epochs 5/5/4. No checkpoint or learning rate was chosen on test performance.

```bash
python methods/roberta/prepare_model.py
python methods/roberta/run_encoder.py train
python methods/roberta/run_encoder.py final-eval --confirm-locked-test
python methods/roberta/analyze_results.py
```

Data reconstruction and the TF-IDF/Qwen predictions are prerequisites for paired comparisons. Local weights/predictions are not included. `results/validation_selection.json` records the selected checkpoints; `test_summary.json` records per-seed scores and encoder-only timing; `comparison_metrics.csv` and `paired_comparisons.csv` provide reference comparisons and uncertainty. Timings exclude model loading/tokenization and are not matched end-to-end speed comparisons with Qwen.

The camera-ready [RoBERTa masking extension](../../analysis/robustness/roberta_masking/README.md) reports all three selected seeds and is kept separate from the original three-model snapshots.
