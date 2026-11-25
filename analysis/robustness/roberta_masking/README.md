# RoBERTa masking (camera-ready addition)

Frozen RoBERTa-base checkpoints were evaluated post hoc under formatting-marker removal,
diagnosis-tail removal, and diagnosis-term masking. Seeds 20260718/19/20 are all reported;
20260718 is the primary comparison. The other model results remain in the
[cue study](../cue_masking/README.md) and [term study](../diagnosis_masking/README.md).

## Published results

[Summary](results/roberta_masking_summary.csv) has all nine seed/intervention rows.
[Checks](results/roberta_masking_checks.json) records the original input, checkpoint,
prediction, and experimental-script hashes and software environment. Each seed reproduced
all 1,500 archived unmasked predictions; all 409 narratives unchanged by term masking
had identical predictions. These are recorded experiment checks, not a new release-time
GPU replication.

For the primary seed: marker change is -0.003; tail-masked macro-F1 is 0.465
(change -0.433, CI [-0.459,-0.411]); term-masked macro-F1 is 0.519
(drop 0.379, CI [0.362,0.400]). Distinct-254 primary accuracy changes from
0.787 to 0.445 under term masking. CSV drops use original minus masked;
the cue figure uses masked minus original. Differences precede rounding.

## Reproduction

Follow the [data and encoder instructions](../../../methods/roberta/README.md)
to reconstruct the original split and obtain frozen checkpoints and per-seed unmasked
predictions. Narrative inputs, weights and case-level predictions are not bundled.
Commands run from the repository root:

```bash
python analysis/robustness/cue_masking/run_cue_masking_analysis.py --stage prepare
python analysis/robustness/diagnosis_masking/run_test.py prepare
python analysis/robustness/roberta_masking/run_roberta_masking.py --help
python analysis/robustness/roberta_masking/run_roberta_masking.py
```

The runner uses `methods/roberta/local/runs/` by default; override with
`--runs` or `NEISS_ROBERTA_RUNS`. Use `--original-predictions` for a different
directory containing `seed_20260718.csv`, `seed_20260719.csv`, and `seed_20260720.csv`.
CUDA, the recorded scientific dependencies, and all four narrative inputs are needed.
Outputs default to ignored `methods_runs/roberta_masking/`; published snapshots are
not overwritten. By default checkpoint hashes must match the original experiment.
For independently retrained checkpoints, use `--allow-new-checkpoints` with their
corresponding unmasked predictions and interpret outputs as a separate replication.

The [original script](provenance/original_run_roberta_masking.py.txt) is retained
verbatim as provenance, not as a portable entry point. Its hash matches the recorded
experiment. The adapted runner has a separate hash in
[release adaptation metadata](provenance/release_adaptation.json); relocation does
not change the recorded experimental hash.

## Figure from aggregates only

```bash
python analysis/robustness/roberta_masking/make_cue_masking_figure.py
```

This uses included aggregate CSVs and plotting dependencies, not private predictions
or another checkout. Run the full publication-figure command to refresh the manifest.
Masking removes evidence and changes the input distribution; larger drops do not
establish the internal mechanism of prediction.
