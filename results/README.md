# Released results

See the [authoritative result map](../docs/RESULT_MAP.md) for source tables, commands, and required inputs.

All files in this directory are aggregate or diagnosis-level artifacts without
injury narratives or raw model responses.

- `primary/`: overall metrics, bootstrap confidence intervals, paired exact
  McNemar tests, and per-diagnosis precision/recall/F1.
- `prompt_ablation/`: development-only prompt-component ablation.
- `cue_masking/`: post-hoc masking summaries.
- `contrasts/`: diagnosis-pair subset metrics, including encoder accuracies.
- `secondary_all_cases/`: exploratory secondary-diagnosis summaries.
- `error_pairs/`: aggregate error-pair counts and review protocol.

Condition identifiers use `hosted_llama_*` for the institutionally
hosted endpoint. This label is presentation-only and does not imply a stable
public model checkpoint.
