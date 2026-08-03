# Released results

All files in this directory are aggregate or diagnosis-level artifacts without
injury narratives or raw model responses.

- `primary/`: overall metrics, bootstrap confidence intervals, paired exact
  McNemar tests, and per-diagnosis precision/recall/F1.
- `prompt_ablation/`: development-only prompt-component ablation.
- `cue_masking/`: post-hoc masking summaries.
- `contrasts/`: natural contrast subset metrics.
- `secondary/`: exploratory secondary-diagnosis summaries.
- `error_analysis/`: aggregate error-pair counts and review protocol.

Condition identifiers use `hosted_llama_*` for the anonymous institutionally
hosted endpoint. This label is presentation-only and does not imply a stable
public model checkpoint.
