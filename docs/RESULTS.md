# Results and interpretation

See the [authoritative result map](RESULT_MAP.md) for source tables, commands, and required inputs.

## Main model comparison

All scores below use frozen models. The balanced and proportional tests each contain 1,500 records but are disjoint. The proportional sample approximates the unweighted 2025 source distribution; it is not a national prevalence estimate.

| Model | Balanced macro-F1 | Balanced accuracy | Proportional accuracy | Proportional weighted F1 |
|---|---:|---:|---:|---:|
| TF-IDF + logistic regression, 6K | 0.8745 | 0.8720 | 0.8147 | 0.8282 |
| RoBERTa-base, 6K, seed 20260718 | 0.8982 | 0.8980 | 0.8640 | 0.8725 |
| Qwen SFT, 6K | 0.9052 | 0.9067 | 0.9020 | 0.9071 |
| Mistral SFT, 3K | 0.8844 | 0.8853 | 0.8740 | 0.8814 |

Sources: [balanced baseline results](../results/primary/model_summary.csv), [encoder results](../methods/roberta/results/test_summary.json), and [proportional results with intervals](../analysis/robustness/proportional_test/results/metrics.csv). The baseline table contains all 15 original prompting, classical, and SFT conditions. Encoder runs supplement those conditions. Two additional encoder seeds are reported in the encoder and proportional result files; they are not the primary comparison.

RoBERTa recovers much of Qwen's balanced-test gain over TF-IDF with fewer parameters. This does not establish equivalence, a matched speedup, or lower end-to-end cost. Parameter counts and encoder-only timing are distinct from deployment measurements.

## Category-level performance

[Proportional category tables](../analysis/robustness/proportional_test/PER_CATEGORY_RESULTS.md) report precision, recall, F1, and support; [the CSV](../analysis/robustness/proportional_test/results/per_class.csv) retains full precision and confusion counts. Zero-support categories have unavailable recall/F1, and categories with no predictions have unavailable precision. Two categories have no proportional-test cases; three have only one. Read rare-category scores with their counts.

The balanced test supports comparisons across all 30 categories. Its macro-F1 is not directly comparable to supported-label macro-F1 over the proportional test's 28 observed classes. Accuracy and weighted F1 are the primary prevalence-sensitive summaries.

## Training size

The nested 1K/3K/6K SFT runs use one seed per condition. Mistral's balanced macro-F1 is 0.884 at 3K and 0.841 at 6K; this is a descriptive result, not evidence that additional data generally harms Mistral. Qwen's observed trend likewise does not establish a stable scaling relationship. Test-set bootstrap intervals do not quantify training variability or variation across independently sampled subsets.

## Diagnostic wording and co-occurring diagnoses

The [term-masking experiment](../analysis/robustness/diagnosis_masking/RESULTS.md) masks 1,565 spans in 1,091 of 1,500 narratives. Macro-F1 changes from 0.875 to 0.651 for TF-IDF, 0.905 to 0.631 for Qwen, and 0.884 to 0.634 for Mistral. The automated lexicon is non-exhaustive and not clinician-validated; masking can remove useful evidence and create distribution shift. These findings measure sensitivity, not internal reasoning.

[Secondary-diagnosis analysis](../analysis/error_analysis/secondary_diagnoses/README.md) separates 254 cases with distinct primary/secondary codes from 78 repeated-code cases. It measures primary, secondary, joint ordered predictions, and swaps; single-label classifiers provide primary predictions only. The full 332-case and distinct 254-case analyses answer different questions and are retained separately.

## Qualitative examples and audit

[Selected burn examples](../analysis/error_analysis/qualitative_errors/README.md) show both Qwen successes and failures. Qwen correctly classifies 49/50 electrical burns but only 18/50 thermal burns; higher overall performance does not imply dominance in every category. Cases were selected retrospectively and are not mechanistic or causal evidence.

[The 100-case audit](../analysis/data_quality/label_audit/README.md) records 93% observed agreement and kappa 0.593 for binary label appropriateness. Reviewer independence is confirmed; qualifications and preservation of pre-adjudication decisions in the edited worksheet remain undocumented. This is not independent diagnosis-code annotation, adjudicated correction, or a dataset-wide error estimate.

## Statistical scope

Baseline pairwise accuracy tests use exact McNemar tests with Holm correction over 105 comparisons. Encoder comparisons form a separate six-test family. Bootstrap intervals condition on fitted models; encoder seed SD describes a different source of uncertainty. No detected difference does not establish equivalence. Post-hoc analyses, unweighted sampling, and rare-class support limit deployment conclusions.
