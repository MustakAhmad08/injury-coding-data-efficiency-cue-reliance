# Difficult Diagnosis Contrast Analysis

## Scope

This analysis evaluates five prespecified diagnosis pairs using the naturally
occurring locked-test cases (50 cases per code). It is a **post hoc natural-case
contrast audit**, not a controlled synthetic minimal-pair experiment. The
50-case reviewer worksheet remains unvalidated until a qualified co-author
completes its review fields.

## Focus-model results

| Contrast | Qwen SFT 6K | Mistral SFT 3K | TF-IDF logistic | GPT-5.4 10-shot |
|---|---:|---:|---:|---:|
| Scald burn vs Thermal burn | 0.660 [0.590, 0.730] | 0.760 [0.690, 0.830] | 0.820 [0.740, 0.890] | 0.890 [0.830, 0.950] |
| Electrical burn vs Electric shock | 0.960 [0.920, 0.990] | 0.920 [0.870, 0.970] | 0.880 [0.810, 0.940] | 0.920 [0.860, 0.970] |
| Concussion vs Internal injury | 0.880 [0.820, 0.940] | 0.930 [0.880, 0.970] | 0.900 [0.840, 0.950] | 0.800 [0.730, 0.860] |
| Laceration vs Avulsion | 0.950 [0.910, 0.990] | 0.940 [0.900, 0.980] | 0.930 [0.880, 0.970] | 0.960 [0.920, 0.990] |
| Ingestion vs Aspiration | 0.890 [0.830, 0.950] | 0.940 [0.890, 0.980] | 0.900 [0.840, 0.950] | 0.850 [0.780, 0.910] |

Accuracy treats predictions outside the two target codes as errors. The full
CSV also reports directional recall, macro F1, direct cross-pair errors, and
other errors for all 15 conditions.

## Mean accuracy across the five contrast subsets

| Model | Mean contrast accuracy |
|---|---:|
| Qwen SFT 6K | 0.868 |
| Mistral SFT 3K | 0.898 |
| TF-IDF + logistic regression | 0.886 |
| GPT-5.4 10-shot | 0.884 |

## Interpretation boundary

- These subsets identify performance on difficult category pairs but do not
  prove sensitivity to a single changed linguistic feature.
- Do not call this a validated minimal-pair experiment until the review
  worksheet is completed and a genuinely controlled contrast set is created.
- Because the locked-test results were already examined, report this as post
  hoc exploratory analysis.

## Reproducibility

- Seed: `20260718`
- Bootstrap: `1000` diagnosis-stratified resamples per pair
- Source predictions: the frozen `test_results.csv` files for all 15
  primary-diagnosis conditions
