# Proportional test: per-category performance

All six frozen model runs use the same 1,500-case proportional test. Values are fractions (0 to 1), rounded to three decimals; the CSV retains full precision.

- **Precision** = TP / (TP + FP): the fraction of predictions for this category that are correct.
- **Recall** = TP / (TP + FN): the fraction of actual cases in the category correctly classified.
- **F1** = 2TP / (2TP + FP + FN): combines precision and recall.
- **N** is the number of true cases. Recall and F1 are reported as N/A when N = 0. Precision is N/A when the model makes no predictions for a category. Estimates for very small categories are unstable.

The primary comparison uses RoBERTa seed **20260718**, matching TF-IDF, Qwen, and Mistral. The other two seeds are supplementary. The reporting seed was chosen after evaluation to align seeds, not to maximize test performance.

Source: `results/per_class.csv`, derived from recorded confusion counts. When local predictions are available, the generator also checks them. No retraining or new inference was performed. All three RoBERTa seeds are retained separately.

## TF-IDF 6K

| Code | Category | N | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| 41 | Ingestion | 19 | 1.000 | 0.895 | 0.944 |
| 42 | Aspiration | 1 | 0.200 | 1.000 | 0.333 |
| 46 | Electrical burn | 0 | 0.000 | N/A | N/A |
| 47 | Burn, not specified | 0 | 0.000 | N/A | N/A |
| 48 | Scald burn | 10 | 0.889 | 0.800 | 0.842 |
| 49 | Chemical burn | 2 | 0.667 | 1.000 | 0.800 |
| 50 | Amputation | 2 | 0.400 | 1.000 | 0.571 |
| 51 | Thermal burn | 11 | 0.800 | 0.727 | 0.762 |
| 52 | Concussion | 28 | 0.867 | 0.929 | 0.897 |
| 53 | Contusion/abrasion | 143 | 0.845 | 0.685 | 0.757 |
| 54 | Crushing injury | 3 | 0.333 | 1.000 | 0.500 |
| 55 | Dislocation | 23 | 0.700 | 0.913 | 0.792 |
| 56 | Foreign body | 30 | 0.800 | 0.933 | 0.862 |
| 57 | Fracture | 240 | 0.947 | 0.900 | 0.923 |
| 58 | Hematoma | 23 | 0.611 | 0.957 | 0.746 |
| 59 | Laceration | 214 | 0.930 | 0.874 | 0.901 |
| 60 | Dental injury | 8 | 0.636 | 0.875 | 0.737 |
| 61 | Nerve damage | 8 | 0.206 | 0.875 | 0.333 |
| 62 | Internal injury | 193 | 0.782 | 0.834 | 0.807 |
| 63 | Puncture | 9 | 0.750 | 1.000 | 0.857 |
| 64 | Strain/sprain | 113 | 0.939 | 0.823 | 0.877 |
| 65 | Anoxia | 4 | 0.667 | 1.000 | 0.800 |
| 66 | Hemorrhage | 5 | 0.263 | 1.000 | 0.417 |
| 67 | Electric shock | 1 | 1.000 | 1.000 | 1.000 |
| 68 | Poisoning | 14 | 0.317 | 0.929 | 0.473 |
| 69 | Submersion | 2 | 0.500 | 1.000 | 0.667 |
| 71 | Unknown/not stated | 371 | 0.890 | 0.701 | 0.784 |
| 72 | Avulsion | 13 | 0.611 | 0.846 | 0.710 |
| 73 | Radiation injury | 1 | 1.000 | 1.000 | 1.000 |
| 74 | Dermatitis/conjunctivitis | 9 | 0.692 | 1.000 | 0.818 |

## Qwen SFT 6K

| Code | Category | N | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| 41 | Ingestion | 19 | 1.000 | 0.895 | 0.944 |
| 42 | Aspiration | 1 | 0.333 | 1.000 | 0.500 |
| 46 | Electrical burn | 0 | N/A | N/A | N/A |
| 47 | Burn, not specified | 0 | 0.000 | N/A | N/A |
| 48 | Scald burn | 10 | 0.900 | 0.900 | 0.900 |
| 49 | Chemical burn | 2 | 0.667 | 1.000 | 0.800 |
| 50 | Amputation | 2 | 0.667 | 1.000 | 0.800 |
| 51 | Thermal burn | 11 | 0.667 | 0.182 | 0.286 |
| 52 | Concussion | 28 | 0.871 | 0.964 | 0.915 |
| 53 | Contusion/abrasion | 143 | 0.944 | 0.937 | 0.940 |
| 54 | Crushing injury | 3 | 0.500 | 1.000 | 0.667 |
| 55 | Dislocation | 23 | 0.821 | 1.000 | 0.902 |
| 56 | Foreign body | 30 | 0.833 | 1.000 | 0.909 |
| 57 | Fracture | 240 | 0.964 | 1.000 | 0.982 |
| 58 | Hematoma | 23 | 0.733 | 0.957 | 0.830 |
| 59 | Laceration | 214 | 0.980 | 0.935 | 0.957 |
| 60 | Dental injury | 8 | 0.667 | 1.000 | 0.800 |
| 61 | Nerve damage | 8 | 0.364 | 1.000 | 0.533 |
| 62 | Internal injury | 193 | 0.859 | 0.855 | 0.857 |
| 63 | Puncture | 9 | 0.818 | 1.000 | 0.900 |
| 64 | Strain/sprain | 113 | 0.953 | 0.903 | 0.927 |
| 65 | Anoxia | 4 | 0.800 | 1.000 | 0.889 |
| 66 | Hemorrhage | 5 | 0.625 | 1.000 | 0.769 |
| 67 | Electric shock | 1 | 1.000 | 1.000 | 1.000 |
| 68 | Poisoning | 14 | 0.545 | 0.857 | 0.667 |
| 69 | Submersion | 2 | 0.667 | 1.000 | 0.800 |
| 71 | Unknown/not stated | 371 | 0.971 | 0.819 | 0.889 |
| 72 | Avulsion | 13 | 1.000 | 0.846 | 0.917 |
| 73 | Radiation injury | 1 | 1.000 | 1.000 | 1.000 |
| 74 | Dermatitis/conjunctivitis | 9 | 0.529 | 1.000 | 0.692 |

## Mistral SFT 3K

| Code | Category | N | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| 41 | Ingestion | 19 | 0.900 | 0.947 | 0.923 |
| 42 | Aspiration | 1 | 0.200 | 1.000 | 0.333 |
| 46 | Electrical burn | 0 | 0.000 | N/A | N/A |
| 47 | Burn, not specified | 0 | 0.000 | N/A | N/A |
| 48 | Scald burn | 10 | 0.667 | 1.000 | 0.800 |
| 49 | Chemical burn | 2 | 0.400 | 1.000 | 0.571 |
| 50 | Amputation | 2 | 0.500 | 1.000 | 0.667 |
| 51 | Thermal burn | 11 | 1.000 | 0.455 | 0.625 |
| 52 | Concussion | 28 | 0.757 | 1.000 | 0.862 |
| 53 | Contusion/abrasion | 143 | 0.926 | 0.881 | 0.903 |
| 54 | Crushing injury | 3 | 0.750 | 1.000 | 0.857 |
| 55 | Dislocation | 23 | 0.917 | 0.957 | 0.936 |
| 56 | Foreign body | 30 | 0.871 | 0.900 | 0.885 |
| 57 | Fracture | 240 | 0.971 | 0.963 | 0.967 |
| 58 | Hematoma | 23 | 0.759 | 0.957 | 0.846 |
| 59 | Laceration | 214 | 0.975 | 0.921 | 0.947 |
| 60 | Dental injury | 8 | 0.889 | 1.000 | 0.941 |
| 61 | Nerve damage | 8 | 0.700 | 0.875 | 0.778 |
| 62 | Internal injury | 193 | 0.757 | 0.886 | 0.816 |
| 63 | Puncture | 9 | 0.643 | 1.000 | 0.783 |
| 64 | Strain/sprain | 113 | 0.947 | 0.947 | 0.947 |
| 65 | Anoxia | 4 | 0.667 | 1.000 | 0.800 |
| 66 | Hemorrhage | 5 | 0.500 | 1.000 | 0.667 |
| 67 | Electric shock | 1 | 1.000 | 1.000 | 1.000 |
| 68 | Poisoning | 14 | 0.244 | 0.714 | 0.364 |
| 69 | Submersion | 2 | 0.667 | 1.000 | 0.800 |
| 71 | Unknown/not stated | 371 | 0.972 | 0.739 | 0.839 |
| 72 | Avulsion | 13 | 0.667 | 0.769 | 0.714 |
| 73 | Radiation injury | 1 | 1.000 | 1.000 | 1.000 |
| 74 | Dermatitis/conjunctivitis | 9 | 0.800 | 0.889 | 0.842 |

## RoBERTa (seed 20260718)

| Code | Category | N | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| 41 | Ingestion | 19 | 1.000 | 0.947 | 0.973 |
| 42 | Aspiration | 1 | 0.250 | 1.000 | 0.400 |
| 46 | Electrical burn | 0 | N/A | N/A | N/A |
| 47 | Burn, not specified | 0 | 0.000 | N/A | N/A |
| 48 | Scald burn | 10 | 0.889 | 0.800 | 0.842 |
| 49 | Chemical burn | 2 | 1.000 | 1.000 | 1.000 |
| 50 | Amputation | 2 | 0.500 | 1.000 | 0.667 |
| 51 | Thermal burn | 11 | 0.875 | 0.636 | 0.737 |
| 52 | Concussion | 28 | 0.900 | 0.964 | 0.931 |
| 53 | Contusion/abrasion | 143 | 0.964 | 0.741 | 0.838 |
| 54 | Crushing injury | 3 | 0.600 | 1.000 | 0.750 |
| 55 | Dislocation | 23 | 1.000 | 0.957 | 0.978 |
| 56 | Foreign body | 30 | 0.763 | 0.967 | 0.853 |
| 57 | Fracture | 240 | 0.967 | 0.979 | 0.973 |
| 58 | Hematoma | 23 | 0.733 | 0.957 | 0.830 |
| 59 | Laceration | 214 | 0.971 | 0.930 | 0.950 |
| 60 | Dental injury | 8 | 0.533 | 1.000 | 0.696 |
| 61 | Nerve damage | 8 | 0.333 | 1.000 | 0.500 |
| 62 | Internal injury | 193 | 0.793 | 0.855 | 0.823 |
| 63 | Puncture | 9 | 0.600 | 1.000 | 0.750 |
| 64 | Strain/sprain | 113 | 0.945 | 0.912 | 0.928 |
| 65 | Anoxia | 4 | 0.571 | 1.000 | 0.727 |
| 66 | Hemorrhage | 5 | 0.333 | 1.000 | 0.500 |
| 67 | Electric shock | 1 | 1.000 | 1.000 | 1.000 |
| 68 | Poisoning | 14 | 0.400 | 0.857 | 0.545 |
| 69 | Submersion | 2 | 0.400 | 1.000 | 0.571 |
| 71 | Unknown/not stated | 371 | 0.929 | 0.744 | 0.826 |
| 72 | Avulsion | 13 | 0.571 | 0.923 | 0.706 |
| 73 | Radiation injury | 1 | 0.333 | 1.000 | 0.500 |
| 74 | Dermatitis/conjunctivitis | 9 | 0.562 | 1.000 | 0.720 |

## Supplementary seed runs

These runs are retained for robustness; seed 20260718 is used in the primary comparison.

## RoBERTa (seed 20260719)

| Code | Category | N | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| 41 | Ingestion | 19 | 1.000 | 0.895 | 0.944 |
| 42 | Aspiration | 1 | 0.125 | 1.000 | 0.222 |
| 46 | Electrical burn | 0 | N/A | N/A | N/A |
| 47 | Burn, not specified | 0 | 0.000 | N/A | N/A |
| 48 | Scald burn | 10 | 0.889 | 0.800 | 0.842 |
| 49 | Chemical burn | 2 | 0.667 | 1.000 | 0.800 |
| 50 | Amputation | 2 | 0.500 | 1.000 | 0.667 |
| 51 | Thermal burn | 11 | 1.000 | 0.818 | 0.900 |
| 52 | Concussion | 28 | 0.929 | 0.929 | 0.929 |
| 53 | Contusion/abrasion | 143 | 0.933 | 0.678 | 0.785 |
| 54 | Crushing injury | 3 | 0.429 | 1.000 | 0.600 |
| 55 | Dislocation | 23 | 0.846 | 0.957 | 0.898 |
| 56 | Foreign body | 30 | 0.735 | 0.833 | 0.781 |
| 57 | Fracture | 240 | 0.975 | 0.975 | 0.975 |
| 58 | Hematoma | 23 | 0.656 | 0.913 | 0.764 |
| 59 | Laceration | 214 | 0.965 | 0.897 | 0.930 |
| 60 | Dental injury | 8 | 0.571 | 1.000 | 0.727 |
| 61 | Nerve damage | 8 | 0.533 | 1.000 | 0.696 |
| 62 | Internal injury | 193 | 0.769 | 0.881 | 0.821 |
| 63 | Puncture | 9 | 0.429 | 1.000 | 0.600 |
| 64 | Strain/sprain | 113 | 0.912 | 0.912 | 0.912 |
| 65 | Anoxia | 4 | 0.500 | 1.000 | 0.667 |
| 66 | Hemorrhage | 5 | 0.250 | 1.000 | 0.400 |
| 67 | Electric shock | 1 | 0.500 | 1.000 | 0.667 |
| 68 | Poisoning | 14 | 0.324 | 0.857 | 0.471 |
| 69 | Submersion | 2 | 0.333 | 1.000 | 0.500 |
| 71 | Unknown/not stated | 371 | 0.937 | 0.717 | 0.812 |
| 72 | Avulsion | 13 | 0.667 | 0.923 | 0.774 |
| 73 | Radiation injury | 1 | 1.000 | 1.000 | 1.000 |
| 74 | Dermatitis/conjunctivitis | 9 | 0.600 | 1.000 | 0.750 |

## RoBERTa (seed 20260720)

| Code | Category | N | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| 41 | Ingestion | 19 | 1.000 | 0.947 | 0.973 |
| 42 | Aspiration | 1 | 0.500 | 1.000 | 0.667 |
| 46 | Electrical burn | 0 | N/A | N/A | N/A |
| 47 | Burn, not specified | 0 | 0.000 | N/A | N/A |
| 48 | Scald burn | 10 | 0.818 | 0.900 | 0.857 |
| 49 | Chemical burn | 2 | 0.667 | 1.000 | 0.800 |
| 50 | Amputation | 2 | 0.500 | 1.000 | 0.667 |
| 51 | Thermal burn | 11 | 0.889 | 0.727 | 0.800 |
| 52 | Concussion | 28 | 0.818 | 0.964 | 0.885 |
| 53 | Contusion/abrasion | 143 | 0.920 | 0.727 | 0.812 |
| 54 | Crushing injury | 3 | 0.600 | 1.000 | 0.750 |
| 55 | Dislocation | 23 | 0.846 | 0.957 | 0.898 |
| 56 | Foreign body | 30 | 0.784 | 0.967 | 0.866 |
| 57 | Fracture | 240 | 0.975 | 0.958 | 0.966 |
| 58 | Hematoma | 23 | 0.611 | 0.957 | 0.746 |
| 59 | Laceration | 214 | 0.947 | 0.921 | 0.934 |
| 60 | Dental injury | 8 | 0.533 | 1.000 | 0.696 |
| 61 | Nerve damage | 8 | 0.381 | 1.000 | 0.552 |
| 62 | Internal injury | 193 | 0.768 | 0.839 | 0.802 |
| 63 | Puncture | 9 | 0.692 | 1.000 | 0.818 |
| 64 | Strain/sprain | 113 | 0.962 | 0.894 | 0.927 |
| 65 | Anoxia | 4 | 0.667 | 1.000 | 0.800 |
| 66 | Hemorrhage | 5 | 0.357 | 1.000 | 0.526 |
| 67 | Electric shock | 1 | 1.000 | 1.000 | 1.000 |
| 68 | Poisoning | 14 | 0.522 | 0.857 | 0.649 |
| 69 | Submersion | 2 | 0.500 | 1.000 | 0.667 |
| 71 | Unknown/not stated | 371 | 0.933 | 0.784 | 0.852 |
| 72 | Avulsion | 13 | 0.706 | 0.923 | 0.800 |
| 73 | Radiation injury | 1 | 0.500 | 1.000 | 0.667 |
| 74 | Dermatitis/conjunctivitis | 9 | 0.692 | 1.000 | 0.818 |
