# Qualitative error analysis

[All analyses](../../README.md) / [Error analysis](../README.md)

Four frozen models are compared on the original balanced test: TF-IDF 6K, RoBERTa 6K (seed 20260718), Qwen 6K, and Mistral 3K. Descriptions are paraphrases; models received original narratives. Examples were selected retrospectively and were not independently adjudicated.

## Thermal burns

| Example | Narrative summary | Dataset gold label | TF-IDF | RoBERTa | Qwen | Mistral |
|---|---|---|---|---|---|---|
| A | A child burned a palm after touching a hot pan; the diagnosis was recorded simply as a burn. | Thermal burn (51) | Thermal burn (51), correct | Thermal burn (51), correct | Thermal burn (51), correct | Scald (48), incorrect |
| B | A child touched a hot clothes iron; the diagnosis described a partial-thickness burn of the left palm. | Thermal burn (51) | Thermal burn (51), correct | Thermal burn (51), correct | Unspecified burn (47), incorrect | Electrical burn (46), incorrect |

In Example A, three models matched the thermal-burn label, whereas Mistral assigned scald despite the narrative describing contact with a hot solid object rather than hot liquid or steam. Example B shows different errors on the same narrative: Qwen assigned an unspecified burn, while Mistral assigned an electrical burn. The narrative describes contact with a hot iron but does not state an electric shock or current exposure. These predictions illustrate two different error types relative to the dataset labels: failure to preserve the burn subtype and assignment of a different injury mechanism. TF-IDF and RoBERTa matched the gold label in both examples.

The examples are consistent with differences across all 50 thermal-burn cases. TF-IDF correctly classified 39/50 (recall 0.780), RoBERTa 36/50 (0.720), Qwen 18/50 (0.360), and Mistral 27/50 (0.540). Qwen's most frequent error was unspecified burn (29 cases), whereas Mistral's was scald (15 cases). Thus, the models share a difficulty distinguishing neighboring burn categories, but they differ in which labels they substitute; a higher overall score does not imply better performance in every category.

These examples were selected retrospectively to illustrate contrasting predictions, not to estimate the prevalence of error mechanisms. We interpret only the observed outputs: the records do not provide reliable evidence of the models' internal reasoning, and these cases were not independently adjudicated. The gold labels remain the original dataset labels.


## Electrical burns

| Example | Narrative summary | Gold label | TF-IDF | RoBERTa | Qwen | Mistral |
|---|---|---|---|---|---|---|---|
| A | A child received a shock while unplugging a charging cord, burning a hand. The diagnosis stated low-voltage electrical shock. | Electrical burn (46) | Electric shock (67) | Electric shock (67) | Electrical burn (46), correct | Electric shock (67) |
| B | A person sustained a facial flash burn while changing a subpanel breaker box. The diagnosis stated partial-thickness facial burn. | Electrical burn (46) | Thermal burn (51) | Thermal burn (51) | Electrical burn (46), correct | Radiation injury (73) |

### Interpretation

We examined two cases from the electrical-burn category to compare model predictions on the same narratives. In the first case, a child received a shock while unplugging a charging cord and burned a hand. Although the diagnosis phrase stated electrical shock, the dataset gold code was electrical burn. Qwen matched this code, whereas TF-IDF, RoBERTa, and Mistral all predicted electric shock. This example illustrates a shared error among the three comparison models when shock and burn descriptions co-occur.

In the second case, a person sustained a facial flash burn while changing a subpanel breaker box. Qwen again matched the electrical-burn gold label. TF-IDF and RoBERTa predicted thermal burn, while Mistral predicted radiation injury. Here, the comparison models differed in which alternative category they assigned to the same narrative. Qwen's prediction was consistent with the electrical context and recorded gold label, but the output alone does not establish how the model used that information.

Across all 50 electrical-burn cases, Qwen correctly classified 49 (recall 0.980), compared with 44 for TF-IDF (0.880), 43 for RoBERTa (0.860), and 44 for Mistral (0.880). These category-level results provide context for the selected examples. The examples describe observed output differences and are not proof of contextual reasoning. They were selected retrospectively, and no independent adjudication of their gold labels was conducted.

## Selection and verification

A search of all 1,500 balanced-test cases found 15 where Qwen matched the gold primary diagnosis and all three comparison models did not. Two of these had gold code 46; both are shown above. Case identifiers and the input-file hashes are retained in the local verification record, rather than this shareable report. These examples complement the previously prepared thermal-burn examples, where Qwen was less accurate at the category level. Neither set alone describes the entire benchmark.

