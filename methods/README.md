# Methods

Data construction, experiment configuration, training, and prediction generation live here. Scoring, error studies, ablations, and figures live under [analysis](../analysis/README.md).

| Directory | Purpose |
|---|---|
| [data_preparation](data_preparation/README.md) | Reconstruct and verify temporal splits |
| [experiments](experiments/README.md) | Prepare demonstrations/configuration and run local conditions |
| [tfidf](tfidf/README.md) | Tune and evaluate TF-IDF logistic regression |
| [sft](sft/README.md) | Shared local generation and QLoRA training implementation |
| [roberta](roberta/README.md) | Encoder training, development selection, and comparisons |
| [gpt54](gpt54/README.md) | GPT-5.4 batch preparation, submission, and collection |
| [hosted_llama](hosted_llama/README.md) | Resumable hosted-endpoint evaluations |

The sequence is source verification, data preparation, development selection, frozen evaluation, and then [benchmark aggregation](../analysis/benchmark/README.md). Run documented commands from the repository root. Existing `methods_runs/` and study `local/` locations remain ignored storage for private data and model artifacts.

See [reproduction](../docs/REPRODUCIBILITY.md), [environments](../docs/ENVIRONMENT.md), and the [result map](../docs/RESULT_MAP.md).
