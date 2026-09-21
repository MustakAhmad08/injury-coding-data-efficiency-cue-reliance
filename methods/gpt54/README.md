# GPT-5.4 batch evaluation

Produce frozen zero-shot and ten-shot predictions through the Batch API.

Run commands from the repository root.

| Entry point | Purpose |
|---|---|
| [run_gpt54_evaluation.py](run_gpt54_evaluation.py) | Prepare, submit, inspect, or collect zero-shot/ten-shot batch jobs. |
| [run_gpt54_10shot_chunks.py](run_gpt54_10shot_chunks.py) | Split a prepared ten-shot batch into 250-case chunks and submit or collect them sequentially. |
| [orchestrate_gpt54_batches.py](orchestrate_gpt54_batches.py) | Monitor the prepared zero-shot job, then coordinate the six ten-shot chunks. |

Job state, private responses, and predictions stay under `methods_runs/gpt54_*/`. These are cooperating entry points, not three separate experiments. The orchestrator currently uses Windows registry support for its credential fallback; on other platforms use the individual batch commands. Submission and orchestration can make paid API requests.

[Published outputs / reference](../../results/primary) ? [Reproduction commands](../../docs/REPRODUCIBILITY.md) ? [Authoritative result map](../../docs/RESULT_MAP.md)
