# Prompt artifacts

These files preserve the system prompts, operational rules, codebook content,
ordered demonstration messages, and SHA-256 manifests used in the 2024
development ablation.

The operational-rules and demonstrations conditions intentionally share the
same system prompt. The experimental difference is the additional ordered
user/assistant message sequence in `demonstration_messages.jsonl`; comparing
only the system-prompt text would therefore omit the treatment difference.
