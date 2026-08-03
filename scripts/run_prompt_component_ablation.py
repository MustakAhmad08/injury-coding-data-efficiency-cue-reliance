"""Nested prompt-component ablation on the 2024 development set.

This script never reads the locked test set. It evaluates base Mistral and Qwen
with: minimal instructions, codebook definitions, current operational rules,
and ten fixed demonstrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Injury_LLM_SFT import (
    DIAGNOSIS_CODES,
    load_fewshot_examples,
    load_model_and_tokenizer,
    parse_model_response,
    render_chat_template,
)


VALIDATION = ROOT / "data" / "neiss_temporal_2021_2025" / "validation.csv"
FEWSHOT = ROOT / "data" / "neiss_temporal_2021_2025" / "fewshot_10.jsonl"
OUT = ROOT / "methods_runs" / "prompt_component_ablation"
SEED = 20260718
N_BOOT = 1000
MODELS = {
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B"),
    "qwen": ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
}
ORDER = ("minimal", "codebook", "operational_rules", "demonstrations")

MINIMAL = """You are a NEISS injury coder. Read the emergency-department narrative and
assign its primary NEISS diagnosis code. Respond only with valid JSON:
{"Diagnosis_1_Code": <integer>}"""

RULES = """Apply these current operational coding rules:
- If multiple injuries are present, choose the most severe as the primary diagnosis.
- Do not code unrelated illnesses, behavioral disorders, or symptoms already captured by an injury.
- If only a probable, possible, or rule-out diagnosis is available, code that diagnosis.
- If no injury is stated, use code 71.
- Concussion is 52; closed head injury, subdural hematoma, cerebral contusion, and intracranial injury are 62.
- Hot liquid or steam burns are 48; flame or hot-surface burns are 51; chemical burns are 49; electrical burns are 46.
- Standard cuts are 59; degloving and skin tears are 72.
- Choking without other information is aspiration, code 42.

These are the study's operational rules; their line-by-line provenance against
the official NEISS manual has not yet been independently verified."""


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("run", "report"))
    parser.add_argument("--condition", choices=tuple(MODELS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompts() -> dict[str, str]:
    codebook = "\n".join(f"- {code}: {label}" for code, label in DIAGNOSIS_CODES.items())
    return {
        "minimal": MINIMAL,
        "codebook": f"{MINIMAL}\n\nAllowed codebook:\n{codebook}",
        "operational_rules": f"{MINIMAL}\n\nAllowed codebook:\n{codebook}\n\n{RULES}",
        "demonstrations": f"{MINIMAL}\n\nAllowed codebook:\n{codebook}\n\n{RULES}",
    }


def build_messages(
    condition: str, system: str, narrative: str, demonstrations: list[dict]
) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    if condition == "demonstrations":
        for example in demonstrations:
            code = int(example["diagnosis_code"])
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": f"Injury Narrative: {example['narrative']}",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"Diagnosis_1_Code": code}),
                    },
                ]
            )
    messages.append({"role": "user", "content": f"Injury Narrative: {narrative}"})
    return messages


def write_prompt_artifacts(
    destination: Path, variants: dict[str, str], demonstrations: list[dict]
) -> None:
    """Save system prompts and the separate few-shot chat turns unambiguously."""
    for name, value in variants.items():
        (destination / f"system_prompt_{name}.txt").write_text(
            value + "\n", encoding="utf-8"
        )

    demo_path = destination / "demonstration_messages.jsonl"
    with demo_path.open("w", encoding="utf-8") as stream:
        for order, example in enumerate(demonstrations, start=1):
            code = int(example["diagnosis_code"])
            record = {
                "order": order,
                "case_id": int(example["case_id"]),
                "messages": [
                    {
                        "role": "user",
                        "content": f"Injury Narrative: {example['narrative']}",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"Diagnosis_1_Code": code}),
                    },
                ],
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    artifact_manifest = {
        "system_prompt_files": {
            name: {
                "path": f"system_prompt_{name}.txt",
                "sha256": sha256(destination / f"system_prompt_{name}.txt"),
            }
            for name in ORDER
        },
        "demonstration_messages": {
            "path": demo_path.name,
            "sha256": sha256(demo_path),
            "count": len(demonstrations),
            "ordered_case_ids": [int(item["case_id"]) for item in demonstrations],
        },
        "implementation_note": (
            "The demonstrations condition intentionally shares the operational-rules "
            "system prompt. Its additional information is stored as ten ordered "
            "user/assistant message pairs in demonstration_messages.jsonl."
        ),
    }
    (destination / "prompt_artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8"
    )


def run(args_ns: argparse.Namespace) -> None:
    if not args_ns.condition:
        raise ValueError("--condition is required for --stage run")
    model_id, label = MODELS[args_ns.condition]
    destination = OUT / args_ns.condition
    destination.mkdir(parents=True, exist_ok=True)
    variants = prompts()
    for name, value in variants.items():
        (destination / f"prompt_{name}.txt").write_text(value + "\n", encoding="utf-8")
    demonstrations = load_fewshot_examples(FEWSHOT)
    if len(demonstrations) != 10:
        raise ValueError("Expected exactly ten fixed demonstrations")
    write_prompt_artifacts(destination, variants, demonstrations)
    pending = [
        name
        for name in ORDER
        if args_ns.overwrite or not (destination / name / "validation_results.csv").exists()
    ]
    if not pending:
        print(f"All prompt conditions already exist for {args_ns.condition}")
        return

    frame = pd.read_csv(VALIDATION)
    if len(frame) != 300 or set(frame["year"].astype(int)) != {2024}:
        raise ValueError("Expected the frozen 300-case 2024 validation set")
    model, tokenizer = load_model_and_tokenizer(model_id)
    model.eval()
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for condition in pending:
        rendered = [
            render_chat_template(
                tokenizer,
                build_messages(condition, variants[condition], str(narrative), demonstrations),
                add_generation_prompt=True,
            )
            for narrative in frame["narrative"]
        ]
        predictions, raw_outputs, parse_errors = [], [], []
        for start in range(0, len(rendered), 8):
            batch = tokenizer(
                rendered[start : start + 8], return_tensors="pt", padding=True
            ).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **batch,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            width = batch.input_ids.shape[-1]
            for output in generated:
                raw = tokenizer.decode(output[width:], skip_special_tokens=True)
                parsed = parse_model_response(raw)
                try:
                    code = int(float(parsed.get("Diagnosis_1_Code")))
                except (TypeError, ValueError):
                    code = -1
                predictions.append(code)
                raw_outputs.append(raw)
                parse_errors.append(code == -1)
            completed = min(start + 8, len(rendered))
            if completed % 100 == 0 or completed == len(rendered):
                print(f"{args_ns.condition}/{condition}: {completed}/{len(rendered)}")
        output = pd.DataFrame(
            {
                "case_id": frame["case_id"],
                "year": frame["year"],
                "true_diag1": frame["diag1"].astype(int),
                "pred_diag1": predictions,
                "diag1_correct": frame["diag1"].astype(int).to_numpy()
                == np.asarray(predictions),
                "parse_error": parse_errors,
                "raw_output": raw_outputs,
            }
        )
        target = destination / condition
        target.mkdir(parents=True, exist_ok=True)
        output.to_csv(target / "validation_results.csv", index=False)
        print(
            f"{label}/{condition}: accuracy={output['diag1_correct'].mean():.4f}, "
            f"parse errors={output['parse_error'].mean():.4f}"
        )
    tokenizer.padding_side = old_padding
    metadata = {
        "model_family": args_ns.condition,
        "model_id": model_id,
        "model_label": label,
        "split": "2024 validation/development only",
        "n_cases": len(frame),
        "validation_sha256": sha256(VALIDATION),
        "fewshot_sha256": sha256(FEWSHOT),
        "conditions": list(ORDER),
        "prompt_sha256": {
            name: sha256(destination / f"prompt_{name}.txt") for name in ORDER
        },
        "seed": SEED,
        "decoding": {"do_sample": False, "max_new_tokens": 32, "batch_size": 8},
        "provenance_warning": (
            "The operational-rule condition uses current study rules. Independent "
            "line-by-line verification against the official manual is pending."
        ),
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def metric(gold: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    labels = sorted(np.unique(gold).tolist())
    return (
        float(accuracy_score(gold, pred)),
        float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
    )


def bootstrap(gold: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(gold == label) for label in sorted(np.unique(gold))]
    scores = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        scores.append(metric(gold[idx], pred[idx])[1])
    return float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


def report(_: argparse.Namespace) -> None:
    rows, comparisons = [], []
    for family, (_, model_label) in MODELS.items():
        frames = {}
        for condition in ORDER:
            frame = pd.read_csv(OUT / family / condition / "validation_results.csv")
            frames[condition] = frame.set_index("case_id")
            gold = frame["true_diag1"].astype(int).to_numpy()
            pred = frame["pred_diag1"].astype(int).to_numpy()
            acc, macro = metric(gold, pred)
            lo, hi = bootstrap(gold, pred)
            rows.append(
                {
                    "family": family,
                    "model": model_label,
                    "condition": condition,
                    "accuracy": acc,
                    "macro_f1": macro,
                    "macro_f1_ci_low": lo,
                    "macro_f1_ci_high": hi,
                    "parse_error_rate": float(frame["parse_error"].mean()),
                }
            )
        for earlier, later in zip(ORDER, ORDER[1:]):
            left, right = frames[earlier], frames[later]
            if not left.index.equals(right.index):
                raise ValueError("Prompt-ablation case alignment failed")
            left_ok = left["diag1_correct"].astype(bool).to_numpy()
            right_ok = right["diag1_correct"].astype(bool).to_numpy()
            n10 = int(np.sum(left_ok & ~right_ok))
            n01 = int(np.sum(~left_ok & right_ok))
            p = float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
            comparisons.append(
                {
                    "model": model_label,
                    "earlier": earlier,
                    "later": later,
                    "earlier_only_correct": n10,
                    "later_only_correct": n01,
                    "mcnemar_exact_p": p,
                }
            )
    summary = pd.DataFrame(rows)
    tests = pd.DataFrame(comparisons)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "prompt_ablation_summary.csv", index=False)
    tests.to_csv(OUT / "adjacent_mcnemar_tests.csv", index=False)
    lines = [
        "# Prompt-component ablation",
        "",
        "## Scope",
        "",
        "This nested ablation uses only the 300-case **2024 development set** and "
        "the unfine-tuned base models. It does not consume the locked 2025 test set "
        "and is not part of the main test benchmark.",
        "",
        "## Development results",
        "",
        "| Model | Prompt condition | Accuracy | Macro F1 (95% CI) | Parse errors |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['condition'].replace('_', ' ')} | "
            f"{row['accuracy']:.3f} | {row['macro_f1']:.3f} "
            f"[{row['macro_f1_ci_low']:.3f}, {row['macro_f1_ci_high']:.3f}] | "
            f"{row['parse_error_rate']:.3f} |"
        )
    lines += [
        "",
        "## Adjacent-condition paired tests",
        "",
        "| Model | Comparison | Earlier-only correct | Later-only correct | Exact McNemar p |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['model']} | {row['earlier']} -> {row['later']} | "
            f"{row['earlier_only_correct']} | {row['later_only_correct']} | "
            f"{row['mcnemar_exact_p']:.4g} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- Results are development-set diagnostics, not additional held-out test results.",
        "- Components are nested, so each row estimates the cumulative prompt rather than "
        "an isolated factorial effect.",
        "- The rules are called **operational rules**, not verified manual rules. "
        "Independent line-by-line provenance verification remains pending.",
        "- The ten demonstrations are fixed; alternative example selections were not tested.",
    ]
    (OUT / "PROMPT_COMPONENT_ABLATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(summary)} prompt-ablation rows")


def main() -> None:
    parsed = args()
    (run if parsed.stage == "run" else report)(parsed)


if __name__ == "__main__":
    main()
