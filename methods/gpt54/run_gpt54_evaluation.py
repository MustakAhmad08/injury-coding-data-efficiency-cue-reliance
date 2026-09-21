"""Run frozen GPT-5.4 zero-shot/10-shot NEISS evaluation through Batch API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.sft.train import SFT_INSTRUCTION, SYSTEM_PROMPT


TEST_FILE = PROJECT_ROOT / "data" / "neiss_temporal_2021_2025" / "locked_test.csv"
FEWSHOT_FILE = PROJECT_ROOT / "data" / "neiss_temporal_2021_2025" / "fewshot_10.jsonl"
MODEL = "gpt-5.4-2026-03-05"
SEED = 20260718

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "Diagnosis_1_Code": {"type": "integer"},
        "Diagnosis_1_Label": {"type": "string"},
        "Body_Part_1_Code": {"type": "integer"},
        "Diagnosis_2_Code": {"type": ["integer", "null"]},
        "Body_Part_2_Code": {"type": ["integer", "null"]},
        "Rationale": {"type": "string"},
    },
    "required": [
        "Diagnosis_1_Code",
        "Diagnosis_1_Label",
        "Body_Part_1_Code",
        "Diagnosis_2_Code",
        "Body_Part_2_Code",
        "Rationale",
    ],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=("gpt54_0shot", "gpt54_10shot"))
    parser.add_argument("--stage", required=True, choices=("prepare", "submit", "status", "collect"))
    parser.add_argument("--confirm-locked-test", action="store_true")
    return parser.parse_args()


def require_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_locked_test:
        raise PermissionError("Add --confirm-locked-test to access the frozen 2025 test cases")


def require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not visible. Fully restart VS Code/Codex after setting the user environment variable."
        )


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def messages_for(narrative: str, shots: int, demonstrations: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in demonstrations[:shots]:
        instruction = example.get("instruction", SFT_INSTRUCTION)
        messages.extend([
            {
                "role": "user",
                "content": f"{instruction}\n\nInjury Narrative: {example['narrative']}",
            },
            {"role": "assistant", "content": example["response"]},
        ])
    messages.append({
        "role": "user",
        "content": f"{SFT_INSTRUCTION}\n\nInjury Narrative: {narrative}",
    })
    return messages


def output_dir(condition: str) -> Path:
    return PROJECT_ROOT / "methods_runs" / condition


def prepare(args: argparse.Namespace) -> None:
    require_confirmation(args)
    shots = 10 if args.condition.endswith("10shot") else 0
    demonstrations = read_jsonl(FEWSHOT_FILE) if shots else []
    if len(demonstrations) != shots:
        raise ValueError(f"Expected {shots} demonstrations, found {len(demonstrations)}")

    test = pd.read_csv(TEST_FILE)
    if len(test) != 1500 or test["case_id"].nunique() != 1500:
        raise ValueError("Locked test integrity failure: expected 1,500 unique case IDs")

    out = output_dir(args.condition)
    out.mkdir(parents=True, exist_ok=True)
    batch_path = out / "batch_input.jsonl"
    digest = hashlib.sha256()
    with batch_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in test.itertuples(index=False):
            request = {
                "custom_id": f"case-{int(row.case_id)}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL,
                    "input": messages_for(str(row.narrative), shots, demonstrations),
                    "reasoning": {"effort": "none"},
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "neiss_coding_result",
                            "strict": True,
                            "schema": OUTPUT_SCHEMA,
                        }
                    },
                    "max_output_tokens": 256,
                    "store": False,
                },
            }
            line = json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))

    manifest = {
        "condition": args.condition,
        "model": MODEL,
        "shots": shots,
        "reasoning_effort": "none",
        "structured_outputs": True,
        "max_output_tokens": 256,
        "store": False,
        "seed": SEED,
        "test_file": str(TEST_FILE),
        "fewshot_file": str(FEWSHOT_FILE) if shots else None,
        "n_cases": len(test),
        "batch_input_sha256": digest.hexdigest(),
    }
    (out / "run_manifest_gpt54.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(test)} requests: {batch_path}")


def submit(args: argparse.Namespace) -> None:
    require_confirmation(args)
    require_api_key()
    from openai import OpenAI

    out = output_dir(args.condition)
    batch_path = out / "batch_input.jsonl"
    if not batch_path.exists():
        raise FileNotFoundError("Run --stage prepare first")
    if (out / "batch_job.json").exists():
        raise FileExistsError("batch_job.json already exists; refusing accidental duplicate submission")

    client = OpenAI()
    with batch_path.open("rb") as stream:
        uploaded = client.files.create(file=stream, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"condition": args.condition, "model": MODEL},
    )
    job = {"batch_id": batch.id, "input_file_id": uploaded.id, "status": batch.status}
    (out / "batch_job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    print(json.dumps(job, indent=2))


def load_job(condition: str) -> tuple[Path, dict]:
    out = output_dir(condition)
    job_path = out / "batch_job.json"
    if not job_path.exists():
        raise FileNotFoundError("No batch_job.json; run --stage submit first")
    return out, json.loads(job_path.read_text(encoding="utf-8"))


def status(args: argparse.Namespace) -> None:
    require_api_key()
    from openai import OpenAI

    _, job = load_job(args.condition)
    batch = OpenAI().batches.retrieve(job["batch_id"])
    print(json.dumps(batch.model_dump(mode="json"), indent=2))


def extract_output_text(body: dict) -> str | None:
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text")
    return None


def collect(args: argparse.Namespace) -> None:
    require_confirmation(args)
    require_api_key()
    from openai import OpenAI

    out, job = load_job(args.condition)
    client = OpenAI()
    batch = client.batches.retrieve(job["batch_id"])
    if batch.status != "completed":
        raise RuntimeError(f"Batch is not complete; current status: {batch.status}")
    if not batch.output_file_id:
        raise RuntimeError("Completed batch has no output_file_id")

    raw = client.files.content(batch.output_file_id).content
    raw_path = out / "batch_output.jsonl"
    raw_path.write_bytes(raw)
    lines = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    by_case = {int(item["custom_id"].removeprefix("case-")): item for item in lines}

    test = pd.read_csv(TEST_FILE)
    expected = set(test["case_id"].astype(int))
    if set(by_case) != expected:
        missing = sorted(expected - set(by_case))[:10]
        extra = sorted(set(by_case) - expected)[:10]
        raise ValueError(f"Batch case-ID mismatch; missing={missing}, extra={extra}")

    results = []
    for row in test.itertuples(index=False):
        case_id = int(row.case_id)
        item = by_case[case_id]
        response = item.get("response") or {}
        body = response.get("body") or {}
        text = extract_output_text(body)
        parsed, error = None, None
        if response.get("status_code") != 200:
            error = json.dumps(item.get("error") or response, ensure_ascii=False)
        elif text is None:
            error = "No output_text in response"
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                error = f"JSON parse error: {exc}"
        parsed = parsed or {}
        pred_diag1 = parsed.get("Diagnosis_1_Code")
        pred_body1 = parsed.get("Body_Part_1_Code")
        results.append({
            "case_id": case_id,
            "year": int(row.year),
            "narrative": row.narrative,
            "true_diag1": int(row.diag1),
            "true_diagnosis_category": row.diagnosis_category,
            "pred_diag1": pred_diag1,
            "diag1_correct": int(row.diag1) == pred_diag1,
            "parse_error": error is not None,
            "raw_output": text or "",
            "true_body1": int(row.body1),
            "pred_body1": pred_body1,
            "true_diag2": row.diag2,
            "true_body2": row.body2,
            "pred_diag2": parsed.get("Diagnosis_2_Code"),
            "body1_correct": int(row.body1) == pred_body1,
            "api_error": error,
            "response_id": body.get("id"),
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(out / "test_results.csv", index=False)
    y_true = result_df["true_diag1"].astype(int)
    y_pred = result_df["pred_diag1"].fillna(-1).astype(int)
    metrics = {
        "condition": args.condition,
        "model": MODEL,
        "n_examples": len(result_df),
        "diagnosis_accuracy": float(accuracy_score(y_true, y_pred)),
        "body_part_accuracy": float(result_df["body1_correct"].mean()),
        "parse_error_rate": float(result_df["parse_error"].mean()),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=sorted(y_true.unique()),
            zero_division=0,
            output_dict=True,
        ),
    }
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in (
        "condition", "n_examples", "diagnosis_accuracy", "body_part_accuracy", "parse_error_rate"
    )}, indent=2))


def main() -> None:
    args = parse_args()
    {"prepare": prepare, "submit": submit, "status": status, "collect": collect}[args.stage](args)


if __name__ == "__main__":
    main()
