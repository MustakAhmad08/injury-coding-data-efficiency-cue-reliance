"""Run resumable zero-shot/10-shot evaluation on an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.sft.train import SFT_INSTRUCTION, SYSTEM_PROMPT, parse_model_response


TEST_FILE = PROJECT_ROOT / "data" / "neiss_temporal_2021_2025" / "locked_test.csv"
FEWSHOT_FILE = PROJECT_ROOT / "data" / "neiss_temporal_2021_2025" / "fewshot_10.jsonl"
DEFAULT_MODEL = "llama4:latest"
SEED = 20260718


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=("hosted_llama_0shot", "hosted_llama_10shot"))
    parser.add_argument("--stage", required=True, choices=("run", "collect"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HOSTED_LLM_ENDPOINT"),
        help="OpenAI-compatible chat-completions URL (or set HOSTED_LLM_ENDPOINT).",
    )
    parser.add_argument(
        "--api-key-env",
        default="HOSTED_LLM_API_KEY",
        help="Name of the environment variable containing the endpoint API key.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int, help="Smoke-test only; do not use for final results.")
    parser.add_argument("--confirm-locked-test", action="store_true")
    return parser.parse_args()


def require_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_locked_test:
        raise PermissionError("Add --confirm-locked-test to access the frozen 2025 test cases")


def require_api_key(args: argparse.Namespace) -> str:
    if not args.endpoint:
        raise EnvironmentError("Provide --endpoint or set HOSTED_LLM_ENDPOINT")
    key = os.environ.get(args.api_key_env)
    if not key:
        raise EnvironmentError(f"API key environment variable {args.api_key_env!r} is missing")
    return key


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


def load_test(args: argparse.Namespace) -> pd.DataFrame:
    require_confirmation(args)
    test = pd.read_csv(TEST_FILE)
    if len(test) != 1500 or test["case_id"].nunique() != 1500:
        raise ValueError("Locked test integrity failure: expected 1,500 unique case IDs")
    if args.limit:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        test = test.head(args.limit).copy()
    return test


def request_case(
    row: dict,
    *,
    key: str,
    endpoint: str,
    model: str,
    shots: int,
    demonstrations: list[dict],
    request_delay: float,
) -> dict:
    body = {
        "model": model,
        "messages": messages_for(str(row["narrative"]), shots, demonstrations),
        "stream": False,
        "temperature": 0,
        "max_tokens": 256,
    }
    last_error = None
    for attempt in range(1, 11):
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body,
                timeout=180,
            )
            if response.status_code == 200:
                payload = response.json()
                text = payload["choices"][0]["message"]["content"]
                time.sleep(request_delay)
                return {
                    "case_id": int(row["case_id"]),
                    "response": text,
                    "response_id": payload.get("id"),
                    "actual_model": payload.get("model"),
                    "usage": payload.get("usage"),
                    "api_error": None,
                }
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            is_rate_limit = response.status_code == 429 or (
                response.status_code == 400 and "rate limit" in response.text.lower()
            )
            if response.status_code not in {408, 429, 500, 502, 503, 504} and not is_rate_limit:
                break
            if is_rate_limit:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(30 * attempt, 180)
                time.sleep(delay)
                continue
        except (requests.RequestException, KeyError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(2 ** attempt, 30))
    return {
        "case_id": int(row["case_id"]),
        "response": "",
        "response_id": None,
        "actual_model": None,
        "usage": None,
        "api_error": last_error,
    }


def run(args: argparse.Namespace) -> None:
    key = require_api_key(args)
    test = load_test(args)
    shots = 10 if args.condition.endswith("10shot") else 0
    demonstrations = read_jsonl(FEWSHOT_FILE) if shots else []
    if len(demonstrations) != shots:
        raise ValueError(f"Expected {shots} demonstrations, found {len(demonstrations)}")
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    if args.request_delay < 0:
        raise ValueError("--request-delay cannot be negative")

    out = output_dir(args.condition)
    out.mkdir(parents=True, exist_ok=True)
    response_path = out / "api_responses.jsonl"
    completed: set[int] = set()
    if response_path.exists():
        completed = {
            int(item["case_id"])
            for item in read_jsonl(response_path)
            if not item.get("api_error")
        }
    pending = [row._asdict() for row in test.itertuples(index=False) if int(row.case_id) not in completed]

    manifest_path = out / "run_manifest_hosted_llama.json"
    manifest = {
        "condition": args.condition,
        "provider": "OpenAI-compatible hosted endpoint",
        "endpoint": args.endpoint,
        "requested_model": args.model,
        "shots": shots,
        "temperature": 0,
        "max_tokens": 256,
        "workers": args.workers,
        "request_delay_seconds": args.request_delay,
        "seed": SEED,
        "test_file": str(TEST_FILE),
        "test_file_sha256": hashlib.sha256(TEST_FILE.read_bytes()).hexdigest(),
        "fewshot_file": str(FEWSHOT_FILE) if shots else None,
        "n_expected": len(test),
        "smoke_test_limit": args.limit,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(completed)} already complete; {len(pending)} pending")

    with response_path.open("a", encoding="utf-8", newline="\n") as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    request_case,
                    row,
                    key=key,
                    endpoint=args.endpoint,
                    model=args.model,
                    shots=shots,
                    demonstrations=demonstrations,
                    request_delay=args.request_delay,
                ): int(row["case_id"])
                for row in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                status = "ok" if not result["api_error"] else "error"
                print(f"[{len(completed) + index}/{len(test)}] case={result['case_id']} {status}", flush=True)

    print("API run finished. Run --stage collect to validate and compute metrics.")


def collect(args: argparse.Namespace) -> None:
    test = load_test(args)
    out = output_dir(args.condition)
    response_path = out / "api_responses.jsonl"
    if not response_path.exists():
        raise FileNotFoundError("No api_responses.jsonl; run --stage run first")
    response_items = read_jsonl(response_path)
    by_case = {int(item["case_id"]): item for item in response_items}
    expected = set(test["case_id"].astype(int))
    if set(by_case) != expected:
        missing = sorted(expected - set(by_case))[:10]
        extra = sorted(set(by_case) - expected)[:10]
        raise ValueError(f"Case-ID mismatch; missing={missing}, extra={extra}")

    results = []
    for row in test.itertuples(index=False):
        item = by_case[int(row.case_id)]
        parsed = parse_model_response(item.get("response") or "")
        pred_diag1 = parsed.get("Diagnosis_1_Code")
        pred_body1 = parsed.get("Body_Part_1_Code")
        parse_error = bool(parsed.get("parse_error")) or pred_diag1 is None
        results.append({
            "case_id": int(row.case_id),
            "year": int(row.year),
            "narrative": row.narrative,
            "true_diag1": int(row.diag1),
            "true_diagnosis_category": row.diagnosis_category,
            "pred_diag1": pred_diag1,
            "diag1_correct": int(row.diag1) == pred_diag1,
            "parse_error": parse_error,
            "raw_output": item.get("response") or "",
            "true_body1": int(row.body1),
            "pred_body1": pred_body1,
            "true_diag2": row.diag2,
            "true_body2": row.body2,
            "pred_diag2": parsed.get("Diagnosis_2_Code"),
            "body1_correct": int(row.body1) == pred_body1,
            "api_error": item.get("api_error"),
            "response_id": item.get("response_id"),
            "actual_model": item.get("actual_model"),
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(out / "test_results.csv", index=False)
    y_true = result_df["true_diag1"].astype(int)
    y_pred = pd.to_numeric(result_df["pred_diag1"], errors="coerce").fillna(-1).astype(int)
    metrics = {
        "condition": args.condition,
        "requested_model": args.model,
        "actual_models": sorted(result_df["actual_model"].dropna().unique().tolist()),
        "n_examples": len(result_df),
        "diagnosis_accuracy": float(accuracy_score(y_true, y_pred)),
        "body_part_accuracy": float(result_df["body1_correct"].mean()),
        "parse_error_rate": float(result_df["parse_error"].mean()),
        "api_error_rate": float(result_df["api_error"].notna().mean()),
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
        "condition", "actual_models", "n_examples", "diagnosis_accuracy",
        "body_part_accuracy", "parse_error_rate", "api_error_rate",
    )}, indent=2))


def main() -> None:
    args = parse_args()
    {"run": run, "collect": collect}[args.stage](args)


if __name__ == "__main__":
    main()
