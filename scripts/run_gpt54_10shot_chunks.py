"""Run the prepared GPT-5.4 10-shot batch sequentially in 250-case chunks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "methods_runs" / "gpt54_10shot"
SOURCE = OUT / "batch_input.jsonl"
STATE = OUT / "batch_chunks.json"
TEST_FILE = PROJECT_ROOT / "data" / "neiss_temporal_2021_2025" / "locked_test.csv"
CHUNK_SIZE = 250
TERMINAL = {"completed", "failed", "expired", "cancelled"}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "submit-next", "status", "collect"))
    parser.add_argument("--confirm-locked-test", action="store_true")
    return parser.parse_args()


def confirm(ns: argparse.Namespace) -> None:
    if not ns.confirm_locked_test:
        raise PermissionError("Add --confirm-locked-test")


def key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not visible")


def load_state() -> dict:
    if not STATE.exists():
        raise FileNotFoundError("Run --stage prepare first")
    return json.loads(STATE.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def prepare(ns: argparse.Namespace) -> None:
    confirm(ns)
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1500:
        raise ValueError(f"Expected 1,500 requests, found {len(lines)}")
    chunks = []
    for number, start in enumerate(range(0, len(lines), CHUNK_SIZE), 1):
        path = OUT / f"batch_input_part_{number:02d}.jsonl"
        selected = lines[start:start + CHUNK_SIZE]
        path.write_text("\n".join(selected) + "\n", encoding="utf-8")
        chunks.append({
            "part": number,
            "path": str(path),
            "requests": len(selected),
            "batch_id": None,
            "input_file_id": None,
            "output_file_id": None,
            "status": "prepared",
        })
    save_state({"condition": "gpt54_10shot", "chunk_size": CHUNK_SIZE, "chunks": chunks})
    print(f"Prepared {len(chunks)} sequential chunks totaling {sum(c['requests'] for c in chunks)} requests")


def refresh(client, state: dict) -> None:
    for chunk in state["chunks"]:
        if chunk["batch_id"] and chunk["status"] not in TERMINAL:
            batch = client.batches.retrieve(chunk["batch_id"])
            chunk["status"] = batch.status
            chunk["output_file_id"] = batch.output_file_id
            chunk["request_counts"] = batch.request_counts.model_dump(mode="json")
            if batch.errors:
                chunk["errors"] = batch.errors.model_dump(mode="json")


def submit_next(ns: argparse.Namespace) -> None:
    confirm(ns)
    key()
    from openai import OpenAI

    client = OpenAI()
    state = load_state()
    refresh(client, state)
    active = [c for c in state["chunks"] if c["status"] not in TERMINAL | {"prepared"}]
    if active:
        save_state(state)
        print(f"Part {active[0]['part']} remains {active[0]['status']}; no new chunk submitted")
        return
    failed = [c for c in state["chunks"] if c["status"] in {"failed", "expired", "cancelled"}]
    if failed:
        save_state(state)
        raise RuntimeError(f"Chunk failure requires review: parts {[c['part'] for c in failed]}")
    pending = [c for c in state["chunks"] if c["status"] == "prepared"]
    if not pending:
        save_state(state)
        print("All chunks have completed")
        return
    chunk = pending[0]
    with Path(chunk["path"]).open("rb") as stream:
        uploaded = client.files.create(file=stream, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"condition": "gpt54_10shot", "part": str(chunk["part"])},
    )
    chunk.update({
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "status": batch.status,
    })
    save_state(state)
    print(json.dumps(chunk, indent=2))


def status(ns: argparse.Namespace) -> None:
    key()
    from openai import OpenAI

    state = load_state()
    refresh(OpenAI(), state)
    save_state(state)
    print(json.dumps([{k: c.get(k) for k in ("part", "status", "batch_id", "request_counts")} for c in state["chunks"]], indent=2))


def output_text(body: dict) -> str | None:
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text")
    return None


def collect(ns: argparse.Namespace) -> None:
    confirm(ns)
    key()
    from openai import OpenAI

    client = OpenAI()
    state = load_state()
    refresh(client, state)
    if any(c["status"] != "completed" for c in state["chunks"]):
        raise RuntimeError("All chunks must be completed before collection")
    lines = []
    for chunk in state["chunks"]:
        raw = client.files.content(chunk["output_file_id"]).content.decode("utf-8")
        path = OUT / f"batch_output_part_{chunk['part']:02d}.jsonl"
        path.write_text(raw, encoding="utf-8")
        lines.extend(json.loads(line) for line in raw.splitlines() if line.strip())
    (OUT / "batch_output.jsonl").write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n",
        encoding="utf-8",
    )
    by_case = {int(x["custom_id"].removeprefix("case-")): x for x in lines}
    test = pd.read_csv(TEST_FILE)
    expected = set(test.case_id.astype(int))
    if set(by_case) != expected:
        raise ValueError("Combined chunk outputs do not match the 1,500 locked case IDs")
    results = []
    for row in test.itertuples(index=False):
        item = by_case[int(row.case_id)]
        response = item.get("response") or {}
        body = response.get("body") or {}
        text = output_text(body)
        error = None
        parsed = {}
        if response.get("status_code") != 200:
            error = json.dumps(item.get("error") or response, ensure_ascii=False)
        elif text is None:
            error = "No output_text"
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                error = f"JSON parse error: {exc}"
        pred_diag = parsed.get("Diagnosis_1_Code")
        pred_body = parsed.get("Body_Part_1_Code")
        results.append({
            "case_id": int(row.case_id), "year": int(row.year), "narrative": row.narrative,
            "true_diag1": int(row.diag1), "true_diagnosis_category": row.diagnosis_category,
            "pred_diag1": pred_diag, "diag1_correct": int(row.diag1) == pred_diag,
            "parse_error": error is not None, "raw_output": text or "",
            "true_body1": int(row.body1), "pred_body1": pred_body,
            "true_diag2": row.diag2, "true_body2": row.body2,
            "pred_diag2": parsed.get("Diagnosis_2_Code"),
            "body1_correct": int(row.body1) == pred_body, "api_error": error,
            "response_id": body.get("id"),
        })
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "test_results.csv", index=False)
    y_true = frame.true_diag1.astype(int)
    y_pred = frame.pred_diag1.fillna(-1).astype(int)
    metrics = {
        "condition": "gpt54_10shot", "model": "gpt-5.4-2026-03-05",
        "n_examples": len(frame),
        "diagnosis_accuracy": float(accuracy_score(y_true, y_pred)),
        "body_part_accuracy": float(frame.body1_correct.mean()),
        "parse_error_rate": float(frame.parse_error.mean()),
        "classification_report": classification_report(
            y_true, y_pred, labels=sorted(y_true.unique()), zero_division=0, output_dict=True
        ),
    }
    (OUT / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_state(state)
    print(json.dumps({k: metrics[k] for k in ("n_examples", "diagnosis_accuracy", "body_part_accuracy", "parse_error_rate")}, indent=2))


def main() -> None:
    ns = args()
    {"prepare": prepare, "submit-next": submit_next, "status": status, "collect": collect}[ns.stage](ns)


if __name__ == "__main__":
    main()
