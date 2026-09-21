"""Monitor zero-shot, then run the six GPT-5.4 ten-shot chunks sequentially."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import winreg
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
ZERO_JOB = ROOT / "methods_runs" / "gpt54_0shot" / "batch_job.json"
CHUNK_STATE = ROOT / "methods_runs" / "gpt54_10shot" / "batch_chunks.json"
TERMINAL = {"completed", "failed", "expired", "cancelled"}
POLL_SECONDS = 30


def configure_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
        value, _ = winreg.QueryValueEx(handle, "OPENAI_API_KEY")
    if not value:
        raise EnvironmentError("User-level OPENAI_API_KEY is missing")
    os.environ["OPENAI_API_KEY"] = value


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("Running:", " ".join(arguments), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def wait_for(client: OpenAI, batch_id: str, label: str) -> None:
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"{label}: status={batch.status}, completed={counts.completed}, "
            f"failed={counts.failed}, total={counts.total}",
            flush=True,
        )
        if batch.status in TERMINAL:
            if batch.status != "completed":
                details = batch.errors.model_dump(mode="json") if batch.errors else None
                raise RuntimeError(f"{label} ended with {batch.status}: {details}")
            return
        time.sleep(POLL_SECONDS)


def main() -> None:
    configure_key()
    client = OpenAI()

    zero = json.loads(ZERO_JOB.read_text(encoding="utf-8"))
    wait_for(client, zero["batch_id"], "gpt54_0shot")
    run(
        str(SCRIPT_DIR / "run_gpt54_evaluation.py"),
        "gpt54_0shot", "--stage", "collect", "--confirm-locked-test",
    )

    while True:
        state = json.loads(CHUNK_STATE.read_text(encoding="utf-8"))
        if all(chunk["status"] == "completed" for chunk in state["chunks"]):
            break
        run(
            str(SCRIPT_DIR / "run_gpt54_10shot_chunks.py"),
            "--stage", "submit-next", "--confirm-locked-test",
        )
        state = json.loads(CHUNK_STATE.read_text(encoding="utf-8"))
        active = [
            chunk for chunk in state["chunks"]
            if chunk["batch_id"] and chunk["status"] not in TERMINAL
        ]
        if active:
            chunk = active[0]
            wait_for(client, chunk["batch_id"], f"gpt54_10shot_part_{chunk['part']:02d}")
        else:
            failed = [chunk for chunk in state["chunks"] if chunk["status"] in TERMINAL - {"completed"}]
            if failed:
                raise RuntimeError(f"Ten-shot chunk failure: {failed}")

    # Refresh the last completed state, then combine and score all six chunks.
    run(str(SCRIPT_DIR / "run_gpt54_10shot_chunks.py"), "--stage", "status")
    run(
        str(SCRIPT_DIR / "run_gpt54_10shot_chunks.py"),
        "--stage", "collect", "--confirm-locked-test",
    )
    print("Both GPT-5.4 conditions completed and were collected successfully.", flush=True)


if __name__ == "__main__":
    main()
