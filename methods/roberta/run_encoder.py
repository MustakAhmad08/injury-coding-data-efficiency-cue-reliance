"""Reproducible additional encoder baseline; no hosted inference.

Run train before final-eval. Model selection never opens the test file.
Local checkpoints and case-level outputs stay under ignored local/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_split(path, years, count):
    df = pd.read_csv(path, dtype={"case_id": str})
    if not {"case_id", "year", "narrative", "diag1"}.issubset(df.columns):
        raise ValueError("Required columns missing")
    if len(df) != count or set(df.year.astype(int)) != years:
        raise ValueError(f"Unexpected row count/year for {Path(path).name}")
    if df.case_id.isna().any() or df.case_id.duplicated().any():
        raise ValueError("Missing or duplicated case identifiers")
    if df.narrative.isna().any() or df.narrative.str.strip().eq("").any():
        raise ValueError("Missing or empty narrative")
    df.diag1 = df.diag1.astype(int)
    if df.diag1.nunique() != 30 or df.diag1.value_counts().nunique() != 1:
        raise ValueError("Expected 30 equally represented diagnosis classes")
    return df


def check_disjoint(a, b):
    if set(a.case_id) & set(b.case_id):
        raise ValueError("Case leakage across splits")
    norm = lambda f: set(f.narrative.str.upper().str.replace(r"\s+", " ", regex=True).str.strip())
    if norm(a) & norm(b):
        raise ValueError("Exact normalized narrative leakage across splits")
    if set(a.diag1) != set(b.diag1):
        raise ValueError("Diagnosis inventories differ")


def encode(frame, tokenizer, codes, config):
    text = frame.narrative.tolist()
    full = tokenizer(text, truncation=False, padding=False)
    lengths = [len(x) for x in full["input_ids"]]
    enc = tokenizer(text, truncation=True, max_length=config["max_length"], padding=False)
    mapping = {code: i for i, code in enumerate(codes)}
    rows = [{**{k: v[i] for k, v in enc.items()}, "labels": mapping[int(frame.diag1.iloc[i])]} for i in range(len(frame))]
    stats = {"count": len(rows), "max_tokens": max(lengths), "truncated": sum(x > config["max_length"] for x in lengths)}
    return rows, stats


def loader(rows, tokenizer, config, train=False):
    return DataLoader(rows, batch_size=config["batch_size" if train else "eval_batch_size"], shuffle=train,
                      num_workers=0, pin_memory=True, collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))


def metrics(gold, predicted, codes):
    return {"accuracy": float(accuracy_score(gold, predicted)),
            "macro_f1": float(f1_score(gold, predicted, labels=codes, average="macro", zero_division=0))}


@torch.inference_mode()
def predict(model, batches, codes):
    model.eval()
    predictions = []
    for batch in batches:
        batch.pop("labels", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**{k: v.cuda(non_blocking=True) for k, v in batch.items()}).logits
        predictions.extend(codes[i] for i in logits.argmax(-1).cpu().tolist())
    return predictions


def load_model(path, codes):
    return AutoModelForSequenceClassification.from_pretrained(
        str(path), num_labels=len(codes), id2label={i: str(c) for i, c in enumerate(codes)},
        label2id={str(c): i for i, c in enumerate(codes)}, local_files_only=True,
        attn_implementation="sdpa",
    ).cuda()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def train(config):
    if (HERE / "results/test_summary.json").exists():
        raise RuntimeError("Test evaluation already exists. Do not retune this frozen experiment.")
    train_path, val_path = ROOT / config["train_file"], ROOT / config["validation_file"]
    training = load_split(train_path, {2021, 2022, 2023}, 6000)
    validation = load_split(val_path, {2024}, 300)
    check_disjoint(training, validation)
    codes = sorted(int(x) for x in training.diag1.unique())
    model_path = HERE / "local/model"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    train_rows, train_stats = encode(training, tokenizer, codes, config)
    val_rows, val_stats = encode(validation, tokenizer, codes, config)
    fingerprint = {"config_sha256": digest(HERE / "config.json"), "train_sha256": digest(train_path),
                   "validation_sha256": digest(val_path), "base_weights_sha256": digest(model_path / "model.safetensors")}
    trial_results = []
    for lr in config["learning_rates"]:
        for seed in config["seeds"]:
            trial = HERE / "local/runs" / f"lr_{lr:g}_seed_{seed}"
            record_path = trial / "trial.json"
            if record_path.exists():
                record = read_json(record_path)
                if record["fingerprint"] != fingerprint or digest(trial / "checkpoint/model.safetensors") != record["checkpoint_sha256"]:
                    raise RuntimeError("Completed trial has changed; cannot silently reuse it")
                trial_results.append(record); continue
            seed_everything(seed)
            model = load_model(model_path, codes)
            n_params = sum(p.numel() for p in model.parameters())
            batches = loader(train_rows, tokenizer, config, True)
            val_batches = loader(val_rows, tokenizer, config)
            decay, no_decay = [], []
            for name, p in model.named_parameters():
                (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(p)
            optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config["weight_decay"]},
                                          {"params": no_decay, "weight_decay": 0.0}], lr=lr, eps=1e-8)
            steps = len(batches) * config["epochs"]
            scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(config["warmup_ratio"] * steps), steps)
            best = -1.0; history = []; fit_seconds = 0.0
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
            for epoch in range(1, config["epochs"] + 1):
                model.train(); loss_sum = 0.0; torch.cuda.synchronize(); epoch_start = time.perf_counter()
                for batch in batches:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(**{k: v.cuda(non_blocking=True) for k, v in batch.items()}).loss
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                    optimizer.step(); scheduler.step(); loss_sum += loss.item()
                torch.cuda.synchronize(); fit_seconds += time.perf_counter() - epoch_start
                predicted = predict(model, val_batches, codes)
                m = metrics(validation.diag1, predicted, codes)
                history.append({"epoch": epoch, "training_loss": loss_sum / len(batches), **m})
                if m["macro_f1"] > best:
                    best = m["macro_f1"]; best_epoch = epoch
                    model.save_pretrained(trial / "checkpoint", safe_serialization=True)
                    tokenizer.save_pretrained(trial / "checkpoint")
                print(f"lr={lr:g} seed={seed} epoch={epoch} dev_macro_f1={m['macro_f1']:.6f} fit_seconds={fit_seconds:.1f}", flush=True)
            torch.cuda.synchronize()
            record = {"learning_rate": lr, "seed": seed, "best_epoch": best_epoch, "validation_macro_f1": best,
                      "history": history, "training_loop_seconds": fit_seconds,
                      "trial_seconds_including_validation_and_checkpoint_io": time.perf_counter() - started,
                      "peak_training_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                      "parameters": n_params, "fingerprint": fingerprint,
                      "checkpoint_sha256": digest(trial / "checkpoint/model.safetensors"),
                      "checkpoint": str((trial / "checkpoint").relative_to(HERE)).replace("\\", "/")}
            write_json(record_path, record); trial_results.append(record)
            del model, optimizer, scheduler, batches, val_batches, decay, no_decay, p, loss
            gc.collect(); torch.cuda.empty_cache()
    scores = [{"learning_rate": lr, "mean_validation_macro_f1": float(np.mean([r["validation_macro_f1"] for r in trial_results if r["learning_rate"] == lr]))} for lr in config["learning_rates"]]
    selected_lr = max(scores, key=lambda x: x["mean_validation_macro_f1"])["learning_rate"]
    selection = {"config": config, "fingerprint": fingerprint, "codes": codes, "grid": scores,
                 "selected_learning_rate": selected_lr, "trials": trial_results,
                 "selected_trials": [r for r in trial_results if r["learning_rate"] == selected_lr],
                 "tokenization": {"training": train_stats, "validation": val_stats},
                 "torch": torch.__version__, "transformers": transformers.__version__,
                 "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0),
                 "determinism": "Seeded RNGs; deterministic algorithms requested with warn_only=True; bitwise reproducibility not guaranteed.",
                 "selection_frozen_utc": datetime.now(timezone.utc).isoformat()}
    write_json(HERE / "results/validation_selection.json", selection)
    print(f"FROZEN selected learning rate: {selected_lr:g}", flush=True)


def evaluate(config, confirmed):
    if not confirmed:
        raise ValueError("--confirm-locked-test is required after freezing development selection")
    selection_path = HERE / "results/validation_selection.json"
    selection = read_json(selection_path)
    if selection["fingerprint"]["config_sha256"] != digest(HERE / "config.json"):
        raise ValueError("Configuration changed after selection")
    result_path = HERE / "results/test_summary.json"
    if result_path.exists():
        raise FileExistsError("Final evaluation exists; refusing replacement")
    for role in ("train", "validation"):
        if digest(ROOT / config[f"{role}_file"]) != selection["fingerprint"][f"{role}_sha256"]:
            raise ValueError("Selection data changed")
    test = load_split(ROOT / config["test_file"], {2025}, 1500)
    for key, years, count in (("train_file", {2021,2022,2023}, 6000), ("validation_file", {2024}, 300)):
        check_disjoint(load_split(ROOT / config[key], years, count), test)
    codes = selection["codes"]; results = []; reports = []
    for trial in selection["selected_trials"]:
        seed = trial["seed"]; checkpoint = HERE / trial["checkpoint"]
        if digest(checkpoint / "model.safetensors") != trial["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed")
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
        rows, stats = encode(test, tokenizer, codes, config)
        batches = loader(rows, tokenizer, config)
        model = load_model(checkpoint, codes)
        predict(model, loader(rows[:config["eval_batch_size"]], tokenizer, config), codes)
        timings = []; torch.cuda.reset_peak_memory_stats()
        for _ in range(config["timing_repeats"]):
            torch.cuda.synchronize(); started = time.perf_counter()
            predicted = predict(model, batches, codes)
            torch.cuda.synchronize(); timings.append(time.perf_counter() - started)
        target = HERE / "local/test_predictions" / f"seed_{seed}.csv"; target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"case_id": test.case_id, "true_diag1": test.diag1, "pred_diag1": predicted}).to_csv(target, index=False)
        m = {"seed": seed, **metrics(test.diag1, predicted, codes), "inference_seconds": timings,
             "mean_inference_seconds": float(np.mean(timings)), "examples_per_second": len(test) / float(np.mean(timings)),
             "peak_inference_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
             "predictions_sha256": digest(target)}
        results.append(m)
        report = classification_report(test.diag1, predicted, labels=codes, output_dict=True, zero_division=0)
        for code in codes:
            reports.append({"seed": seed, "diagnosis_code": code, **report[str(code)]})
        print(f"TEST seed={seed} macro_f1={m['macro_f1']:.6f} accuracy={m['accuracy']:.6f}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    summary = {"runs": results, "macro_f1_mean": float(np.mean([r["macro_f1"] for r in results])),
               "macro_f1_sd": float(np.std([r["macro_f1"] for r in results], ddof=1)),
               "accuracy_mean": float(np.mean([r["accuracy"] for r in results])),
               "accuracy_sd": float(np.std([r["accuracy"] for r in results], ddof=1)),
               "selection_sha256": digest(selection_path), "test_sha256": digest(ROOT / config["test_file"]),
               "test_tokenization": stats, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "timing_scope": "Warm GPU forward inference including dynamic collation, host-to-device transfer and argmax transfer; excludes tokenization, model loading and disk output. Three repeated passes per model; no selection uses test results.",
               "interpretation": "SD is training-seed variability on a fixed training subset and fixed test set; it is not a test-sampling confidence interval. No best test seed is selected."}
    pd.DataFrame(reports).to_csv(HERE / "results/per_category_metrics.csv", index=False)
    write_json(result_path, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "final-eval"])
    parser.add_argument("--confirm-locked-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This frozen protocol requires a CUDA GPU supporting bfloat16")
    config = read_json(HERE / "config.json")
    if args.stage == "train": train(config)
    else: evaluate(config, args.confirm_locked_test)


if __name__ == "__main__":
    main()
