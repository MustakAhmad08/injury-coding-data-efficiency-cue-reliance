"""Compare all encoder seeds to the two fixed reference prediction sets.

Six planned comparisons form their own Holm family. Bootstrap intervals for
macro-F1 differences resample cases within diagnosis and pair model predictions.
These intervals concern test cases, not training-run uncertainty.
"""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def score(gold, pred, labels):
    tp = np.array([np.sum((gold == x) & (pred == x)) for x in labels])
    actual = np.array([np.sum(gold == x) for x in labels])
    predicted = np.array([np.sum(pred == x) for x in labels])
    denominator = actual + predicted
    return float(np.mean(np.divide(2 * tp, denominator, out=np.zeros(len(labels)), where=denominator != 0)))


def load(path):
    f = pd.read_csv(path, dtype={"case_id": str})[["case_id", "true_diag1", "pred_diag1"]]
    if f.case_id.duplicated().any() or f.isna().any().any():
        raise ValueError("Invalid case-level predictions")
    return f.sort_values("case_id").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tfidf", type=Path, default=ROOT / "methods_runs/tfidf_logreg_6000/test_results.csv")
    p.add_argument("--qwen", type=Path, default=ROOT / "methods_runs/qwen_sft_6000/test_results.csv")
    args = p.parse_args()
    summary = json.loads((HERE / "results/test_summary.json").read_text())
    refs = {"tfidf_logreg_6000": load(args.tfidf), "qwen_sft_6000": load(args.qwen)}
    frames = {**refs, **{f"roberta_seed_{r['seed']}": load(HERE / f"local/test_predictions/seed_{r['seed']}.csv") for r in summary["runs"]}}
    reference = next(iter(refs.values()))
    for name, f in frames.items():
        if not f[["case_id", "true_diag1"]].equals(reference[["case_id", "true_diag1"]]):
            raise ValueError(f"Unpaired cases or labels: {name}")
    gold = reference.true_diag1.to_numpy(); labels = np.unique(gold)
    pred = {name: f.pred_diag1.to_numpy() for name, f in frames.items()}
    comparisons = [(name, ref) for name in frames if name.startswith("roberta_") for ref in refs]
    rng = np.random.default_rng(20260718)
    groups = [np.flatnonzero(gold == code) for code in labels]
    draws = {pair: [] for pair in comparisons}; model_draws = {name: [] for name in frames}
    for _ in range(1000):
        ix = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        scores = {name: score(gold[ix], values[ix], labels) for name, values in pred.items()}
        for name, value in scores.items(): model_draws[name].append(value)
        for pair in comparisons: draws[pair].append(scores[pair[0]] - scores[pair[1]])
    table = []
    for name, values in pred.items():
        lo, hi = np.percentile(model_draws[name], [2.5, 97.5])
        table.append({"condition": name, "macro_f1": score(gold, values, labels),
                      "accuracy": float(np.mean(values == gold)), "macro_f1_ci_low": lo, "macro_f1_ci_high": hi})
    rows = []
    for encoder, ref in comparisons:
        a, b = pred[encoder] == gold, pred[ref] == gold
        encoder_only = int(np.sum(a & ~b)); reference_only = int(np.sum(~a & b))
        n = encoder_only + reference_only
        raw_p = binomtest(encoder_only, n, p=0.5).pvalue if n else 1.0
        lo, hi = np.percentile(draws[(encoder, ref)], [2.5, 97.5])
        rows.append({"encoder": encoder, "reference": ref, "macro_f1_difference": score(gold,pred[encoder],labels)-score(gold,pred[ref],labels),
                     "difference_ci_low": lo, "difference_ci_high": hi, "encoder_only_correct": encoder_only,
                     "reference_only_correct": reference_only, "mcnemar_exact_p": raw_p})
    running = 0.0
    for rank, index in enumerate(sorted(range(len(rows)), key=lambda i: rows[i]["mcnemar_exact_p"])):
        running = max(running, min(1.0, rows[index]["mcnemar_exact_p"] * (len(rows)-rank)))
        rows[index]["holm_p_six_comparisons"] = running
    pd.DataFrame(table).to_csv(HERE / "results/comparison_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "results/paired_comparisons.csv", index=False)
    meta = {"bootstrap_resamples": 1000, "seed": 20260718, "case_count": len(gold),
            "holm_family": "Six encoder-seed versus reference comparisons (3 seeds x TF-IDF/Qwen). Separate from the original 105 comparisons.",
            "inference": "McNemar tests accuracy; bootstrap differences concern macro-F1. Intervals are unadjusted and do not combine training-seed and test-sampling uncertainty. Non-significance is not equivalence.",
            "reference_prediction_sha256": {"tfidf": hashlib.sha256(args.tfidf.read_bytes()).hexdigest(), "qwen": hashlib.sha256(args.qwen.read_bytes()).hexdigest()}}
    (HERE / "results/comparison_metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(pd.DataFrame(table).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__": main()
