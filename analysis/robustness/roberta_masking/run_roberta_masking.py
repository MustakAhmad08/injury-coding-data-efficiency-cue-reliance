"""Reproduce RoBERTa masking with reconstructed private inputs and frozen models.

Published results are never overwritten by default. --help needs only Python.
See README.md for preparation, checkpoint identity, and replication limitations.
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path(os.environ.get(
        'NEISS_ROBERTA_RUNS', str(ROOT / 'methods/roberta/local/runs'))))
    parser.add_argument('--original-predictions', type=Path,
                        default=ROOT / 'methods/roberta/local/test_predictions')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'methods_runs/roberta_masking')
    parser.add_argument('--allow-new-checkpoints', action='store_true',
                        help='Permit separately trained checkpoints; report a replication, not an exact archived-model check.')
    args = parser.parse_args()
    import hashlib, json, re
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np, pandas as pd, torch
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

    P = ROOT
    RUNS = args.runs.resolve()
    OUT = args.output_dir.resolve(); OUT.mkdir(parents=True, exist_ok=True)
    LOCAL = OUT / 'local'; LOCAL.mkdir(exist_ok=True)
    SEEDS = [20260718, 20260719, 20260720]
    SEED = 20260718; N_BOOT = 1000

    def digest(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
        return h.hexdigest()

    def code(v):
        m = re.fullmatch(r'\s*(\d{2})(?:\.0+)?(?:\s*\([^\n]*\))?\s*', str(v))
        return int(m.group(1)) if m else -1

    test = pd.read_csv(P / 'data/neiss_temporal_2021_2025/locked_test.csv', dtype={'case_id': str})
    assert digest(P / 'data/neiss_temporal_2021_2025/locked_test.csv').startswith('19797d363074')
    ids = test.case_id.tolist()
    variants = {'original': test}
    for v in ['marker_only', 'diagnosis_tail']:
        variants[v] = pd.read_csv(P / f'methods_runs/cue_masking/{v}.csv', dtype={'case_id': str})
    variants['term_mask'] = pd.read_csv(P / 'analysis/robustness/diagnosis_masking/local/masked_test.csv', dtype={'case_id': str})
    for k, f in variants.items():
        variants[k] = f.set_index('case_id').loc[ids].reset_index()
        assert variants[k].diag1.astype(int).tolist() == test.diag1.astype(int).tolist(), k
    y = test.diag1.astype(int).to_numpy()
    codes = sorted(set(y.tolist()))

    @torch.inference_mode()
    def predict(model, tok, texts):
        enc = tok(texts, truncation=True, max_length=512, padding=False)
        rows = [{k: enc[k][i] for k in enc} for i in range(len(texts))]
        dl = DataLoader(rows, batch_size=64, shuffle=False, collate_fn=DataCollatorWithPadding(tok, pad_to_multiple_of=8))
        out = []
        for b in dl:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(**{k: v.cuda() for k, v in b.items()}).logits
            out.extend(codes[i] for i in logits.argmax(-1).cpu().tolist())
        return np.array(out)

    def metric(g, p):
        labels = sorted(np.unique(g).tolist())
        return float(accuracy_score(g, p)), float(f1_score(g, p, labels=labels, average='macro', zero_division=0))

    def paired_drop_ci(g, a, b, labels):
        rng = np.random.default_rng(SEED)
        groups = [np.flatnonzero(g == l) for l in labels]
        ad, fd = [], []
        for _ in range(N_BOOT):
            ix = np.concatenate([rng.choice(gr, len(gr), replace=True) for gr in groups])
            oa, of = metric(g[ix], a[ix]); ma, mf = metric(g[ix], b[ix])
            ad.append(oa - ma); fd.append(of - mf)
        q = lambda x: (float(np.quantile(x, .025)), float(np.quantile(x, .975)))
        return q(ad) + q(fd)

    d2 = test.diag2.map(code).to_numpy()
    distinct = (d2 != -1) & (d2 != y)
    assert distinct.sum() == 254
    changed_term = (test.narrative != variants['term_mask'].narrative).to_numpy()

    rows, checks = [], []
    for s in SEEDS:
        ck = RUNS / f'lr_5e-05_seed_{s}/checkpoint'
        prov = json.load(open(P / f'analysis/robustness/proportional_test/results/roberta_seed_{s}_provenance.json'))
        actual_checkpoint_hash = digest(ck / 'model.safetensors')
        if not args.allow_new_checkpoints:
            assert actual_checkpoint_hash == prov['checkpoint_sha256'], 'Checkpoint differs from the published experiment; use --allow-new-checkpoints for a separately labelled replication.'
        tok = AutoTokenizer.from_pretrained(str(ck), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(ck), local_files_only=True, attn_implementation='sdpa').cuda().eval()
        torch.manual_seed(s)
        preds = {k: predict(model, tok, f.narrative.astype(str).tolist()) for k, f in variants.items()}
        arch = pd.read_csv(args.original_predictions / f'seed_{s}.csv', dtype={'case_id': str}).set_index('case_id').loc[ids]
        agree = int((arch.pred_diag1.astype(int).to_numpy() == preds['original']).sum())
        checks.append(dict(seed=s, checkpoint_sha256=actual_checkpoint_hash, published_checkpoint_match=(actual_checkpoint_hash == prov['checkpoint_sha256']), archived_prediction_agreement=agree, n=1500,
                           term_unchanged_disagreements=int((preds['original'][~changed_term] != preds['term_mask'][~changed_term]).sum()),
                           term_unchanged_n=int((~changed_term).sum())))
        pd.DataFrame({'case_id': ids, 'true_diag1': y, **{f'pred_{k}': v for k, v in preds.items()}}).to_csv(LOCAL / f'predictions_seed_{s}.csv', index=False)
        base = arch.pred_diag1.astype(int).to_numpy()  # archived predictions define the reported original
        oa, of = metric(y, base)
        for v in ['marker_only', 'diagnosis_tail', 'term_mask']:
            ma, mf = metric(y, preds[v])
            ci = paired_drop_ci(y, base, preds[v], np.array(codes))
            rows.append(dict(seed=s, variant=v, original_accuracy=oa, masked_accuracy=ma, original_macro_f1=of, masked_macro_f1=mf,
                             macro_f1_drop=of - mf, f1_drop_ci_low=ci[2], f1_drop_ci_high=ci[3],
                             accuracy_drop=oa - ma, acc_drop_ci_low=ci[0], acc_drop_ci_high=ci[1],
                             distinct254_original_acc=float((base[distinct] == y[distinct]).mean()),
                             distinct254_masked_acc=float((preds[v][distinct] == y[distinct]).mean())))
        del model; torch.cuda.empty_cache()

    res = pd.DataFrame(rows); res.to_csv(OUT / 'roberta_masking_summary.csv', index=False)
    inputs = {k: digest(p) for k, p in {
        'locked_test.csv': P / 'data/neiss_temporal_2021_2025/locked_test.csv',
        'marker_only.csv': P / 'methods_runs/cue_masking/marker_only.csv',
        'diagnosis_tail.csv': P / 'methods_runs/cue_masking/diagnosis_tail.csv',
        'term_masked_test.csv': P / 'analysis/robustness/diagnosis_masking/local/masked_test.csv'}.items()}
    for c in checks:
        c['predictions_sha256'] = digest(LOCAL / f"predictions_seed_{c['seed']}.csv")
    json.dump(dict(replication=True, input_sha256=inputs, script_sha256=digest(__file__), checks=checks, env=dict(torch=torch.__version__, cuda=torch.version.cuda,
                   transformers=__import__('transformers').__version__, gpu=torch.cuda.get_device_name(0))),
              open(OUT / 'roberta_masking_checks.json', 'w'), indent=2)
    pd.set_option('display.width', 250)
    print(json.dumps(checks, indent=1))
    print(res.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
