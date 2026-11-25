"""Validate camera-ready aggregates and provenance without data, GPUs or networks."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {20260718, 20260719, 20260720}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    study = ROOT / 'analysis/robustness/roberta_masking'
    with (study / 'results/roberta_masking_summary.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    expected = {(s,v) for s in SEEDS for v in ('marker_only','diagnosis_tail','term_mask')}
    assert len(rows) == 9 and {(int(r['seed']),r['variant']) for r in rows} == expected
    encoder = json.loads((ROOT/'methods/roberta/results/test_summary.json').read_text())
    original = {int(r['seed']):r for r in encoder['runs']}
    primary = {}
    for row in rows:
        seed = int(row['seed'])
        for metric in ('accuracy','macro_f1'):
            assert math.isclose(float(row['original_'+metric]),original[seed][metric],abs_tol=1e-12)
        drop = float(row['original_macro_f1']) - float(row['masked_macro_f1'])
        assert math.isclose(drop,float(row['macro_f1_drop']),abs_tol=1e-12)
        assert float(row['f1_drop_ci_low']) <= drop <= float(row['f1_drop_ci_high'])
        for k in ('original_accuracy','masked_accuracy','original_macro_f1','masked_macro_f1',
                  'distinct254_original_acc','distinct254_masked_acc'):
            assert 0 <= float(row[k]) <= 1
        if seed == 20260718: primary[row['variant']] = row
    for variant,drop in [('marker_only',.003),('diagnosis_tail',.433),('term_mask',.379)]:
        assert round(float(primary[variant]['macro_f1_drop']),3) == drop
    assert round(float(primary['term_mask']['masked_macro_f1']),3) == .519
    assert round(float(primary['diagnosis_tail']['masked_macro_f1']),3) == .465
    assert math.isclose(float(primary['term_mask']['distinct254_original_acc']),200/254)
    assert math.isclose(float(primary['term_mask']['distinct254_masked_acc']),113/254)
    checks = json.loads((study/'results/roberta_masking_checks.json').read_text())
    assert {r['seed'] for r in checks['checks']} == SEEDS
    assert checks['input_sha256']['locked_test.csv'] == encoder['test_sha256']
    for r in checks['checks']:
        assert r['n'] == r['archived_prediction_agreement'] == 1500
        assert r['term_unchanged_n'] == 409 and r['term_unchanged_disagreements'] == 0
        prov = json.loads((ROOT/f"analysis/robustness/proportional_test/results/roberta_seed_{r['seed']}_provenance.json").read_text())
        assert r['checkpoint_sha256'] == prov['checkpoint_sha256']
    adaptation = json.loads((study/'provenance/release_adaptation.json').read_text())
    assert checks['script_sha256'] == digest(study/'provenance/original_run_roberta_masking.py.txt')
    assert adaptation['original_experiment_script_sha256'] == checks['script_sha256']
    assert adaptation['adapted_script_sha256'] == digest(ROOT/adaptation['adapted_script'])
    with (ROOT/'results/contrasts/encoder_pair_accuracy.csv').open(newline='',encoding='utf-8') as f:
        pairs = list(csv.DictReader(f))
    assert len(pairs) == 15 and len({(r['model_key'],r['contrast_id']) for r in pairs}) == 15
    with (ROOT/'methods/roberta/results/per_category_metrics.csv').open(newline='',encoding='utf-8') as f:
        per_code = {(int(r['seed']),int(r['diagnosis_code'])):r for r in csv.DictReader(f)}
    for r in pairs:
        seed = int(r['model_key'].rsplit('_',1)[1])
        correct = sum(round(float(per_code[seed,int(r[c])]['recall'])*50) for c in ('code_a','code_b'))
        assert int(r['n']) == 100 and correct == int(r['correct'])
        assert math.isclose(float(r['accuracy']),correct/100)
    assert math.isclose(sum(float(r['accuracy']) for r in pairs if r['model_key']=='roberta_seed_20260718')/5,.894)
    manifest = json.loads((ROOT/'figures/figure_manifest.json').read_text())
    for name,expected_hash in manifest['inputs'].items():
        assert digest(ROOT/name) == expected_hash, name
    for name,expected_hash in manifest['output_sha256'].items():
        assert digest(ROOT/name) == expected_hash, name
    assert manifest['generator_sha256'] == digest(ROOT/'analysis/publication_figures/generate_publication_figures.py')
    assert set(manifest['outputs']) == set(manifest['output_sha256'])
    assert (ROOT/'VERSION').read_text().strip() == '2.1.0'
    assert 'version: 2.1.0' in (ROOT/'CITATION.cff').read_text()
    assert 'Anonymous Authors' not in (ROOT/'LICENSE').read_text()
    print('Camera-ready checks passed: nine masking rows, subgroup scores, provenance, encoder subsets, and figure hashes.')


if __name__ == '__main__': main()
