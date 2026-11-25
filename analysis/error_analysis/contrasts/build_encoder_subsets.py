"""Derive diagnosis-pair accuracy from published encoder per-category recall.

Each pair has 50 cases per code. No private predictions are needed. This does
not estimate pair-restricted F1 or bootstrap intervals from rounded metrics.
"""
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
PAIRS = [('scald_vs_thermal_burn',48,51), ('electrical_burn_vs_shock',46,67),
         ('concussion_vs_internal_injury',52,62), ('laceration_vs_avulsion',59,72),
         ('ingestion_vs_aspiration',41,42)]

def build_rows():
    with (ROOT / 'methods/roberta/results/per_category_metrics.csv').open(newline='', encoding='utf-8') as f:
        metrics = {(int(r['seed']), int(r['diagnosis_code'])): r for r in csv.DictReader(f)}
    rows = []
    for seed in (20260718,20260719,20260720):
        for name,a,b in PAIRS:
            pair = [metrics[seed,c] for c in (a,b)]
            assert all(float(r['support']) == 50 for r in pair)
            correct = sum(round(float(r['recall'])*50) for r in pair)
            rows.append(dict(contrast_id=name, code_a=a, code_b=b,
                             model_key=f'roberta_seed_{seed}', n=100,
                             correct=correct, accuracy=correct/100))
    return rows

def main():
    rows=build_rows()
    target=ROOT / 'results/contrasts/encoder_pair_accuracy.csv'
    with target.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__ == '__main__': main()
