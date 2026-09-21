"""Check complete analysis outputs without requiring private narratives or weights."""
import csv
import json
import re
from pathlib import Path
from verify_release import main as verify_original

ROOT=Path(__file__).resolve().parents[1]
def verify_analysis_layout():
    groups = {
        'data_quality': {'dataset_composition', 'source_distribution', 'label_audit'},
        'robustness': {'proportional_test', 'diagnosis_masking'},
        'error_analysis': {'secondary_diagnoses', 'qualitative_errors'},
    }
    analysis = ROOT / 'analysis'
    def directories(folder):
        return {p.name for p in folder.iterdir()
                if p.is_dir() and not p.name.startswith('.') and p.name != '__pycache__'}
    actual = directories(analysis)
    assert actual == set(groups), f'Unexpected analysis folders (possibly obsolete copies): {actual ^ set(groups)}'
    for group, studies in groups.items():
        assert (analysis / group / 'README.md').is_file()
        actual_studies = directories(analysis / group)
        assert actual_studies == studies, (group, actual_studies ^ studies)
        for study in studies:
            assert (analysis / group / study / 'README.md').is_file()
            assert (analysis / group / study / 'results').is_dir()

def main():
    verify_analysis_layout()
    assert (ROOT / "methods/sft/train.py").is_file()
    assert (ROOT / "results/error_pairs/error_pair_counts.csv").is_file()
    assert (ROOT / "results/secondary_all_cases/secondary_diagnosis_summary.csv").is_file()
    verify_original()
    assert (ROOT/'VERSION').read_text().strip()=='2.0.0'
    p=ROOT/'analysis/robustness/proportional_test/results'
    with (p/'metrics.csv').open(newline='',encoding='utf-8') as f:metrics=[r for r in csv.DictReader(f) if not r['condition'].startswith('roberta') or r['condition']=='roberta_seed_20260718']
    assert {r['condition'] for r in metrics}=={'tfidf_logreg_6000','qwen_sft_6000','mistral_sft_3000','roberta_seed_20260718'}
    assert all(int(r['n'])==1500 for r in metrics)
    with (p/'per_class.csv').open(newline='',encoding='utf-8') as f:rows=[r for r in csv.DictReader(f) if not r['condition'].startswith('roberta') or r['condition']=='roberta_seed_20260718']
    assert len(rows)==120 and len({(r['condition'],r['diagnosis_code']) for r in rows})==120
    assert all('one_vs_rest_accuracy' not in r and all(k in r for k in ['precision','recall','f1']) for r in rows)
    for condition in {r['condition'] for r in rows}:
        group=[r for r in rows if r['condition']==condition]
        assert sum(int(r['support']) for r in group)==1500
        assert sum(int(r['support'])==0 for r in group)==2
    audit=json.loads((ROOT/'analysis/data_quality/label_audit/results/summary.json').read_text())
    assert sum(audit['counts'].values())==audit['n']==100
    assert abs(audit['observed_agreement']-.93)<1e-12
    assert abs(audit['cohens_kappa']-.5930232558139534)<1e-12
    assert audit['stored_agreement_column_mismatches']==13
    restricted=[p for p in ROOT.rglob('*') if p.is_file() and (p.suffix in {'.safetensors','.joblib'} or 'local' in p.relative_to(ROOT).parts)]
    assert not restricted,restricted
    assert not (ROOT/'revisions').exists()
    assert not list(ROOT.rglob('*.docx'))
    for page in ROOT.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',page.read_text(encoding='utf-8')):
            target=target.strip('<>').split('#')[0]
            if target and '://' not in target and not target.startswith('mailto:'):
                assert (page.parent/target).exists(), (page.relative_to(ROOT),target)
    print('V2 checks passed: primary seed, 4 x 30 proportional metrics, sample counts, audit snapshot, no local/model artifacts.')

if __name__=='__main__':main()
