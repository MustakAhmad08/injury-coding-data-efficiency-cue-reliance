"""Report original eligible 2025 record frequencies beside existing recalls.

No model inference, new evaluation sample, or prevalence-weighted performance
estimate is produced. The published output contains aggregates only.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REQUIRED = ['CPSC_Case_Number', 'Diagnosis', 'Body_Part', 'Narrative_1']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def eligible_records(raw, codes):
    """Match read_year's relevant preprocessing, in the same order."""
    audit = {'raw_rows': len(raw)}
    clean = raw.dropna(subset=REQUIRED).copy()
    audit['missing_required_fields_removed'] = len(raw) - len(clean)
    for column in ['CPSC_Case_Number', 'Diagnosis', 'Body_Part']:
        clean[column] = clean[column].astype('int64')
    normalized = clean.Narrative_1.map(lambda x: re.sub(r'\s+', ' ', str(x).strip()).upper())
    keep = normalized.str.len().gt(0)
    audit['empty_narratives_removed'] = int((~keep).sum())
    clean = clean.loc[keep].drop(columns='Narrative_1')
    n = len(clean)
    clean = clean.drop_duplicates('CPSC_Case_Number', keep='first')
    audit['duplicate_case_ids_removed'] = n-len(clean)
    audit['cleaned_rows_all_codes'] = len(clean)
    outside = clean.loc[~clean.Diagnosis.isin(codes)]
    audit['out_of_inventory_counts'] = {str(k):int(v) for k,v in outside.Diagnosis.value_counts().items()}
    audit['out_of_inventory_rows_removed'] = len(outside)
    selected = clean.loc[clean.Diagnosis.isin(codes)]
    audit['eligible_rows_30_codes'] = len(selected)
    assert sum(audit[k] for k in ['missing_required_fields_removed','empty_narratives_removed','duplicate_case_ids_removed','out_of_inventory_rows_removed','eligible_rows_30_codes']) == len(raw)
    return selected, audit


def merge_recalls(counts, reference, encoder, codes):
    output = counts.copy()
    for condition, name in [('tfidf_logreg_6000','tfidf_recall'),('qwen_sft_6000','qwen_recall')]:
        f = reference.loc[reference.condition.eq(condition)]
        if len(f)!=30 or f.diagnosis_code.duplicated().any() or set(f.diagnosis_code)!=set(codes) or not f.support.eq(50).all():
            raise ValueError(f'Invalid balanced-test metrics: {condition}')
        output = output.merge(f[['diagnosis_code','recall']].rename(columns={'recall':name}),on='diagnosis_code',validate='one_to_one')
    if len(encoder)!=90 or encoder.duplicated(['seed','diagnosis_code']).any() or encoder.seed.nunique()!=3 or not encoder.support.eq(50).all():
        raise ValueError('Expected three seeds x 30 diagnoses, each with support 50')
    for seed, f in encoder.groupby('seed'):
        if set(f.diagnosis_code)!=set(codes): raise ValueError(f'Incomplete seed: {seed}')
    aggregate = encoder.groupby('diagnosis_code').recall.agg(['mean','std']).reset_index().rename(columns={'mean':'roberta_recall_mean','std':'roberta_recall_sd'})
    output = output.merge(aggregate,on='diagnosis_code',validate='one_to_one')
    if output.isna().any().any(): raise ValueError('Missing table entries')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-file', type=Path, required=True)
    p.add_argument('--reference-metrics', type=Path)
    p.add_argument('--encoder-metrics', type=Path, default=ROOT/'methods/roberta/results/per_category_metrics.csv')
    args = p.parse_args()
    config = json.loads((HERE/'config.json').read_text())
    if sha(args.source_file)!=config['source_sha256']:
        raise ValueError('Source differs from the workbook used for the original benchmark')
    refs = args.reference_metrics or ROOT/'methods_runs/combined_evaluation/per_category_metrics.csv'
    if not refs.exists(): refs = ROOT/'results/primary/per_category_metrics.csv'
    print('Reading the verified original 2025 workbook...',flush=True)
    raw = pd.read_excel(args.source_file,usecols=REQUIRED)
    selected, audit = eligible_records(raw, config['diagnosis_codes'])
    counts = selected.Diagnosis.value_counts().reindex(config['diagnosis_codes'],fill_value=0)
    total = int(counts.sum())
    if total==0: raise ValueError('No eligible records')
    table = pd.DataFrame({'diagnosis_code':counts.index, 'diagnosis_label':[config['diagnosis_labels'][str(x)] for x in counts.index],
                          'original_2025_count':counts.values, 'original_2025_percent':counts.values/total*100,
                          'balanced_test_count':50, 'balanced_test_percent':100/30})
    table = merge_recalls(table,pd.read_csv(refs),pd.read_csv(args.encoder_metrics),config['diagnosis_codes'])
    table = table[['diagnosis_code','diagnosis_label','original_2025_count','original_2025_percent','balanced_test_count','balanced_test_percent','tfidf_recall','roberta_recall_mean','roberta_recall_sd','qwen_recall']]
    assert len(table)==30 and table.original_2025_count.sum()==total
    assert abs(table.original_2025_percent.sum()-100)<1e-9
    assert table.balanced_test_count.sum()==1500
    out=HERE/'results'; out.mkdir(exist_ok=True)
    table.to_csv(out/'original_2025_distribution_and_recall.csv',index=False)
    audit.update({'source_file':args.source_file.name,'source_sha256':sha(args.source_file),
                  'reference_metrics_sha256':sha(refs),'encoder_metrics_sha256':sha(args.encoder_metrics),
                  'builder_sha256':sha(Path(__file__)), 'config_sha256':sha(HERE/'config.json'),
                  'denominator':'Eligible unique records in the supplied 2025 workbook restricted to the frozen 30-code inventory, after the original required-field and narrative cleaning.',
                  'frequency_scope':'Unweighted source-record frequencies, not survey-weighted national injury estimates.',
                  'recall_scope':'Previously measured recall on 50 balanced-test cases per diagnosis. Encoder mean and sample SD across three training seeds on the same cases. No inference on the remaining source records.',
                  'pandas_version':pd.__version__, 'new_model_evaluation':False})
    (out/'provenance.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)
    print(table.sort_values('original_2025_count',ascending=False).head(5).to_string(index=False),flush=True)


if __name__=='__main__': main()
