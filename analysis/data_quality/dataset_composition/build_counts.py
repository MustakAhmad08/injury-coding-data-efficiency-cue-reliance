"""Audit frozen split counts and nesting; export only diagnosis-level counts."""
import hashlib
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'data/neiss_temporal_2021_2025'


def main():
    manifest=json.loads((DATA/'dataset_manifest.json').read_text())
    codes=manifest['selected_diagnosis_codes']
    spec={'train_1k':'ablations/train_1000.csv','train_3k':'ablations/train_3000.csv',
          'train_6k':'ablations/train_6000.csv','development':'validation.csv','test':'locked_test.csv'}
    frames={}; hashes={}
    table=pd.DataFrame({'diagnosis_code':codes,'diagnosis_label':[manifest['selected_diagnosis_labels'][str(c)] for c in codes]})
    for name,filename in spec.items():
        path=DATA/filename; digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==manifest['output_sha256'][filename],filename
        frame=pd.read_csv(path,usecols=['case_id','diag1','year'],dtype={'case_id':str})
        assert not frame.case_id.duplicated().any()
        expected={2024} if name=='development' else {2025} if name=='test' else {2021,2022,2023}
        assert set(frame.year)==expected
        assert set(frame.diag1)==set(codes)
        table[name]=frame.diag1.value_counts().reindex(codes).to_numpy()
        frames[name]=frame; hashes[filename]=digest
    assert set(frames['train_1k'].case_id)<set(frames['train_3k'].case_id)<set(frames['train_6k'].case_id)
    for a,b in [('train_6k','development'),('train_6k','test'),('development','test')]:
        assert not set(frames[a].case_id)&set(frames[b].case_id)
    for size in [1000,3000,6000]:
        col=f'train_{size//1000}k'
        assert table[col].sum()==size
        assert dict(zip(map(str,codes),map(int,table[col])))==manifest['ablation_category_counts'][str(size)]
    assert table.development.eq(10).all() and table.test.eq(50).all()
    assert table.train_3k.eq(100).all() and table.train_6k.eq(200).all()
    assert table.train_1k.value_counts().to_dict()=={33:20,34:10}
    out=HERE/'results'; out.mkdir(exist_ok=True)
    table.to_csv(out/'per_class_split_counts.csv',index=False)
    (out/'provenance.json').write_text(json.dumps({'input_sha256':hashes,'checks':'Frozen file hashes, diagnosis inventory, years, unique IDs, nested training subsets, disjoint temporal splits, manifest counts and totals verified.', 'totals':{k:len(v) for k,v in frames.items()},'new_evaluation':False},indent=2)+'\n')
    m=HERE/'results'; m.mkdir(exist_ok=True)
    rows=[' & '.join([str(r.diagnosis_code),r.diagnosis_label]+[str(getattr(r,k)) for k in spec])+r' \\' for r in table.itertuples()]
    tex=r'''\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{rlrrrrr}
\hline
Code & Diagnosis & Train 1K & Train 3K & Train 6K & Dev. & Test \\
\hline
'''+ '\n'.join(rows)+r'''
\hline
 & Total & 1,000 & 3,000 & 6,000 & 300 & 1,500 \\
\hline
\end{tabular}
\caption{Exact diagnosis counts in the frozen benchmark. The 1K and 3K training sets are nested subsets of the 6K training set, not additional records. Training uses 2021--2023, development 2024, and test 2025. The 6K, 3K, development and test sets are exactly class-balanced; 1K is approximately balanced because 1,000 is not divisible by 30. Original 2025 frequencies are reported separately in Table~\ref{tab:original-distribution}.}
\label{tab:split-class-counts}
\end{table*}
'''
    (m/'counts_table.tex').write_text(tex,encoding='utf-8')
    lines=['# Exact per-class dataset counts','', 'All counts were computed from the frozen CSV files and checked against the original manifest. Training subsets are nested and must not be added together.', '', '| Code | Diagnosis | Train 1K | Train 3K | Train 6K | Development | Test |','|---|---|---:|---:|---:|---:|---:|']
    for r in table.itertuples(): lines.append('| '+' | '.join([str(r.diagnosis_code),r.diagnosis_label]+[str(getattr(r,k)) for k in spec])+' |')
    lines += ['', '**Balance:** all 30 categories have 200 examples in the full training set, 100 in the 3K subset, 10 in development, and 50 in test. The 1K subset has 34 examples for ten categories and 33 for twenty categories.', '', '**Representativeness:** this deliberately balanced benchmark is not representative of diagnosis prevalence in injury-surveillance records. In the supplied 2025 source data, Unknown/Not Stated is 24.74%, fracture 15.99%, and laceration 14.25%, rather than 3.33% each. These source frequencies are unweighted record proportions, not national estimates. Existing benchmark scores do not establish natural-prevalence performance.', '', 'See the source-distribution analysis for the complete original-frequency table. This count analysis does not train models or alter datasets.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Verified exact counts for all 30 classes and all five datasets; nesting and temporal separation passed.')


if __name__=='__main__': main()
