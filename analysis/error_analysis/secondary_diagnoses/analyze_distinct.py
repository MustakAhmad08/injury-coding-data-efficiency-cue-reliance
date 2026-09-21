"""Descriptive analysis of frozen predictions on 254 distinct-diagnosis cases.

No inference, prompt changes or diagnosis-term masking is performed.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def code(value):
    m=re.fullmatch(r'\s*(\d{2})(?:\.0+)?(?:\s*\([^\n]*\))?\s*',str(value))
    return int(m.group(1)) if m else -1


def wilson(k,n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,mid-half),min(1,mid+half)


def subset(gold):
    a=gold.diag1.map(code); b=gold.diag2.map(code)
    return gold.loc[b.ne(-1)&b.ne(a)].copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir',type=Path,default=ROOT/'methods_runs')
    parser.add_argument('--test-file',type=Path,default=ROOT/'data/neiss_temporal_2021_2025/locked_test.csv')
    parser.add_argument('--encoder-predictions',type=Path,default=ROOT/'methods/roberta/local/test_predictions')
    args=parser.parse_args()
    gold=pd.read_csv(args.test_file,dtype={'case_id':str})
    assert len(gold)==1500 and not gold.case_id.duplicated().any()
    assert hashlib.sha256(args.test_file.read_bytes()).hexdigest()=='19797d36307469b7522fc21947e72c74fa76dae8264b499ef74ca91ab5044d15'
    distinct=subset(gold).set_index('case_id').sort_index()
    all_secondary=gold.diag2.map(code).ne(-1)
    assert all_secondary.sum()==332 and len(distinct)==254
    codes=set(gold.diag1.astype(int)); n=len(distinct)
    paths={name:args.runs_dir/name/'test_results.csv' for name in ['tfidf_logreg_6000','qwen_sft_6000','mistral_sft_3000','gpt54_10shot']}
    paths.update({f'roberta_seed_{seed}':args.encoder_predictions/f'seed_{seed}.csv' for seed in [20260718,20260719,20260720]})
    rows=[]; hashes={}
    for name,path in paths.items():
        p=pd.read_csv(path,dtype={'case_id':str})
        assert len(p)==1500 and not p.case_id.duplicated().any()
        assert set(p.case_id)==set(gold.case_id)
        aligned=p.set_index('case_id').sort_index()
        truth=gold.set_index('case_id').sort_index()
        assert np.array_equal(aligned.true_diag1.map(code),truth.diag1.map(code))
        if 'true_diag2' in aligned:
            assert np.array_equal(aligned.true_diag2.map(code),truth.diag2.map(code))
        p=aligned.loc[distinct.index]
        g1=distinct.diag1.map(code).to_numpy(); g2=distinct.diag2.map(code).to_numpy()
        p1=p.pred_diag1.map(code).to_numpy()
        primary=p1==g1
        k=int(primary.sum()); lo,hi=wilson(k,n)
        row={'condition':name,'n':n,'primary_correct':k,'primary_accuracy':k/n,'primary_wilson_low':lo,'primary_wilson_high':hi,
             'gold_secondary_in_primary_count':int((p1==g2).sum()),'secondary_evaluated':'pred_diag2' in p}
        if 'pred_diag2' in p:
            p2=p.pred_diag2.map(code).to_numpy()
            p2=np.array([v if v in codes else -1 for v in p2])
            for label,mask in [('secondary',p2==g2),('both_ordered',primary&(p2==g2)),('exact_swap',(p1==g2)&(p2==g1))]:
                k=int(mask.sum()); lo,hi=wilson(k,n)
                row.update({f'{label}_correct' if label!='exact_swap' else 'exact_swap_count':k,f'{label}_accuracy' if label!='exact_swap' else 'exact_swap_rate':k/n,f'{label}_wilson_low':lo,f'{label}_wilson_high':hi})
        rows.append(row); hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    out=HERE/'results'; out.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows); frame.to_csv(out/'distinct_254_metrics.csv',index=False)
    manifest={'test_sha256':hashlib.sha256(args.test_file.read_bytes()).hexdigest(),'prediction_sha256':hashes,
              'n_test':1500,'n_secondary_present':332,'n_repeated_codes':78,'n_distinct_codes':254,
              'subset_definition':'Gold secondary diagnosis present and different from the gold primary diagnosis; not selected by model outputs.',
              'interval':'95% Wilson binomial intervals for each fixed-model proportion; descriptive, unadjusted, not training-seed uncertainty or a model comparison test.',
              'secondary_scope':'Secondary/both-ordered/swap metrics apply only to generative models that predict both slots. Missing or invalid secondary predictions count as incorrect. Blank fields for primary-only models mean not applicable.',
              'inference':'Retrospective descriptive subgroup analysis. Narratives are unmodified and may still contain explicit diagnosis terms. These metrics cannot distinguish lexical extraction from contextual reasoning.',
              'new_model_inference':False,'diagnosis_term_only_masking_performed':False}
    (out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(frame[['condition','n','primary_accuracy','secondary_accuracy','both_ordered_accuracy','exact_swap_rate']].to_string(index=False))


if __name__=='__main__': main()
