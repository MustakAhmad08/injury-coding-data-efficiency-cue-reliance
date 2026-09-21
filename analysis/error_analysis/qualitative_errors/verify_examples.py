"""Verify two thermal-burn examples against existing frozen predictions.

Case identifiers are provided in ignored local/selected_cases.json.
Only aggregate counts and this script are copied into the release.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PATHS={
    'TF-IDF':'methods_runs/tfidf_logreg_6000/test_results.csv',
    'RoBERTa':'methods/roberta/local/test_predictions/seed_20260718.csv',
    'Qwen':'methods_runs/qwen_sft_6000/test_results.csv',
    'Mistral':'methods_runs/mistral_sft_3000/test_results.csv',
}
EXPECTED={'A':{'TF-IDF':51,'RoBERTa':51,'Qwen':51,'Mistral':48},
          'B':{'TF-IDF':51,'RoBERTa':51,'Qwen':47,'Mistral':46}}

def main():
    cases=json.loads((HERE/'local/selected_cases.json').read_text())
    goldpath=ROOT/'data/neiss_temporal_2021_2025/locked_test.csv'
    gold=pd.read_csv(goldpath,dtype={'case_id':str}).set_index('case_id')
    selected=gold.index[gold.diag1.eq(51)]
    assert len(selected)==50
    rows=[];hashes={'balanced_test':hashlib.sha256(goldpath.read_bytes()).hexdigest()}
    for name,relative in PATHS.items():
        path=ROOT/relative
        frame=pd.read_csv(path,dtype={'case_id':str}).set_index('case_id')
        assert frame.index.is_unique and set(frame.index)==set(gold.index)
        assert (frame.loc[gold.index,'true_diag1'].to_numpy()==gold.diag1.to_numpy()).all()
        for label,case in cases.items():
            assert gold.loc[case,'diag1']==51
            assert frame.loc[case,'pred_diag1']==EXPECTED[label][name]
        counts=frame.loc[selected,'pred_diag1'].value_counts()
        for code,count in counts.items():
            rows.append({'model':name,'gold_code':51,'predicted_code':int(code),'count':int(count),'category_n':50})
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE/'results').mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE/'results/thermal_burn_confusions.csv',index=False)
    (HERE/'results/verification.json').write_text(json.dumps({'verified':True,'seed':20260718,'test':'original balanced 2025 test','gold_category':51,'category_n':50,'examples':EXPECTED,'input_sha256':hashes,'selection':'Retrospective illustrative selection; no model inference or gold-label adjudication.'},indent=2)+'\n')
    print('Verified 8 example predictions and all 200 category-level predictions.')

if __name__=='__main__':main()
