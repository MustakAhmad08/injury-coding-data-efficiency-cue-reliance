"""Post hoc 1,500-case proportional test, with frozen local classifiers."""
import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.prepare_neiss_temporal_dataset import read_year, normalized_narrative, DIAGNOSIS_LABELS
SEED=20260918
DATA=ROOT/'data/neiss_temporal_2021_2025'
TEST=HERE/'local/proportional_test_1500.csv'
CONDITIONS=['tfidf_logreg_6000','qwen_sft_6000','mistral_sft_3000']+[f'roberta_seed_{s}' for s in [20260718,20260719,20260720]]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')

def allocate(counts,n=1500):
    counts=counts.sort_index()
    exact=counts/counts.sum()*n
    quota=np.floor(exact).astype(int)
    ranked=sorted(counts.index,key=lambda c:(-(exact[c]-quota[c]),c))
    for c in ranked[:n-int(quota.sum())]:quota[c]+=1
    assert quota.sum()==n
    return quota

def prepare(source):
    if TEST.exists():raise FileExistsError('Sample already frozen; do not resample.')
    assert sha(source)=='1cf24d2b2586d320e53e2d63049a5f32136c500efb5dda654ca531f0792ab563'
    original=read_year(Path(source).parent,2025)
    codes=sorted(pd.read_csv(DATA/'locked_test.csv').diag1.unique())
    original=original[original.diag1.isin(codes)].copy()
    assert len(original)==410201
    counts=original.diag1.value_counts().reindex(codes,fill_value=0).sort_index()
    quota=allocate(counts)
    pool=original.copy(); exclusions={}; hashes={}
    previous=[]
    for name in ['train.csv','validation.csv','locked_test.csv']:
        frame=pd.read_csv(DATA/name);previous.append(frame)
        hashes[name]=sha(DATA/name)
        match=pool.case_id.isin(frame.case_id)
        exclusions[name+'_case_ids']=int(match.sum());pool=pool.loc[~match].copy()
        match=pool.narrative.isin(frame.narrative.map(normalized_narrative))
        exclusions[name+'_additional_exact_narratives']=int(match.sum());pool=pool.loc[~match].copy()
    before=len(pool)
    pool=pool.sort_values('case_id').drop_duplicates('narrative',keep='first')
    exclusions['within_pool_duplicate_narratives']=before-len(pool)
    available=pool.diag1.value_counts().reindex(codes,fill_value=0)
    assert (available>=quota).all()
    pieces=[pool[pool.diag1.eq(c)].sample(n=int(quota[c]),random_state=SEED+int(c)*101) for c in codes if quota[c]>0]
    test=pd.concat(pieces).sample(frac=1,random_state=SEED).reset_index(drop=True)
    test['diagnosis_category']=test.diag1.map(DIAGNOSIS_LABELS)
    assert len(test)==1500 and test.case_id.nunique()==1500 and test.narrative.nunique()==1500
    for f in previous:
        assert not set(test.case_id)&set(f.case_id)
        assert not set(test.narrative)&set(f.narrative.map(normalized_narrative))
    TEST.parent.mkdir(parents=True,exist_ok=True);test.to_csv(TEST,index=False)
    test[['case_id']].to_csv(HERE/'local/selected_case_ids.csv',index=False)
    dist=pd.DataFrame({'diagnosis_code':codes,'diagnosis_category':[DIAGNOSIS_LABELS[c] for c in codes],
        'source_count':[int(counts[c]) for c in codes],'source_fraction':[counts[c]/counts.sum() for c in codes],
        'available_after_exclusions':[int(available[c]) for c in codes],'test_count':[int(quota[c]) for c in codes]})
    dist['test_fraction']=dist.test_count/1500
    (HERE/'results').mkdir(exist_ok=True);dist.to_csv(HERE/'results/class_distribution.csv',index=False)
    manifest={'design':'Post hoc proportional-stratified sample, fixed n=1500; not an independent prospective validation.',
      'seed':SEED,'source_sha256':sha(source),'source_n':410201,'test_n':1500,'test_sha256':sha(TEST),
      'previous_split_sha256':hashes,'exclusions':exclusions,'available_n':len(pool),
      'allocation':'Largest remainder allocation from original 2025 counts BEFORE overlap exclusions; ties by ascending code. Random sampling without replacement within each class. No minimum per class.',
      'target':'Unweighted observed 2025 source-record diagnosis proportions, approximated by integer quotas. Not national survey-weighted prevalence.',
      'sampling_limitation':'Sampling within class is restricted to records with unseen exact normalized narratives; duplicate removal and previous-test exclusion may alter within-class composition. This is not a simple random sample of all source records.',
      'zero_support_codes':[int(c) for c in codes if quota[c]==0],'supported_classes':int((quota>0).sum()),
      'models':CONDITIONS,'evaluation':'Unmasked narratives; frozen original prompts and checkpoints; no tuning or model selection. Accuracy and support-weighted F1 are primary; macro F1 over supported labels and fixed 30-label macro F1 are explicitly distinguished.',
      'uncertainty':'1000 paired stratified bootstrap samples, preserving allocated class counts; conditional on this allocation and each frozen model. Three encoder seeds reported separately and as mean/sample SD.',
      'builder_sha256':sha(__file__)}
    write_json(HERE/'results/sampling_manifest.json',manifest)
    print(dist.to_string(index=False));print('Exclusions:',exclusions,flush=True)

def infer(condition):
    protocol=json.loads((HERE/'results/sampling_manifest.json').read_text())
    assert sha(TEST)==protocol['test_sha256']
    target=HERE/'local'/condition/'test_results.csv';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        meta=json.loads((HERE/'results'/f'{condition}_provenance.json').read_text())
        assert meta['test_sha256']==sha(TEST) and meta['predictions_sha256']==sha(target)
        print('Verified existing results:',condition);return
    d=pd.read_csv(TEST)
    if condition.startswith('tfidf'):
        import joblib
        path=ROOT/'methods_runs/tfidf_logreg_6000/tfidf_logreg_pipeline.joblib'
        pred=joblib.load(path).predict(d.narrative)
        result=pd.DataFrame(dict(case_id=d.case_id,true_diag1=d.diag1,pred_diag1=pred))
        meta={'pipeline_sha256':sha(path)}
    elif condition.startswith('roberta'):
        from methods.roberta import run_encoder as e
        seed=int(condition.rsplit('_',1)[1]);e.seed_everything(seed)
        selection=json.loads((e.HERE/'results/validation_selection.json').read_text())
        config=selection['config'];codes=selection['codes']
        trial=next(t for t in selection['selected_trials'] if t['seed']==seed)
        ckpt=e.HERE/trial['checkpoint'];assert sha(ckpt/'model.safetensors')==trial['checkpoint_sha256']
        tokenizer=e.AutoTokenizer.from_pretrained(str(ckpt),local_files_only=True)
        rows,stats=e.encode(d,tokenizer,codes,config)
        model=e.load_model(ckpt,codes)
        pred=e.predict(model,e.loader(rows,tokenizer,config),codes)
        result=pd.DataFrame(dict(case_id=d.case_id,true_diag1=d.diag1,pred_diag1=pred))
        meta={'seed':seed,'checkpoint_sha256':trial['checkpoint_sha256'],'selection_sha256':sha(e.HERE/'results/validation_selection.json'),'tokenization':stats}
        del model;gc.collect();e.torch.cuda.empty_cache()
    else:
        from transformers import set_seed
        from methods.sft.train import load_model_and_tokenizer,PeftModel,evaluate_test_set
        set_seed(20260718)
        base={'qwen_sft_6000':'Qwen/Qwen2.5-7B-Instruct','mistral_sft_3000':'mistralai/Mistral-7B-Instruct-v0.3'}[condition]
        adapter=Path(os.environ.get('NEISS_ADAPTER_ROOT', str(ROOT/'methods_runs')))/condition
        model,tokenizer=load_model_and_tokenizer(base)
        old=json.loads((ROOT/'configs/recorded_sft_settings.json').read_text())[condition]
        assert model.config._commit_hash==old['model_commit']
        model=PeftModel.from_pretrained(model,adapter);model.eval()
        meta={'base_model':base,'base_commit':model.config._commit_hash,'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'batch_size':8,'max_new_tokens':256,'greedy':True,'inference_seed':20260718,'test_sha256':sha(TEST)}
        state=target.parent/'run_state.json'
        if state.exists():assert json.loads(state.read_text())==meta
        else:write_json(state,meta)
        chunks=[]
        for start in range(0,len(d),80):
            part=d.iloc[start:start+80];saved=target.parent/f'chunk_{start:04d}.csv'
            if saved.exists():
                out=pd.read_csv(saved)
                assert out.case_id.tolist()==part.case_id.tolist() and out.narrative.tolist()==part.narrative.tolist()
            else:
                inp=target.parent/f'input_{start:04d}.csv';part.to_csv(inp,index=False)
                out,_=evaluate_test_set(str(inp),model,tokenizer,data_mode='neiss',max_new_tokens=256,inference_batch_size=8)
                temporary=saved.with_suffix('.tmp');out.to_csv(temporary,index=False);temporary.replace(saved)
            chunks.append(out);print(f'Checkpointed {start+len(part)}/1500',flush=True)
        result=pd.concat(chunks,ignore_index=True)
    result.to_csv(target,index=False)
    meta.update(test_sha256=sha(TEST),predictions_sha256=sha(target))
    write_json(HERE/'results'/f'{condition}_provenance.json',meta)
    print('Saved',condition,len(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer']);p.add_argument('--source-file',type=Path);p.add_argument('--condition',choices=CONDITIONS);a=p.parse_args()
    if a.stage=='prepare':prepare(a.source_file)
    else:infer(a.condition)
