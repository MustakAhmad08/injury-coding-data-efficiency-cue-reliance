"""Frozen-model, post hoc diagnosis-term masking. No labels used to locate spans."""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Conservative diagnostic nouns/abbreviations; deliberately not mechanism verbs
# (fell, scraped, swallowed), generic injury/pain, or anatomical locations.
TERMS = [
 r'fractures?', r'fxs?', r'lacerations?', r'lacs?', r'contusions?', r'abrasions?',
 r'hematomas?', r'haematomas?', r'concussions?', r'dislocations?', r'subluxations?',
 r'sprains?', r'strains?', r'avulsions?', r'amputations?', r'punctures?',
 r'burns?', r'scalds?', r'sunburns?', r'dermatitis', r'conjunctivitis',
 r'hemorrhag(?:e|es|ing)', r'haemorrhag(?:e|es|ing)', r'epistaxis',
 r'anoxia', r'hypoxia', r'asphyxia(?:tion)?', r'aspiration', r'ingestion',
 r'poisoning', r'overdos(?:e|es)', r'electrocution', r'electric(?:al)?\s+shock',
 r'drowning', r'submersion', r'foreign\s+bod(?:y|ies)',
 r'crush(?:ing)?\s+injur(?:y|ies)', r'nerve\s+damage', r'neuropathy',
 r'pneumothora(?:x|ces)', r'hemothora(?:x|ces)', r'rhabdomyolysis',
 r'traumatic\s+brain\s+injur(?:y|ies)', r'TBI', r'CHI',
]
PATTERN = re.compile(r'\b(?:' + '|'.join(sorted(TERMS, key=len, reverse=True)) + r')\b', re.I)
SEED = 20260718
DATA = ROOT/'data/neiss_temporal_2021_2025/locked_test.csv'

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def mask(text):
    matches = list(PATTERN.finditer(text))
    return PATTERN.sub('[MASK]', text), matches

def prepare():
    assert digest(DATA) == '19797d36307469b7522fc21947e72c74fa76dae8264b499ef74ca91ab5044d15'
    d = pd.read_csv(DATA)
    assert len(d) == 1500 and d.case_id.nunique() == 1500
    local = HERE/'local'; local.mkdir(exist_ok=True)
    results = HERE/'results'; results.mkdir(exist_ok=True)
    spans=[]; counts=[]
    for row in d.itertuples():
        masked, matches = mask(row.narrative)
        counts.append(len(matches))
        for m in matches:
            spans.append(dict(case_id=row.case_id,start=m.start(),end=m.end(),term=m.group(),original=row.narrative,masked=masked))
        # Check all characters outside the explicit matched spans are retained.
        cursor=0; chunks=[]
        for m in matches:
            chunks += [row.narrative[cursor:m.start()], '[MASK]']; cursor=m.end()
        chunks.append(row.narrative[cursor:])
        assert ''.join(chunks)==masked
        d.loc[d.case_id.eq(row.case_id),'narrative']=masked
    d.to_csv(local/'masked_test.csv',index=False)
    pd.DataFrame(spans).to_csv(local/'span_audit.csv',index=False)
    pd.Series([s['term'].lower() for s in spans]).value_counts().rename_axis('matched_term').to_csv(results/'term_counts.csv',header=['count'])
    source=pd.read_csv(DATA); source['masked_spans']=counts
    coverage=source.groupby(['diag1','diagnosis_category']).agg(n=('case_id','size'),changed=('masked_spans',lambda x:int(x.gt(0).sum())),spans=('masked_spans','sum')).reset_index()
    coverage.to_csv(results/'coverage.csv',index=False)
    manifest=dict(test_sha256=digest(DATA),masked_sha256=digest(local/'masked_test.csv'),script_sha256=digest(__file__),seed=SEED,n=1500,changed_cases=sum(c>0 for c in counts),matched_spans=sum(counts),patterns=TERMS,mask_token='[MASK]',scope='All narrative positions, all listed terms, independent of gold labels; case-insensitive word boundaries.',limitations='Conservative automated lexicon, not expert-validated exhaustive diagnosis annotation. Synonyms and misspellings may remain. Diagnostic nouns can themselves carry legitimate context. Mask tokens introduce distribution shift. No mechanistic proof of reasoning.',selection='Frozen before masked model inference; no model selection or tuning.')
    (results/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['n','changed_cases','matched_spans']}))

def infer(condition):
    target=HERE/'local'/condition/'test_results.csv'
    if target.exists():
        print('Existing predictions retained:',condition); return
    target.parent.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(HERE/'local/masked_test.csv')
    if condition=='tfidf_logreg_6000':
        import joblib
        p=joblib.load(ROOT/'methods_runs/tfidf_logreg_6000/tfidf_logreg_pipeline.joblib')
        pred=p.predict(d.narrative)
        result=pd.DataFrame(dict(case_id=d.case_id,true_diag1=d.diag1,pred_diag1=pred))
        metadata={'pipeline_sha256':digest(ROOT/'methods_runs/tfidf_logreg_6000/tfidf_logreg_pipeline.joblib')}
    else:
        from transformers import set_seed
        from methods.sft.train import load_model_and_tokenizer, PeftModel, evaluate_test_set
        set_seed(SEED)
        base={'qwen_sft_6000':'Qwen/Qwen2.5-7B-Instruct','mistral_sft_3000':'mistralai/Mistral-7B-Instruct-v0.3'}[condition]
        adapter=Path(os.environ.get('NEISS_ADAPTER_ROOT', str(ROOT/'methods_runs')))/condition
        model,tokenizer=load_model_and_tokenizer(base)
        model=PeftModel.from_pretrained(model,adapter); model.eval()
        # Save each 200-case block so an interrupted session can resume. Blocks
        # are divisible by batch size 8, preserving the original batch groups.
        chunks=[]
        for start in range(0,len(d),200):
            saved=target.parent/f'chunk_{start:04d}.csv'
            part=d.iloc[start:start+200]
            if saved.exists():
                out=pd.read_csv(saved)
                assert out.case_id.astype(str).tolist()==part.case_id.astype(str).tolist()
            else:
                inp=target.parent/f'input_{start:04d}.csv';part.to_csv(inp,index=False)
                out,_=evaluate_test_set(str(inp),model,tokenizer,data_mode='neiss',max_new_tokens=256,inference_batch_size=8)
                out.to_csv(saved,index=False)
            chunks.append(out)
            print(f'Checkpointed {start+len(part)}/{len(d)} cases',flush=True)
        result=pd.concat(chunks,ignore_index=True)
        metadata={'base_model':base,'base_commit':getattr(model.config,'_commit_hash',None),'adapter_sha256':digest(adapter/'adapter_model.safetensors'),'max_new_tokens':256,'batch_size':8,'checkpoint_block_size':200,'greedy':True,'seed':SEED}
    result.to_csv(target,index=False)
    metadata.update(predictions_sha256=digest(target),masked_sha256=digest(HERE/'local/masked_test.csv'))
    (HERE/'results'/f'{condition}_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Saved',condition,len(result),flush=True)

def report():
    from analysis.error_analysis.secondary_diagnoses.analyze_distinct import code
    from scripts.run_cue_masking_analysis import paired_drop_ci
    d=pd.read_csv(DATA,dtype={'case_id':str}).set_index('case_id')
    masked=pd.read_csv(HERE/'local/masked_test.csv',dtype={'case_id':str}).set_index('case_id').loc[d.index]
    changed=d.narrative.ne(masked.narrative).to_numpy()
    distinct=(d.diag2.map(code).ne(-1)&d.diag2.map(code).ne(d.diag1.map(code))).to_numpy()
    assert distinct.sum()==254
    rows=[]; category_rows=[]; checks=[]
    for name in ['tfidf_logreg_6000','qwen_sft_6000','mistral_sft_3000']:
        a=pd.read_csv(ROOT/'methods_runs'/name/'test_results.csv',dtype={'case_id':str}).set_index('case_id')
        b=pd.read_csv(HERE/'local'/name/'test_results.csv',dtype={'case_id':str}).set_index('case_id')
        assert a.index.is_unique and b.index.is_unique and set(a.index)==set(b.index)==set(d.index)
        a=a.loc[d.index];b=b.loc[d.index]
        y=d.diag1.map(code).to_numpy(); pa=a.pred_diag1.map(code).to_numpy();pb=b.pred_diag1.map(code).to_numpy()
        assert np.array_equal(a.true_diag1.map(code),y) and np.array_equal(b.true_diag1.map(code),y)
        checks.append(dict(condition=name,unchanged_cases=int((~changed).sum()),unchanged_primary_prediction_disagreements=int((pa[~changed]!=pb[~changed]).sum()),original_predictions_sha256=digest(ROOT/'methods_runs'/name/'test_results.csv')))
        for label in sorted(set(y)):
            ix=y==label
            category_rows.append(dict(condition=name,diagnosis_code=label,n=int(ix.sum()),changed=int(changed[ix].sum()),original_recall=float((pa[ix]==y[ix]).mean()),masked_recall=float((pb[ix]==y[ix]).mean())))
        for scope,ix in [('all_1500',np.ones(len(d),dtype=bool)),('changed_only',changed),('distinct_254',distinct)]:
            labels=sorted(set(y[ix]))
            row=dict(condition=name,scope=scope,n=int(ix.sum()),original_accuracy=accuracy_score(y[ix],pa[ix]),masked_accuracy=accuracy_score(y[ix],pb[ix]),original_macro_f1=f1_score(y[ix],pa[ix],labels=labels,average='macro',zero_division=0),masked_macro_f1=f1_score(y[ix],pb[ix],labels=labels,average='macro',zero_division=0))
            if scope=='all_1500':
                ci=paired_drop_ci(y,pa,pb,np.array(labels))
                row.update(zip(['accuracy_drop_ci_low','accuracy_drop_ci_high','f1_drop_ci_low','f1_drop_ci_high'],ci))
            if scope=='distinct_254' and 'pred_diag2' in b:
                g2=d.diag2.map(code).to_numpy()
                for prefix,frame,p1 in [('original',a,pa),('masked',b,pb)]:
                    p2=frame.pred_diag2.map(code).to_numpy()
                    row[prefix+'_secondary_accuracy']=float((p2[ix]==g2[ix]).mean())
                    row[prefix+'_both_ordered_accuracy']=float(((p1[ix]==y[ix])&(p2[ix]==g2[ix])).mean())
                    row[prefix+'_exact_swaps']=int(((p1[ix]==g2[ix])&(p2[ix]==y[ix])).sum())
            rows.append(row)
    pd.DataFrame(rows).to_csv(HERE/'results/metrics.csv',index=False)
    pd.DataFrame(category_rows).to_csv(HERE/'results/per_class.csv',index=False)
    (HERE/'results/comparison_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','report']);p.add_argument('--condition',choices=['tfidf_logreg_6000','qwen_sft_6000','mistral_sft_3000']);args=p.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='infer':infer(args.condition)
    else:report()
