"""Fixed-model estimates on the proportional test and original balanced test."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from experiment import HERE,ROOT,DATA,TEST,CONDITIONS,SEED,sha,write_json
from analysis.error_analysis.secondary_diagnoses.analyze_distinct import wilson

def scores(y,p,codes):
    present=sorted(set(y))
    return {'accuracy':float(accuracy_score(y,p)),
        'weighted_f1':float(f1_score(y,p,labels=codes,average='weighted',zero_division=0)),
        'macro_f1_supported':float(f1_score(y,p,labels=present,average='macro',zero_division=0)),
        'macro_f1_fixed30':float(f1_score(y,p,labels=codes,average='macro',zero_division=0))}

def predictions(path,gold):
    p=pd.read_csv(path,dtype={'case_id':str}).set_index('case_id')
    assert p.index.is_unique and set(p.index)==set(gold.index)
    p=p.loc[gold.index]
    assert np.array_equal(p.true_diag1.astype(int),gold.diag1.astype(int))
    return pd.to_numeric(p.pred_diag1,errors='coerce').fillna(-1).astype(int).to_numpy()

def main():
    protocol=json.loads((HERE/'results/sampling_manifest.json').read_text())
    assert sha(TEST)==protocol['test_sha256']
    gold=pd.read_csv(TEST,dtype={'case_id':str}).set_index('case_id')
    balanced=pd.read_csv(DATA/'locked_test.csv',dtype={'case_id':str}).set_index('case_id')
    assert not set(gold.index)&set(balanced.index)
    codes=sorted(balanced.diag1.unique());y=gold.diag1.astype(int).to_numpy()
    assert len(y)==1500 and len(set(y))==28
    rng=np.random.default_rng(SEED)
    groups=[np.flatnonzero(y==c) for c in codes if np.any(y==c)]
    boot=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(1000)]
    rows=[];class_rows=[];confusion_rows=[];hashes={}
    for name in CONDITIONS:
        path=HERE/'local'/name/'test_results.csv'
        meta=json.loads((HERE/'results'/f'{name}_provenance.json').read_text())
        assert meta['test_sha256']==sha(TEST) and meta['predictions_sha256']==sha(path)
        p=predictions(path,gold);m=scores(y,p,codes)
        samples=[scores(y[ix],p[ix],codes) for ix in boot]
        row={'condition':name,'n':1500,'supported_classes':28,'invalid_predictions':int((~np.isin(p,codes)).sum()),**m}
        for key in m:
            values=[v[key] for v in samples]
            row[key+'_ci_low'],row[key+'_ci_high']=np.quantile(values,[.025,.975])
        if name.startswith('roberta'):
            original=ROOT/'methods/roberta/local/test_predictions'/f"seed_{name.rsplit('_',1)[1]}.csv"
        else:original=ROOT/'methods_runs'/name/'test_results.csv'
        bp=predictions(original,balanced)
        row.update({'balanced_'+k:v for k,v in scores(balanced.diag1.to_numpy(),bp,codes).items()})
        rows.append(row);hashes[name]={'new':sha(path),'balanced':sha(original)}
        precision,recall,f1,support=precision_recall_fscore_support(y,p,labels=codes,zero_division=0)
        for c,pr,re,ff,n in zip(codes,precision,recall,f1,support):
            actual=y==c;predicted=p==c;correct=int((actual&predicted).sum())
            class_rows.append({'condition':name,'diagnosis_code':int(c),'support':int(n),'predicted_count':int(predicted.sum()),'true_positives':correct,'false_negatives':int(n)-correct,'false_positives':int(predicted.sum())-correct,'precision':pr if predicted.any() else np.nan,'recall':re if n else np.nan,'f1':ff if n else np.nan})
            lo,hi=wilson(correct,int(n)) if n else (np.nan,np.nan)
            class_rows[-1].update(recall_wilson_low=lo,recall_wilson_high=hi)
            tn=int((~actual & ~predicted).sum())
            class_rows[-1].update(true_negatives=tn)
            for guess in sorted(set(p[actual])):
                confusion_rows.append({'condition':name,'true_code':int(c),'predicted_code':int(guess),'count':int((actual&(p==guess)).sum())})
        print(name,m,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(HERE/'results/metrics.csv',index=False)
    pd.DataFrame(class_rows).to_csv(HERE/'results/per_class.csv',index=False)
    from report_per_category import main as report_per_category
    report_per_category()
    pd.DataFrame(confusion_rows).to_csv(HERE/'results/confusion_counts.csv',index=False)
    encoder=frame[frame.condition.str.startswith('roberta')]
    aggregate={k:{'mean':float(encoder[k].mean()),'sd':float(encoder[k].std(ddof=1))} for k in list(scores(y,y,codes))+['balanced_accuracy','balanced_weighted_f1','balanced_macro_f1_fixed30']}
    write_json(HERE/'results/encoder_summary.json',aggregate)
    write_json(HERE/'results/analysis_manifest.json',{'test_sha256':sha(TEST),'prediction_sha256':hashes,'bootstrap_seed':SEED,'bootstrap_resamples':1000,'bootstrap':'Shared within-class resamples with fixed proportional quotas, conditional on sample design and fitted models; intervals are unadjusted descriptive intervals, not simultaneous tests. No class-prevalence or training-seed uncertainty included.','macro':'Supported-label macro averages 28 classes; fixed30 macro averages all 30 using zero_division=0. Zero-support class recall is unavailable.','script_sha256':sha(__file__)})

if __name__=='__main__':main()
