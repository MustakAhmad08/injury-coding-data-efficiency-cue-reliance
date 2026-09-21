"""Audit frozen sample, split disjointness, and completed evaluation artifacts."""
import argparse
import importlib.metadata as packages
import json
import numpy as np
import pandas as pd
from experiment import HERE,ROOT,DATA,TEST,CONDITIONS,allocate,normalized_narrative,sha,write_json

def main(data_only=False):
    manifest=json.loads((HERE/'results/sampling_manifest.json').read_text())
    assert sha(TEST)==manifest['test_sha256']
    for name,expected_sha in manifest['previous_split_sha256'].items():
        assert sha(DATA/name)==expected_sha
    assert sha(DATA/'locked_test.csv')=='19797d36307469b7522fc21947e72c74fa76dae8264b499ef74ca91ab5044d15'
    selection=json.loads((ROOT/'methods/roberta/results/validation_selection.json').read_text())
    for role in ['train','validation']:
        assert sha(ROOT/selection['config'][role+'_file'])==selection['fingerprint'][role+'_sha256']
    test=pd.read_csv(TEST)
    assert len(test)==test.case_id.nunique()==test.narrative.nunique()==1500
    assert set(test.year)=={2025}
    counts=pd.read_csv(HERE/'results/class_distribution.csv').set_index('diagnosis_code')
    expected=allocate(counts.source_count)
    observed=test.diag1.value_counts().reindex(counts.index,fill_value=0).sort_index()
    assert observed.index.equals(expected.index)
    assert np.array_equal(observed,expected) and np.array_equal(observed,counts.test_count)
    assert len(observed[observed>0])==28
    audit={}
    for name in ['train.csv','validation.csv','locked_test.csv','ablations/train_1000.csv','ablations/train_3000.csv','ablations/train_6000.csv']:
        f=pd.read_csv(DATA/name)
        assert not set(test.case_id)&set(f.case_id)
        assert not set(test.narrative)&set(f.narrative.map(normalized_narrative))
        audit[name]={'case_id_overlap':0,'exact_normalized_narrative_overlap':0,'sha256':sha(DATA/name)}
    results={'n':1500,'unique_narratives':1500,'supported_classes':28,'quota_check_passed':True,'test_sha256':sha(TEST),'case_ids_sha256':sha(HERE/'local/selected_case_ids.csv'),'split_checks':audit,'evaluation_verified':False}
    if not data_only:
        metrics=pd.read_csv(HERE/'results/metrics.csv').set_index('condition')
        per=pd.read_csv(HERE/'results/per_class.csv')
        for name in CONDITIONS:
            path=HERE/'local'/name/'test_results.csv'
            meta=json.loads((HERE/'results'/f'{name}_provenance.json').read_text())
            assert meta['test_sha256']==sha(TEST) and meta['predictions_sha256']==sha(path)
            p=pd.read_csv(path).set_index('case_id');g=test.set_index('case_id')
            assert p.index.is_unique and set(p.index)==set(g.index)
            p=p.loc[g.index]
            assert np.array_equal(p.true_diag1,g.diag1)
            acc=float(p.pred_diag1.eq(g.diag1).mean())
            assert np.isclose(acc,metrics.loc[name,'accuracy'])
            part=per[per.condition.eq(name)]
            assert part.support.sum()==1500 and part.false_negatives.sum()==round((1-acc)*1500)
            assert part[part.support.eq(0)].recall.isna().all()
            if name in ['qwen_sft_6000','mistral_sft_3000']:
                old=json.loads((ROOT/'methods_runs'/name/'run_manifest_sft_final_evaluation.json').read_text())
                assert meta['base_commit']==old['model_commit']
        results['evaluation_verified']=True
    write_json(HERE/'results/verification.json',results)
    write_json(HERE/'results/environment.json',{k:packages.version(k) for k in ['torch','transformers','peft','bitsandbytes','scikit-learn','pandas','numpy']})
    print('Sample, quotas, and six split-overlap checks passed. Evaluation verified:',results['evaluation_verified'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-only',action='store_true');a=p.parse_args();main(a.data_only)
