"""Report category metrics from frozen counts; independently verify predictions."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NAMES = {
    'tfidf_logreg_6000': 'TF-IDF 6K',
    'qwen_sft_6000': 'Qwen SFT 6K',
    'mistral_sft_3000': 'Mistral SFT 3K',
    **{f'roberta_seed_{s}': f'RoBERTa (seed {s})' for s in (20260718,20260719,20260720)},
}

def main():
    frame = pd.read_csv(HERE/'results/per_class.csv')
    metrics = pd.read_csv(HERE/'results/metrics.csv').set_index('condition')
    classes = pd.read_csv(HERE/'results/class_distribution.csv').set_index('diagnosis_code')
    totals = frame.condition.map(metrics.n)
    frame['true_negatives'] = totals-frame.true_positives-frame.false_positives-frame.false_negatives
    frame = frame.drop(columns=['one_vs_rest_accuracy'], errors='ignore')
    frame['diagnosis_category'] = frame.diagnosis_code.map(classes.diagnosis_category)
    assert len(frame)==180 and not frame.duplicated(['condition','diagnosis_code']).any()
    for condition, group in frame.groupby('condition',sort=False):
        prediction_file = HERE/'local'/condition/'test_results.csv'
        if not prediction_file.is_file():
            continue
        predictions = pd.read_csv(prediction_file)
        y = predictions.true_diag1.astype(int).to_numpy()
        p = pd.to_numeric(predictions.pred_diag1,errors='coerce').fillna(-1).astype(int).to_numpy()
        assert len(y)==1500 and group.support.sum()==len(y)
        for row in group.itertuples():
            actual, predicted = y==row.diagnosis_code,p==row.diagnosis_code
            tp = int((actual & predicted).sum())
            fp = int((~actual & predicted).sum())
            fn = int((actual & ~predicted).sum())
            tn = int((~actual & ~predicted).sum())
            assert (tp,fp,fn,tn)==(row.true_positives,row.false_positives,row.false_negatives,row.true_negatives)
            if predicted.any():
                assert np.isclose(row.precision,tp/(tp+fp))
            else:
                assert pd.isna(row.precision)
            if actual.any():
                assert np.isclose(row.recall,tp/(tp+fn))
                assert np.isclose(row.f1,2*tp/(2*tp+fp+fn))
            else:
                assert pd.isna(row.recall) and pd.isna(row.f1)
    frame.to_csv(HERE/'results/per_class.csv',index=False)
    lines = ['# Proportional test: per-category performance','',
        'All six frozen model runs use the same 1,500-case proportional test. Values are fractions (0 to 1), rounded to three decimals; the CSV retains full precision.','',
        '- **Precision** = TP / (TP + FP): the fraction of predictions for this category that are correct.',
        '- **Recall** = TP / (TP + FN): the fraction of actual cases in the category correctly classified.',
        '- **F1** = 2TP / (2TP + FP + FN): combines precision and recall.',
        '- **N** is the number of true cases. Recall and F1 are reported as N/A when N = 0. Precision is N/A when the model makes no predictions for a category. Estimates for very small categories are unstable.','',
        'The primary comparison uses RoBERTa seed **20260718**, matching TF-IDF, Qwen, and Mistral. The other two seeds are supplementary. The reporting seed was chosen after evaluation to align seeds, not to maximize test performance.','',
        'Source: `results/per_class.csv`, derived from recorded confusion counts. When local predictions are available, the generator also checks them. No retraining or new inference was performed. All three RoBERTa seeds are retained separately.','']
    def fmt(value):
        return 'N/A' if pd.isna(value) else f'{value:.3f}'
    for condition,group in frame.groupby('condition',sort=False):
        if condition=='roberta_seed_20260719':
            lines += ['## Supplementary seed runs','','These runs are retained for robustness; seed 20260718 is used in the primary comparison.','']
        lines += [f'## {NAMES[condition]}','','| Code | Category | N | Precision | Recall | F1 |','|---|---|---:|---:|---:|---:|']
        for row in group.sort_values('diagnosis_code').itertuples():
            lines.append(f'| {row.diagnosis_code} | {row.diagnosis_category} | {row.support} | {fmt(row.precision)} | {fmt(row.recall)} | {fmt(row.f1)} |')
        lines.append('')
    (HERE/'PER_CATEGORY_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Reported 180 category/model rows; checked any locally available predictions.')

if __name__=='__main__':
    main()
