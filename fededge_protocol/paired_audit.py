"""Paired, subject-disjoint same-test comparison of actual predictions.

No dummy results. Not suitable for comparing independently published AUROCs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fededge_protocol.metrics import binary_metrics


def paired_report(a,b,bootstrap=1000,seed=20265796,threshold=0.5):
    required={'subject_id','y_true','y_prob','split'}
    for name,f in [('A',a),('B',b)]:
        if not required.issubset(f.columns):raise ValueError(f'{name}: missing prediction columns')
        if f.subject_id.duplicated().any():raise ValueError(f'{name}: duplicated subject_id')
        if set(f['split']) != {'test'}:raise ValueError('only locked test split accepted')
    x=a.set_index('subject_id').sort_index()
    y=b.set_index('subject_id').sort_index()
    if set(x.index)!=set(y.index):raise ValueError('patient sets do not match')
    y=y.loc[x.index]
    if not np.array_equal(x.y_true.to_numpy(),y.y_true.to_numpy()):
        raise ValueError('actual outcomes differ for same patient')
    labels=x.y_true.to_numpy();p1=x.y_prob.to_numpy();p2=y.y_prob.to_numpy()
    pa=binary_metrics(labels,p1,threshold);pb=binary_metrics(labels,p2,threshold)
    names=['auroc','auprc','f1','brier','ece','accuracy']
    deltas={k:(None if pa[k] is None or pb[k] is None else float(pb[k]-pa[k])) for k in names}
    rng=np.random.default_rng(seed)
    n=len(labels);samples={k:[] for k in names}
    for _ in range(bootstrap):
        idx=rng.integers(0,n,n)
        if len(set(labels[idx]))<2:continue
        m1=binary_metrics(labels[idx],p1[idx],threshold);m2=binary_metrics(labels[idx],p2[idx],threshold)
        for k in names:samples[k].append(float(m2[k]-m1[k]))
    if len(samples['auroc'])<max(20,int(bootstrap*.5)):
        raise ValueError('Insufficient valid class-balanced paired replicates')
    ci={k:[float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]
        for k,vals in samples.items()}
    return {'n_matched_patients':n,'model_A':pa,'model_B':pb,'delta_B_minus_A':deltas,
            'paired_bootstrap_95ci':ci,'bootstrap_valid_replicates':len(samples['auroc']),
            'bootstrap_seed':seed,'endpoint_provenance':'supplied genuine test predictions; verify independently'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a',required=True);p.add_argument('--b',required=True)
    p.add_argument('--out',required=True);p.add_argument('--bootstrap',type=int,default=1000)
    z=p.parse_args()
    report=paired_report(pd.read_csv(z.a),pd.read_csv(z.b),bootstrap=z.bootstrap)
    Path(z.out).write_text(json.dumps(report,indent=2),encoding='utf8')
    print('Wrote matched-patient comparison to',z.out)
if __name__=='__main__':main()
