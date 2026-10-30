"""Patient-paired stratified bootstrap for same-test-split baseline comparisons.
Only reads held-out actual subject-level prediction CSVs; never simulates labels.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score


def paired_compare(a,b,repetitions=1000,seed=20265796):
    for df in (a,b):
        needed={'subject_id','y_true','y_prob','split'}
        if not needed.issubset(df.columns):raise ValueError('Missing predictions columns')
        if len(df)!=df.subject_id.nunique() or set(df['split'])!={'test'}:
            raise ValueError('Must contain unique held-out test patients only')
    df=a.merge(b,on='subject_id',suffixes=('_a','_b'),validate='one_to_one')
    if len(df)!=len(a) or len(df)!=len(b):raise ValueError('Paired comparison requires identical patients')
    if not np.array_equal(df.y_true_a,df.y_true_b):raise ValueError('Mismatched ground truth')
    y=df.y_true_a.to_numpy(dtype=int);p=df.y_prob_a.to_numpy(float);q=df.y_prob_b.to_numpy(float)
    if set(np.unique(y))!={0,1}:raise ValueError('Both classes required')
    if not (np.isfinite(p).all() and np.isfinite(q).all() and
            ((p>=0)&(p<=1)).all() and ((q>=0)&(q<=1)).all()):
        raise ValueError('Probabilities must be finite in [0,1]')
    rng=np.random.default_rng(seed)
    strata=[np.flatnonzero(y==i) for i in (0,1)]
    metrics={'auroc':roc_auc_score,'auprc':average_precision_score}
    output={'n_paired':len(y),'n_positive':int(y.sum()),'bootstrap_iterations':repetitions,
            'seed':seed,'comparison_protocol':'patient-paired outcome-stratified bootstrap; same fixed test IDs'}
    for name,func in metrics.items():
        observed=float(func(y,p)-func(y,q));boot=[]
        for _ in range(repetitions):
            idx=np.concatenate([rng.choice(ix,size=len(ix),replace=True) for ix in strata])
            boot.append(float(func(y[idx],p[idx])-func(y[idx],q[idx])))
        ci=np.quantile(boot,[.025,.975])
        # Bootstrap sign-tail exploratory estimate; preregister comparison family & adjust in final study.
        pval=min(1.,2*min(np.mean(np.asarray(boot)<=0),np.mean(np.asarray(boot)>=0)))
        output[name]={'delta_a_minus_b':observed,'bootstrap_95_ci':[float(ci[0]),float(ci[1])],
                      'bootstrap_sign_test_p_exploratory':float(pval)}
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--out',required=True);p.add_argument('--repetitions',type=int,default=2000)
    a=p.parse_args();ans=paired_compare(pd.read_csv(a.model),pd.read_csv(a.baseline),a.repetitions)
    Path(a.out).write_text(json.dumps(ans,indent=2))
