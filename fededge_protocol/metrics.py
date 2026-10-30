"""Audit-grade binary evaluation; no default or fabricated performance values."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import brier_score_loss, log_loss


def expected_calibration_error(y, p, bins=15):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    breaks=np.linspace(0,1,bins+1)
    indices=np.minimum(np.digitize(p,breaks[1:-1]),bins-1)
    out=0.0
    for i in range(bins):
        sub=indices==i
        if sub.any():out+=sub.mean()*abs(y[sub].mean()-p[sub].mean())
    return float(out)


def binary_metrics(y, p, threshold=0.5):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if not set(np.unique(y)).issubset({0,1}):raise ValueError('binary labels required')
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('probabilities must be finite in [0,1]')
    yh=(p>=threshold).astype(int)
    out={'n':int(len(y)),'positive_n':int(y.sum()),'threshold':float(threshold),
         'accuracy':float(accuracy_score(y,yh)),
         'precision':float(precision_score(y,yh,zero_division=0)),
         'recall':float(recall_score(y,yh,zero_division=0)),
         'specificity':float((yh[y==0]==0).mean()) if (y==0).any() else None,
         'f1':float(f1_score(y,yh,zero_division=0)),
         'brier':float(brier_score_loss(y,p)),
         'ece':expected_calibration_error(y,p),
         'nll':float(log_loss(y,np.column_stack([1-np.clip(p,1e-12,1-1e-12),np.clip(p,1e-12,1-1e-12)]),labels=[0,1]))}
    out['auroc']=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
    out['auprc']=float(average_precision_score(y,p)) if y.sum()>0 else None
    return out


def patient_bootstrap(df,nboot=1000,seed=20265796,threshold=0.5):
    """Cluster resample subjects: never resample waveform windows as patients."""
    required={'subject_id','y_true','y_prob'}
    if not required.issubset(df.columns):raise ValueError('Missing columns')
    rng=np.random.default_rng(seed)
    groups={pid:g for pid,g in df.groupby('subject_id')}
    pids=np.array(sorted(groups,key=str),dtype=object)
    results=[]
    for _ in range(nboot):
        draw=rng.choice(pids,size=len(pids),replace=True)
        b=pd.concat([groups[x] for x in draw],ignore_index=True)
        if b.y_true.nunique()<2:continue
        results.append(binary_metrics(b.y_true,b.y_prob,threshold))
    if len(results)<max(10,int(nboot*.5)):
        raise ValueError('Too few valid positive-and-negative bootstrap replicates')
    keys=['auroc','auprc','accuracy','f1','brier','ece']
    return {key:[float(np.nanpercentile([r[key] for r in results],v)) for v in (2.5,97.5)]
            for key in keys}


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--predictions',required=True)
    pa.add_argument('--out',required=True)
    pa.add_argument('--threshold',type=float,default=0.5)
    pa.add_argument('--bootstrap',type=int,default=1000)
    a=pa.parse_args()
    frame=pd.read_csv(a.predictions)
    if not {'subject_id','y_true','y_prob','split'}.issubset(frame.columns):
        raise ValueError('Needs subject_id,y_true,y_prob,split')
    if set(frame.split.unique())!={'test'}:
        raise ValueError('Only LOCKED TEST predictions are permitted here; fit thresholds on calibration')
    report={'point_estimates':binary_metrics(frame.y_true,frame.y_prob,a.threshold),
            'patient_bootstrap_95ci':patient_bootstrap(frame,nboot=a.bootstrap,threshold=a.threshold)}
    Path(a.out).write_text(json.dumps(report,indent=2),encoding='utf8')
    print('Wrote real-data evaluation',a.out)

if __name__=='__main__':main()
