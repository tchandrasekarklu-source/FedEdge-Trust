"""Calibration-only temperature fitting and lockable held-out evaluation.

No synthetic clinical accuracy claims. Only supplied authorized predictions are used.
Required CSV columns: subject_id,split,y_true,logit.
Tuning logits use the calibration split; untouched test logits receive the fixed temperature.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import log_loss
from .metrics import binary_metrics, patient_bootstrap


def _check_records(df):
    required={'subject_id','split','y_true','logit'}
    if not required.issubset(df):
        raise ValueError('Missing required fields: '+','.join(sorted(required-set(df))))
    if df.subject_id.duplicated().any():
        raise ValueError('Exactly one prediction per patient')
    if not df.split.isin(['calibration','test']).all():
        raise ValueError('Only calibration/test records accepted; tuning and training are separate')
    if not set(df.y_true.unique()).issubset({0,1}):raise ValueError('Binary labels required')
    if not np.isfinite(df.logit.to_numpy(dtype=float)).all():raise ValueError('Finite logits required')
    if not (df.split=='calibration').any() or not (df.split=='test').any():
        raise ValueError('Both calibration and test records required')


def fit_temperature_and_evaluate(frame, nboot=1000, seed=20265796):
    _check_records(frame)
    cal=frame[frame.split=='calibration'].copy()
    test=frame[frame.split=='test'].copy()
    # Optimize temperature on calibration only; no test labels consulted until locked.
    y=cal.y_true.to_numpy(dtype=int)
    z=cal.logit.to_numpy(dtype=float)
    if len(set(y)) != 2: raise ValueError('Calibration split must contain both outcome classes')
    def loss(logT):
        pr=expit(z/np.exp(logT))
        return float(log_loss(y,np.clip(pr,1e-10,1-1e-10),labels=[0,1]))
    opt=minimize_scalar(loss,bounds=(-4,4),method='bounded',options={'xatol':1e-6})
    if not opt.success:raise RuntimeError('Temperature fitting failed')
    T=float(np.exp(opt.x))
    p=expit(test.logit.to_numpy(dtype=float)/T)
    evaluation=test[['subject_id','y_true']].copy()
    evaluation['y_prob']=p
    result={'temperature':T, 'calibration_count':int(len(cal)), 'test_count':int(len(test)),
            'calibration_positive_count':int(sum(y)), 'test_positive_count':int(sum(test.y_true)),
            'test_metrics':binary_metrics(test.y_true,p),
            'test_95ci':patient_bootstrap(evaluation,nboot=nboot,seed=seed),
            'note':'Empirical only when source input is credentialed real-patient predictions'}
    return result, evaluation


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',required=True)
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--bootstrap',type=int,default=1000)
    args=ap.parse_args()
    path=Path(args.predictions)
    df=pd.read_csv(path)
    report,test_predictions=fit_temperature_and_evaluate(df,nboot=args.bootstrap)
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    report['input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'calibrated_test_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    test_predictions.to_csv(out/'restricted_calibrated_test_predictions.csv',index=False)
    print('Wrote calibrated held-out-test evaluation; inputs must come from real model runs.')

if __name__=='__main__':main()
