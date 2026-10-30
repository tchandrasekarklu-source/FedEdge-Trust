"""Reproducible DP budget calculator from actual client run logs.

CSV required columns client_id, sigma, q, steps, delta, max_grad_norm, run_log_sha256,run_log_path.
Never infer epsilon from sigma alone. Uses integer RDP upper-bound accountant.
"""
import argparse,json,re,hashlib
from pathlib import Path
import pandas as pd
from .privacy import rdp_accountant

def report(path):
    df=pd.read_csv(path,dtype={'client_id':str,'run_log_sha256':str})
    req={'client_id','sigma','q','steps','delta','max_grad_norm','run_log_sha256','run_log_path'}
    if not req.issubset(df.columns):raise ValueError('Privacy run-log fields missing: '+repr(req-set(df)))
    if df.empty or df[list(req)].isnull().any().any():raise ValueError('Missing DP accounting evidence fields')
    if df.client_id.duplicated().any():raise ValueError('Duplicate client entry: compose repeated releases explicitly')
    result=[]
    for row in df.itertuples(index=False):
        if not re.fullmatch(r'[0-9a-fA-F]{64}',row.run_log_sha256):
            raise ValueError('Verified training-log SHA256 is mandatory')
        if row.max_grad_norm<=0 or int(row.steps)!=row.steps:raise ValueError('Invalid clipping or step count')
        path=Path(row.run_log_path)
        if not path.is_file():raise ValueError('Source training log does not exist: '+str(path))
        if hashlib.sha256(path.read_bytes()).hexdigest().lower()!=row.run_log_sha256.lower():
            raise ValueError('Training log hash verification failed')
        r=rdp_accountant(float(row.q),float(row.sigma),int(row.steps),float(row.delta))
        r.update({'client_id':row.client_id,'max_grad_norm':float(row.max_grad_norm),
                 'run_log_sha256':row.run_log_sha256})
        result.append(r)
    return {'clients':result,'parallel_disjoint_clients_epsilon_upper_bound':max(x['epsilon_upper_bound'] for x in result),
            'assumptions':'Clients contain disjoint training patients, training logs exact, private validation/model selection released only under accounted mechanisms; secure aggregation separate'}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--log-csv',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();Path(args.out).write_text(json.dumps(report(args.log_csv),indent=2))
