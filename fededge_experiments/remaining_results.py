"""Evidence-gated held-out subgroup, site, robustness and abstention reports.

Never generates labels, imputes clinical results or authenticates a clinical data source.
All computations are performed on user-supplied *already evaluated*, patient-level
predictions, and require a manifest binding that file to its recorded SHA-256 hash.
Site comparisons are descriptive and are not called external validation unless a
manually attested, genuinely independent healthcare institution is provided.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from fededge_protocol.metrics import binary_metrics

CORE = {'subject_id', 'split', 'y_true', 'y_prob', 'model', 'seed'}

def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def validate_predictions(df):
    missing=sorted(CORE-set(df.columns))
    if missing: raise ValueError('Prediction columns missing: '+', '.join(missing))
    if df.empty or df.isna().any().any():
        raise ValueError('No empty inputs and no missing values (use dedicated subgroup columns only when complete)')
    if set(df['split'].astype(str)) != {'test'}:
        raise ValueError('Results can only use frozen TEST records')
    if df['subject_id'].isna().any() or df.duplicated(['subject_id','model','seed']).any():
        raise ValueError('Duplicate patient/model/seed or null subject ID')
    if not df['y_true'].isin([0,1]).all(): raise ValueError('Binary outcome required')
    p=pd.to_numeric(df['y_prob'],errors='coerce')
    if not np.isfinite(p).all() or not p.between(0,1).all():
        raise ValueError('Probabilities out of bounds')
    for _,grp in df.groupby('subject_id'):
        if grp['y_true'].nunique()!=1:
            raise ValueError('Patient labels differ across model/seed records')
    return df

def _finite_metric(df, threshold):
    if len(df)<2 or df['y_true'].nunique()<2:
        return {'n':int(len(df)),'events':int(df['y_true'].sum()),'reportable':False,
                'reason':'both outcome classes required'}
    metrics=binary_metrics(df['y_true'].to_numpy(),df['y_prob'].to_numpy(),threshold)
    counts=confusion_matrix(df['y_true'],df['y_prob'].to_numpy()>=threshold,labels=[0,1])
    tn,fp,fn,tp=[int(x) for x in counts.ravel()]
    metrics.update({'events':int(df['y_true'].sum()),'false_positive_rate':fp/(fp+tn) if fp+tn else None,
                    'true_positive_rate':tp/(tp+fn) if tp+fn else None, 'reportable':True})
    return metrics

def _auc_ci(df, seed=20265796, nboot=300):
    """Within-subgroup class-stratified patient bootstrap, conditional on eligible group."""
    yy=df['y_true'].to_numpy(dtype=int);pp=df['y_prob'].to_numpy(dtype=float)
    strata=[np.flatnonzero(yy==v) for v in (0,1)]
    if any(len(x)==0 for x in strata):return None
    rng=np.random.default_rng(seed); draws=[]
    for _ in range(nboot):
        idx=np.concatenate([rng.choice(x,len(x),replace=True) for x in strata])
        draws.append(roc_auc_score(yy[idx],pp[idx]))
    return [round(float(x),6) for x in np.quantile(draws,[.025,.975])]


def subgroup_report(df,group_key,threshold=0.5,min_n=30,min_events=5):
    if group_key not in df.columns:
        return {'status':'NOT AVAILABLE','reason':'subgroup key not recorded'}
    reports=[]
    for key,g in df.groupby(group_key,dropna=False,sort=True):
        n=len(g); ev=int(g['y_true'].sum()); other=n-ev
        if n<min_n or ev<min_events or other<min_events:
            reports.append({'subgroup':str(key),'n':n,'events':ev,'status':'SUPPRESSED: insufficient support'})
            continue
        metrics=_finite_metric(g,threshold)
        metrics['auroc_95ci_patient_bootstrap']=_auc_ci(g)
        reports.append({'subgroup':str(key),'status':'CALCULATED',**metrics})
    acceptable=[r for r in reports if r['status']=='CALCULATED']
    equalized_odds_gap=None
    if len(acceptable)>=2:
        tpr=[r['true_positive_rate'] for r in acceptable]
        fpr=[r['false_positive_rate'] for r in acceptable]
        equalized_odds_gap=max(max(tpr)-min(tpr),max(fpr)-min(fpr))
    return {'group':group_key,'groups':reports,'equalized_odds_gap':equalized_odds_gap,
            'interpretation':'Descriptive conditional disparities only; no causal fairness guarantee'}

def risk_coverage_report(df, uncertainty_col, cutoff, threshold):
    if uncertainty_col not in df: return {'status':'NOT AVAILABLE','reason':'no frozen-model uncertainty scores'}
    u=pd.to_numeric(df[uncertainty_col],errors='coerce')
    if not np.isfinite(u).all():raise ValueError('Missing/nonfinite uncertainty scores')
    selected=df.loc[u<=cutoff]
    abstain=df.loc[u>cutoff]
    return {'cutoff_from_calibration':float(cutoff),'coverage':len(selected)/len(df),
            'abstained_patients':int(len(abstain)), 'accepted_patients':int(len(selected)),
            'accepted':_finite_metric(selected,threshold),
            'caution':'Threshold/cutoff must be frozen using calibration data before reading TEST outcomes'}

def missing_modality_report(df,reference='clean',condition_col='stress_condition',threshold=0.5):
    if condition_col not in df: return {'status':'NOT AVAILABLE','reason':'no stress-condition prediction manifest'}
    # Multiple stress rows per patient are represented as a separate experimental matrix;
    # each condition must refer to exactly the same patient labels for valid paired deltas.
    groups={str(k):g for k,g in df.groupby(condition_col)}
    if reference not in groups:return {'status':'NOT AVAILABLE','reason':'no clean reference condition'}
    base=groups[reference]
    if base['subject_id'].duplicated().any():raise ValueError('Duplicated clean reference patient')
    reports=[]
    for key,arm in groups.items():
        if arm['subject_id'].duplicated().any():raise ValueError('Repeated patient inside stress condition')
        merged=base[['subject_id','y_true']].merge(arm[['subject_id','y_true']],on='subject_id',suffixes=('_base','_stress'),validate='one_to_one')
        if len(merged)!=len(base) or len(merged)!=len(arm):raise ValueError('Stress comparison requires identical patients')
        if (merged.y_true_base!=merged.y_true_stress).any():raise ValueError('Outcome mismatch between stress conditions')
        reports.append({'condition':key,'metrics':_finite_metric(arm,threshold)})
    return {'status':'CALCULATED','conditions':reports,'note':'Perturbations on genuine matched test patients must be disclosed as stress simulations, never independent cohorts'}

def result_suite(df, threshold, cutoff=None, uncertainty_col='uncertainty', min_n=30, min_events=5):
    """Single frozen model and seed only; use matched comparator script for multi-arm results."""
    validate_predictions(df)
    if len(df[['model','seed']].drop_duplicates())!=1:
        raise ValueError('Select exactly one model and seed to prevent mixing independent predictions')
    if df['subject_id'].duplicated().any():raise ValueError('One row per held-out patient required')
    out={'clinical_provenance':'INPUT FILE ATTESTATION REQUIRED: metrics are arithmetic only, not proof of source authenticity',
         'n':int(len(df)),'metrics':_finite_metric(df,threshold),
         'subgroups':{key:subgroup_report(df,key,threshold,min_n,min_events) for key in ('sex','age_band','diabetes','cohort')},
         'actual_institution_sites': 'NOT MEASURED: independent site provenance must be verified separately'}
    if 'site_id' in df:
        out['site_descriptive']=subgroup_report(df,'site_id',threshold,min_n,min_events)
        out['actual_institution_sites']='DESCRIPTIVE ONLY: never infer independent sites from virtual federated clients'
    if cutoff is not None:
        out['selective_prediction']=risk_coverage_report(df,uncertainty_col,cutoff,threshold)
    else:out['selective_prediction']='NOT MEASURED: require pre-registered calibration-set cutoff'
    out['missing_modality_stress']='NOT MEASURED in this single-row-per-patient suite; use separate condition-wise records and missing_modality_report'
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True);p.add_argument('--threshold',required=True,type=float)
    p.add_argument('--uncertainty-cutoff',type=float)
    p.add_argument('--model',required=True);p.add_argument('--seed',required=True,type=int)
    args=p.parse_args()
    input_sha=file_sha256(args.predictions)
    manifest=json.loads(Path(args.manifest).read_text())
    if manifest.get('predictions_sha256')!=input_sha:
        raise ValueError('Prediction digest mismatch; cannot audit unbound result inputs')
    for field in ('dataset_version','cohort_version','split_fingerprint','endpoint','source_provenance'):
        if not manifest.get(field):raise ValueError('Manifest missing '+field)
    df=pd.read_csv(args.predictions)
    validate_predictions(df)
    df=df.loc[(df['model'].astype(str)==args.model)&(df['seed']==args.seed)].copy()
    if df.empty:raise ValueError('Requested model/seed absent')
    report=result_suite(df,args.threshold,args.uncertainty_cutoff)
    report.update({'input_sha256':input_sha,'provenance_fields':manifest,'model':args.model,'seed':args.seed})
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(report,indent=2,default=str))
    print('Computed held-out analysis from SUPPLIED inputs; clinical source not independently authenticated:',args.out)

if __name__=='__main__':main()
