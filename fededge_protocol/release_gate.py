"""Fail-closed, privacy-conscious cohort integrity release gate.

Consumes a *locally constructed, restricted* patient-level manifest, never public
identifiers or manufactured predictions. It produces aggregate-only QC JSON.
Passing is necessary for an empirical study, never evidence of model accuracy.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

PARTITIONS = ('train','tune','calibration','test')
REQUIRED = {'subject_id','echo_study_id','echo_time','ecg_study_id','ecg_time',
            'ecg_clock_qc','hadm_id','label','split','client_id',
            'mortality_followup_hours','mask_echo','mask_ecg','mask_ehr'}


def validate_release_manifest(df, max_lag_hours=48):
    absent = sorted(REQUIRED - set(df.columns))
    if absent:
        raise ValueError('Restricted cohort manifest missing fields: '+', '.join(absent))
    x = df.copy()
    reasons=[]
    if x.empty: reasons.append('no eligible patients')
    if x['subject_id'].isna().any() or x['subject_id'].duplicated().any():
        reasons.append('null or repeated patient IDs')
    if x['echo_study_id'].isna().any() or x['echo_study_id'].duplicated().any():
        reasons.append('missing or repeated index echo studies')
    if x['ecg_study_id'].isna().any() or (x['ecg_study_id'].astype(str).str.len()==0).any():
        reasons.append('missing ECG waveforms')
    if x['hadm_id'].isna().any(): reasons.append('hospital admission identifiers missing')
    if not x['label'].isin([0,1]).all(): reasons.append('non-binary or unverified outcome labels')
    if not x['split'].isin(PARTITIONS).all(): reasons.append('unknown patient-level split values')
    if not x['ecg_clock_qc'].eq('verified').all():
        reasons.append('unverified ECG timestamps in the primary cohort')
    for col in ('mask_echo','mask_ecg','mask_ehr'):
        if not x[col].eq(1).all(): reasons.append(f'{col} is not present for every primary tri-modal record')
    t_echo=pd.to_datetime(x['echo_time'],errors='coerce',utc=True)
    t_ecg=pd.to_datetime(x['ecg_time'],errors='coerce',utc=True)
    if t_echo.isna().any() or t_ecg.isna().any():
        reasons.append('missing or invalid echo/ECG timestamps')
    else:
        delta=(t_echo-t_ecg).dt.total_seconds()/3600
        if not delta.between(0,max_lag_hours).all():
            reasons.append('post-index ECG or time lag outside prespecified window')
    followup=pd.to_numeric(x['mortality_followup_hours'],errors='coerce')
    if followup.isna().any() or not (followup>0).all():
        reasons.append('missing or nonpositive time from echo index to discharge')
    istrain=x['split'].eq('train')
    badtrain=x.loc[istrain,'client_id'].fillna('').astype(str).eq('').any()
    badhold=x.loc[~istrain,'client_id'].fillna('').astype(str).ne('').any()
    if badtrain or badhold: reasons.append('unassigned TRAIN client or holdout assigned to a client')
    if istrain.any():
        clients=x.loc[istrain,'client_id'].nunique()
        if clients != 5: reasons.append(f'expected 5 nonempty virtual TRAIN clients; found {clients}')
    else: reasons.append('no training patients')
    counts=[]
    for split in PARTITIONS:
        subset=x[x['split'].eq(split)]
        n=int(len(subset));events=int(subset.label.eq(1).sum())
        counts.append({'split':split,'patients':n,'events':events,'nonevents':n-events})
        if not n or events==0 or events==n:
            reasons.append(f'{split} split has zero or single-class observations')
    client_counts=[]
    for c,rows in x.loc[istrain].groupby('client_id',dropna=False):
        client_counts.append({'client':str(c),'patients':int(len(rows)),
                              'events':int(rows['label'].eq(1).sum())})
    result={'primary_cohort_manifest_checked':True,
            'analysis_unit':'one patient, one echo-index admission',
            'cohort_patients':int(len(x)), 'counts_by_split':counts,
            'virtual_client_counts':client_counts,
            'numeric_evidence_present':False,
            'ready_for_experiments':not reasons,
            'eligible_for_clinical_effectiveness_claims':False,
            'unresolved_release_reasons':sorted(set(reasons)),
            'disclosure':'QC does not verify video availability, DP guarantees, model fitting or clinical performance.'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restricted-manifest',required=True)
    p.add_argument('--report',required=True)
    args=p.parse_args()
    report=validate_release_manifest(pd.read_csv(args.restricted_manifest))
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf8')
    print('Cohort ready for modeling:',report['ready_for_experiments'])
    print('Quantitative clinical results established: NO')
    if not report['ready_for_experiments']:
        raise SystemExit('Cohort QC failed; inspect local aggregate report before any model result')

if __name__=='__main__':main()
