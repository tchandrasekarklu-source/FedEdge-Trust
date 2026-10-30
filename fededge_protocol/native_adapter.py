"""Normalize original credentialed MIMIC-IV-Echo/ECG/Hosp table metadata.

Native field names are drawn from official MIMIC-IV documentation. Use on a
secure workstation only. This module does not fetch restricted files.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def _require(df, keys, name):
    missing=set(keys)-set(df)
    if missing:
        raise ValueError(f'{name}: missing native columns {sorted(missing)}')


def normalize_native(echo_studies, ecg_machine, admissions, labevents):
    """Return normalized cohort inputs; no cross-modal sample pairing here."""
    _require(echo_studies,['subject_id','study_id','study_datetime'],'echo-study-list')
    _require(ecg_machine,['subject_id','study_id','ecg_time'],'machine_measurements')
    _require(admissions,['subject_id','hadm_id','admittime','dischtime','hospital_expire_flag'],'admissions')
    _require(labevents,['subject_id','hadm_id','charttime','storetime','itemid'],'labevents')
    e=echo_studies[['subject_id','study_id','study_datetime']].rename(
        columns={'study_id':'echo_study_id','study_datetime':'echo_time'}).drop_duplicates()
    ecg_cols=['subject_id','study_id','ecg_time']
    if 'ecg_clock_qc' in ecg_machine.columns: ecg_cols.append('ecg_clock_qc')
    c=ecg_machine[ecg_cols].rename(
        columns={'study_id':'ecg_study_id'}).drop_duplicates()
    if 'ecg_clock_qc' not in c.columns: c['ecg_clock_qc']='unverified'
    if not c['ecg_clock_qc'].isin(['verified','unverified','rejected']).all():
        raise ValueError('ecg_clock_qc must be verified, unverified, or rejected')
    a=admissions[['subject_id','hadm_id','admittime','dischtime','hospital_expire_flag']].drop_duplicates()
    v=labevents[['subject_id','hadm_id','charttime','storetime','itemid']].rename(
        columns={'charttime':'event_time','storetime':'available_time','itemid':'event_type'}).dropna(subset=['hadm_id','event_time','available_time'])
    for f,col in [(e,'echo_time'),(c,'ecg_time'),(a,'admittime'),(a,'dischtime'),(v,'event_time'),(v,'available_time')]:
        f[col]=pd.to_datetime(f[col],utc=True,errors='raise')
    for name,f in [('echo',e),('ecg',c),('admissions',a),('events',v)]:
        if f['subject_id'].isna().any():
            raise ValueError(name+': null subject_id')
    if e.duplicated(['subject_id','echo_study_id']).any():
        raise ValueError('one echo study ID maps to duplicate records; investigate')
    if c.duplicated(['subject_id','ecg_study_id']).any():
        raise ValueError('one ECG ID maps to duplicate records; investigate')
    # Flag identity conflicts, even if same study happens to be duplicated across subjects.
    if e.echo_study_id.nunique()!=len(e) or c.ecg_study_id.nunique()!=len(c):
        raise ValueError('Study ID reused across subjects')
    return {'echo':e,'ecg':c,'admissions':a,'events':v}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ['echo-study-list','ecg-machine','admissions','labevents']:
        p.add_argument('--'+x,required=True)
    p.add_argument('--outdir',required=True)
    args=p.parse_args()
    src={x:pd.read_csv(getattr(args,x.replace('-','_')),compression='infer',low_memory=False)
         for x in ['echo-study-list','ecg-machine','admissions','labevents']}
    output=normalize_native(src['echo-study-list'],src['ecg-machine'],src['admissions'],src['labevents'])
    dest=Path(args.outdir)
    dest.mkdir(parents=True,exist_ok=True)
    manifest={}
    for name,data in output.items():
        data.to_csv(dest/(name+'.csv'),index=False)
        manifest[name]={'rows':int(len(data)),'unique_subjects':int(data.subject_id.nunique())}
    (dest/'native_normalization_qc.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
