"""Deterministic, patient-linked MIMIC-IV-Echo/ECG/EHR cohort construction.

Input tables MUST be normalized by a credentialed user, under the terms of the
original sources. This module never pairs different patients or uses future EHR
information. Uses one qualifying index echo examination per patient.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {
    'echo': {'subject_id', 'echo_study_id', 'echo_time'},
    'ecg': {'subject_id', 'ecg_study_id', 'ecg_time'},
    'admissions': {'subject_id', 'hadm_id', 'admittime', 'dischtime', 'hospital_expire_flag'},
    'events': {'subject_id', 'hadm_id', 'event_time', 'available_time', 'event_type'},
}

def _validate_frame(name, frame):
    absent = REQUIRED[name].difference(frame.columns)
    if absent:
        raise ValueError(f'{name}: missing required columns: {sorted(absent)}')
    if frame['subject_id'].isna().any():
        raise ValueError(f'{name}: null subject_id is forbidden')


def _datetime(frame, *columns):
    for col in columns:
        frame[col] = pd.to_datetime(frame[col], errors='raise', utc=True)
    return frame


def make_cohort(echo, ecg, admissions, events, window_hours=48, history_hours=24,
                require_ecg=True, require_ehr=True, require_verified_ecg=True):
    """Create a constrained patient- and time-matched cohort.

    Features are available strictly before or at echo index time. The label is
    recorded hospital outcome (not a model estimate). The time window applies
    to the most recent ECG, <= echo time; no outcome or future event enters the
    feature definition. Returns cohort, quality-control report.
    """
    frames = {n: v.copy() for n, v in dict(echo=echo, ecg=ecg,
                                              admissions=admissions,
                                              events=events).items()}
    for name, frame in frames.items(): _validate_frame(name, frame)
    e, c, a, v = (frames[k] for k in ['echo','ecg','admissions','events'])
    _datetime(e, 'echo_time'); _datetime(c, 'ecg_time')
    _datetime(a, 'admittime','dischtime'); _datetime(v,'event_time','available_time')
    e = e.drop_duplicates(['subject_id','echo_study_id']).sort_values(
        ['subject_id','echo_time','echo_study_id']).reset_index(drop=True)
    if 'ecg_clock_qc' not in c.columns:
        c['ecg_clock_qc'] = 'unverified'
    if not c['ecg_clock_qc'].isin(['verified','unverified','rejected']).all():
        raise ValueError('ecg_clock_qc must be verified/unverified/rejected')
    c = c.drop_duplicates(['subject_id','ecg_study_id']).sort_values(
        ['subject_id','ecg_time','ecg_study_id']).reset_index(drop=True)
    a = a.drop_duplicates(['subject_id','hadm_id']).sort_values(
        ['subject_id','admittime','hadm_id']).reset_index(drop=True)
    v = v.sort_values(['subject_id','event_time']).reset_index(drop=True)
    q = {'echo_studies_input': int(len(e)), 'ecg_records_input': int(len(c)),
         'admissions_input': int(len(a)), 'ehr_events_input': int(len(v)),
         'rejected_outside_admission': 0, 'rejected_missing_ecg': 0,
         'rejected_missing_ehr': 0, 'rejected_ambiguous_admission': 0,
         'rejected_outcome_unknown': 0, 'rejected_nonpositive_followup': 0,
         'rejected_unverified_ecg_time': 0, 'included_unverified_ecg_time': 0,
         'postindex_result_records_excluded': 0, 'event_availability_unknown_excluded': 0}
    # Group dictionaries are faster than large cartesian patient joins.
    C = {k: grp for k,grp in c.groupby('subject_id',sort=False)}
    A = {k: grp for k,grp in a.groupby('subject_id',sort=False)}
    V = {k: grp for k,grp in v.groupby('subject_id',sort=False)}
    output = []
    for patient, ep in e.groupby('subject_id',sort=False):
        candidates = A.get(patient)
        if candidates is None:
            q['rejected_outside_admission'] += len(ep);continue
        used = False
        for _, x in ep.iterrows():
            index_time = x.echo_time
            matches = candidates[(candidates.admittime <= index_time) &
                                 (candidates.dischtime > index_time)]
            if len(matches) == 0:
                q['rejected_outside_admission'] += 1;continue
            if len(matches) != 1:
                q['rejected_ambiguous_admission'] += 1;continue
            ac = matches.iloc[0]
            if pd.isna(ac.hospital_expire_flag):
                q['rejected_outcome_unknown']+=1;continue
            if ac.hospital_expire_flag not in (0,1):
                raise ValueError('hospital_expire_flag must be binary 0/1')
            pc = C.get(patient)
            if pc is None: pc = c.iloc[0:0]
            pc = pc[(pc.ecg_time <= index_time) &
                    (pc.ecg_time >= index_time-pd.Timedelta(hours=window_hours))]
            pc = pc[pc.ecg_clock_qc != 'rejected']
            if require_verified_ecg:
                old_len = len(pc)
                pc = pc[pc.ecg_clock_qc == 'verified']
                if old_len and not len(pc): q['rejected_unverified_ecg_time'] += 1
            pc = pc.sort_values(['ecg_time','ecg_study_id'],ascending=[False,True])
            if require_ecg and len(pc)==0:
                q['rejected_missing_ecg']+=1;continue
            pv = V.get(patient)
            if pv is None: pv = v.iloc[0:0]
            pv = pv[(pv.event_time <= index_time) &
                    (pv.event_time >= index_time-pd.Timedelta(hours=history_hours)) &
                    (pv.event_time >= ac.admittime) &
                    (pv.hadm_id == ac.hadm_id)]
            # Chart/sample collection time alone is insufficient: the clinical
            # result may have become available only after the index echo.
            q['event_availability_unknown_excluded'] += int(pv.available_time.isna().sum())
            q['postindex_result_records_excluded'] += int((pv.available_time > index_time).sum())
            pv = pv[pv.available_time.notna() &
                    (pv.available_time <= index_time) &
                    (pv.available_time >= pv.event_time)]
            if require_ehr and len(pv)==0:
                q['rejected_missing_ehr']+=1;continue
            matched = pc.iloc[0] if len(pc) else None
            row = dict(subject_id=patient, echo_study_id=x.echo_study_id,
                       echo_time=index_time.isoformat(), hadm_id=ac.hadm_id,
                       ecg_study_id=(matched.ecg_study_id if matched is not None else ''),
                       ecg_time=(matched.ecg_time.isoformat() if matched is not None else ''),
                       ecg_clock_qc=(str(matched.ecg_clock_qc) if matched is not None else 'missing'),
                       ehr_event_count=int(len(pv)),
                       mortality_followup_hours=float((ac.dischtime-index_time).total_seconds()/3600),
                       label=int(ac.hospital_expire_flag),
                       mask_echo=1, mask_ecg=int(matched is not None),
                       mask_ehr=int(len(pv)>0),
                       ecg_lag_hours=float((index_time-matched.ecg_time).total_seconds()/3600)
                                     if matched is not None else np.nan,
                       history_hours=history_hours)
            if matched is not None and matched.ecg_clock_qc != 'verified':
                q['included_unverified_ecg_time'] += 1
            output.append(row)
            used = True
            break   # At most one qualifying index per subject
    frame = pd.DataFrame(output)
    if len(frame):
        assert frame.subject_id.is_unique
        assert frame.echo_study_id.notna().all()
        observed = frame.ecg_lag_hours.notna()
        assert ((frame.loc[observed,'ecg_lag_hours'] >= 0)&
                (frame.loc[observed,'ecg_lag_hours'] <= window_hours)).all()
        assert frame.label.isin([0,1]).all()
    q['included_subjects'] = int(len(frame))
    q['included_positive'] = int(frame.label.sum()) if len(frame) else 0
    q['included_negative'] = int(len(frame)-frame.label.sum()) if len(frame) else 0
    q['strict_triple_modality'] = bool(require_ecg and require_ehr)
    q['clock_qc_required'] = bool(require_verified_ecg)
    return frame,q


def _stratified_subject_splits(cohort, seed=20265796,
                               fractions=(.60,.15,.10,.15)):
    """Disjoint train/tune/calibration/test at patient level.

    Stratifies at patient level without repeated records. Output independent of
    source row order (ids are sorted before permuting). Never changes splits
    after looking at model outcomes.
    """
    if not np.isclose(sum(fractions),1): raise ValueError('fractions must sum to 1')
    if cohort.empty: raise ValueError('Empty cohort: do not invent sample counts')
    if cohort.subject_id.duplicated().any(): raise ValueError('duplicate patient')
    rng = np.random.Generator(np.random.PCG64(seed))
    labels=('train','tune','calibration','test')
    assigned = {}
    for target in (0,1):
        ids = sorted(cohort.loc[cohort.label==target,'subject_id'].tolist(),key=str)
        rng.shuffle(ids)
        counts = np.floor(np.asarray(fractions)*len(ids)).astype(int)
        remainder = len(ids)-counts.sum()
        for idx in np.argsort(-np.asarray(fractions))[:remainder]:counts[idx]+=1
        i=0
        for part,num in zip(labels,counts):
            for sid in ids[i:i+num]:assigned[sid]=part
            i+=num
    out=cohort.copy()
    out['split']=out.subject_id.map(assigned)
    assert out.split.notna().all()
    return out


def add_virtual_clients(cohort, seed=20265796, num_clients=5, alpha=0.5):
    """Assign TRAINING patients exclusively to K virtual clients.

    Label-conditional Dirichlet skew (non-IID) gives *virtual partitions*, not
    actual hospitals. Test, tuning and calibration subjects are unassigned.
    """
    if num_clients<=0 or alpha<=0: raise ValueError('K and alpha must be positive')
    if 'split' not in cohort: raise ValueError('split must be created first')
    out=cohort.copy();out['client_id']=''
    rng=np.random.Generator(np.random.PCG64(seed+1))
    train=out[out.split=='train']
    if len(train)<num_clients: raise ValueError('not enough training subjects for clients')
    for target in sorted(train.label.unique()):
        ids=sorted(train.loc[train.label==target,'subject_id'].tolist(),key=str)
        rng.shuffle(ids)
        proportions=rng.dirichlet(np.full(num_clients,alpha))
        allocations=rng.multinomial(len(ids),proportions)
        cursor=0
        for k,n in enumerate(allocations):
            subset=ids[cursor:cursor+n]
            out.loc[(out.split=='train') & out.subject_id.isin(subset),'client_id']=f'virtual_{k+1}'
            cursor+=n
    # Dirichlet can yield an empty client, do not silently mislabel such a run.
    actual=set(out.loc[out.split=='train','client_id'])
    if len(actual)!=num_clients:
        raise ValueError(f'Non-IID assignment produced only {len(actual)} nonempty clients; revise seed or sample size')
    if (out.loc[out.split!='train','client_id']!='').any():
        raise AssertionError('test/tune/calibration subject was placed in a client')
    return out


def audit_assignment(frame):
    if frame.subject_id.duplicated().any(): raise AssertionError('Patient appears multiple times')
    expected={'train','tune','calibration','test'}
    if not set(frame.split.unique()).issubset(expected): raise AssertionError('Invalid split')
    if (frame.loc[frame.split!='train','client_id']!='').any(): raise AssertionError('Holdout client leakage')
    stats=frame.groupby(['split','client_id','label'],dropna=False).size().reset_index(name='n')
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in REQUIRED:p.add_argument(f'--{name}',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--seed',type=int,default=20265796)
    p.add_argument('--clients',type=int,default=5)
    p.add_argument('--alpha',type=float,default=0.5)
    p.add_argument('--sensitivity-allow-unverified-ecg-clock',action='store_true',
                   help='ONLY for an explicitly labeled exploratory sensitivity cohort; PRIMARY requires verified ECG timing')
    args=p.parse_args()
    data={name:pd.read_csv(getattr(args,name)) for name in REQUIRED}
    cohort,q=make_cohort(**data,require_verified_ecg=not args.sensitivity_allow_unverified_ecg_clock)
    dest=Path(args.outdir);dest.mkdir(parents=True,exist_ok=True)
    q['analysis_tier']='exploratory_unverified_ECG' if args.sensitivity_allow_unverified_ecg_clock else 'primary_verified_ECG'
    (dest/'qc_report.json').write_text(json.dumps(q,indent=2),encoding='utf8')
    if cohort.empty:
        (dest/'INFEASIBLE.txt').write_text('No eligible patients. Do not report a tri-modal result or relax QC after viewing labels.\n',encoding='utf8')
        raise SystemExit('Eligibility failed: no valid matched patients; see qc_report.json')
    split=_stratified_subject_splits(cohort,seed=args.seed)
    split=add_virtual_clients(split,seed=args.seed,num_clients=args.clients,alpha=args.alpha)
    stats=audit_assignment(split)
    # Local restricted workspace; do not publish indices without checking dataset terms.
    split.to_csv(dest/'restricted_patient_manifest.csv',index=False)
    stats.to_csv(dest/'client_split_counts.csv',index=False)
    (dest/'qc_report.json').write_text(json.dumps(q,indent=2),encoding='utf8')
    print('Cohort QC:',q)
    print('Client/split counts written to',dest)

if __name__=='__main__':main()
