import pandas as pd
from fededge_protocol.release_gate import validate_release_manifest

def fake_manifest():
    rows=[]
    specs=[('train',20),('tune',10),('calibration',10),('test',10)]
    k=0
    for split,n in specs:
        for j in range(n):
            rows.append(dict(subject_id=10000+k,echo_study_id=40000+k,
                echo_time='2180-02-08 10:00:00',ecg_study_id=50000+k,
                ecg_time='2180-02-07 16:00:00',ecg_clock_qc='verified',
                hadm_id=70000+k,label=j%2,split=split,
                client_id=(f'virtual_{j%5+1}' if split=='train' else ''),
                mortality_followup_hours=32,mask_echo=1,mask_ecg=1,mask_ehr=1))
            k+=1
    return pd.DataFrame(rows)

def test_passed_gate_not_clinical_result():
    r=validate_release_manifest(fake_manifest())
    assert r['ready_for_experiments']
    assert not r['eligible_for_clinical_effectiveness_claims']
    assert r['cohort_patients']==50

def test_clock_flag_fails_primary():
    x=fake_manifest();x.loc[0,'ecg_clock_qc']='unverified'
    r=validate_release_manifest(x)
    assert not r['ready_for_experiments']
    assert any('unverified ECG' in reason for reason in r['unresolved_release_reasons'])

def test_cross_partition_client_leakage_fails():
    x=fake_manifest();x.loc[x.split=='test','client_id']='virtual_1'
    assert not validate_release_manifest(x)['ready_for_experiments']

def test_duplicate_patient_fails():
    x=fake_manifest();x.loc[1,'subject_id']=x.loc[0,'subject_id']
    assert not validate_release_manifest(x)['ready_for_experiments']

def test_future_ecg_fails():
    x=fake_manifest();x.loc[0,'ecg_time']='2180-02-09 00:00:00'
    assert not validate_release_manifest(x)['ready_for_experiments']

def test_oneclass_holdout_fails():
    x=fake_manifest();x.loc[x.split=='test','label']=0
    assert not validate_release_manifest(x)['ready_for_experiments']
