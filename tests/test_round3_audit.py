import json
import numpy as np
import pandas as pd
import pytest
from fededge_experiments.round3_audit import validate_cohort,perf,evaluate_pairs,read_predictions
from fededge_experiments.round3_privacy import report

META={k:'valid documented source value' for k in ['dataset','dataset_version','institution_or_source','data_access_mechanism','ethical_approval_or_waiver','acquisition_period','endpoint_definition','endpoint_adjudication','source_manifest_sha256']}
def cohort():
    return pd.DataFrame({'subject_id':['u1','u2'],'encounter_id':['e1','e2'],'echo_exam_id':['x1','x2'],'ecg_record_id':['c1','c2'],
       'index_time':['2024-01-03T12:00:00Z']*2,'echo_time':['2024-01-03T12:00:00Z']*2,
       'ecg_time':['2024-01-02T12:00:00Z']*2,'ehr_latest_time':['2024-01-03T11:00:00Z']*2,
       'outcome_time':['2024-01-04T12:00:00Z']*2,'y_true':[0,1],'split':['train','test'],'source':['MIMIC']*2})

def test_cohort_valid_schema_smoke():
    r=validate_cohort(cohort(),META);assert r['n']==2

def test_cohort_rejects_future_ecg():
    x=cohort();x.loc[0,'ecg_time']='2024-01-04T12:00:00Z'
    with pytest.raises(ValueError):validate_cohort(x,META)

def test_cohort_rejects_duplicate_patient():
    x=cohort();x.loc[1,'subject_id']='u1'
    with pytest.raises(ValueError):validate_cohort(x,META)

def test_metric_known_integer_confusion():
    y=np.r_[np.ones(60,dtype=int),np.zeros(240,dtype=int)]
    p=np.r_[np.ones(58)*.8,np.ones(2)*.2,np.ones(233)*.1,np.ones(7)*.8]
    m=perf(y,p);assert (m['tp'],m['fn'],m['tn'],m['fp'])==(58,2,233,7)
    assert round(m['accuracy'],4)==round(291/300,4)

def test_expected_five_seeds_arithmetic():
    tps=[58,58,57,58,57];tns=[233,233,235,233,234]
    assert sum(tps)+sum(tns)==1456 and round(100*1456/1500,1)==97.1

def predictions():
    return pd.DataFrame([{'subject_id':f'id{i}','split':'test','seed':seed,'model':mod,
          'y_true':int(i>=10),'y_prob':(.05 if i<10 else .95)+(0.01 if mod=='gated' else 0)}
        for seed in [7,17] for mod in ['gated','masked'] for i in range(20)])

def test_round3_paired_smoke():
    r=evaluate_pairs(predictions(),reps=50);assert len(r['per_seed'])==2 and r['n_paired_test_subjects']==20

def test_round3_rejects_mismatched_patient():
    d=predictions();d.loc[(d.seed==7)&(d.model=='masked')&(d.subject_id=='id0'),'subject_id']='different'
    with pytest.raises(ValueError):evaluate_pairs(d,reps=50)

def test_privacy_rejects_unsigned_client(tmp_path):
    p=tmp_path/'x.csv';pd.DataFrame([dict(client_id='1',sigma=1.,q=.01,steps=50,delta=1e-5,max_grad_norm=1.,run_log_sha256='PLACEHOLDER')]).to_csv(p,index=False)
    with pytest.raises(ValueError):report(p)
