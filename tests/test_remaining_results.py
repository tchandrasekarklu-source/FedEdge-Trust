import numpy as np
import pandas as pd
import pytest
from fededge_experiments.remaining_results import (validate_predictions,subgroup_report,
    risk_coverage_report,result_suite,missing_modality_report)

def fixture():
    return pd.DataFrame({'subject_id':[f'P{x}' for x in range(12)],'split':['test']*12,
        'y_true':[0,1]*6,'y_prob':[.2,.8,.45,.6,.15,.9,.3,.65,.1,.55,.35,.95],
        'model':['full']*12,'seed':[7]*12,'sex':['F']*6+['M']*6,
        'age_band':['adult']*12, 'uncertainty':[.1]*8+[.8]*4})

def test_valid_binary_paired():
    d=fixture();validate_predictions(d)
    assert result_suite(d,.5)['metrics']['n']==12

def test_reject_leakage_and_duplicate():
    d=fixture();d.loc[0,'split']='calibration'
    with pytest.raises(ValueError):validate_predictions(d)
    d=fixture();d.loc[1,'subject_id']=d.loc[0,'subject_id']
    with pytest.raises(ValueError):validate_predictions(d)

def test_bad_outcome_and_probability():
    d=fixture();d.loc[0,'y_true']=2
    with pytest.raises(ValueError):validate_predictions(d)
    d=fixture();d.loc[0,'y_prob']=-.1
    with pytest.raises(ValueError):validate_predictions(d)

def test_subgroups_and_sample_gating():
    d=fixture(); r=subgroup_report(d,'sex',min_n=3,min_events=2)
    assert r['equalized_odds_gap'] is not None
    r=subgroup_report(d,'sex',min_n=50)
    assert all('SUPPRESSED' in x['status'] for x in r['groups'])

def test_risk_coverage():
    d=fixture(); r=risk_coverage_report(d,'uncertainty',.5,.5)
    assert r['accepted_patients']==8 and r['abstained_patients']==4

def test_require_one_model_seed():
    d=fixture();d.loc[0,'seed']=8
    with pytest.raises(ValueError):result_suite(d,.5)

def test_no_false_external_claim():
    d=fixture(); d['site_id']=['site-0']*6+['site-1']*6
    r=result_suite(d,.5,min_n=3,min_events=2)
    assert 'DESCRIPTIVE ONLY' in r['actual_institution_sites']

def test_stress_patient_alignment():
    d=fixture(); d['stress_condition']='clean'
    b=d.copy();b['stress_condition']='missing_ecg'
    both=pd.concat([d,b]);report=missing_modality_report(both)
    assert report['status']=='CALCULATED'
    b=b.iloc[:-1]
    with pytest.raises(ValueError): missing_modality_report(pd.concat([d,b]))
