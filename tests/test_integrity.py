import pandas as pd
import numpy as np
import torch
from fededge_protocol.cohort import make_cohort, _stratified_subject_splits, add_virtual_clients, audit_assignment
from fededge_protocol.metrics import binary_metrics,patient_bootstrap
from fededge_protocol.fusion import QualityTimeAttention


def example_data(n=80):
    base=pd.Timestamp('2025-01-02T12:00:00Z')
    e=[]; c=[]; a=[]; v=[]
    for i in range(n):
        pid=i+1000;t=base+pd.Timedelta(days=i)
        e.append(dict(subject_id=pid,echo_study_id=f'e{i}',echo_time=t))
        c.append(dict(subject_id=pid,ecg_study_id=f'c{i}',ecg_time=t-pd.Timedelta(hours=2),ecg_clock_qc='verified'))
        # future ECG should NEVER be used
        c.append(dict(subject_id=pid,ecg_study_id=f'future{i}',ecg_time=t+pd.Timedelta(hours=1),ecg_clock_qc='verified'))
        a.append(dict(subject_id=pid,hadm_id=i+20,admittime=t-pd.Timedelta(days=1),
                      dischtime=t+pd.Timedelta(days=1),hospital_expire_flag=i%3==0))
        v.append(dict(subject_id=pid,hadm_id=i+20,event_time=t-pd.Timedelta(hours=4),available_time=t-pd.Timedelta(hours=3),event_type='HR',value=72))
        v.append(dict(subject_id=pid,hadm_id=i+20,event_time=t+pd.Timedelta(hours=1),available_time=t+pd.Timedelta(hours=2),event_type='FUTURE_OUTCOME',value=1))
    return tuple(pd.DataFrame(z) for z in (e,c,a,v))


def test_linkage_and_future_leakage_exclusion():
    cohort,q=make_cohort(*example_data())
    assert len(cohort)==80
    assert set(cohort.ecg_study_id)=={f'c{i}' for i in range(80)}
    assert (cohort.ecg_lag_hours==2).all()
    assert (cohort.ehr_event_count==1).all()
    assert cohort.subject_id.is_unique
    assert q['strict_triple_modality']


def test_partition_disjoint_and_deterministic():
    cohort,_=make_cohort(*example_data())
    s=_stratified_subject_splits(cohort)
    s2=_stratified_subject_splits(cohort.sample(frac=1,random_state=101))
    assert s.set_index('subject_id').split.to_dict()==s2.set_index('subject_id').split.to_dict()
    c=add_virtual_clients(s);audit_assignment(c)
    assert set(c.loc[c.split!='train','client_id'])=={''}
    assert c.loc[c.split=='train','client_id'].str.startswith('virtual_').all()


def test_evaluation_and_patient_bootstrap():
    f=pd.DataFrame({'subject_id':range(20),'y_true':[0,1]*10,
                    'y_prob':[.1,.9]*10})
    r=binary_metrics(f.y_true,f.y_prob)
    assert r['accuracy']==1 and r['auroc']==1
    ci=patient_bootstrap(f,nboot=50)
    assert ci['auroc']==[1,1]


def test_masked_fusion_and_abstention():
    model=QualityTimeAttention(dim=8)
    embedding=torch.randn(2,3,8)
    mask=torch.tensor([[True,False,True],[False,False,False]])
    score,w,valid=model(embedding,mask,torch.ones(2,3),torch.zeros(2,3))
    assert valid.tolist()==[True,False]
    assert w[0,1].item()==0
    assert w[1].abs().sum().item()==0
    assert torch.isfinite(score).all()


def test_zero_quality_excludes_modality():
    model=QualityTimeAttention(dim=8)
    z=torch.randn(1,3,8)
    logits, weights, valid=model(z, torch.tensor([[True,True,False]]),
                                  torch.tensor([[0.,1.,0.]]),torch.zeros(1,3))
    assert valid.item()
    assert weights[0,0].item()==0
    assert weights[0,1].item()==1


def test_missing_embedding_cannot_change_output():
    # Invariance is essential: a masked modality is not permitted to
    # influence the query, keys, weights, logits, or final risk.
    torch.manual_seed(20)
    model=QualityTimeAttention(dim=8)
    model.eval()
    original=torch.randn(1,3,8)
    presence=torch.tensor([[True,False,True]])
    quality=torch.ones(1,3)
    lag=torch.zeros(1,3)
    s1,w1,valid1=model(original,presence,quality,lag)
    perturbed=original.clone()
    perturbed[:,1,:]=10000*torch.randn(1,8)
    s2,w2,valid2=model(perturbed,presence,quality,lag)
    assert valid1.item() and valid2.item()
    assert torch.allclose(s1,s2,atol=1e-6)
    assert torch.allclose(w1,w2,atol=1e-6)


def test_cross_admission_data_must_not_leak():
    echo, ecg, admissions, events=example_data(n=15)
    # Only one old-admission EHR event exists for the first patient.
    events.loc[events.subject_id==1000,'hadm_id']=999999
    _, qc=make_cohort(echo,ecg,admissions,events)
    assert qc['included_subjects']==14
    assert qc['rejected_missing_ehr']>=1

def test_outcome_at_index_is_not_a_prediction_target():
    echo,ecg,admissions,events=example_data(n=15)
    admissions.loc[admissions.subject_id==1000,'dischtime']=echo.loc[0,'echo_time']
    cohort,qc=make_cohort(echo,ecg,admissions,events)
    assert len(cohort)==14
    assert 1000 not in cohort.subject_id.values


def test_native_adapter_correct_mapping_and_encounter_isolation():
    from fededge_protocol.native_adapter import normalize_native
    echo,ecg,adm,evt=example_data(n=20)
    ne=echo.rename(columns={'echo_study_id':'study_id','echo_time':'study_datetime'})
    nc=ecg.rename(columns={'ecg_study_id':'study_id'})
    nv=evt.rename(columns={'event_time':'charttime','available_time':'storetime','event_type':'itemid'})
    transformed=normalize_native(ne,nc,adm,nv)
    rows,qc=make_cohort(**transformed)
    assert len(rows)==20
    assert rows.mortality_followup_hours.min()>0
    assert qc['included_subjects']==20


def test_paired_statistical_comparator_checks_patient_identity():
    from fededge_protocol.paired_audit import paired_report
    a=pd.DataFrame({'subject_id':range(40),'y_true':[0,1]*20,
                    'y_prob':[.2,.8]*20,'split':['test']*40})
    b=a.copy();b.y_prob=b.y_prob.clip(.05,.95)
    result=paired_report(a,b,bootstrap=60)
    assert result['n_matched_patients']==40
    assert abs(result['delta_B_minus_A']['auroc'])<1e-9
    b.loc[0,'subject_id']=999
    import pytest
    with pytest.raises(ValueError):paired_report(a,b,bootstrap=60)


def test_hardware_benchmark_emits_no_fabricated_energy():
    from fededge_protocol.benchmark import benchmark
    mod=torch.nn.Linear(8,1)
    result=benchmark(mod,torch.randn(1,8),warmup=1,repetitions=12)
    assert result['energy_mJ_per_sample'] is None
    assert result['latency_ms'][95]>=0
    assert result['measured_iterations']==12


def test_fedavg_fedprox_reference_uses_only_provided_clients():
    from fededge_protocol.federated_reference import run_federated
    X=np.random.default_rng(123).normal(size=(24,4))
    y=np.array([0,1]*12)
    clients=np.array(['virtual_1']*12+['virtual_2']*12)
    m1, report1=run_federated(X,y,clients,rounds=2,epochs=2,seed=1)
    m2, report2=run_federated(X,y,clients,rounds=2,epochs=2,mu=.1,seed=1)
    assert report1['optimizer']=='FedAvg' and report2['optimizer']=='FedProx'
    assert report1['client_patient_counts']=={'virtual_1':12,'virtual_2':12}
    assert torch.isfinite(next(m1.parameters())).all()
    assert torch.isfinite(next(m2.parameters())).all()


def test_independent_clock_verification_gate():
    echo,ecg,admissions,events=example_data(n=14)
    ecg['ecg_clock_qc']='unverified'
    cohort,q=make_cohort(echo,ecg,admissions,events,require_verified_ecg=False)
    assert q['included_unverified_ecg_time']==14
    empty,q2=make_cohort(echo,ecg,admissions,events,require_verified_ecg=True)
    assert len(empty)==0 and q2['rejected_unverified_ecg_time']>=14
    ecg.loc[ecg.ecg_study_id=='c0','ecg_clock_qc']='verified'
    only_one,q3=make_cohort(echo,ecg,admissions,events,require_verified_ecg=True)
    assert len(only_one)==1 and only_one.iloc[0].subject_id==1000


def test_pairwise_operator_missing_modality_invariance():
    from fededge_protocol.pairwise_fusion import PairwiseQualityTimeAttention
    torch.manual_seed(303)
    model=PairwiseQualityTimeAttention(dim=8,modalities=3)
    model.eval()
    z=torch.randn(2,3,8)
    present=torch.tensor([[True,False,True],[False,False,False]])
    quality=torch.tensor([[1.,0.7,0.8],[1.,1.,1.]])
    lag=torch.tensor([[0.,4.,18.],[0.,0.,0.]])
    score,att,valid=model(z,present,quality,lag)
    fake=z.clone();fake[:,1,:]*=300000
    score2,att2,valid2=model(fake,present,quality,lag)
    assert valid.tolist()==[True,False]
    assert torch.allclose(score,score2,atol=1e-5)
    assert torch.allclose(att,att2,atol=1e-5)
    assert att[0,:,1].abs().sum().item()==0
    assert att[0,1,:].abs().sum().item()==0
    assert att[1].abs().sum().item()==0


def test_strict_pairwise_clock_gate_and_evidence_release_gate(tmp_path):
    from fededge_protocol.pairwise_fusion import PairwiseQualityTimeAttention
    from fededge_protocol.evidence_gate import audit_evidence
    model=PairwiseQualityTimeAttention(dim=4,modalities=3)
    z=torch.randn(1,3,4)
    mask=torch.ones(1,3,dtype=torch.bool)
    q=torch.ones(1,3)
    lag=torch.zeros(1,3)
    import pytest
    with pytest.raises(ValueError):
        model(z,mask,q,lag,require_clock_verified=True)
    score,att,valid=model(z,mask,q,lag,clock_verified=torch.tensor([[True,False,True]]),require_clock_verified=True)
    assert valid.item() and att[:,1,:].abs().sum().item()==0
    assert att[:,:,1].abs().sum().item()==0
    incomplete={'claim_type':'performance','evidence_status':'target','n_subjects':0}
    failures=audit_evidence(incomplete,tmp_path)
    assert 'evidence is not measured' in failures
    assert 'verified subject count absent' in failures


def test_native_adapter_preserves_external_clock_review():
    from fededge_protocol.native_adapter import normalize_native
    echo,ecg,adm,evt=example_data(n=3)
    native_e=echo.rename(columns={'echo_study_id':'study_id','echo_time':'study_datetime'})
    native_c=ecg.rename(columns={'ecg_study_id':'study_id'})
    native_c['ecg_clock_qc']='unverified'
    native_c.loc[native_c.study_id=='c0','ecg_clock_qc']='verified'
    native_v=evt.rename(columns={'event_time':'charttime','available_time':'storetime','event_type':'itemid'})
    normalized=normalize_native(native_e,native_c,adm,native_v)
    assert set(normalized['ecg'].ecg_clock_qc)=={'verified','unverified'}
    cohort,info=make_cohort(**normalized,require_verified_ecg=True)
    assert cohort.subject_id.tolist()==[1000]


def test_primary_clock_qc_strict_default_and_relaxed_explicit():
    echo,ecg,adm,events=example_data(n=12)
    ecg['ecg_clock_qc']='unverified'
    strict,qc=make_cohort(echo,ecg,adm,events)
    assert strict.empty and qc['clock_qc_required'] is True
    exploratory,eq=make_cohort(echo,ecg,adm,events,require_verified_ecg=False)
    assert len(exploratory)==12 and eq['included_unverified_ecg_time']==12


def test_temperature_fitted_only_on_calibration():
    from fededge_protocol.calibration import fit_temperature_and_evaluate
    import pytest
    rng=np.random.default_rng(100)
    f=pd.DataFrame({'subject_id':range(120),
      'split':['calibration']*60+['test']*60,
      'y_true':[0,1]*60,
      'logit':np.tile([-2.,2.],60)+rng.normal(0,.1,120)})
    a,_=fit_temperature_and_evaluate(f,nboot=24)
    f2=f.copy()
    f2.loc[f2.split=='test','y_true']=1-f2.loc[f2.split=='test','y_true'].to_numpy()
    b,_=fit_temperature_and_evaluate(f2,nboot=24)
    assert np.isclose(a['temperature'],b['temperature'])
    assert a['test_metrics']['brier']!=b['test_metrics']['brier']
    duplicate=pd.concat([f,f.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError):fit_temperature_and_evaluate(duplicate,nboot=24)


def test_plots_reject_non_test_and_handle_observed_input(tmp_path):
    from fededge_protocol.plots import observed_figures
    import pytest
    d=pd.DataFrame({'subject_id':range(20), 'y_true':[0,1]*10,
                    'y_prob':[.2,.8]*10,'split':['test']*20})
    result=observed_figures(d,tmp_path/'plots')
    assert result['n_patients']==20
    assert (tmp_path/'plots'/'observed_roc.png').exists()
    d.loc[0,'split']='train'
    with pytest.raises(ValueError): observed_figures(d,tmp_path/'reject')


def test_primary_cli_persists_feasibility_failure(tmp_path):
    import subprocess,sys
    echo,ecg,adm,events=example_data(n=8)
    ecg['ecg_clock_qc']='unverified'
    mapping={'echo':echo,'ecg':ecg,'admissions':adm,'events':events}
    for name,table in mapping.items():table.to_csv(tmp_path/(name+'.csv'),index=False)
    out=tmp_path/'primary_output'
    args=[sys.executable,'-m','fededge_protocol.cohort']
    for name in mapping:args.extend(['--'+name,str(tmp_path/(name+'.csv'))])
    args.extend(['--outdir',str(out)])
    execution=subprocess.run(args,capture_output=True,text=True)
    assert execution.returncode!=0
    assert (out/'qc_report.json').exists()
    assert (out/'INFEASIBLE.txt').exists()
    assert not (out/'restricted_patient_manifest.csv').exists()


def test_preindex_lab_sample_postindex_result_rejected():
    # A sample collected BEFORE echo but its result stored AFTER echo is
    # not permitted as a feature at echo index time.
    echo,ecg,adm,events=example_data(n=6)
    events.loc[(events.subject_id==1000) & (events.event_type=='HR'),
               'available_time']=pd.to_datetime(echo.loc[0,'echo_time'])+pd.Timedelta(hours=1)
    cohort,q=make_cohort(echo,ecg,adm,events)
    assert 1000 not in cohort.subject_id.to_numpy()
    assert len(cohort)==5
    assert q['postindex_result_records_excluded']>=1

def test_missing_availability_field_cannot_enter_primary():
    from pytest import raises
    echo,ecg,adm,events=example_data(n=6)
    with raises(ValueError,match='available_time'):
        make_cohort(echo,ecg,adm,events.drop(columns=['available_time']))

def test_native_adapter_requires_lab_result_storetime():
    from pytest import raises
    from fededge_protocol.native_adapter import normalize_native
    echo,ecg,adm,events=example_data(n=6)
    e=echo.rename(columns={'echo_study_id':'study_id','echo_time':'study_datetime'})
    c=ecg.rename(columns={'ecg_study_id':'study_id'})
    v=events.rename(columns={'event_time':'charttime','available_time':'storetime','event_type':'itemid'})
    with raises(ValueError,match='storetime'):
        normalize_native(e,c,adm,v.drop(columns=['storetime']))
