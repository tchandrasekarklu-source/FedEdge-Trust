"""All records constructed here are artificial software test fixtures, never clinical data."""
import numpy as np
import pandas as pd
import pytest
import torch
from fededge_experiments.model import ITGFusion
from fededge_experiments.privacy import rdp_accountant, private_step
from fededge_experiments.data import LinkedCohort
from fededge_experiments.compare import paired_compare
from fededge_experiments.edge_device import integrated_energy_joules
from fededge_experiments.experiments import calibrate_temperature,report_metrics,run,parser


def mock_npz(path,n=28):
    rng=np.random.default_rng(10)
    split=np.array(['train']*16+['tune']*4+['calibration']*4+['test']*4)
    labels=np.array(([0,1]*8)+[0,1,0,1]+[0,1,0,1]+[0,1,0,1])
    client=np.array(['a']*8+['b']*8+['']*12)
    np.savez(path,echo=rng.normal(size=(n,2,1,16,16)).astype('f4'),
             ecg=rng.normal(size=(n,12,32)).astype('f4'),
             ehr=rng.normal(size=(n,3,8)).astype('f4'),
             adj=np.broadcast_to(np.eye(3),(n,3,3)).astype('f4'),
             presence=np.ones((n,3),dtype=np.uint8),quality=np.ones((n,3),dtype='f4'),
             lag_hours=np.zeros((n,3),dtype='f4'),clock_verified=np.ones((n,3),dtype=np.uint8),
             y=labels,subject_id=np.array([f'mock_{i:03d}' for i in range(n)]),
             split=split,client_id=client)


def test_model_training_gradient_and_shapes():
    model=ITGFusion(dim=16,frames=2,height=16,width=16,ehr_features=8)
    x=(torch.randn(2,2,1,16,16),torch.randn(2,12,32),torch.randn(2,3,8),torch.eye(3).repeat(2,1,1),
       torch.ones(2,3,dtype=torch.bool),torch.ones(2,3),torch.zeros(2,3),torch.ones(2,3,dtype=torch.bool))
    y=model(*x)
    assert y.shape==(2,) and y.isfinite().all()
    torch.nn.functional.binary_cross_entropy_with_logits(y,torch.tensor([1.,0.])).backward()
    assert model.echo.patch.weight.grad is not None
    assert model.ecg.cnn[0].weight.grad is not None
    assert model.ehr.graph.weight.grad is not None
    assert model.fusion.query.weight.grad is not None


def test_primary_clock_validation(tmp_path):
    f=tmp_path/'mock.npz';mock_npz(f)
    assert len(LinkedCohort(f).client_indices())==2
    with np.load(f) as z:data={k:z[k] for k in z.files}
    data['clock_verified']=data['clock_verified'].copy();data['clock_verified'][0,1]=0
    np.savez(f,**data)
    with pytest.raises(ValueError,match='Unverified'):LinkedCohort(f)


def test_rdp_monotonic():
    x=rdp_accountant(.05,1.1,20,1e-5)
    y=rdp_accountant(.05,1.1,40,1e-5)
    z=rdp_accountant(.05,1.8,40,1e-5)
    assert 0<x['epsilon_upper_bound']<y['epsilon_upper_bound']
    assert z['epsilon_upper_bound'] < y['epsilon_upper_bound']
    assert rdp_accountant(1.,1.,10,1e-5)['epsilon_upper_bound']>0


def test_private_step_runs_and_clips():
    class Linear(torch.nn.Module):
        def __init__(self): super().__init__();self.lin=torch.nn.Linear(2,1)
        def forward(self, x): return self.lin(x).squeeze(-1)
    torch.manual_seed(1)
    m=Linear();opt=torch.optim.SGD(m.parameters(),lr=.02)
    x=torch.randn(8,2);y=torch.tensor([0.,1.]*4)
    k=private_step(m,(x,),y,opt,.5,1.0,1.0,rng=torch.Generator().manual_seed(10))
    assert 0<=k<=8 and torch.isfinite(m.lin.weight).all()


def test_compare_subject_pairing():
    d=pd.DataFrame({'subject_id':['a','b','c','d','e','f'], 'y_true':[0,1,0,1,0,1],
                    'y_prob':[.1,.9,.3,.8,.2,.7], 'split':['test']*6})
    out=paired_compare(d,d.copy(),50)
    assert out['auroc']['delta_a_minus_b']==0
    bad=d.copy();bad.loc[0,'subject_id']='changed'
    with pytest.raises(ValueError,match='identical patients'):paired_compare(d,bad,50)


def test_energy_integration():
    df=pd.DataFrame({'epoch_seconds':[0,1,2,3],'watts':[2.,2.,2.,2.]})
    assert abs(integrated_energy_joules(df,.5,2.5)-4.)<1e-10
    assert abs(integrated_energy_joules(df,.5,2.5,idle_power=1.)-2.)<1e-10


def test_run_actual_computation_on_mock_only(tmp_path):
    torch.set_num_threads(1)
    f=tmp_path/'mock.npz';mock_npz(f)
    out=tmp_path/'results'
    args=parser().parse_args(['--cohort',str(f),'--outdir',str(out),'--models','full,late',
             '--seeds','9','--epochs','1','--dim','16','--batch','4','--device','cpu'])
    results=run(args)
    assert len(results)==2 and all(r['test_n']==4 for r in results)
    assert (out/'central_full_seed9'/'model.pt').is_file()
    assert (out/'central_full_seed9'/'locked_test_predictions_RESTRICTED.csv').is_file()
    assert not any('clinical' in str(x).lower() for x in (out/'all_seed_metrics.csv').read_text().splitlines())
    with pytest.raises(FileExistsError):run(args)

def test_multi_release_privacy_composition():
    from fededge_experiments.privacy_release_ledger import compose_dp_releases
    rel=[]
    for rid in ['a','b']:
        for cid in ['one','two']:
            rel.append({'release_id':rid,'client_id':cid,'epsilon':1.2,'delta':1e-5,
                        'subject_disjoint_across_clients':True})
    ledger=compose_dp_releases(rel)
    assert ledger['parallel_client_epsilon_upper_bound']==2.4
    assert abs(ledger['parallel_client_delta_upper_bound']-2e-5)<1e-12
    rel[0]['subject_disjoint_across_clients']=False
    with pytest.raises(ValueError):compose_dp_releases(rel)
