"""Small, truthful FedAvg/FedProx reference optimizer on *precomputed* features.

Not the paper's full ViT + CNN-BiLSTM + TGNN clinical training. Provides
reproducible local-client update mechanics. DP and secure aggregation are NOT
implemented. Do not report synthetic test accuracy as patient evidence.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def _model(n_features):
    return nn.Sequential(nn.Linear(n_features,1))


def run_federated(X,y,client_ids,rounds=5,epochs=2,mu=0.,lr=0.05,seed=20265796):
    X=np.asarray(X,dtype=np.float32);y=np.asarray(y,dtype=np.float32)
    client_ids=np.asarray(client_ids)
    if X.ndim!=2 or y.shape!=(len(X),) or client_ids.shape!=(len(X),):
        raise ValueError('expected X=(patients,features), y and client IDs vectors')
    if not np.isfinite(X).all() or not np.isin(y,[0,1]).all():
        raise ValueError('invalid feature matrix or binary labels')
    clients=sorted({str(k) for k in client_ids})
    if len(clients)<2 or any(k=='' for k in clients):
        raise ValueError('training must have >=2 nonempty virtual client groups')
    if rounds<1 or epochs<1 or lr<=0 or mu<0:raise ValueError('invalid hyperparameters')
    torch.manual_seed(seed)
    model=_model(X.shape[1]);criterion=nn.BCEWithLogitsLoss()
    rounds_log=[]
    for round_idx in range(rounds):
        snapshots=[];sizes=[]; losses=[]
        reference=[p.detach().clone() for p in model.parameters()]
        for client in clients:
            rows=np.where(client_ids.astype(str)==client)[0]
            if len(rows)==0:raise ValueError('empty client')
            local=_model(X.shape[1]);local.load_state_dict(model.state_dict())
            opt=torch.optim.SGD(local.parameters(),lr=lr)
            xx=torch.as_tensor(X[rows]);yy=torch.as_tensor(y[rows]).reshape(-1,1)
            for _ in range(epochs):
                opt.zero_grad()
                empirical=criterion(local(xx),yy)
                penalty=sum(((p-p0)**2).sum() for p,p0 in zip(local.parameters(),reference))
                loss=empirical+(mu/2)*penalty
                loss.backward();opt.step()
            snapshots.append({k:v.detach().clone() for k,v in local.state_dict().items()})
            sizes.append(len(rows));losses.append(float(empirical.detach()))
        combined={}
        for key in model.state_dict():
            combined[key]=sum(snap[key]*(sz/sum(sizes)) for snap,sz in zip(snapshots,sizes))
        model.load_state_dict(combined)
        rounds_log.append({'round':round_idx+1,'weighted_local_train_bce':float(np.average(losses,weights=sizes)),
                           'train_unique_patients':int(sum(sizes))})
    return model,{'optimizer':'FedProx' if mu>0 else 'FedAvg','mu':mu,'lr':lr,'seed':seed,
                  'local_epochs':epochs,'rounds':rounds,'client_patient_counts':dict(zip(clients,sizes)),
                  'training_history':rounds_log,
                  'evidence_boundary':'PRECOMPUTED FEATURES; not full multimodal training; no privacy mechanism'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz',required=True,help='Restricted npz with X, y, split, client_id')
    p.add_argument('--outdir',required=True)
    p.add_argument('--rounds',type=int,default=5);p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--lr',type=float,default=.05);p.add_argument('--mu',type=float,default=0.)
    a=p.parse_args()
    z=np.load(a.npz,allow_pickle=False)
    for key in ('X','y','split','client_id'):
        if key not in z:raise ValueError('missing '+key)
    if set(z['split'].astype(str)) != {'train'}:
        raise ValueError('this optimizer accepts TRAIN ONLY; test and calibration excluded')
    model,report=run_federated(z['X'],z['y'],z['client_id'],a.rounds,a.epochs,a.mu,a.lr)
    dest=Path(a.outdir);dest.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),dest/'reference_federated_linear_weights.pt')
    (dest/'reference_federated_training_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('Reference optimizer trained on supplied features; full system not validated')
if __name__=='__main__':main()
