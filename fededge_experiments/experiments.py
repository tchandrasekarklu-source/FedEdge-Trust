"""Fixed-split multi-run multimodal, matched baselines/ablations, FedAvg/FedProx+DP.

This module reports results ONLY from a user-supplied verified patient cohort.
It does NOT ship clinical results, fabricated labels or generated patient records.
"""
import argparse, copy, hashlib, json, math, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score,recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, brier_score_loss, log_loss)
from .data import LinkedCohort
from .model import ITGFusion
from .privacy import private_step, rdp_accountant


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def predict(model, cohort, idx, device, batch_size=16):
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(idx),batch_size):
            x,_=cohort.batch(idx[start:start+batch_size],device)
            out.extend(model(*x).detach().cpu().numpy().tolist())
    return np.asarray(out,dtype=np.float64)


def train_epoch(model,cohort,idx,optimizer,device,batch_size=8,seed=0,ref=None,mu=0.):
    model.train();rr=np.random.default_rng(seed);perm=rr.permutation(idx)
    losses=[]
    for start in range(0,len(perm),batch_size):
        x,y=cohort.batch(perm[start:start+batch_size],device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(*x)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
        if mu and ref is not None:
            loss=loss+(mu/2)*sum(((p-ref[k])**2).sum() for k,p in model.named_parameters())
        loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def calibrate_temperature(logits, labels):
    """Fit ONLY on calibration labels; never touches the test split."""
    xx=torch.tensor(np.asarray(logits),dtype=torch.float64)
    yy=torch.tensor(np.asarray(labels),dtype=torch.float64)
    theta=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))
    opt=torch.optim.LBFGS([theta],lr=.2,max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(xx/torch.exp(theta),yy)
        loss.backward();return loss
    opt.step(closure)
    return float(torch.exp(theta.detach()).clamp(min=.05,max=20.))


def sigmoid(x):
    z=np.asarray(x,dtype=np.float64)
    return 1/(1+np.exp(-np.clip(z,-70,70)))


def operating_threshold(probs, labels):
    # Calibration split only. Set an explicit clinical sensitivity target in the final clinical protocol.
    grid=np.linspace(.05,.95,91)
    scores=[f1_score(labels,probs>=t,zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def report_metrics(labels,probs,threshold):
    labels=np.asarray(labels).astype(int); probs=np.asarray(probs)
    prediction=(probs>=threshold).astype(int)
    bins=np.linspace(0,1,16); ece=0.
    for a,b in zip(bins[:-1],bins[1:]):
        m=(probs>=a)&(probs<b if b<1 else probs<=b)
        if m.any():ece+=(m.mean())*abs(float(labels[m].mean())-float(probs[m].mean()))
    cm=confusion_matrix(labels,prediction,labels=[0,1])
    return {'n':int(len(labels)),'events':int(labels.sum()),
            'auroc':float(roc_auc_score(labels,probs)) if len(np.unique(labels))==2 else None,
            'auprc':float(average_precision_score(labels,probs)) if labels.sum() else None,
            'accuracy':float(accuracy_score(labels,prediction)),
            'precision':float(precision_score(labels,prediction,zero_division=0)),
            'sensitivity':float(recall_score(labels,prediction,zero_division=0)),
            'specificity':float(cm[0,0]/max(1,cm[0].sum())),
            'f1':float(f1_score(labels,prediction,zero_division=0)),
            'mcc':float(matthews_corrcoef(labels,prediction)),
            'brier':float(brier_score_loss(labels,probs)),
            'nll':float(log_loss(labels,np.clip(probs,1e-8,1-1e-8),labels=[0,1])),
            'ece_15':float(ece),'tn_fp_fn_tp':cm.ravel().astype(int).tolist()}


def make_model(cohort,variant,dim,device):
    dims=cohort.dims()
    if dims['height'] % 8 or dims['width'] % 8 or dim%4:
        raise ValueError('Height/width multiple of 8 and dimension divisible by 4 required')
    return ITGFusion(dim=dim,variant=variant,**dims).to(device)


def federated_train(model,cohort,device,rounds,local_steps,lr,batch,mu,dp_sigma=0.,clip=1.,delta=1e-5,seed=42):
    clients=cohort.client_indices()
    if len(clients)<2:raise ValueError('Requires >=2 nonempty disjoint virtual clients')
    all_train=np.concatenate(list(clients.values()))
    if len(np.unique(all_train)) != len(all_train) or set(all_train) != set(cohort.indices('train')):
        raise ValueError('Federated partition fails train-only isolation')
    accountant={k:0 for k in clients};rates={k:min(1.,batch/len(v)) for k,v in clients.items()}
    history=[]
    for round_idx in range(rounds):
        global_state=copy.deepcopy(model.state_dict())
        local_states=[];sizes=[]
        for c,(client,idx) in enumerate(clients.items()):
            loc=copy.deepcopy(model)
            opt=torch.optim.SGD(loc.parameters(),lr=lr)
            ref={k:p.detach().clone() for k,p in loc.named_parameters()}
            for step in range(local_steps):
                if dp_sigma:
                    x,y=cohort.batch(idx,device)
                    gr=torch.Generator(device=device).manual_seed(seed+round_idx*100003+c*1009+step)
                    ref_id={id(p):ref[k] for k,p in loc.named_parameters()}
                    private_step(loc,x,y,opt,rates[client],dp_sigma,clip,ref_id,mu,gr)
                    accountant[client]+=1
                else:
                    train_epoch(loc,cohort,idx,opt,device,batch,seed+round_idx*10000+c*100+step,ref,mu)
            local_states.append({k:v.detach().clone() for k,v in loc.state_dict().items()})
            sizes.append(len(idx))
        model.load_state_dict({k:sum(state[k]*(n/sum(sizes)) for state,n in zip(local_states,sizes))
                               for k in global_state})
        history.append({'round':round_idx+1,'client_count':len(clients)})
    privacy=None
    if dp_sigma:
        privacy={c:rdp_accountant(rates[c],dp_sigma,steps,delta) for c,steps in accountant.items()}
        privacy={'per_client':privacy,'epsilon_parallel_composition_upper_bound':max(v['epsilon_upper_bound'] for v in privacy.values()),
                 'delta_per_client':delta,'no_secure_aggregation':True,
                 'other_private_processing_not_accounted_for':True}
    return model,history,privacy


def run(args):
    path=Path(args.outdir);path.mkdir(parents=True,exist_ok=True)
    if (path/'FINAL_LOCKED_TEST_RESULTS.json').exists() and not args.allow_overwrite:
        raise FileExistsError('Locked test outputs already exist; prevent selective reruns. Use new output directory.')
    cohort=LinkedCohort(args.cohort,require_primary=not args.secondary_missing_modality_analysis)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
    device=torch.device(args.device)
    models=args.models.split(',');seeds=[int(x) for x in args.seeds.split(',')]
    if len(set(seeds))!=len(seeds) or len(seeds)<1:raise ValueError('Seeds must be unique')
    if args.dp_sigma and args.method not in ('fedavg','fedprox'):
        raise ValueError('DP-SGD is supported only for federated methods')
    split_info={s:cohort.indices(s) for s in ('train','tune','calibration','test')}
    if args.dp_sigma and not(0<args.delta<1/len(split_info['train'])):
        raise ValueError('For this study choose delta < 1/n_training_subjects')
    unique=np.unique(cohort.raw['subject_id'].astype(str))
    split_fingerprint=hashlib.sha256(('|'.join(sorted(unique))+'|'+str([(k,len(v)) for k,v in split_info.items()])).encode()).hexdigest()
    all_results=[]
    for variant in models:
        for seed in seeds:
            set_seed(seed)
            model=make_model(cohort,variant,args.dim,device)
            best=None;privacy=None;history=[]
            if args.method=='central':
                opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
                best_loss=np.inf
                for epoch in range(args.epochs):
                    tr_loss=train_epoch(model,cohort,split_info['train'],opt,device,args.batch,seed+epoch)
                    tune_logits=predict(model,cohort,split_info['tune'],device,args.batch)
                    yy=cohort.raw['y'][split_info['tune']]
                    tune_nll=float(np.mean(np.logaddexp(0,tune_logits)-yy*tune_logits))
                    history.append({'epoch':epoch+1,'train_loss':tr_loss,'tune_nll':tune_nll})
                    if tune_nll<best_loss:
                        best_loss=tune_nll;best=copy.deepcopy(model.state_dict())
                model.load_state_dict(best)
            else:
                model,history,privacy=federated_train(model,cohort,device,args.rounds,args.local_steps,
                           args.lr,args.batch,args.prox_mu if args.method=='fedprox' else 0.,
                           args.dp_sigma,args.clip,args.delta,seed)
            cal=split_info['calibration'];test=split_info['test']
            yy_cal=cohort.raw['y'][cal].astype(int)
            cal_logits=predict(model,cohort,cal,device,args.batch)
            temp=calibrate_temperature(cal_logits,yy_cal)
            thresh=operating_threshold(sigmoid(cal_logits/temp),yy_cal)
            te_logits=predict(model,cohort,test,device,args.batch)
            probs=sigmoid(te_logits/temp)
            yy_te=cohort.raw['y'][test].astype(int)
            result={'variant':variant,'seed':seed,'method':args.method,
                    'temperature_from_calibration':temp,'threshold_from_calibration':thresh,
                    'metrics':report_metrics(yy_te,probs,thresh),'training_history':history,
                    'privacy':privacy,'train_n':int(len(split_info['train'])),'tune_n':int(len(split_info['tune'])),
                    'calibration_n':int(len(cal)),'test_n':int(len(test)),
                    'split_fingerprint':split_fingerprint,
                    'evidence_note':'Only valid with verified genuine patient-linked NPZ and source audit; no prospective/external validation.'}
            label=f'{args.method}_{variant}_seed{seed}'
            dest=path/label;dest.mkdir(exist_ok=True)
            torch.save({'state_dict':model.cpu().state_dict(),'model_kwargs':{'dim':args.dim,'variant':variant,**cohort.dims()},
                        'config':vars(args)},dest/'model.pt')
            (dest/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
            pd.DataFrame({'subject_id':cohort.raw['subject_id'][test].astype(str),
                          'y_true':yy_te,'y_prob':probs,'split':'test','seed':seed,
                          'variant':variant}).to_csv(dest/'locked_test_predictions_RESTRICTED.csv',index=False)
            all_results.append(result)
    summary=[{'variant':r['variant'],'seed':r['seed'],'method':r['method'],
              **{k:v for k,v in r['metrics'].items() if not isinstance(v,list)}} for r in all_results]
    (path/'FINAL_LOCKED_TEST_RESULTS.json').write_text(json.dumps({'results':summary,'source_file':str(args.cohort),
                   'restricted_subject_level_predictions':True,'split_fingerprint':split_fingerprint},indent=2))
    pd.DataFrame(summary).to_csv(path/'all_seed_metrics.csv',index=False)
    print('Measured outputs saved from supplied linked cohort; check data authenticity externally:',path)
    return all_results


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',required=True,help='Restricted linked patient tensors NPZ; never upload to public GitHub')
    p.add_argument('--outdir',required=True)
    p.add_argument('--models',default='full,late,echo_only,ecg_only,ehr_only,no_quality_time')
    p.add_argument('--seeds',default='7,17,27,37,47')
    p.add_argument('--method',choices=['central','fedavg','fedprox'],default='central')
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--local-steps',type=int,default=5)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--dim',type=int,default=32);p.add_argument('--device',default='cpu')
    p.add_argument('--prox-mu',type=float,default=.01)
    p.add_argument('--dp-sigma',type=float,default=0.,help='positive for actual Poisson DP-SGD; may be very slow')
    p.add_argument('--clip',type=float,default=1.);p.add_argument('--delta',type=float,default=1e-5)
    p.add_argument('--secondary-missing-modality-analysis',action='store_true')
    p.add_argument('--allow-overwrite',action='store_true')
    return p
if __name__=='__main__':run(parser().parse_args())
