"""Third-review evidence gates. Accept genuine investigator-supplied data only.

Cohort CSV schema: subject_id,encounter_id,echo_exam_id,ecg_record_id,
index_time,echo_time,ecg_time,ehr_latest_time,outcome_time,y_true,split,source.
All timestamps ISO-8601. One patient-one index sample; no time leakage allowed.
Authorized acquisition metadata must be supplied in JSON and signed off by the PI.

Predictions CSV schema: subject_id,split,seed,model,y_true,y_prob.
Strict same-patient, same-label, same split matching; outputs per seed + bootstrap
paired H1 comparisons across patients (the *patient* is the resampling unit).
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss, log_loss, matthews_corrcoef

REQUIRED_META = ['dataset','dataset_version','institution_or_source','data_access_mechanism',
                 'ethical_approval_or_waiver','acquisition_period','endpoint_definition',
                 'endpoint_adjudication','source_manifest_sha256']
REQUIRED_COHORT = ['subject_id','encounter_id','echo_exam_id','ecg_record_id','index_time',
                   'echo_time','ecg_time','ehr_latest_time','outcome_time','y_true','split','source']


def sha256(path):
    with open(path,'rb') as f:
        h=hashlib.sha256()
        while True:
            b=f.read(1024*1024)
            if not b:break
            h.update(b)
        return h.hexdigest()


def validate_cohort(df,meta,source_file=None,max_ecg_hours=72):
    missing=set(REQUIRED_COHORT)-set(df.columns)
    if missing:raise ValueError('Cohort missing columns: '+', '.join(sorted(missing)))
    for k in REQUIRED_META:
        if not isinstance(meta.get(k),str) or len(meta[k].strip())<4:
            raise ValueError('Missing independently verifiable acquisition/ethics metadata: '+k)
    if source_file and sha256(source_file)!=meta['source_manifest_sha256']:
        raise ValueError('Cohort source-manifest SHA256 mismatch')
    if df.empty or df[REQUIRED_COHORT].isna().any().any():
        raise ValueError('Empty cohort or missing mandatory values')
    if (df.subject_id.astype(str).str.strip()=='').any() or df.subject_id.duplicated().any():
        raise ValueError('Patient overlap or duplicate index encounters; one subject per cohort row')
    if set(df.split.unique())-{'train','tune','calibration','test'}:
        raise ValueError('Unexpected split label')
    if not set(np.unique(df.y_true))<= {0,1}:
        raise ValueError('Binary label expected; dataset endpoint must match manifest')
    times={k:pd.to_datetime(df[k],utc=True,errors='raise') for k in
           ['index_time','echo_time','ecg_time','ehr_latest_time','outcome_time']}
    index=times['index_time']
    if (times['echo_time']>index).any() or (times['ecg_time']>index).any():
        raise ValueError('Future echo or ECG used as a pre-index predictor')
    if ((index-times['ecg_time']).dt.total_seconds()>max_ecg_hours*3600).any():
        raise ValueError('ECG exceeds the prespecified pre-index temporal window')
    if (times['ehr_latest_time']>index).any() or (times['outcome_time']<=index).any():
        raise ValueError('Post-index feature leakage / non-post-index outcome')
    return {'n':len(df),'events':int(df.y_true.sum()),'non_events':int(len(df)-df.y_true.sum()),
            'split':{s:{'n':len(g),'events':int(g.y_true.sum())} for s,g in df.groupby('split')},
            'subject_id_hash_sha256':hashlib.sha256('\n'.join(sorted(df.subject_id.astype(str))).encode()).hexdigest(),
            'valid_source_timing':True,'note':'Metadata validation is not independent confirmation that clinical access was authorized.'}


def ece(y,p,bins=10):
    y=np.asarray(y);p=np.asarray(p)
    ix=np.minimum((p*bins).astype(int),bins-1)
    return float(sum(np.mean(ix==b)*abs(y[ix==b].mean()-p[ix==b].mean()) for b in range(bins) if np.any(ix==b)))


def perf(y,p,threshold=.5):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if len(y)<2 or set(np.unique(y))!={0,1}:raise ValueError('Both outcome classes required')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Finite probabilities within [0,1] required')
    yhat=(p>=threshold).astype(int)
    tn,fp,fn,tp=map(int,confusion_matrix(y,yhat,labels=[0,1]).ravel())
    pp=tp/(tp+fp) if tp+fp else 0
    r=tp/(tp+fn) if tp+fn else 0
    return {'n':int(len(y)),'events':int(y.sum()),'tn':tn,'fp':fp,'fn':fn,'tp':tp,
      'accuracy':float((tp+tn)/len(y)),'precision':float(pp),'sensitivity':float(r),
      'specificity':float(tn/(tn+fp)),'f1':float(2*pp*r/(pp+r)) if pp+r else 0.,
      'mcc':float(matthews_corrcoef(y,yhat)),'auroc':float(roc_auc_score(y,p)),
      'auprc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),
      'nll':float(log_loss(y,p,labels=[0,1])),'ece':ece(y,p),'threshold':threshold}


def read_predictions(path):
    df=pd.read_csv(path,dtype={'subject_id':'string','model':'string','split':'string'})
    req={'subject_id','split','seed','model','y_true','y_prob'}
    if not req.issubset(df):raise ValueError('Missing prediction fields: '+str(req-set(df)))
    if df[list(req)].isnull().any().any():raise ValueError('Null prediction field')
    if set(df.split.unique())!={'test'}:raise ValueError('Only locked test predictions permitted')
    if df.duplicated(['subject_id','seed','model']).any():raise ValueError('Duplicate per-patient prediction')
    if not set(df.y_true.unique())<={0,1}:raise ValueError('Nonbinary outcome')
    if not np.isfinite(df.y_prob).all() or ((df.y_prob<0)|(df.y_prob>1)).any():
        raise ValueError('Probability outside [0,1]')
    return df


def evaluate_pairs(df,a='gated',b='masked',reps=500,seed=20265796):
    if reps<50:raise ValueError('At least 50 bootstrap draws')
    models=set(df.model.unique())
    if not {a,b}.issubset(models):raise ValueError('Required baseline and target models absent')
    seeds=sorted(set(df.seed.unique()))
    y_all=[];pa_all=[];pb_all=[];n=None;ids_ref=None; per_seed=[]
    for sd in seeds:
        x=df[(df.seed==sd)&(df.model==a)].sort_values('subject_id')
        z=df[(df.seed==sd)&(df.model==b)].sort_values('subject_id')
        if len(x)!=len(z) or not np.array_equal(x.subject_id.values,z.subject_id.values):
            raise ValueError('Same-patient matched model runs required')
        if n is None:
            n=len(x);ids_ref=x.subject_id.to_numpy()
        if len(x)!=n or not np.array_equal(ids_ref,x.subject_id.to_numpy()):
            raise ValueError('The SAME locked patient IDs are required for every seed')
        if not np.array_equal(x.y_true.to_numpy(),z.y_true.to_numpy()):
            raise ValueError('Outcome mismatch across baselines')
        ya=x.y_true.to_numpy(dtype=int)
        if y_all and not np.array_equal(ya,y_all[0]):raise ValueError('Labels differ across seeds')
        y_all.append(ya);pa_all.append(x.y_prob.to_numpy(float));pb_all.append(z.y_prob.to_numpy(float))
        per_seed.append({'seed':int(sd),'model_a':perf(ya,pa_all[-1]),'model_b':perf(ya,pb_all[-1])})
    y=y_all[0]
    rng=np.random.default_rng(seed)
    strata=[np.flatnonzero(y==j) for j in (0,1)]
    def diffs(ix):
        aa=[perf(y[ix],p[ix]) for p in pa_all]
        bb=[perf(y[ix],p[ix]) for p in pb_all]
        return {k:float(np.mean([aa[i][k]-bb[i][k] for i in range(len(seeds))]))
                for k in ('auroc','auprc','brier','ece','accuracy','f1')}
    observed=diffs(np.arange(n));v={k:[] for k in observed}
    for _ in range(reps):
        ix=np.concatenate([rng.choice(z,len(z),replace=True) for z in strata]);v0=diffs(ix)
        for k in observed:v[k].append(v0[k])
    outcome={k:{'mean_paired_delta':observed[k],
                'patient_bootstrap_95_CI':list(map(float,np.quantile(v[k],[.025,.975])))} for k in observed}
    return {'n_paired_test_subjects':int(n),'n_test_events':int(y.sum()),'seeds':list(map(int,seeds)),
            'comparison':f'{a} minus {b}','per_seed':per_seed,'paired_effects':outcome,
            'H1_rule':'AUROC CI lower > 0 and Brier CI upper < 0 (direction pre-specified); report effect size even if null',
            'H1_supported_by_this_comparison':bool(outcome['auroc']['patient_bootstrap_95_CI'][0]>0 and outcome['brier']['patient_bootstrap_95_CI'][1]<0),
            'scope':'Valid only for provenance-authenticated clinical held-out predictions; software cannot certify dataset origin'}


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('cohort');c.add_argument('--csv',required=True);c.add_argument('--metadata',required=True);c.add_argument('--out',required=True)
    e=sub.add_parser('compare');e.add_argument('--predictions',required=True);e.add_argument('--model',default='gated');e.add_argument('--baseline',default='masked');e.add_argument('--reps',type=int,default=1000);e.add_argument('--out',required=True)
    a=p.parse_args()
    if a.cmd=='cohort': result=validate_cohort(pd.read_csv(a.csv),json.loads(Path(a.metadata).read_text()),a.csv)
    else: result=evaluate_pairs(read_predictions(a.predictions),a.model,a.baseline,a.reps)
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({'output':a.out,'ok':True,'type':a.cmd}))

if __name__=='__main__':main()
