"""Reproducible LARGE *SYNTHETIC ONLY* multimodal input fixture.

No actual hospital/patient/electrocardiogram/echocardiography data are included.
Source video frames have meaningful temporal/anatomic patterns plus speckle;
this is an engineering stress fixture, NOT a clinical research dataset.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

FRAMES=64; RES=48; PATCH4=np.array([4,20,36,52]); SEED=20261029


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()


def generate(out, n=1024, chunk=64, seed=SEED):
    out=Path(out); videos=out/'source_echo_frames_SYNTHETIC_ONLY'; videos.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    # Proportions are a software fixture, not clinical cohort sizes or event prevalences.
    age=rng.integers(30,90,size=n,dtype=np.int16)
    sex=rng.integers(0,2,size=n,dtype=np.int8)
    diabetes=(rng.random(n)<.22).astype(np.int8)
    z=.55*((age.astype(float)-55)/17)+.36*diabetes+.08*sex+rng.normal(size=n)
    probs=1/(1+np.exp(-(z-.9)))
    y=(rng.random(n)<probs).astype(np.int8)
    splits=np.full(n,'',dtype='<U11')
    # Disjoint stratified train / tune / calibration / locked-test fixture.
    for cls in (0,1):
        idx=np.flatnonzero(y==cls);rng.shuffle(idx);m=len(idx)
        b1=int(round(.6*m));b2=b1+int(round(.15*m));b3=b2+int(round(.10*m))
        for ix,key in [(idx[:b1],'train'),(idx[b1:b2],'tune'),(idx[b2:b3],'calibration'),(idx[b3:],'test')]:splits[ix]=key
    cid=np.full(n,'',dtype='<U12');tr=np.flatnonzero(splits=='train')
    # Each TRAIN subject receives one and only one synthetic virtual client.
    for cls in (0,1):
        idx=tr[y[tr]==cls];rng.shuffle(idx)
        weights=rng.dirichlet(np.repeat(.8,5));counts=rng.multinomial(len(idx),weights)
        offset=0
        for j,count in enumerate(counts):
            cid[idx[offset:offset+count]]=f'virtual_{j+1}';offset+=count
    assert len(set(cid[tr]))==5
    # Derive actual trainable tensors from the high-resolution source fixture.
    echo=np.empty((n,4,1,32,32),dtype=np.float32)
    ecg=np.empty((n,12,128),dtype=np.float32)
    ehr=np.empty((n,6,16),dtype=np.float32)
    adj=np.zeros((n,6,6),dtype=np.float32)
    presence=np.ones((n,3),dtype=np.int8)
    quality=rng.uniform(.75,.98,size=(n,3)).astype(np.float32)
    lag_hours=rng.uniform(0,32,size=(n,3)).astype(np.float32)
    clock_verified=np.ones((n,3),dtype=np.int8)
    xaxis=np.arange(RES,dtype=np.float32)-((RES-1)/2)
    yy,xx=np.meshgrid(xaxis,xaxis,indexing='ij')
    t=np.arange(FRAMES,dtype=np.float32)
    # Synthetic cardiac ring with a label-dependent radius and rhythmic motion.
    for start in range(0,n,chunk):
        count=min(chunk,n-start)
        pix=rng.integers(12,245,size=(count,FRAMES,RES,RES),dtype=np.uint8)
        for j in range(count):
            i=start+j
            cyclic=np.sin(2*np.pi*t/(19+(i%5)))
            r=10.0 + 2.6*float(y[i]) + 1.6*cyclic
            # Faint moving ring -> valid signal, combined with synthetic speckle.
            rad=np.sqrt(xx[None,:,:]**2+yy[None,:,:]**2)
            ring=(np.abs(rad-r[:,None,None])<1.1)
            pix[j][ring]=np.clip(pix[j][ring].astype(np.int16)+45,0,255).astype(np.uint8)
            echo[i,:,0]=(pix[j,PATCH4,8:40,8:40].astype(np.float32)/255.0)
            time=np.linspace(0,1,128,dtype=np.float32)
            bpm=1.0+(float(y[i])*.38)+(.07*(i%4))
            base=np.sin(2*np.pi*bpm*7*time)+.23*np.sin(2*np.pi*bpm*14*time)
            ecg[i]=(base[None,:]+rng.normal(0,.12,size=(12,128))+.13*float(y[i])*np.cos(2*np.pi*2*time)[None,:]).astype(np.float32)
            ehr[i]=rng.normal(size=(6,16)).astype(np.float32)*.3
            ehr[i,:,0]=float((age[i]-55)/18)+rng.normal(size=6)*.05
            ehr[i,:,1]=float(diabetes[i])
            ehr[i,:,2]=float(y[i])*.5+rng.normal(size=6)*.2
            adj[i]=np.eye(6,dtype=np.float32)
            adj[i,np.arange(5),np.arange(1,6)]=1
            adj[i,np.arange(1,6),np.arange(5)]=1
        path=videos/f'synthetic_echo_frames_{start:04d}_{start+count-1:04d}.npy'
        np.save(path,pix,allow_pickle=False)
    ids=np.array([f'SYNTHETIC_{i:05d}' for i in range(n)])
    dct=dict(echo=echo,ecg=ecg,ehr=ehr,adj=adj,presence=presence,quality=quality,
             lag_hours=lag_hours,clock_verified=clock_verified,y=y,subject_id=ids,split=splits,client_id=cid)
    np.savez_compressed(out/'synthetic_all_1024_linked_tensors.npz',**dct)
    # A compact balanced smoke-test NPZ with all four splits and five clients.
    selected=[]
    for label in ('train','tune','calibration','test'):
        si=np.flatnonzero(splits==label)
        for cls in (0,1):
            c=si[y[si]==cls]; selected.extend(c[:min(8,len(c))].tolist())
    # Client coverage on train rows may need explicit supplementation.
    for client in [f'virtual_{i}' for i in range(1,6)]:
        ix=np.flatnonzero((splits=='train')&(cid==client))
        if ix.size: selected.extend(ix[:4].tolist())
    selected=np.unique(selected)
    np.savez_compressed(out/'synthetic_tiny_smoke_test.npz',**{k:v[selected] for k,v in dct.items()})
    df=pd.DataFrame(dict(subject_id=ids,split=splits,client_id=cid,age=age,sex_code=sex,
                         diabetes_code=diabetes,y_true=y,source='SYNTHETIC_ONLY'))
    df.to_csv(out/'synthetic_subject_manifest_NO_REAL_PATIENTS.csv',index=False)
    files=sorted(videos.glob('*.npy'))
    manifest={'is_synthetic':True,'is_clinical_evidence':False,'seed':seed,'n':n,'frames_per_video':FRAMES,
              'source_video_resolution':[RES,RES], 'modality_names':['echo','ecg','ehr'],
              'simulation_not_dicom':True,'positive_count':int(y.sum()),
              'split_counts':{k:int((splits==k).sum()) for k in ['train','tune','calibration','test']},
              'client_counts':{k:int((cid==k).sum()) for k in np.unique(cid) if k},
              'shards':[{'file':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],
              'derived_tensor_sha256':sha(out/'synthetic_all_1024_linked_tensors.npz'),
              'generator':'tools/make_synthetic_cohort.py',
              'notice':'Synthetic procedural engineering data; no clinical efficacy, privacy guarantees or site generalization inferred.'}
    (out/'synthetic_generation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='shards'},indent=2))
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='synthetic_fixtures')
    ap.add_argument('--n',type=int,default=1024);ap.add_argument('--chunk',type=int,default=64)
    ap.add_argument('--seed',type=int,default=SEED)
    args=ap.parse_args(); generate(args.out,args.n,args.chunk,args.seed)
