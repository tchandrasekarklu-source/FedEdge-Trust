"""Locked patient-level tensors from a *separately audited* linked cohort.

No IDs/patient histories embedded in this project. Data source authenticity
cannot be proven from an NPZ alone; retain licensed linkage manifest locally.
"""
from pathlib import Path
import numpy as np
import torch

FIELDS=('echo','ecg','ehr','adj','presence','quality','lag_hours','clock_verified','y','subject_id','split','client_id')
INPUT_FIELDS=FIELDS[:8]
SPLITS=('train','tune','calibration','test')

class LinkedCohort:
    def __init__(self,path,require_primary=True):
        if not Path(path).is_file(): raise FileNotFoundError(path)
        with np.load(path,allow_pickle=False) as z:
            missing=set(FIELDS)-set(z.files)
            if missing: raise ValueError(f'Missing cohort fields: {sorted(missing)}')
            self.raw={k:z[k].copy() for k in FIELDS}
        d=self.raw; n=len(d['y'])
        if n<8 or any(len(v)!=n for v in d.values()):raise ValueError('N too small or inconsistent')
        if len(set(map(str,d['subject_id'])))!=n:
            raise ValueError('One record per subject for this initial evaluation; duplicate/overlapping IDs forbidden')
        if set(map(str,d['split']))!=set(SPLITS):
            raise ValueError('Must have nonempty train/tune/calibration/test splits')
        if np.any(~np.isin(d['y'],[0,1])):
            raise ValueError('Binary labels only')
        if d['echo'].ndim!=5 or d['ecg'].ndim!=3 or d['ehr'].ndim!=3 or d['adj'].ndim!=3:
            raise ValueError('Invalid modalities dimensions')
        if d['ecg'].shape[1]!=12 or d['echo'].shape[2]!=1 or d['adj'].shape[1:]!=(d['ehr'].shape[1],)*2:
            raise ValueError('Wrong ECG channels, echo channel or EHR graph shape')
        if any(d[k].shape!=(n,3) for k in ('presence','quality','lag_hours','clock_verified')):
            raise ValueError('Modality fields must be [N,3]')
        if any(not np.isfinite(d[k]).all() for k in ('echo','ecg','ehr','adj','quality','lag_hours')):
            raise ValueError('Nonfinite tensor values')
        if not np.isfinite(d['quality']).all() or (d['quality']<0).any() or (d['quality']>1).any():
            raise ValueError('Invalid quality')
        if (d['lag_hours']<0).any(): raise ValueError('Future or negative feature-to-index lag')
        # Only train subjects may be members of federated client training partitions.
        tr=d['split'].astype(str)=='train'
        if (d['client_id'][~tr].astype(str)!='').any():
            raise ValueError('Nontraining row assigned to a virtual federated client')
        if (d['client_id'][tr].astype(str)=='').any():
            raise ValueError('Training client assignment missing')
        self.primary=require_primary
        if require_primary:
            if not np.all(d['presence'].astype(bool).all(1)):
                raise ValueError('Primary tri-modal cohort requires all three real modalities')
            if not np.all(d['clock_verified'].astype(bool).all(1)):
                raise ValueError('Unverified ECG/echo/EHR clocks cannot enter strict primary cohort')
        self.n=n
    def indices(self,split):
        if split not in SPLITS: raise ValueError(split)
        return np.flatnonzero(self.raw['split'].astype(str)==split)
    def batch(self,indices,device='cpu'):
        d=self.raw
        return tuple(torch.as_tensor(d[k][indices], device=device,
                dtype=torch.bool if k in ('presence','clock_verified') else torch.float32)
                for k in INPUT_FIELDS),torch.as_tensor(d['y'][indices], device=device,dtype=torch.float32)
    def dims(self):
        d=self.raw
        return dict(frames=d['echo'].shape[1],height=d['echo'].shape[-2],width=d['echo'].shape[-1],
                    ehr_features=d['ehr'].shape[-1])
    def client_indices(self):
        d=self.raw; tr=self.indices('train')
        return {str(key):tr[d['client_id'][tr].astype(str)==str(key)]
                for key in sorted(set(d['client_id'][tr].astype(str)))}
