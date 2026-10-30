"""Create deterministic tiny cohort from full synthetic records, at least 4 train cases/client."""
import argparse
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--dir',default='synthetic_fixtures');a=p.parse_args()
 d=Path(a.dir)
 with np.load(d/'synthetic_all_1024_linked_tensors.npz',allow_pickle=False) as z:
  raw={k:z[k] for k in z.files}
 y=raw['y'];splits=raw['split'];cid=raw['client_id'];selected=[]
 for label in ('train','tune','calibration','test'):
  si=np.flatnonzero(splits==label)
  for cls in (0,1):
   c=si[y[si]==cls];selected.extend(c[:min(8,len(c))].tolist())
 for client in [f'virtual_{i}' for i in range(1,6)]:
  ix=np.flatnonzero((splits=='train')&(cid==client));selected.extend(ix[:4].tolist())
 selected=np.unique(selected)
 np.savez_compressed(d/'synthetic_tiny_smoke_test.npz',**{k:v[selected] for k,v in raw.items()})
 print('Synthetic tiny suite',len(selected),'cases, clients', {i:int(np.sum(cid[selected]==i)) for i in sorted(set(cid[selected])) if i})

if __name__=='__main__':main()
