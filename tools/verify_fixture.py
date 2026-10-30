"""Verify provided large synthetic fixture provenance and model input compatibility."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from fededge_experiments.data import LinkedCohort

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 a=argparse.ArgumentParser();a.add_argument('--dir',default='synthetic_fixtures');x=a.parse_args()
 root=Path(x.dir);m=json.loads((root/'synthetic_generation_manifest.json').read_text())
 assert m['is_synthetic'] and not m['is_clinical_evidence']
 full=LinkedCohort(root/'synthetic_all_1024_linked_tensors.npz',require_primary=True)
 assert full.n==m['n']
 assert sha(root/'synthetic_all_1024_linked_tensors.npz')==m['derived_tensor_sha256']
 for item in m['shards']:
  p=root/item['file'];assert p.stat().st_size==item['bytes'];assert sha(p)==item['sha256']
 for item in (m['shards'][0],m['shards'][-1]):
  p=root/item['file'];part=np.load(p,allow_pickle=False)
  start=int(p.stem.split('_')[-2]);n=len(part)
  mini=part[:,[4,20,36,52],8:40,8:40].astype('float32')/255
  assert np.array_equal(mini,full.raw['echo'][start:start+n,:,0])
 LinkedCohort(root/'synthetic_tiny_smoke_test.npz',require_primary=True)
 print('PASS:',len(m['shards']),'high-resolution video shards; aligned compact tensors;',full.n,'synthetic subjects')

if __name__=='__main__':main()
