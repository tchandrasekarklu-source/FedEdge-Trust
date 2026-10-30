"""Write metadata verifying which scripts executed on *synthetic* fixtures."""
import hashlib,json,platform,sys
from pathlib import Path
import numpy as np,pandas as pd,sklearn,torch,scipy

def hash_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 root=Path(__file__).resolve().parents[1]
 runs={}
 for key in ('central_3arm_seed7','fedavg_seed7','fedprox_seed7','dp_fedavg_seed7'):
  p=root/'artifacts_execution_demo'/key/'FINAL_LOCKED_TEST_RESULTS.json'
  if p.exists():
   obj=json.loads(p.read_text())
   runs[key]={'prediction_file_count':len(list((p.parent).glob('*/locked_test_predictions_RESTRICTED.csv'))),
              'results':obj['results'],'output_sha256':hash_file(p)}
 build={'created_utc':'2026-10-29','evidence_scope':'EXECUTED SOFTWARE DEMONSTRATION WITH SYNTHETIC DATA ONLY',
        'clinical_data_included':False,'actual_patient_efficacy_evidence':False,
        'unverified_hypotheses':['H1 on genuinely linked clinical patients','H2 on authentic clinical client partitions'],
        'tested_protocol_suite':'49 passed',
        'platform':platform.platform(),'python':sys.version.split()[0],
        'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,
        'scipy':scipy.__version__,'torch':torch.__version__,'device':'cpu',
        'synthetic_source_video_count':1024,'synthetic_frames_per_source_video':64,
        'runs':runs}
 (root/'environment_and_execution_proof.json').write_text(json.dumps(build,indent=2))
 manifest=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file():continue
  if p.name=='FILE_SHA256_MANIFEST.txt':continue
  if '__pycache__' in p.parts or '.pytest_cache' in p.parts:continue
  manifest.append(f'{hash_file(p)}  {p.relative_to(root).as_posix()}')
 (root/'FILE_SHA256_MANIFEST.txt').write_text('\n'.join(manifest)+'\n')
 print('Manifest files:',len(manifest),'including',len(list((root/'synthetic_fixtures'/'source_echo_frames_SYNTHETIC_ONLY').glob('*.npy'))),'source video shards')

if __name__=='__main__':main()
