import json
from pathlib import Path
import numpy as np
import pandas as pd
from fededge_experiments.data import LinkedCohort
from fededge_experiments.round3_audit import perf,read_predictions

ROOT=Path(__file__).resolve().parents[1]

def test_fixture_generated_from_synthetic_only():
 m=json.loads((ROOT/'synthetic_fixtures'/'synthetic_generation_manifest.json').read_text())
 assert m['is_synthetic'] is True and m['is_clinical_evidence'] is False
 assert m['n']==1024 and len(m['shards'])==16
 assert sum(m['split_counts'].values())==m['n']

def test_training_fixture_has_strict_splits():
 cohort=LinkedCohort(ROOT/'synthetic_fixtures'/'synthetic_tiny_smoke_test.npz')
 for name,idx in cohort.client_indices().items():
  assert len(idx)>=2 and np.all(cohort.raw['split'][idx]=='train')
 assert len(cohort.indices('test'))>=10

def test_paired_test_predictions_and_accuracy_integer():
 p=ROOT/'artifacts_execution_demo'/'central_3arm_seed7'/'SYNTHETIC_ONLY_paired_test_predictions.csv'
 df=read_predictions(p)
 assert set(df.model)=={'gated','masked','late'}
 base=df[df.model=='gated'].sort_values('subject_id')
 n=len(base)
 for model in ('late','masked'):
  other=df[df.model==model].sort_values('subject_id')
  assert np.array_equal(other.subject_id.to_numpy(),base.subject_id.to_numpy())
  assert np.array_equal(other.y_true.to_numpy(),base.y_true.to_numpy())
  m=perf(other.y_true.to_numpy(),other.y_prob.to_numpy())
  assert m['accuracy']*n==m['tp']+m['tn']

def test_private_demo_accountant_discloses_limitations():
 r=json.loads((ROOT/'artifacts_execution_demo'/'dp_fedavg_seed7'/'fedavg_full_seed7'/'results.json').read_text())
 p=r['privacy']; assert p['epsilon_parallel_composition_upper_bound']>0
 assert p['no_secure_aggregation'] is True
 assert p['other_private_processing_not_accounted_for'] is True
