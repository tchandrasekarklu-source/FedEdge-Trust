"""Create traceable SYNTHETIC-ONLY plots and paired H1 reports from executed demo predictions."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
from fededge_experiments.round3_audit import read_predictions,evaluate_pairs


def main():
 root=Path('artifacts_execution_demo/central_3arm_seed7')
 rows=[]
 for src,dst in [('full','gated'),('no_quality_time','masked'),('late','late')]:
  f=root/f'central_{src}_seed7'/'locked_test_predictions_RESTRICTED.csv'
  df=pd.read_csv(f,dtype={'subject_id':str})
  df['model']=dst
  rows.append(df[['subject_id','split','seed','model','y_true','y_prob']])
 frame=pd.concat(rows,ignore_index=True)
 frame.to_csv(root/'SYNTHETIC_ONLY_paired_test_predictions.csv',index=False)
 original=read_predictions(root/'SYNTHETIC_ONLY_paired_test_predictions.csv')
 for base in ('masked','late'):
  report=evaluate_pairs(original,'gated',base,reps=300,seed=20261029)
  report['evidence_class']='SYNTHETIC SOFTWARE FUNCTIONALITY TEST — NOT CLINICAL PERFORMANCE'
  (root/f'SYNTHETIC_ONLY_H1_gated_vs_{base}.json').write_text(json.dumps(report,indent=2))
 summary=[]
 for name in ('gated','masked','late'):
  z=original[original.model==name]
  yy=z.y_true.to_numpy();p=z.y_prob.to_numpy()
  summary.append({'model':name,'n':len(z),'positive':int(yy.sum()),'synthetic_auroc':roc_auc_score(yy,p),
                  'synthetic_auprc':average_precision_score(yy,p)})
 for method in ('roc','pr'):
  fig,ax=plt.subplots(figsize=(7,5))
  for name in ('gated','masked','late'):
   z=original[original.model==name];yy=z.y_true.to_numpy();p=z.y_prob.to_numpy()
   if method=='roc':x,y,_=roc_curve(yy,p);ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate')
   else:y,x,_=precision_recall_curve(yy,p);ax.set_xlabel('Recall');ax.set_ylabel('Precision')
   ax.plot(x,y,label=name)
  ax.legend();ax.grid(alpha=.25)
  ax.set_title('SYNTHETIC SOFTWARE SMOKE TEST — NOT CLINICAL RESULTS')
  fig.tight_layout();fig.savefig(root/f'SYNTHETIC_ONLY_{method.upper()}_curves.png',dpi=150);plt.close(fig)
 pd.DataFrame(summary).to_csv(root/'SYNTHETIC_ONLY_three_arm_metrics.csv',index=False)
 (root/'SYNTHETIC_ONLY_NOT_CLINICAL.txt').write_text(
  'ALL outputs in artifacts_execution_demo are derived from artificial ultrasound-like frames and '
  'synthetic ECG/EHR signals. These are executed software tests with synthetic labels, NOT actual '
  'patient-level cardiovascular performance. Do not transfer them into manuscript Tables 7–15.\n')
 print('Generated software-demo ROC/PR charts, paired H1 reports and three-arm metric CSV')

if __name__=='__main__':main()
