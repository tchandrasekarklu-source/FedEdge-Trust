"""Render ONLY from user-supplied observed locked-test predictions.
Input columns: subject_id,y_true,y_prob,split (all split=test). Figures are descriptive
and not released as clinical evidence without accompanying cohort and model audit.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import average_precision_score, confusion_matrix


def observed_figures(frame,outdir,threshold=0.5):
    required={'subject_id','y_true','y_prob','split'}
    if not required.issubset(frame.columns):raise ValueError(f'Missing {sorted(required-set(frame.columns))}')
    if frame.subject_id.duplicated().any():raise ValueError('One patient once only')
    if set(frame.split.unique())!={'test'}:raise ValueError('Only locked TEST data permitted')
    if len(frame)<2 or set(frame.y_true.unique())!={0,1}:raise ValueError('Need both outcome classes')
    if not np.isfinite(frame.y_prob).all() or not frame.y_prob.between(0,1).all():
        raise ValueError('Valid [0,1] probabilities required')
    if not 0<=threshold<=1:raise ValueError('Invalid threshold')
    y=frame.y_true.to_numpy(dtype=int);p=frame.y_prob.to_numpy(dtype=float)
    if not np.isin(y,[0,1]).all():raise ValueError('Invalid labels')
    dest=Path(outdir);dest.mkdir(parents=True,exist_ok=True)
    fpr,tpr,_=roc_curve(y,p)
    fig,ax=plt.subplots(figsize=(5.7,4.4));ax.plot(fpr,tpr,label=f'AUROC = {roc_auc_score(y,p):.3f}');ax.plot([0,1],[0,1],'--',alpha=.6)
    ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title='Observed locked-test ROC');ax.legend();fig.tight_layout();fig.savefig(dest/'observed_roc.png',dpi=220);plt.close(fig)
    prec,rec,_=precision_recall_curve(y,p)
    fig,ax=plt.subplots(figsize=(5.7,4.4));ax.plot(rec,prec,label=f'Average precision = {average_precision_score(y,p):.3f}')
    ax.axhline(y.mean(),ls='--',alpha=.5,label='Prevalence');ax.set(xlabel='Recall',ylabel='Precision',title='Observed locked-test precision-recall');ax.legend();fig.tight_layout();fig.savefig(dest/'observed_pr.png',dpi=220);plt.close(fig)
    bins=np.linspace(0,1,11);idx=np.minimum(np.digitize(p,bins[1:-1]),9)
    mean_p=[];obs=[];n=[]
    for k in range(10):
        m=idx==k
        if m.any():mean_p.append(float(p[m].mean()));obs.append(float(y[m].mean()));n.append(int(m.sum()))
    fig,ax=plt.subplots(figsize=(5.7,4.4));ax.plot([0,1],[0,1],'--',alpha=.5,label='Perfect calibration');ax.plot(mean_p,obs,'o-',label='Observed bins')
    ax.set(xlabel='Predicted event probability',ylabel='Observed event frequency',title='Locked-test reliability (occupied bins)');ax.legend();fig.tight_layout();fig.savefig(dest/'observed_reliability.png',dpi=220);plt.close(fig)
    cm=confusion_matrix(y,(p>=threshold).astype(int),labels=[0,1])
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    for i in range(2):
        for j in range(2):ax.text(j,i,str(int(cm[i,j])),ha='center',va='center',color='black')
    ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xticklabels(['No event','Event']);ax.set_yticklabels(['No event','Event']);ax.set(xlabel='Predicted',ylabel='Observed',title=f'Locked-test confusion matrix (threshold={threshold:g})');fig.tight_layout();fig.savefig(dest/'observed_confusion_matrix.png',dpi=220);plt.close(fig)
    manifest={'n_patients':len(frame),'observed_events':int(y.sum()),'threshold':threshold,'confusion_matrix':cm.tolist(),
              'note':'These figures require real locked-test inputs for empirical publication.'}
    (dest/'figure_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',required=True);ap.add_argument('--outdir',required=True)
    ap.add_argument('--threshold',type=float,default=.5)
    a=ap.parse_args(); observed_figures(pd.read_csv(a.predictions),a.outdir,a.threshold)

if __name__=='__main__':main()
