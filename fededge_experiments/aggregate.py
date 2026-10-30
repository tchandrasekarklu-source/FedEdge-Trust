"""Matched-seed summary from actual locked-test model runs (not synthetic estimates).

Pairing is across the exact same predeclared seeds; t intervals reflect
*training-seed variation* and do not substitute for patient bootstrap CIs.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t


def aggregate(df,reference='full',metric='auroc'):
    required={'seed','variant','method',metric}
    if not required.issubset(df.columns):raise ValueError('Missing columns')
    if df.duplicated(['seed','variant','method']).any():raise ValueError('Duplicate seed/model/method run')
    if df['method'].nunique()!=1: raise ValueError('Do not pool central and FL methods')
    ref=df[df.variant==reference].set_index('seed')
    if len(ref)<2:raise ValueError('At least two repeated runs required; five is recommended')
    output=[]
    for variant,g in df.groupby('variant'):
        sample=g.set_index('seed')
        if set(sample.index)!=set(ref.index):raise ValueError('All comparisons must use IDENTICAL seeds')
        paired=(ref.loc[sorted(ref.index),metric]-sample.loc[sorted(ref.index),metric]).to_numpy(float)
        actual=sample[metric].to_numpy(float)
        if not np.isfinite(actual).all():raise ValueError('Metrics undefined or nonfinite')
        def interval(arr):
            mean=float(np.mean(arr));sd=float(np.std(arr,ddof=1));half=float(t.ppf(.975,len(arr)-1)*sd/np.sqrt(len(arr)))
            return [mean-half,mean+half]
        output.append({'variant':variant,'method':str(g.method.iloc[0]),'seed_count':len(actual),
                       'mean_metric':float(actual.mean()),'sd_across_seeds':float(actual.std(ddof=1)),
                       'seed_mean_95_t_interval':interval(actual),
                       'reference_minus_variant_mean':float(paired.mean()),
                       'paired_seed_difference_95_t_interval':interval(paired),
                       'metric':metric,
                       'caution':'Seed uncertainty only; patient-level held-out bootstrap must also be reported'})
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-metrics',required=True);p.add_argument('--out',required=True)
    p.add_argument('--reference',default='full');p.add_argument('--metric',default='auroc')
    a=p.parse_args()
    Path(a.out).write_text(json.dumps(aggregate(pd.read_csv(a.seed_metrics),a.reference,a.metric),indent=2))
