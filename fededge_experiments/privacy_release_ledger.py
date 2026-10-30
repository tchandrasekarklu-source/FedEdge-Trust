"""Conservative longitudinal composition for multiple released DP training runs.

If same patient participates in several separately published checkpoints,
privacy costs add; selecting the smallest epsilon per run is NOT a valid
account of their combined release. Requires shared certified patient IDs.
"""
import argparse,json
from pathlib import Path


def compose_dp_releases(records):
    """records list: client_id, epsilon, delta, release_id, subject_disjoint_across_clients."""
    if not records:raise ValueError('No documented private model releases')
    ledger={}; ids=set()
    for r in records:
        for key in ('client_id','release_id','epsilon','delta','subject_disjoint_across_clients'):
            if key not in r:raise ValueError('Missing '+key)
        if not r['subject_disjoint_across_clients']:
            raise ValueError('Client partition overlap not certified; cannot use parallel composition')
        key=(str(r['release_id']),str(r['client_id']))
        if key in ids:raise ValueError('Duplicate client/release')
        ids.add(key)
        eps=float(r['epsilon']);delta=float(r['delta'])
        if eps<0 or not (0<delta<1):raise ValueError('Invalid privacy parameter')
        c=str(r['client_id'])
        totals=ledger.setdefault(c,{'epsilon_upper_bound':0.,'delta_upper_bound':0.,'model_releases':0})
        totals['epsilon_upper_bound']+=eps
        totals['delta_upper_bound']+=delta
        totals['model_releases']+=1
    if any(x['delta_upper_bound']>=1 for x in ledger.values()):
        raise ValueError('Cumulative delta reaches or exceeds 1')
    return {'per_client_composed_privacy':ledger,
            'parallel_client_epsilon_upper_bound':max(x['epsilon_upper_bound'] for x in ledger.values()),
            'parallel_client_delta_upper_bound':max(x['delta_upper_bound'] for x in ledger.values()),
            'composition':'basic sequential across released models, parallel max across certified disjoint clients',
            'not_included':'unaccounted private model selection, calibration, subgroup monitoring, external disclosures'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--releases-json',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    payload=json.loads(Path(a.releases_json).read_text())
    Path(a.out).write_text(json.dumps(compose_dp_releases(payload),indent=2))
