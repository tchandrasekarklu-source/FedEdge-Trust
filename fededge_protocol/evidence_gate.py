"""Conservative claim release checks. No clinical results are shipped here."""
from pathlib import Path


def audit_evidence(record, base_dir):
    """Returns a list of blocking issues; an empty list is NOT regulatory approval.

    Paths must be local evidence files, not declared numeric performance targets.
    Used to ensure that no target gets included as a measured claim accidentally.
    """
    root=Path(base_dir).resolve()
    failures=[]
    if record.get('claim_type') not in ('performance','fairness','privacy','hardware'):
        failures.append('unsupported claim type')
    if record.get('evidence_status')!='measured':
        failures.append('evidence is not measured')
    if not isinstance(record.get('n_subjects'),int) or record.get('n_subjects',0)<=0:
        failures.append('verified subject count absent')
    mandatory=['cohort_manifest','split_manifest','training_config','seed_runs','predictions','evaluation_script']
    for key in mandatory:
        candidate=record.get(key)
        if not candidate:
            failures.append(f'missing {key}')
            continue
        path=(root / candidate).resolve()
        if root not in path.parents or not path.is_file() or path.stat().st_size == 0:
            failures.append(f'{key} is missing, empty, or outside evidence directory')
    if record.get('claim_type')=='hardware':
        if not record.get('device_spec') or not record.get('measurement_protocol'):
            failures.append('device specification and measurement method are required')
    if record.get('claim_type')=='privacy':
        if not record.get('accountant_trace') or not record.get('delta'):
            failures.append('DP accountant trace and delta are required')
    if record.get('claim_type')=='fairness':
        if not record.get('group_counts'):
            failures.append('subgroup denominators are required')
    return failures
