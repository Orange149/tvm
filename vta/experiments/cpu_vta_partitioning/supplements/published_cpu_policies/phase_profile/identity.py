"""Resolve the executor actually used, not the first runner in a manifest."""
from pathlib import Path
from cpu_vta_partitioning.common import read,sha256,write
from policy_branch_eval.analysis import cohort_specs


def execution_identity(bundle):
    bundle=Path(bundle);manifest=read(bundle/'manifest.json')
    plan=bundle/'plans/executors.json'
    if plan.exists():
        entry=read(plan)['policy'];path=entry['path'];expected=entry['sha256']
    else:
        expected=manifest['runner_sha256']
        matches=[n for n,v in manifest['files'].items() if n.startswith('runner/') and v['sha256']==expected]
        if len(matches)!=1:raise ValueError('ambiguous active executor in capsule')
        path=matches[0]
    if sha256(bundle/path)!=expected or manifest['files'][path]['sha256']!=expected:
        raise ValueError('actual executor differs from capsule seal')
    return dict(path=path,runner_sha256=expected,
        runtime_sha256=sha256(bundle/'runtime/libtvm_runtime.so'),source_manifest_sha256=sha256(bundle/'manifest.json'))


def correct_audit(output):
    output=Path(output);sources={}
    for spec in cohort_specs():sources[spec['id']]=execution_identity(spec['bundle'])
    previous=read(output/'FEATURE_FREEZE.json')['proofs']
    value=dict(reason='Earlier audit selected first runner from full component capsule; that capsule also carries an unused baseline runner',
        previous_reported={n:v['runner_sha256'] for n,v in previous.items()},
        actual_execution_identities=sources,original_numeric_features_changed=False,
        source_FEATURE_FREEZE_preserved=True,
        all_measured_cohorts_same_runner=len({v['runner_sha256'] for v in sources.values()})==1,
        all_measured_cohorts_same_runtime=len({v['runtime_sha256'] for v in sources.values()})==1)
    write(output/'EXECUTOR_IDENTITY_CORRECTION.json',value,replace=True)
    return value
