"""Same native execution semantics, with explicit graph size and audited window."""
from copy import deepcopy
from pathlib import Path
from types import FunctionType

from cpu_vta_partitioning.common import digest, read
from cpu_vta_partitioning.transport import SSH
from cpu_policies.open_resources.domain import validate_candidate
from common_top20 import runtime as original


def task(bundle, cfg, phase='short'):
    bundle = Path(bundle)
    manifest = read(bundle / 'manifest.json')
    plan = read(bundle / 'plans/protocol.json')
    if phase not in ('qualify', 'short'):
        raise ValueError('no unrequested confirmation or extra measurement phase')
    validate_candidate(cfg, read(bundle / 'plans/programs.json'), unit_count=manifest['unit_count'],
                       max_vta_islands=plan['max_vta_islands'])
    window = plan['qualification' if phase == 'qualify' else 'performance']
    result = dict(kind='qualify_pipeline' if phase == 'qualify' else 'performance', phase=phase,
        configuration=deepcopy(cfg), stages=deepcopy(cfg['stages']), environment=plan['environment'],
        **window, input_indices=[0], network=manifest['network'], canonical_sha256=manifest['canonical_sha256'],
        executor_sha256=manifest['runner_sha256'], runtime_sha256=manifest['runtime_sha256'],
        bundle_id=manifest['bundle_id'], experiment=plan['experiment'], detailed=False,
        startup_affinity_probe=True)
    result['task_id'] = digest(result)[:24]
    return result


def command(document, bundle, remote, job):
    # Private globals adapt only task validation; source module and binary remain unchanged.
    fn = FunctionType(original.command.__code__, {**original.command.__globals__, 'task': task})
    return fn(document, bundle, remote, job)


def summary(raw, document, bundle):
    fn = FunctionType(original.summary.__code__, {**original.summary.__globals__, 'task': task})
    result = fn(raw, document, bundle)
    if document['phase'] == 'short':
        total = sum(document[k] for k in ('warmup', 'scored', 'tail'))
        for row in result['stage_completion_counts_at_window']:
            if row['counts'] is None:
                raise ValueError('missing stage timestamps: cannot audit the measured window')
            if row['counts'][1] >= total:
                raise ValueError('stage exhausted all tasks inside the scored window; no automatic tail extension')
        result['early_stage_exhaustion'] = False
        first, last = document['warmup'] + 1, document['warmup'] + document['scored']
        result['inventory_at_window'] = [dict(stage=r['stage'], first=r['counts'][0] - first,
                                             last=r['counts'][1] - last)
                                         for r in result['stage_completion_counts_at_window']]
    return result


def transport(config, *, execute_board=False):
    measure = FunctionType(SSH.measure.__code__, {**SSH.measure.__globals__, 'native_command': command})
    return type('ResNet50PipelineSSH', (SSH,), {'measure': measure})(config, execute_board=execute_board)
