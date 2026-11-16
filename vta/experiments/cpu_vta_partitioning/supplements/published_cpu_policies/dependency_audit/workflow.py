"""Isolation audit -> projection -> replay freeze -> diagnostic analysis."""
from pathlib import Path
import shutil

from cpu_vta_partitioning.common import read, sha256, verify_inventory, write
from .common import ROOT, OUTPUT, protocol, sources, freeze, require_stage
from .evidence import audit_isolation
from .trace import load_trace
from .model import replay as model_replay


def audit(output=OUTPUT):
    value = audit_isolation()
    value['controller_sources'] = sources()
    freeze(Path(output)/'AUDIT.json', value)
    return dict(status='ISOLATED_SOURCES_VERIFIED', isolated_receipts=len(value['receipts']),
                diagnostic_performance_read=False, new_native_sessions=0)


def extract(output=OUTPUT):
    output = Path(output)
    if not (output/'AUDIT.json').exists():
        audit(output)
    source = read(output/'AUDIT.json')
    if source['controller_sources'] != sources():
        raise ValueError('extractor differs from audited source')
    parsed, model_cases = {}, {}
    for name in ('overlap4', 'split3'):
        entry = source['cases'][name+'_observed']
        raw = Path(entry['raw'])
        ref = next(r for r in source['receipts'] if r['task_id'] == entry['task_id'])
        verify_inventory(raw, ref['raw_files'])
        trace = load_trace(raw, source['module_sha256'])
        parsed[name] = trace
        if not trace['frame_mapping']:
            continue
        projected = []
        for frame in trace['frames']:
            projected.append(dict(frame_id=frame['frame_id'],
                phase='warmup' if frame['frame_id'] < 2 else 'scored',
                worker_cpu_ns=[k['worker_cpu_ns'] for k in frame['launches']],
                interlaunch_caller_cpu_ns=frame['interlaunch_caller_cpu_ns']))
        model_cases[name] = dict(core_map=trace['core_map'], frames=projected)
    freeze(output/'EXTRACTION.json', dict(cases=parsed,
        no_exact_cross_clock_anchor=True, no_unknown_costs_filled_with_zero=True))
    freeze(output/'MODEL_INPUT.json', dict(schema=1,
        source_role='isolated_original_wait_callbacks', cases=model_cases))
    freeze(output/'INPUT_FREEZE.json', dict(protocol=protocol(), sources=sources(),
        files={name: sha256(output/name) for name in ('AUDIT.json','EXTRACTION.json','MODEL_INPUT.json')}))
    for name, expected in sources().items():
        target = output/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != expected:
            raise ValueError('offline source snapshot changed')
        if not target.exists():
            shutil.copy2(ROOT/name, target)
    return dict(status='ISOLATED_INPUTS_FROZEN', frame_mapped_cases=sorted(model_cases),
                diagnostic_performance_read=False)


def dot(nodes):
    lines = ['digraph callbacks {', 'rankdir=LR;']
    for n in nodes:
        if n['kind'] == 'callback':
            label = f"k{n['kernel']} worker{n['worker']} core{n['core']}\\n{n['duration_ns']} ns"
            shape = 'box'
        else:
            label, shape = n['id'], 'diamond' if n['kind'] == 'join' else 'ellipse'
        lines.append(f'"{n["id"]}" [label="{label}",shape={shape}];')
        lines += [f'"{p}" -> "{n["id"]}";' for p in n['parents']]
    return '\n'.join(lines+['}'])+'\n'


def replay(output=OUTPUT):
    output = Path(output)
    require_stage(output, 'input')
    data = read(output/'MODEL_INPUT.json')
    value = model_replay(data) if data['cases'] else dict(cases={},
        status='INSUFFICIENT_FRAME_MAPPING', full_pipeline_FPS=None)
    freeze(output/'REPLAY.json', value)
    files = {'REPLAY.json': sha256(output/'REPLAY.json')}
    for name, case in value['cases'].items():
        frame = next(f for f in case['frames'] if f['phase'] == 'scored')
        for suffix, key in (('callbacks','timeline'),('with_observed_gaps','timeline_with_observed_gaps')):
            path = output/'graphs'/(name+'_'+suffix+'.dot')
            body = dot(frame[key])
            if path.exists() and path.read_text() != body:
                raise ValueError('frozen dependency graph changed')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
            files[str(path.relative_to(output))] = sha256(path)
    freeze(output/'REPLAY_FREEZE.json', dict(input_freeze_sha256=sha256(output/'INPUT_FREEZE.json'),
        files=files, diagnostic_performance_read=False,
        scope='single-stage work replay, not a shared-pipeline performance model'))
    return dict(status='REPLAY_FROZEN', cases=sorted(value['cases']), full_pipeline_FPS=None)


def validate(output=OUTPUT):
    output = Path(output)
    require_stage(output, 'replay')
    source = read(output/'AUDIT.json')
    for ref in source['receipts']:
        if sha256(ref['path']) != ref['sha256']:
            raise ValueError('isolated source receipt changed')
        verify_inventory(ref['raw'], ref['raw_files'])
    data = read(output/'MODEL_INPUT.json')
    if data['cases'] and model_replay(data) != read(output/'REPLAY.json'):
        raise ValueError('pure replay no longer reproduces its frozen result')
    for name,expected in sources().items():
        if sha256(output/'sources'/name)!=expected:
            raise ValueError('offline source snapshot changed')
    if (output/'analysis/RESULTS.json').exists():
        for ref in read(output/'analysis/RESULTS.json')['diagnostic_receipts']:
            if sha256(ref['path'])!=ref['sha256']:
                raise ValueError('diagnostic receipt changed')
            verify_inventory(ref['raw'],ref['raw_files'])
    # Existing archived deployment inventories only; never invoke board tools.
    old = {}
    for name in ('yolo_wait_diagnosis_01/capsule', 'yolo_affinity_check_01/capsule',
                 'yolo_validation_01/pipeline_bundle', 'bundles/e542003fc5a64fb17ee1e751'):
        bundle = ROOT/'artifacts'/name
        manifest = read(bundle/'manifest.json')
        verify_inventory(bundle, manifest['files'])
        old[name] = manifest['bundle_id']
    return dict(validated=True, isolated_receipts_revalidated=len(source['receipts']),
        old_batches=old, replay_deterministic=True, board_access=False)


def run(output=OUTPUT):
    audit(output)
    extract(output)
    replay(output)
    from .analysis import analyze
    result = analyze(output)
    result['validation'] = validate(output)
    return result
