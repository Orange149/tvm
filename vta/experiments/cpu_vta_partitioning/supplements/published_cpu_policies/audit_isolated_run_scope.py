"""Audit CPU share scope using accepted isolated records only.

The output is an evidence table, not a replacement cost table. No pipeline
records, FPS, board connection, or runtime imports are accepted as inputs.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, write, sha256, verify_inventory
from cpu_policies.open_resources.costs import quantize_ns
from run_budget_reconstruction import reconstruct
from scheduler_gap_audit.curves import csv_write

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/isolated_run_scope_01'
INPUTS = {
    'resnet18': (
        ROOT / 'artifacts/frame_stream_top20_static_validation/isolated_costs.json',
        ROOT / 'artifacts/bundles/8a942b2047c031339168f9d2/plans/programs.json'),
    'yolov3_tiny416': (
        ROOT / 'artifacts/yolo_validation_01/ranking/inputs/isolated_costs.json',
        ROOT / 'artifacts/yolo_validation_01/ranking/inputs/programs.json'),
}


def logical_counters(affinity, row):
    """Old reports lack slots; singleton affinities can identify their order."""
    n = row['threads']
    if affinity['mode'] != 'serial' or len(affinity['stages']) != 1:
        raise ValueError('isolated serial affinity required')
    if not affinity.get('cpu_worker_pinning'):
        raise ValueError('fixed worker binding required')
    stage = affinity['stages'][0]
    mapping = row['worker_core_map']
    if len(mapping) != n or len(set(mapping)) != n:
        raise ValueError('thread mapping incomplete')
    entries = {x['tid']: x for x in stage['effective_affinities']}
    if len(entries) != len(stage['effective_affinities']):
        raise ValueError('duplicate thread ledger')
    slots = stage.get('worker_slots', [])
    if slots:
        if len(slots) != n or {x['slot'] for x in slots} != set(range(n)):
            raise ValueError('logical worker slots incomplete')
        tids = {x['slot']: x['tid'] for x in slots}
        method = 'recorded_logical_slots'
    else:
        tids = {}
        for slot, core in enumerate(mapping):
            matches = [tid for tid, x in entries.items() if x['cpus'] == [core]]
            if len(matches) != 1:
                raise ValueError('singleton affinity cannot identify logical slot')
            tids[slot] = matches[0]
        method = 'inferred_from_unique_singleton_affinities_and_requested_map'
    if len(set(tids.values())) != n or tids[0] != stage['caller_tid']:
        raise ValueError('caller or worker identity differs')
    values = []
    for slot in range(n):
        entry = entries[tids[slot]]
        if entry['cpus'] != [mapping[slot]]:
            raise ValueError('actual worker affinity differs')
        value = entry['cpu_time_delta_ms']
        quantize_ns(value)  # Reject missing, reversed, or nonfinite counters.
        values.append(value)
    extras = [x for tid, x in entries.items() if tid not in tids.values()]
    if any(x['cpu_time_delta_ms'] != 0 for x in extras):
        raise ValueError('extra active thread outside component group')
    return values, method


def inspect_scope(task, frames, affinity, launcher):
    if task.get('kind') != 'calibration' or len(task['stages']) != 1:
        raise ValueError('only isolated calibration inputs allowed')
    row = task['stages'][0]
    if row['device'] != 'cpu' or task['input_indices'] != [0]:
        raise ValueError('CPU input-zero calibration required')
    if (task['warmup'], task['scored'], task['tail']) != (2, 8, 0):
        raise ValueError('isolated window differs')
    if len(frames) != 10 or [r['frame_id'] for r in frames] != list(range(10)):
        raise ValueError('missing, duplicate, or reordered isolated frame')
    if [r['completion_index'] for r in frames] != list(range(10)):
        raise ValueError('completion identity differs')
    for frame in frames:
        if (frame['mode'] != 'serial' or frame['stage_count'] != 1
                or frame['input_index'] != 0
                or frame['stage_cpu_time_scope'] != 'exclusive_process_cpu_time'):
            raise ValueError('concurrent or incompatible run CPU scope')
        quantize_ns(frame['stage0_run_process_cpu_ms'])
    env_lines = [line for line in launcher.splitlines() if line.startswith('env ')]
    if len(env_lines) != 1:
        raise ValueError('unique captured native environment required')
    tokens = shlex.split(env_lines[0])
    for expected in ('TVM_THREAD_POOL_SPIN_COUNT=0', 'TVM_EXCLUDE_WORKER0=1',
                     'TVM_BIND_THREADS=1', '--serial', '--cpu-worker-pinning'):
        if expected not in tokens:
            raise ValueError('quiet fixed-worker environment not established: ' + expected)
    if task.get('environment', {}).get('spin_count', 0) != 0:
        raise ValueError('session requests spinning workers')
    whole, method = logical_counters(affinity, row)
    result = dict(whole_thread_CPU_ms=whole, thread_mapping_evidence=method,
                  measured_all_run_CPU_ns=sum(quantize_ns(r['stage0_run_process_cpu_ms'])
                                             for r in frames),
                  all_recorded_frames=10, scored_frames=8,
                  ledger_resolution_warning='coarse /proc counters; zero is unobservable work, not proof of no work')
    if row['threads'] == 1:
        result.update(status='single_thread_identity', estimated_weights=[1.0])
    elif any(value == 0 for value in whole[1:]):
        result.update(status='retain_original_unobservable_noncaller', estimated_weights=None)
    else:
        try:
            estimate = reconstruct(frames, whole[1:])
        except ValueError as error:
            result.update(status='retain_original_incompatible_remainder',
                          estimated_weights=None, reconstruction_error=str(error))
        else:
            result.update(status='reconstruction_available',
                          estimated_weights=estimate['weights'], reconstruction=estimate)
    return result


def receipt_location(record, index):
    if 'source' in record:
        source = record['source']
        path = Path(source['receipt'])
        expected = source['receipt_sha256']
    else:
        source = record['provenance']
        ref = source['source_receipt']
        expected = ref['sha256']
        relative = Path(ref['path'])
        matches = [p for p in index[relative.name]
                   if str(p).endswith(str(relative)) and sha256(p) == expected]
        if not matches:
            raise ValueError('original isolated receipt missing')
        path = sorted(matches)[0]
    if sha256(path) != expected:
        raise ValueError('original receipt hash differs')
    return path, expected


def raw_directory(path, receipt):
    if 'raw' in receipt:
        root = Path(receipt['raw'])
    else:
        root = path.parents[2]
    verify_inventory(root, receipt['files'])
    matches = [root / p for p in receipt['files'] if Path(p).name == 'result.jsonl']
    if len(matches) != 1:
        raise ValueError('isolated raw result not unique')
    raw = matches[0].parent
    required = ['result.jsonl', 'affinity.json', 'execute.sh', 'boot_id', 'exit_code']
    if any(not any(root / p == raw / name for p in receipt['files']) for name in required):
        raise ValueError('raw scope evidence missing from hashed inventory')
    if (raw / 'exit_code').read_text().strip() != '0':
        raise ValueError('failed native session')
    if (raw / 'boot_id').read_text().strip() != receipt['boot_id']:
        raise ValueError('raw boot differs from accepted receipt')
    return raw


def bundle_manifests():
    found = {}
    for path in sorted((ROOT / 'artifacts').rglob('manifest.json')):
        value = read(path)
        if 'bundle_id' in value and 'files' in value:
            found.setdefault(value['bundle_id'], (path, value))
    return found


def validate_source_program(record, task, programs, bundles):
    row = record['configuration']['stages'][0]
    actual = task['stages'][0]
    for key in ('program', 'device', 'begin', 'end', 'threads', 'mask', 'worker_core_map'):
        if row[key] != actual[key]:
            raise ValueError('isolated signature differs: ' + key)
    if 'source' in record:
        source = record['source']
        boot, bundle, tid = source['boot_id'], source['bundle_id'], source['task_id']
    else:
        source = record['provenance']
        boot, bundle, tid = (source['source_boot_id'], source['source_bundle_id'],
                             source['source_task_id'])
    if task['bundle_id'] != bundle or task['task_id'] != tid:
        raise ValueError('source task or bundle differs')
    manifest_path, manifest = bundles[bundle]
    for filename in ('graph.json', 'module.so', 'params.bin'):
        relative = 'programs/' + row['program'] + '/' + filename
        wanted = programs[row['program']]['files'][filename]['sha256']
        if manifest['files'][relative]['sha256'] != wanted:
            raise ValueError('source module identity differs from frozen cost inventory')
        if 'source' in record and source['program_files'][filename] != wanted:
            raise ValueError('source program hash differs')
    native = task['executor_sha256']
    if native not in {v['sha256'] for k, v in manifest['files'].items() if k.startswith('runner/')}:
        raise ValueError('captured executor absent from source manifest')
    return boot, manifest_path, {
        name: manifest['files']['runtime/' + name]['sha256']
        for name in ('libtvm_runtime.so', 'libvta.so')}


def run():
    if OUT.exists():
        raise FileExistsError('preserve the existing audit; validate or choose a new output identity')
    index = defaultdict(list)
    for p in (ROOT / 'runs').glob('*/accepted/*/*.json'):
        index[p.name].append(p)
    bundles = bundle_manifests()
    sources, rows = {}, []
    for network, (cost_path, program_path) in INPUTS.items():
        sources.update({str(p): sha256(p) for p in (cost_path, program_path)})
        records, programs = read(cost_path)['costs'], read(program_path)
        for key, record in records.items():
            row = record['configuration']['stages'][0]
            if row['device'] != 'cpu':
                continue
            path, expected = receipt_location(record, index)
            receipt = read(path)
            boot, mp, runtime = validate_source_program(record, receipt['task'], programs, bundles)
            if boot != receipt['boot_id']:
                raise ValueError('original boot identity differs')
            raw = raw_directory(path, receipt)
            frames = [json.loads(s) for s in (raw / 'result.jsonl').read_text().splitlines()]
            result = inspect_scope(receipt['task'], frames, read(raw / 'affinity.json'),
                                   (raw / 'execute.sh').read_text())
            # Reproduce the source scored run total without changing its value.
            scored_mean = sum(f['stage0_run_process_cpu_ms'] for f in frames[2:10]) / 8
            if abs(scored_mean - record['cost']['run_core_ms']) > 1e-8:
                raise ValueError('source run total differs from formal isolated cost')
            original = record.get('session_thread_weights')
            whole = result['whole_thread_CPU_ms']
            measured = [v / sum(whole) for v in whole] if sum(whole) > 0 else None
            if original is not None and measured is not None:
                if len(original) != len(measured) or max(abs(a-b) for a,b in zip(original, measured)) > 1e-10:
                    raise ValueError('formal session shares do not match their source ledger')
            if original is None and row['threads'] > 1:
                result.update(status='retain_original_explicit_uniform_fallback', estimated_weights=None)
            proposed = result['estimated_weights']
            result.update(network=network, component_key=key, program=row['program'],
                          threads=row['threads'], source_receipt=str(path), source_boot=boot,
                          source_task_id=receipt['task']['task_id'], runtime_hashes=runtime,
                          original_weights=original,
                          caller_share_change_pp=(100*(proposed[0]-original[0])
                              if proposed is not None and original is not None else None),
                          original_service_run_boundary_unchanged=True,
                          worker_quiet_outside_run_is_an_assumption_not_a_per_record_observation=True)
            rows.append(result)
            sources.update({str(path): expected, str(mp): sha256(mp)})
        print(json.dumps({'network': network, 'CPU_receipts_verified': sum(r['network']==network for r in rows)}), flush=True)
    summaries = []
    for network in INPUTS:
        chosen = [r for r in rows if r['network'] == network]
        multi = [r for r in chosen if r['threads'] > 1]
        available = [r for r in multi if r['status'] == 'reconstruction_available']
        changes = [abs(r['caller_share_change_pp']) for r in available]
        summaries.append(dict(network=network, CPU_records=len(chosen), multithread_records=len(multi),
                              statuses=dict(Counter(r['status'] for r in chosen)),
                              max_abs_caller_share_change_pp=max(changes, default=None),
                              median_abs_caller_share_change_pp=(sorted(changes)[len(changes)//2] if changes else None)))
    result = dict(sources=sources, rows=rows, summaries=summaries,
                  source_files={str(p):sha256(p) for p in (Path(__file__),ROOT/'run_budget_reconstruction.py')},
                  raw_receipt_and_inventory_hashes_verified=True,
                  source_program_identity_matches_frozen_inventory=True,
                  source_binary_files_not_rehashed_by_this_scope_audit=True,
                  new_board_sessions=0, new_DNN_inferences=0,
                  pipeline_results_read=False, formal_costs_modified=False,
                  correction_promoted=False, remaining_pipeline_slowdown_explained=False,
                  hypothesis='remove caller work outside run from isolated run share allocation',
                  limits=['coarse counters', 'initial empty callbacks', 'noncaller quiet outside run assumption',
                          'all recorded runs used rather than scored-only per-thread clocks'])
    OUT.mkdir(parents=True)
    write(OUT/'RESULTS.json', result)
    csv_write(OUT/'CPU_scopes.csv', rows)
    write(OUT/'MANIFEST.json', {'files':{str(p):sha256(p) for p in (OUT/'RESULTS.json',OUT/'CPU_scopes.csv')}})
    print(json.dumps({'summaries':summaries,'output':str(OUT)},ensure_ascii=False,indent=2))
    return result


if __name__ == '__main__':
    run()
