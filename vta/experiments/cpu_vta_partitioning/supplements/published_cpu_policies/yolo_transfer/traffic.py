"""Actual VTA DMA payloads for the new graph; no ResNet stage-cost scaling."""
from copy import deepcopy
from pathlib import Path
from types import FunctionType

from cpu_vta_partitioning.common import (digest, inventory, locked, read, sha256,
                                        sync_tree, verify_inventory, write)


def task(bundle, key):
    bundle = Path(bundle)
    configuration = read(bundle / 'plans/component_configurations.json')[key]
    if configuration['context_id'] != 'shared4' or configuration['stages'][0]['device'] != 'vta':
        raise ValueError('traffic uses the actual shared-host VTA component only')
    doc = {'kind': 'traffic', 'configuration': configuration,
           'stages': deepcopy(configuration['stages']), 'component_key': key,
           'environment': {'poll_sleep_ns': 1000, 'post_start_sleep_ns': 1000},
           'warmup': 1, 'scored': 1, 'tail': 0, 'input_indices': [0],
           'native_warmup_outside_results': True, 'controller_sha256': sha256(Path(__file__)),
           'bundle_id': read(bundle / 'manifest.json')['bundle_id']}
    doc['task_id'] = digest(doc)[:24]
    return doc


def native_command(document, bundle, remote_bundle, remote_job):
    from cpu_vta_partitioning.measurement.tasks import native_command as base
    if document != task(bundle, document['component_key']):
        raise ValueError('traffic task differs')
    argv, inputs = base(document, bundle, remote_bundle, remote_job)
    loader = argv.index(remote_bundle + '/runtime/ld-linux-aarch64.so.1')
    argv[loader:loader] = ['TVM_BIND_THREADS=1', 'TVM_EXCLUDE_WORKER0=1']
    argv[argv.index('--runs') + 1] = '1'
    argv[argv.index('--warmup-runs') + 1] = '1'
    argv[argv.index('--output-mode') + 1] = 'raw'
    argv += ['--serial', '--allow-cpu-affinity-superset', '--cpu-worker-pinning',
             '--control-cpu-affinity', '0,1,2,3', '--vta-runtime-profile-dir', remote_job + '/traffic_profile']
    return argv, inputs


def summary(raw, document, bundle):
    from cpu_vta_partitioning.measurement import affinity, frames, input_sequence
    raw, bundle = Path(raw), Path(bundle)
    rows = frames(raw / 'result.jsonl', 1)
    input_sequence(rows, {**document, 'warmup': 0})
    affinity(read(raw / 'affinity.json'), document['stages'])
    program = read(bundle / 'plans/programs.json')[document['stages'][0]['program']]
    expected = read(bundle / 'plans/expected_outputs.json')
    if len(rows[0].get('raw_outputs', [])) != len(program['output_node_ids']):
        raise ValueError('traffic output frontier incomplete')
    for index, (node, row) in enumerate(zip(program['output_node_ids'], rows[0]['raw_outputs'])):
        if row.get('index') != index or any(row.get(k) != expected[str(node)][k]
                for k in ('shape', 'dtype', 'nbytes', 'fnv1a64')):
            raise ValueError('traffic output does not match independent ARM reference')
    measured = read(raw / 'traffic_profile/single_run_status.json')
    total = read(raw / 'traffic_profile/benchmark_totals_status.json')
    keys = ('synchronize_load_bytes', 'synchronize_store_bytes', 'synchronize_calls')
    if any(type(measured.get(k)) is not int or measured[k] <= 0 or total.get(k) != measured[k] for k in keys):
        raise ValueError('traffic counters missing or not exactly one post-warmup run')
    return {'load_bytes': measured[keys[0]], 'store_bytes': measured[keys[1]],
            'synchronize_calls': measured[keys[2]], 'count': 1, 'hooks_present': True,
            'traffic_scope': 'VTA requested DMA payload, not total AXI traffic', 'correct': True}


def run(bundle, run_dir, local_config, *, execute_board=False, limit=None):
    from cpu_vta_partitioning.transport import SSH
    from .profile import validate
    bundle, run_dir = Path(bundle).resolve(), Path(run_dir).resolve()
    manifest = validate(bundle)
    measure = FunctionType(SSH.measure.__code__, {**SSH.measure.__globals__, 'native_command': native_command},
                           SSH.measure.__name__, SSH.measure.__defaults__, SSH.measure.__closure__)
    ssh = type('YOLOTrafficSSH', (SSH,), {'measure': measure})(read(local_config)['board'], execute_board=execute_board)
    with locked(run_dir):
        boot = ssh.boot_id()
        identity = {'boot_id': boot, 'bundle_id': manifest['bundle_id'], 'controller_sha256': sha256(Path(__file__))}
        marker = run_dir / 'run.json'
        if marker.exists() and read(marker) != identity:
            raise ValueError('traffic boot, capsule, or controller differs')
        if not marker.exists():
            write(marker, identity)
            (run_dir / 'frozen_controller.py').write_bytes(Path(__file__).read_bytes())
        remote = ssh.deploy(bundle, run_dir)
        ssh.wait_idle(boot)
        frequencies = ssh.command('for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do cat "$f"; done').split()
        if frequencies != ['1066666'] * 4:
            raise ValueError('CPU frequency changed; no automatic retuning')
        configurations = read(bundle / 'plans/component_configurations.json')
        keys = [key for key, value in configurations.items()
                if value['stages'][0]['device'] == 'vta' and value['context_id'] == 'shared4']
        keys.sort(key=lambda key: configurations[key]['stages'][0]['program'])
        records = {}
        new = 0
        for key in keys:
            doc = task(bundle, key)
            receipt = run_dir / 'accepted' / boot / (doc['task_id'] + '.json')
            if receipt.exists():
                item = read(receipt)
                if item['task'] != doc or item['boot_id'] != boot:
                    raise ValueError('traffic receipt identity differs')
                verify_inventory(item['raw'], item['files'])
                result = summary(item['raw'], doc, bundle)
                if result != item['summary']:
                    raise ValueError('traffic replay differs')
            else:
                if limit is not None and new >= limit:
                    continue
                raw = ssh.measure(doc, bundle, remote, run_dir, boot)
                result = summary(raw, doc, bundle)
                write(receipt, {'task': doc, 'boot_id': boot, 'raw': str(raw), 'files': inventory(raw), 'summary': result})
                ssh.acknowledge(doc, run_dir, boot)
                new += 1
            name = doc['stages'][0]['program']
            records[name] = {**result, 'source': {'boot_id': boot, 'task_id': doc['task_id'], 'receipt_sha256': sha256(receipt)}}
            print('traffic accepted', len(records), '/', len(keys), name, flush=True)
        result = {'status': 'COMPLETE' if len(records) == len(keys) else 'PARTIAL',
                  'accepted': len(records), 'required': len(keys), 'boot_id': boot,
                  'pipeline_FPS_used': False, 'records': records}
        write(run_dir / 'traffic.json', result, replace=True)
        sync_tree(run_dir)
        return {k: v for k, v in result.items() if k != 'records'}
