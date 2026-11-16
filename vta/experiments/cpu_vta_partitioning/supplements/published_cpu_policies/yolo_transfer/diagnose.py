"""One full-tensor qualification for a rejected component, never a cost input."""
from copy import deepcopy
from pathlib import Path
from types import FunctionType

from cpu_vta_partitioning.common import digest, inventory, locked, read, sha256, write


def run(bundle, source_task, run_dir, local_config, *, execute_board=False):
    import numpy as np
    from cpu_vta_partitioning.transport import SSH
    from . import profile
    from .references import output_arrays
    bundle, run_dir = Path(bundle).resolve(), Path(run_dir).resolve()
    profile.validate(bundle)
    original = read(source_task)
    if original != profile.make_task(bundle, original['component_key']):
        raise ValueError('rejected task identity differs')
    document = {k: deepcopy(v) for k, v in original.items() if k != 'task_id'}
    document.update(kind='qualify_component', warmup=0, scored=1, tail=0,
                    purpose='full_tensor_debug_of_rejected_output_not_a_cost',
                    rejected_task_sha256=sha256(source_task), controller_sha256=sha256(Path(__file__)))
    document['task_id'] = digest(document)[:24]
    def command(doc, bundle, remote_bundle, remote_job):
        if doc != document:
            raise ValueError('diagnostic task differs')
        argv, inputs = profile.native_command(original, bundle, remote_bundle, remote_job)
        argv[argv.index('--runs') + 1] = '1'
        argv[argv.index('--output-mode') + 1] = 'raw_all_stages'
        argv += ['--output-dump-dir', remote_job + '/outputs']
        return argv, inputs
    measure = FunctionType(SSH.measure.__code__, {**SSH.measure.__globals__, 'native_command': command},
                           SSH.measure.__name__, SSH.measure.__defaults__, SSH.measure.__closure__)
    ssh = type('YOLOMismatchSSH', (SSH,), {'measure': measure})(read(local_config)['board'], execute_board=execute_board)
    with locked(run_dir):
        boot = ssh.boot_id()
        marker = run_dir / 'run.json'
        identity = {'boot_id': boot, 'bundle_id': original['bundle_id'], 'diagnostic_task': document['task_id']}
        if marker.exists() and read(marker) != identity:
            raise ValueError('diagnostic boot or task changed')
        if not marker.exists():
            write(marker, identity)
        remote = ssh.deploy(bundle, run_dir)
        raw = ssh.measure(document, bundle, remote, run_dir, boot)
        program, arrays = output_arrays(raw, document, bundle)
        inputs = read(bundle / 'plans/inputs.json')['records'][0]
        records = []
        for node, actual in zip(program['output_node_ids'], arrays):
            expected = np.load(bundle / inputs['frontiers'][str(node)]['npy'], allow_pickle=False)
            difference = actual.astype('float64') - expected.astype('float64')
            np.save(run_dir / ('actual_' + str(node) + '.npy'), actual)
            records.append({'node': node, 'elements': actual.size,
                'wrong_elements': int(np.count_nonzero(actual != expected)),
                'max_abs_difference': float(np.max(np.abs(difference))),
                'mean_abs_difference': float(np.mean(np.abs(difference)))})
        result = {'purpose': 'output_correctness_diagnosis_not_performance', 'outputs': records,
                  'task': document, 'raw': str(raw), 'files': inventory(raw)}
        write(run_dir / 'DIAGNOSIS.json', result, replace=True)
        ssh.acknowledge(document, run_dir, boot)
        return records
