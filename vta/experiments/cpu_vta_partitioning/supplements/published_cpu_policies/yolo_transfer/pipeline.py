"""Frozen Top-20 union validation using the unchanged pinned native executor."""
from copy import deepcopy
import math
from pathlib import Path
import random
import shlex
import shutil
import statistics
import subprocess
import time
from types import FunctionType

from cpu_vta_partitioning.common import (digest, inventory, locked, now, read, sha256,
                                        sync_tree, verify_inventory, write)
from cpu_vta_partitioning.prepare.build import copy_checked

METHODS = ('uniform_strict20', 'weighted_strict20', 'weighted_behavior20')
SEED = 20261108


def validate_lists(rankings, pool):
    from cpu_policies.open_resources.domain import configuration, digest as config_digest
    lists = rankings['lists']
    if set(lists) != set(METHODS):
        raise ValueError('unexpected comparison methods')
    union = set()
    for name in METHODS:
        rows = lists[name]
        if len(rows) != 20 or len({r['id'] for r in rows}) != 20:
            raise ValueError('each method must have exactly twenty distinct candidates')
        for row in rows:
            if row['id'] not in pool or configuration(row) != configuration(pool[row['id']]):
                raise ValueError('list and union execution identities differ')
            if row['id'] != 'cfg_' + config_digest(configuration(row))[:24]:
                raise ValueError('candidate execution identity differs')
            if type(row['score_ns']) is not int or row['score_ns'] < 0:
                raise ValueError('invalid frozen prediction')
            union.add(row['id'])
    if union != set(pool) or len(pool) > 60:
        raise ValueError('union differs from the three frozen Top-20 lists')
    return lists


def prepare(profile_bundle, ranked, output):
    import cpu_vta_partitioning
    from cpu_policies.open_resources.domain import validate_candidate
    from .profile import validate as validate_profile
    from .ranking import verify_artifacts
    profile_bundle, ranked, output = [Path(p).resolve() for p in (profile_bundle, ranked, output)]
    validate_profile(profile_bundle)
    proof = verify_artifacts(ranked)
    ranking_inputs = read(ranked / 'INPUTS.json')
    if (ranking_inputs['profile_capsule_manifest.json']['sha256'] != sha256(profile_bundle / 'manifest.json')
            or ranking_inputs['programs.json']['sha256'] != sha256(profile_bundle / 'plans/programs.json')):
        raise ValueError('ranking and deployed program catalog differ')
    if output.exists():
        raise FileExistsError('pipeline capsules are immutable')
    freeze = read(ranked / 'FREEZE.json')
    if freeze['rankings_sha256'] != sha256(ranked / 'rankings.json') or freeze['pool_sha256'] != sha256(ranked / 'pool.json'):
        raise ValueError('the premeasurement candidate freeze changed')
    rankings, pool = read(ranked / 'rankings.json'), read(ranked / 'pool.json')
    validate_lists(rankings, pool)
    if not rankings['original_verification']['passed'] or not rankings['enhanced_verification']['passed']:
        raise ValueError('independent full-domain enumeration is required')
    full = read(profile_bundle / 'plans/programs.json')
    for cfg in pool.values():
        validate_candidate(cfg, full, unit_count=14, max_vta_islands=3)
    started = time.monotonic()
    with locked(output):
        programs = {s['program']: full[s['program']] for cfg in pool.values() for s in cfg['stages']}
        parent = read(profile_bundle / 'manifest.json')
        for name, info in parent['files'].items():
            if name.startswith(('runtime/', 'runner/', 'firmware/', 'references/')) or name == 'inputs/000.bin':
                copy_checked(profile_bundle / name, output / name, info['sha256'])
        for name in sorted(programs):
            for filename in ('graph.json', 'module.so', 'params.bin', 'record.json'):
                copy_checked(profile_bundle / 'programs' / name / filename, output / 'programs' / name / filename)
        write(output / 'plans/programs.json', programs)
        input_manifest = read(profile_bundle / 'plans/inputs.json')
        for record in input_manifest['records']:
            for frontier in record['frontiers'].values():
                # This capsule runs full pipelines from the image, never isolated
                # intervals. Keep host references and omit unused raw-input paths.
                frontier.pop('raw', None)
        write(output / 'plans/inputs.json', input_manifest)
        copy_checked(profile_bundle / 'plans/expected_outputs.json', output / 'plans/expected_outputs.json')
        copy_checked(profile_bundle / 'manifest.json', output / 'sources/profile_capsule_manifest.json')
        for src, dest in [('rankings.json', 'plans/rankings.json'), ('pool.json', 'plans/pool.json'),
                          ('FREEZE.json', 'plans/ranking_freeze.json'), ('ddr.json', 'provenance/ddr.json')]:
            copy_checked(ranked / src, output / dest)
        for directory in ('original', 'enhanced'):
            shutil.copytree(ranked / directory, output / 'provenance/search' / directory)
        write(output / 'plans/static_domain_verification.json', proof)
        shutil.copytree(ranked / 'inputs', output / 'provenance/static_inputs')
        copy_checked(ranked / 'INPUTS.json', output / 'provenance/static_inputs/INPUTS.json')
        package = Path(__file__).resolve().parent
        for source in package.rglob('*.py'):
            copy_checked(source, output / 'sources/yolo_transfer' / source.relative_to(package))
        core = Path(cpu_vta_partitioning.__file__).resolve().parent
        for source in core.rglob('*.py'):
            copy_checked(source, output / 'sources/base_controller/cpu_vta_partitioning' / source.relative_to(core))
        policies = package.parent / 'cpu_policies/open_resources'
        copy_checked(policies.parent / '__init__.py', output / 'sources/cpu_policies/__init__.py')
        for filename in ('domain.py', 'costs.py', 'runtime.py', 'common.py', '__init__.py'):
            copy_checked(policies / filename, output / 'sources/cpu_policies/open_resources' / filename)
        copy_checked(package / 'pipeline_cli.py', output / 'sources/run_pipeline.py')
        protocol = {'experiment': 'yolov3_tiny416_thread_work_top20', 'seed': SEED, 'input': 0,
                    'qualification': {'warmup': 0, 'scored': 1, 'tail': 0},
                    'performance': {'warmup': 8, 'scored': 16, 'tail': 4}, 'queue_depth': 2,
                    'rounds': 1, 'anchors': False, 'drift_rejection': False, 'automatic_remeasurement': False,
                    'runtime_changes': False, 'executor_changes': False, 'board_reboot': False,
                    'performance_output_mode': 'raw, all eight outputs hashed on every frame',
                    'unmodeled_work': 'producer/collector and output-summary work remain explicit approximations'}
        write(output / 'plans/protocol.json', protocol)
        order = sorted(pool)
        random.Random(SEED).shuffle(order)
        write(output / 'plans/execution_order.json', order)
        write(output / 'plans/board_execution.json', {'boot_policy': 'one_boot_no_automatic_reboot',
              'udmabuf_bytes': 201326592, 'storage_margin_bytes': 268435456})
        emulator = shutil.which('qemu-aarch64-static')
        if not emulator:
            raise RuntimeError('QEMU loader check unavailable')
        check = subprocess.run([emulator, str(output / 'runtime/ld-linux-aarch64.so.1'), '--library-path',
            str(output / 'runtime'), str(output / 'runner/vta_stage_pipeline_runner'), '--help'],
            capture_output=True, text=True, timeout=45)
        if check.returncode or '--cpu-worker-pinning' not in check.stdout + check.stderr:
            raise ValueError('pipeline executor loader or fixed-worker flag unavailable')
        write(output / 'plans/qemu_loader.json', {'passed': True, 'stdout': check.stdout, 'stderr': check.stderr})
        payload = {name: info for name, info in inventory(output).items()
                   if name.startswith(('programs/', 'runtime/', 'runner/', 'firmware/'))
                   and name.rsplit('/', 1)[-1] != 'record.json'}
        for name in ('inputs/000.bin', 'plans/board_execution.json'):
            payload[name] = {'bytes': (output / name).stat().st_size, 'sha256': sha256(output / name)}
        write(output / 'plans/board_payload.json', {'files': payload,
              'unpacked_bytes': sum(v['bytes'] for v in payload.values())})
        write(output / 'plans/preparation_cost.json', {'seconds': time.monotonic() - started,
              'candidate_count': len(pool), 'sessions': 2 * len(pool), 'full_DNN_inferences': 29 * len(pool),
              'new_module_compilations': 0, 'runtime_rebuilds': 0, 'payload_bytes': sum(v['bytes'] for v in payload.values())})
        manifest = {'model': 'YOLOv3-tiny416', 'scope': 'frozen_three_Top20_union',
                    'profile_capsule_id': parent['bundle_id'], 'runner_sha256': parent['runner_sha256'],
                    'candidate_count': len(pool), 'files': inventory(output, ['.lock'])}
        manifest['bundle_id'] = digest(manifest)[:24]
        write(output / 'manifest.json', manifest)
        sync_tree(output)
    return {'bundle': str(output), 'bundle_id': manifest['bundle_id'], 'candidates': len(pool)}


def validate(bundle):
    import cpu_vta_partitioning
    from cpu_policies.open_resources import domain, costs, runtime
    bundle = Path(bundle).resolve()
    manifest = read(bundle / 'manifest.json')
    if manifest['bundle_id'] != digest({k: v for k, v in manifest.items() if k != 'bundle_id'})[:24]:
        raise ValueError('pipeline capsule seal differs')
    verify_inventory(bundle, manifest['files'])
    freeze = read(bundle / 'plans/ranking_freeze.json')
    if freeze['rankings_sha256'] != sha256(bundle / 'plans/rankings.json') or freeze['pool_sha256'] != sha256(bundle / 'plans/pool.json'):
        raise ValueError('frozen candidate lists changed')
    pool = read(bundle / 'plans/pool.json')
    validate_lists(read(bundle / 'plans/rankings.json'), pool)
    from .domain_counts import allocation_counts
    proof = read(bundle / 'plans/static_domain_verification.json')
    if (proof.get('passed') is not True or proof.get('domain_counts') != allocation_counts()
            or proof.get('pipeline_FPS_read') is not False):
        raise ValueError('pipeline capsule lacks the complete static-domain proof')
    programs = read(bundle / 'plans/programs.json')
    for cfg in pool.values():
        domain.validate_candidate(cfg, programs, unit_count=14, max_vta_islands=3)
    package = Path(__file__).resolve().parent
    for source in (bundle / 'sources/yolo_transfer').rglob('*.py'):
        if sha256(source) != sha256(package / source.relative_to(bundle / 'sources/yolo_transfer')):
            raise ValueError('pipeline controller differs from frozen source')
    core = Path(cpu_vta_partitioning.__file__).resolve().parent
    snapshot = bundle / 'sources/base_controller/cpu_vta_partitioning'
    for source in snapshot.rglob('*.py'):
        if sha256(source) != sha256(core / source.relative_to(snapshot)):
            raise ValueError('pipeline transport differs from frozen source')
    for module in (domain, costs, runtime):
        if sha256(module.__file__) != sha256(bundle / 'sources/cpu_policies/open_resources' / Path(module.__file__).name):
            raise ValueError('pipeline domain or affinity validator differs')
    return manifest


def make_task(bundle, name, qualification=False):
    bundle = Path(bundle)
    config = read(bundle / 'plans/pool.json')[name]
    protocol = read(bundle / 'plans/protocol.json')
    window = protocol['qualification' if qualification else 'performance']
    manifest = read(bundle / 'manifest.json')
    document = {'kind': 'qualify_pipeline' if qualification else 'performance',
        'candidate_id': name, 'configuration': deepcopy(config), 'stages': deepcopy(config['stages']),
        'environment': {'cores': 4, 'frequency_khz': 1066666, 'poll_sleep_ns': 1000,
                        'post_start_sleep_ns': 1000, 'spin_count': 0, 'exclude_worker0': 1, 'bind_threads': 1},
        'warmup': window['warmup'], 'scored': window['scored'], 'tail': window['tail'], 'input_indices': [0],
        'executor_sha256': manifest['runner_sha256'], 'bundle_id': manifest['bundle_id'],
        'experiment': protocol['experiment'], 'output_mode': 'all_stage_dumps' if qualification else 'all_final_hashes'}
    document['task_id'] = digest(document)[:24]
    return document


def native_command(document, bundle, remote_bundle, remote_job):
    from cpu_vta_partitioning.measurement.tasks import native_command as base
    if document != make_task(bundle, document['candidate_id'], document['kind'] == 'qualify_pipeline'):
        raise ValueError('pipeline task identity differs')
    argv, inputs = base(document, bundle, remote_bundle, remote_job)
    loader = argv.index(remote_bundle + '/runtime/ld-linux-aarch64.so.1')
    argv[loader:loader] = ['TVM_BIND_THREADS=1', 'TVM_EXCLUDE_WORKER0=1']
    cfg = document['configuration']
    argv[argv.index('--cpu-stage-gate') + 1] = cfg['execution_mode']
    argv += ['--allow-cpu-affinity-superset', '--cpu-worker-pinning',
             '--control-cpu-affinity', ','.join(map(str, cfg['control_mask']))]
    if document['kind'] == 'qualify_pipeline':
        argv.append('--resource-affinity-probe')
    else:
        argv[argv.index('--output-mode') + 1] = 'raw'
    if '--process-cpu-affinity' in argv or '--queue-timing' in argv:
        raise ValueError('unrequested process binding or timing probes')
    return argv, inputs


def check_outputs(actual, program, expected):
    if len(actual) != len(program['output_node_ids']):
        raise ValueError('pipeline output frontier incomplete')
    for index, (node, output) in enumerate(zip(program['output_node_ids'], actual)):
        if output.get('index') != index or any(output.get(k) != expected[str(node)][k]
                for k in ('shape', 'dtype', 'nbytes', 'fnv1a64')):
            raise ValueError('pipeline output hash/type differs from ARM reference: ' + str(node))


def summary(raw, document, bundle):
    from cpu_vta_partitioning.measurement import correctness, frames, input_sequence, window
    from cpu_policies.open_resources.runtime import _affinity
    raw, bundle = Path(raw), Path(bundle)
    total = sum(document[k] for k in ('warmup', 'scored', 'tail'))
    rows = frames(raw / 'result.jsonl', total)
    if any(type(row.get(k)) is not int for row in rows for k in ('frame_id', 'completion_index', 'input_index')):
        raise ValueError('frame/input identities must be integers')
    input_sequence(rows, document)
    cfg = document['configuration']
    qualification = document['kind'] == 'qualify_pipeline'
    affinity_report = read(raw / 'affinity.json')
    ledger, idle = _affinity(affinity_report, cfg, qualification)
    for stage, reported in zip(cfg['stages'], affinity_report['stages']):
        if stage['device'] == 'vta':
            caller = next(e for e in reported['effective_affinities'] if e['tid'] == reported['caller_tid'])
            if caller['cpus'] != stage['mask']:
                raise ValueError('VTA host allowed cores differ from the requested placement')
    programs = read(bundle / 'plans/programs.json')
    expected = read(bundle / 'plans/expected_outputs.json')
    for row in rows:
        if (row.get('cpu_stage_gate_mode') != cfg['execution_mode']
                or row.get('queue_timing_enabled') is not False
                or row.get('defer_result_write_enabled') is not True
                or row.get('stage_cpu_time_scope') != 'disabled_concurrent_pipeline'):
            raise ValueError('pipeline gate/probe/logging mode differs')
        check_outputs(row.get('raw_outputs', []), programs[cfg['stages'][-1]['program']], expected)
    if qualification:
        correctness(rows, document, bundle, raw)
        for row in rows:
            if len(row['stage_raw_outputs']) != len(cfg['stages']):
                raise ValueError('qualification stage outputs incomplete')
            for stage, output in zip(cfg['stages'], row['stage_raw_outputs']):
                check_outputs(output['outputs'], programs[stage['program']], expected)
        return {'correct': True, 'thread_ledger': ledger, 'idle_auxiliary_threads': idle}
    result = window(rows, document['warmup'], document['scored'], document['tail'])
    if any(type(r.get('total_latency_ms')) not in (int, float)
           or not math.isfinite(r['total_latency_ms']) or r['total_latency_ms'] < 0 for r in rows):
        raise ValueError('invalid frame latency')
    clocks = [row.get('completion_process_cpu_ms') for row in rows]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in clocks):
        raise ValueError('invalid whole-process CPU clock')
    if any(b < a for a, b in zip(clocks, clocks[1:])):
        raise ValueError('nonmonotonic process CPU clock')
    first, last = document['warmup'], document['warmup'] + document['scored'] - 1
    work = clocks[last] - clocks[first]
    wall = rows[last]['completion_ms'] - rows[first]['completion_ms']
    if work <= 0 or work / wall > 4.05:
        raise ValueError('whole-process CPU work is inconsistent with four physical cores')
    services = []
    for index in range(len(cfg['stages'])):
        values = [sum(row[f'stage{index}_{op}_ms'] for op in ('set', 'run', 'get'))
                  for row in rows[first:last + 1]]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid pipeline stage service time')
        services.append(statistics.mean(values))
    return {**result, 'fps': 1000 / result['ii_ms'], 'correct': True,
            'process_core_ms_per_completion': work / (document['scored'] - 1),
            'average_cpu_cores': work / wall, 'cpu_time_scope': 'whole_process_between_scored_completions',
            'stage_service_ms': services, 'thread_ledger': ledger, 'idle_auxiliary_threads': idle}


def matched_ssh(config, *, execute_board=False):
    from cpu_vta_partitioning.transport import SSH
    measure = FunctionType(SSH.measure.__code__, {**SSH.measure.__globals__, 'native_command': native_command},
                           SSH.measure.__name__, SSH.measure.__defaults__, SSH.measure.__closure__)
    result = type('YOLOPipelineSSH', (SSH,), {'measure': measure})(config, execute_board=execute_board)
    return result


def run(bundle, run_dir, local_config, *, execute_board=False):
    bundle, run_dir = Path(bundle).resolve(), Path(run_dir).resolve()
    manifest = validate(bundle)
    ssh = matched_ssh(read(local_config)['board'], execute_board=execute_board)
    with locked(run_dir):
        boot = ssh.boot_id()
        identity = {'boot_id': boot, 'bundle_id': manifest['bundle_id'], 'bundle': str(bundle)}
        marker = run_dir / 'run.json'
        if marker.exists() and read(marker) != identity:
            raise ValueError('pipeline run boot or capsule identity changed')
        if not marker.exists():
            write(marker, identity)
            write(run_dir / 'selection_freeze.json', read(bundle / 'plans/ranking_freeze.json'))
        if read(run_dir / 'selection_freeze.json') != read(bundle / 'plans/ranking_freeze.json'):
            raise ValueError('pipeline selection was not frozen')
        remote = ssh.deploy(bundle, run_dir)
        ssh.wait_idle(boot)
        hardware = ssh.idle_command('sha256sum /lib/firmware/vta_hpc.bit\n'
            'cat /sys/class/fpga_manager/fpga0/state\ncat /sys/class/u-dma-buf/udmabuf0/size', boot=boot)
        lines = hardware.splitlines()
        if (len(lines) != 3 or lines[0].split()[0] != sha256(bundle / 'firmware/vta_hpc.bit')
                or lines[1] != 'operating' or lines[2] != '201326592'):
            raise ValueError('actual firmware or DMA identity differs; no implicit hardware change')
        write(run_dir / ('hardware-' + boot + '.json'), {'boot_id': boot, 'raw': hardware}, replace=True)
        frequencies = ssh.command('for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do cat "$f"; done').split()
        if frequencies != ['1066666'] * 4:
            raise ValueError('CPU frequency differs; no automatic retuning')
        order = read(bundle / 'plans/execution_order.json')
        documents = [make_task(bundle, name, qualification) for qualification in (True, False) for name in order]
        for index, document in enumerate(documents, 1):
            receipt = run_dir / 'accepted' / boot / (document['task_id'] + '.json')
            if receipt.exists():
                item = read(receipt)
                if item['task'] != document or item['boot_id'] != boot:
                    raise ValueError('pipeline receipt identity differs')
                verify_inventory(item['raw'], item['files'])
                if summary(item['raw'], document, bundle) != item['summary']:
                    raise ValueError('pipeline receipt does not replay')
            else:
                started_at = now()
                started = time.monotonic()
                raw = ssh.measure(document, bundle, remote, run_dir, boot)
                result = summary(raw, document, bundle)
                write(receipt, {'task': document, 'boot_id': boot, 'raw': str(raw), 'summary': result,
                    'files': inventory(raw), 'measurement_started_at': started_at, 'accepted_at': now(),
                    'measurement_and_collection_seconds': time.monotonic() - started})
                ssh.acknowledge(document, run_dir, boot)
            write(run_dir / 'state.json', {**identity, 'status': 'RUNNING', 'accepted_sessions': index,
                'required_sessions': len(documents), 'updated_at': now()}, replace=True)
            print('pipeline accepted', index, '/', len(documents), document['kind'], document['candidate_id'], flush=True)
        verify = 'cd ' + shlex.quote(remote) + ' && sha256sum -c board.sha256 > post_run_verification.log && sync'
        ssh.idle_command(verify, boot=boot, timeout=600)
        result = {**identity, 'status': 'COMPLETE', 'accepted_sessions': len(documents),
                  'required_sessions': len(documents), 'candidate_count': len(order), 'finished_at': now(),
                  'single_boot_only': True, 'post_deployment_hashes_passed': True}
        write(run_dir / 'state.json', result, replace=True)
        sync_tree(run_dir)
        return result
