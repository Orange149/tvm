#!/usr/bin/env python3
"""Immutable, opt-in board controller for the preregistered CPU/VTA residual experiment."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import tarfile
import uuid

import numpy as np

from cpu_vta_residual_protocol import (DEFAULT_SOURCE, ROOT, freeze_protocol,
    load_protocol, matrix, stages_for, verify_assets)
from analyze_cpu_vta_residual_validation import (analyze_existing, analyze_pipeline,
    analyze_serial, analyze_replicates, compare_services, select_cpu_contrast, write_report)
from thesis_completion_context_reset import reset
from thesis_e2e_ablation_protocol import window_metrics
from thesis_supplement_run import sha256, write_new_json

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlgorithms=+ssh-rsa',
       '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa', 'root@192.168.1.247']
SDK = '/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux'
CONTROL_FILES = ('run_cpu_vta_residual_validation.py', 'analyze_cpu_vta_residual_validation.py',
                 'cpu_vta_residual_protocol.py', 'thesis_completion_context_reset.py',
                 'thesis_supplement_run.py', 'thesis_e2e_ablation_protocol.py')


def read_json(path):
    return json.loads(Path(path).read_text())


def new_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_new_json(Path(path), value)


def freeze_controls(store):
    directory = store.directory / 'raw/controller_sources'
    directory.mkdir(exist_ok=False)
    manifest = {}
    for name in CONTROL_FILES:
        source = Path(__file__).with_name(name)
        shutil.copy2(source, directory / name)
        manifest[name] = {'source_path': str(source), 'sha256': sha256(directory / name)}
    new_json(store.directory / 'analysis/controller_sources.json', manifest)
    store.event('controller_sources_frozen',
                manifest_sha256=sha256(store.directory / 'analysis/controller_sources.json'))


def verify_controls(store):
    path = store.directory / 'analysis/controller_sources.json'
    if not path.exists():
        raise ValueError('No immutable controller snapshot; freeze a new protocol')
    events = [e for e in store.events() if e['event'] == 'controller_sources_frozen']
    if len(events) != 1 or sha256(path) != events[0]['manifest_sha256']:
        raise ValueError('Controller manifest drift')
    for name, record in read_json(path).items():
        if sha256(record['source_path']) != record['sha256'] or sha256(
                store.directory / 'raw/controller_sources' / name) != record['sha256']:
            raise ValueError('Controller source drift; use a new protocol: ' + name)


def checked(store, argv, action, timeout=60, stdin=None):
    result = store.execute(argv, action, 'one_time_preparation', timeout, stdin)
    if result['status'] != 'ok':
        raise RuntimeError(f"{action}: {result['status']}; evidence {result['stderr']}")
    return result


def script_action(store, script, action, timeout=60):
    path = store.directory / 'raw' / f'{action}_{uuid.uuid4().hex[:8]}.sh'
    with path.open('x') as stream:
        stream.write('set -eu\n' + script + '\n')
    return checked(store, SSH + ['sh -s'], action, timeout, path)


def build_runner(store):
    """Cross compile only the runner, linked to the exact copied runtime libraries."""
    binding = store.directory / 'analysis/runner_binding.json'
    if binding.exists():
        result = read_json(binding)
        if sha256(store.directory / result['path']) != result['sha256']:
            raise ValueError('New runner identity changed')
        return result
    directory = store.directory / 'raw/build'
    directory.mkdir(exist_ok=False)
    snapshot = directory / 'source/vta/apps/native_deploy'
    snapshot.mkdir(parents=True)
    source_dir = ROOT / 'vta/apps/native_deploy'
    files = ['vta_stage_pipeline_runner.cc', 'vta_shared_buffer_plan.h',
             'vta_managed_lifecycle_control.h', 'vta_command_capacity_certificate.h']
    for name in files:
        shutil.copy2(source_dir / name, snapshot / name)
    pico = directory / 'source/3rdparty/picojson'
    pico.mkdir(parents=True)
    shutil.copy2(ROOT / '3rdparty/picojson/picojson.h', pico / 'picojson.h')
    output = directory / 'vta_stage_pipeline_runner_residual'
    includes = [ROOT / p for p in ('include', '3rdparty/dlpack/include',
                '3rdparty/dmlc-core/include', '3rdparty/vta-hw/include', 'vta/include')]
    command = ('unset LD_LIBRARY_PATH\nsource ' + shlex.quote(SDK) + '\n' +
        'aarch64-xilinx-linux-g++ -std=c++17 -O2 --sysroot="$SDKTARGETSYSROOT" ' +
        ' '.join('-I ' + shlex.quote(str(p)) for p in includes) + ' ' +
        shlex.quote(str(snapshot / files[0])) + ' -o ' + shlex.quote(str(output)) +
        ' -L ' + shlex.quote(str(store.directory / 'raw/runtime')) +
        ' -Wl,-rpath-link,"$SDKTARGETSYSROOT/lib" -Wl,-rpath-link,"$SDKTARGETSYSROOT/usr/lib"' +
        ' -ltvm_runtime -lvta -ldl -pthread\n')
    script = directory / 'build.sh'
    with script.open('x') as stream:
        stream.write(command)
    checked(store, ['bash', str(script)], 'compile_frozen_runtime_runner', 300)
    result = {'path': str(output.relative_to(store.directory)), 'sha256': sha256(output),
              'source_files': {n: sha256(snapshot / n) for n in files},
              'picojson_sha256': sha256(pico / 'picojson.h'),
              'runtime_bindings': read_json(store.directory / 'analysis/board_artifact_bindings.json'),
              'defaults': {'queue_timing': False, 'defer_result_write': False},
              'source_snapshot': str(directory.relative_to(store.directory))}
    new_json(binding, result)
    store.event('runner_frozen', runner_sha256=result['sha256'], binding_sha256=sha256(binding))
    return result


def preflight(store):
    # Foreign idle RPC parents are allowed; native actors and RPC children are not.
    script = '''
printf 'BOOT %s\\n' "$(cat /proc/sys/kernel/random/boot_id)"
printf 'FPGA %s\\n' "$(cat /sys/class/fpga_manager/fpga0/state)"
printf 'DMA %s\\n' "$(cat /sys/class/u-dma-buf/udmabuf0/size)"
printf 'FIRMWARE_LINK %s\\n' "$(readlink /lib/firmware/vta_hpc.bit)"
residual_host_mask=$(awk '/^Cpus_allowed_list:/ {print $2; exit}' /proc/self/status)
case "$residual_host_mask" in 0-3|0,1,2,3) :;; *) echo "HOST_MASK_DRIFT $residual_host_mask"; exit 46;; esac
for residual_cpu in 0 1 2 3; do
 residual_cp=/sys/devices/system/cpu/cpu$residual_cpu/cpufreq
 printf 'CPU %s %s %s\\n' "$residual_cpu" "$(cat "$residual_cp/scaling_governor")" "$(cat "$residual_cp/scaling_cur_freq")"
done
df -Pk /var/volatile
for residual_exe in /proc/[0-9]*/exe; do
 residual_target=$(readlink "$residual_exe" 2>/dev/null || true)
 case "$residual_target" in
  *vta_stage_pipeline_runner*|*vta_native_runner*|*vta_stage_pair_runner*|*vta_dma_microbench_runner*|*c3_reuse_timed*) echo "BUSY $residual_target"; exit 41;;
  */tvm_rpc)
   residual_pid=${residual_exe#/proc/}; residual_pid=${residual_pid%/exe}
   for residual_status in /proc/[0-9]*/status; do
    residual_parent=$(awk '/^PPid:/ {print $2; exit}' "$residual_status" 2>/dev/null || true)
    test "$residual_parent" != "$residual_pid" || { echo "BUSY_RPC $residual_pid"; exit 42; }
   done;;
 esac
done
'''
    result = script_action(store, script, 'board_preflight')
    lines = (store.directory / result['stdout']).read_text().splitlines()
    def field(name):
        return next(line.split()[1] for line in lines if line.startswith(name + ' '))
    cpus = [line.split()[1:] for line in lines if line.startswith('CPU ')]
    if field('FPGA') != 'operating' or int(field('DMA')) != store.contract['udmabuf_bytes']:
        raise ValueError('FPGA/DMA contract changed')
    if len(cpus) != 4 or any(c[1] != 'userspace' for c in cpus) or len({c[2] for c in cpus}) != 1:
        raise ValueError('CPU frequency is not already frozen in userspace at one frequency')
    return {'boot_id': field('BOOT'), 'cpu_frequency_khz': int(cpus[0][2]),
            'governor': 'userspace', 'fpga_actual_frequency_mhz': 'unknown',
            'fpga_target_frequency_mhz': 100, 'preflight_action': result,
            'foreign_active_vta_processes': False,
            'preceding_firmware_link': field('FIRMWARE_LINK')}


def deploy(store, boot, binding):
    """Deploy to a new owned tmpfs directory; leave existing SD and runtime untouched."""
    board = '/var/volatile/cpu_vta_residual_' + store.contract['run_id'] + '_b' + str(boot['boot_index'])
    boot['board_directory'] = board
    archive_path = store.directory / 'raw' / f'deploy_b{boot["boot_index"]}.tar'
    remote_assets = {}
    with tarfile.open(archive_path, 'x') as archive:
        for relative, identity in store.contract['assets'].items():
            if relative.startswith(('raw/programs/', 'raw/component_inputs/')) or relative == 'raw/inputs/input.bin':
                name = relative
            elif relative.startswith('raw/runtime/') and Path(relative).name != 'u-dma-buf.ko':
                name = Path(relative).name
            else:
                continue
            archive.add(store.directory / relative, arcname=name, recursive=False)
            remote_assets[name] = identity['sha256']
        archive.add(store.directory / binding['path'], arcname='vta_stage_pipeline_runner_residual', recursive=False)
        remote_assets['vta_stage_pipeline_runner_residual'] = binding['sha256']
        checksums = store.directory / 'raw' / f'deploy_b{boot["boot_index"]}.sha256'
        with checksums.open('x') as stream:
            for name, digest in sorted(remote_assets.items()):
                stream.write(digest + '  ' + name + '\n')
        archive.add(checksums, arcname='assets.sha256', recursive=False)
    script_action(store, f'test ! -e {shlex.quote(board)}\nmkdir {shlex.quote(board)}', 'create_owned_tmpfs')
    checked(store, SSH + ['tar -xf - -C ' + shlex.quote(board)], 'deploy_owned_tmpfs', 120, archive_path)
    script_action(store, f'''cd {shlex.quote(board)}
sha256sum -c assets.sha256
chmod +x tvm_rpc vta_stage_pipeline_runner_residual
export LD_LIBRARY_PATH="$PWD" LD_PRELOAD="$PWD/libvta.so"
./tvm_rpc server --host=127.0.0.1 --port=9092 --port-end=9093 > rpc.stdout 2> rpc.stderr < /dev/null &
printf '%s\\n' "$!" > rpc.pid
''', 'start_owned_idle_rpc')
    script_action(store, f'''cd {shlex.quote(board)}
residual_pid=$(cat rpc.pid)
test "$(readlink /proc/$residual_pid/exe)" = "$PWD/tvm_rpc"
''', 'verify_owned_idle_rpc')
    new_json(store.directory / 'analysis' / f'boot_{boot["boot_index"]}.json', boot)
    store.event('board_deployed', boot_id=boot['boot_id'], boot_index=boot['boot_index'], board_directory=board)
    return board


def verify_board_prefix(boot, board):
    return f'''test "$(cat /proc/sys/kernel/random/boot_id)" = {shlex.quote(boot['boot_id'])}
cd {shlex.quote(board)}
sha256sum -c assets.sha256 > /dev/null
for residual_cpu in 0 1 2 3; do
 residual_cp=/sys/devices/system/cpu/cpu$residual_cpu/cpufreq
 test "$(cat "$residual_cp/scaling_governor")" = userspace
 test "$(cat "$residual_cp/scaling_cur_freq")" = {boot['cpu_frequency_khz']}
done
'''


def runner_argv(board, output, stages, condition, mode, runs, telemetry=False, profile=False):
    argv = [board + '/vta_stage_pipeline_runner_residual']
    for index, stage in enumerate(stages):
        prefix = '--stage' + str(index)
        argv += [prefix + '-name', stage['name'], prefix + '-device', stage['device'],
                 prefix + '-graph', board + '/' + stage['graph'],
                 prefix + '-lib', board + '/' + stage['lib'],
                 prefix + '-params', board + '/' + stage['params'],
                 prefix + '-input-names', ','.join(stage['input_names']),
                 prefix + '-runtime-num-threads', str(condition['threads'][index])]
        # For VTA this binds its host caller/pool, never FPGA execution lanes.
        argv += [prefix + '-cpu-affinity', ','.join(map(str, condition['affinity'][index]))]
        if index:
            argv += [prefix + '-input-sources', ','.join(f'stage{index-1}:{j}' for j in range(len(stage['input_names'])))]
    argv += ['--input', board + '/' + stages[0]['input_files'][0], '--runs', str(runs),
             '--queue-depth', str(condition['queue_depth']), '--runtime-num-threads', str(max(condition['threads'])),
             '--output-jsonl', output + '/result.jsonl', '--thread-affinity-report', output + '/affinity.json']
    if len(stages[0]['input_names']) != 1:
        raise ValueError('First frozen stage needs an input list; single-input runner cannot silently drop frontier')
    if mode in ('serial', 'correctness_serial', 'component'):
        argv += ['--serial']
    correct = mode.startswith('correctness')
    argv += ['--output-mode', 'raw_all_stages' if correct else ('raw' if mode == 'component' else 'classification')]
    if correct:
        argv += ['--output-dump-dir', output + '/outputs']
    if telemetry and mode in ('pipeline', 'correctness_pipeline'):
        argv += ['--queue-timing']
    if telemetry:
        argv += ['--defer-result-write']
    if profile:
        argv += ['--vta-runtime-profile-dir', output + '/vta_profile']
    return argv


def safe_extract(archive_source, destination):
    with tarfile.open(archive_source) as archive:
        seen = set()
        for item in archive.getmembers():
            if item.name.startswith('/') or '..' in Path(item.name).parts or not (item.isdir() or item.isfile()):
                raise ValueError('Unsafe result archive member')
            if item.isdir():
                continue
            normalized = Path(item.name)
            if normalized in seen:
                raise ValueError('Duplicate normalized result archive member')
            seen.add(normalized)
            if (destination / item.name).exists():
                raise ValueError('Result archive would overwrite local evidence')
        for item in archive.getmembers():
            if not item.isfile():
                continue
            target = destination / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(item) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)


def verify_affinity(path, condition, stages, mode):
    report = read_json(path)
    if len(report['stages']) != len(stages):
        raise ValueError('Incomplete thread affinity report')
    for index, stage in enumerate(report['stages']):
        expected = condition['affinity'][index]
        requested = expected
        if stage['stage_index'] != index or stage['requested_cpus'] != requested:
            raise ValueError('Requested affinity drift')
        rows = stage['effective_affinities']
        if not rows or sorted(t['tid'] for t in rows) != sorted(stage['thread_ids']):
            raise ValueError('Missing effective thread affinity')
        if stage.get('caller_tid') not in stage['thread_ids']:
            raise ValueError('Missing verified stage caller TID')
        if any(t['cpus'] != expected for t in rows):
            raise ValueError(f'Actual worker mask drift at stage {index}: {rows}')
    return report


def numeric_check(actual, expected, tolerances):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.isfinite(actual).all():
        raise ValueError('Output shape/dtype/finite contract failure')
    passed = np.array_equal(actual, expected) if np.issubdtype(expected.dtype, np.integer) else np.allclose(
        actual, expected, atol=tolerances['float_atol'], rtol=tolerances['float_rtol'])
    if not passed:
        raise ValueError('Complete frontier numeric failure')


def validate_correctness(store, destination, remote, rows, stages, stability):
    with np.load(store.directory / store.contract['paths']['frontiers'], allow_pickle=False) as data:
        references = {key: data[key] for key in data.files}
    checks = 0
    for row in rows:
        descriptors = [(s['output_files'], spec['output_node_ids']) for s, spec in zip(row['stage_raw_outputs'], stages)]
        if len(row['stage_raw_outputs']) != len(stages):
            raise ValueError('Incomplete stage frontier outputs')
        descriptors += [(row['raw_output_files'], stages[-1]['output_node_ids'])]
        for items, nodes in descriptors:
            if len(items) != len(nodes):
                raise ValueError('Incomplete live output frontier')
            for item, node in zip(items, nodes):
                path = destination / Path(item['path']).relative_to(remote)
                values = np.frombuffer(path.read_bytes(), dtype=item['dtype']).reshape(item['shape'])
                numeric_check(values, references[f'node_{node}'], store.contract['numeric'])
                checks += 1
    observations = []
    if stability:
        observations = [json.loads(line.split('[THESIS_STABILITY] ', 1)[1]) for line in
                        (destination / 'native.stdout').read_text().splitlines() if '[THESIS_STABILITY] ' in line]
        if len(observations) != len(rows) * sum(len(s['output_node_ids']) for s in stages):
            raise ValueError('Incomplete immediate/delayed frontier checks')
        for item in observations:
            stage = next(s for s in stages if s['name'] == item['stage'])
            expected = references[f"node_{stage['output_node_ids'][item['output_index']]}"]
            pair = []
            for suffix in ('immediate', 'delayed'):
                path = destination / 'stability' / (item['stem'] + '_' + suffix + '.bin')
                values = np.frombuffer(path.read_bytes(), dtype=item['dtype']).reshape(item['shape'])
                numeric_check(values, expected, store.contract['numeric'])
                pair.append(values.tobytes())
            if pair[0] != pair[1] or not item['stable'] or item['delayed_ms'] - item['immediate_ms'] < 9.999:
                raise ValueError('Output changed during correctness stability check')
    return {'full_output_passed': True, 'numeric_checks': checks, 'stability_checks': len(observations)}


def session(store, boot, condition, mode, block, topology='03:17', telemetry=False,
            label=None, component=None, profile=False):
    verify_assets(store)
    binding = build_runner(store)
    board = boot['board_directory']
    stages = stages_for(store, topology)
    if component is not None:
        stages = [copy.deepcopy(store.contract['programs'][component])]
        stages[0]['index'] = 0
    correct = mode.startswith('correctness')
    runs = 10 if correct else (307 if mode == 'pipeline' else 20)
    identity = f"b{boot['boot_index']}_{topology.replace(':', '_')}_{block}_{condition['condition']}_{label or mode}"
    destination = store.directory / 'raw/sessions' / identity
    destination.mkdir(parents=True, exist_ok=False)
    remote = board + '/sessions/' + identity
    argv = runner_argv(board, remote, stages, condition, mode, runs, telemetry, profile)
    record = {'session_id': identity, 'boot_id': boot['boot_id'], 'boot_index': boot['boot_index'],
              'condition': condition['condition'], 'mode': mode, 'block': block, 'topology': topology,
              'threads': condition['threads'], 'affinity': condition['affinity'],
              'spin_count': condition['spin_count'], 'queue_depth': condition['queue_depth'],
              'runner_sha256': binding['sha256'], 'runtime_bindings': {
                  n: v['sha256'] for n, v in binding['runtime_bindings'].items()},
              'program_identity': [{k: s[k] for k in ('name', 'files', 'output_node_ids', 'input_node_ids')} for s in stages],
              'numeric_input_sha256': store.contract['numeric_input']['input_sha256'],
              'telemetry': telemetry, 'profile': profile, 'runs': runs, 'argv': argv,
              'cpu_frequency_khz': boot['cpu_frequency_khz'], 'fpga_actual_frequency_mhz': 'unknown'}
    new_json(destination / 'session_contract.json', record)
    script_action(store, verify_board_prefix(boot, board), 'verify_before_' + identity)
    # This context reset is the already validated firmware/completion protocol.
    reset(store, boot['boot_index'], boot['boot_id'], identity, board)
    stability_env = ('export THESIS_OUTPUT_STABILITY_DIR=' + shlex.quote(remote + '/stability')) if mode == 'correctness_serial' else 'unset THESIS_OUTPUT_STABILITY_DIR'
    diagnostics = '1' if profile else '0'
    script = destination / 'execute.sh'
    with script.open('x') as stream:
        stream.write('set -eu\n' + verify_board_prefix(boot, board) + f'''
mkdir -p {shlex.quote(str(Path(remote).parent))}
test ! -e {shlex.quote(remote)}
mkdir {shlex.quote(remote)}
export LD_LIBRARY_PATH={shlex.quote(board)} LD_PRELOAD={shlex.quote(board + '/libvta.so')}
export TVM_NUM_THREADS={max(condition['threads'])} TVM_THREAD_POOL_SPIN_COUNT={condition['spin_count']} VTA_REPLAY_POLICY=disabled
export VTA_QUEUE_DIAGNOSTICS={diagnostics} C3_COMPLETION_TRACE={diagnostics}
unset VTA_INSN_BUFFER_BYTES VTA_UOP_BUFFER_BYTES VTA_INSN_SUBMIT_THRESHOLD_BYTES VTA_COMMAND_MANIFEST_ID
{stability_env}
residual_host_mask=$(awk '/^Cpus_allowed_list:/ {{print $2; exit}}' /proc/self/status)
case "$residual_host_mask" in 0-3|0,1,2,3) :;; *) exit 46;; esac
{shlex.join(argv)} > {shlex.quote(remote + '/native.stdout')} 2> {shlex.quote(remote + '/native.stderr')} &
residual_native_pid=$!
printf '%s\\n' "$residual_native_pid" > {shlex.quote(remote + '/native.pid')}
if wait "$residual_native_pid"; then residual_status=0; else residual_status=$?; fi
printf '%s\\n' "$residual_status" > {shlex.quote(remote + '/exit_status')}
tar -cf - -C {shlex.quote(remote)} .
exit "$residual_status"
''')
    action = store.execute(SSH + ['sh -s'], 'native_' + identity,
                           'functional_qualification' if correct else 'performance', 180 if runs == 307 else 90, script)
    archive_source = store.directory / action['stdout']
    if action['status'] != 'ok':
        recovery = script_action(store, f'''
if test -f {shlex.quote(remote + '/native.pid')}; then
 residual_pid=$(cat {shlex.quote(remote + '/native.pid')})
 if test -e /proc/$residual_pid/exe; then
  test "$(readlink /proc/$residual_pid/exe)" = {shlex.quote(board + '/vta_stage_pipeline_runner_residual')}
  kill -TERM "$residual_pid"
  sleep 1
  if test -e /proc/$residual_pid/exe; then kill -KILL "$residual_pid"; fi
 fi
fi
if test -d {shlex.quote(remote)}; then tar -cf - -C {shlex.quote(remote)} .; fi
''', 'recover_' + identity, 30)
        archive_source = store.directory / recovery['stdout']
    if archive_source.stat().st_size:
        safe_extract(archive_source, destination)
    if action['status'] != 'ok':
        raise RuntimeError('Native failure/timeout; partial evidence retained: ' + str(destination))
    rows = [json.loads(line) for line in (destination / 'result.jsonl').read_text().splitlines()]
    if len(rows) != runs or [row['frame_id'] for row in rows] != list(range(runs)):
        raise ValueError('Missing/reordered native frame')
    if mode == 'pipeline' and any(row.get(f'stage{i}_process_cpu_ms') is not None for row in rows for i in range(len(stages))):
        raise ValueError('Pipeline process CPU time was incorrectly attributed to a stage')
    expected = np.load(store.directory / store.contract['paths']['reference_logits'], allow_pickle=False)
    if component is None and any(row['top1'] != int(np.argmax(expected.reshape(-1))) for row in rows):
        raise ValueError('Final top1 drift')
    record['affinity_report'] = verify_affinity(destination / 'affinity.json', condition, stages, mode)
    if correct:
        metrics = validate_correctness(store, destination, remote, rows, stages, mode == 'correctness_serial')
    elif mode == 'pipeline':
        metrics = analyze_pipeline(rows)
        window = window_metrics([row['completion_ms'] / 1000 for row in rows[5:]])
        if abs(metrics['ii_ms'] - 1000 * window['ii_seconds']) > 1e-6:
            raise ValueError('Independent window score mismatch')
    else:
        metrics = analyze_serial(rows)
    record.update(metrics=metrics, raw_jsonl_sha256=sha256(destination / 'result.jsonl'),
                  raw_jsonl=str((destination / 'result.jsonl').relative_to(store.directory)))
    new_json(destination / 'result_summary.json', record)
    store.event('session_validated', session_id=identity, boot_id=boot['boot_id'],
                mode=mode, condition=condition['condition'], ii_ms=metrics.get('ii_ms'), raw_jsonl_sha256=record['raw_jsonl_sha256'])
    # Host archives, raw output tensors and partial evidence are retained permanently.
    script_action(store, f'cd {shlex.quote(board)}\ntest -f {shlex.quote(remote + "/exit_status")}\nrm -rf {shlex.quote(remote)}', 'remove_archived_owned_session_' + identity)
    print(f"PASS {identity}" + (f" II={metrics['ii_ms']:.3f}ms" if 'ii_ms' in metrics else ''), flush=True)
    return record


def pair_services(serial, pipeline):
    a, b = copy.deepcopy(serial['metrics']), copy.deepcopy(pipeline['metrics'])
    a['session_identity'], b['session_identity'] = serial, pipeline
    return compare_services(a, b)


def cleanup_rpc(store, boot):
    board = boot['board_directory']
    script_action(store, f'''cd {shlex.quote(board)}
residual_pid=$(cat rpc.pid)
if test -e /proc/$residual_pid/exe; then
 test "$(readlink /proc/$residual_pid/exe)" = "$PWD/tvm_rpc"
 kill -TERM "$residual_pid"
fi
if test "$(readlink /lib/firmware/vta_hpc.bit)" = "$PWD/vta_hpc.bit"; then
 ln -sf {shlex.quote(boot['preceding_firmware_link'])} /lib/firmware/vta_hpc.bit
fi
''', 'stop_owned_rpc')


def collect_pairs(store, boot, selection, label):
    configs = {c['condition']: c for c in matrix()}
    telemetry = read_json(store.directory / 'analysis/telemetry_gate.json')['primary_telemetry']
    output = []
    for topology in ('03:17', '03:15'):
        for name in (selection['baseline'], selection['intervention']):
            for mode in ('correctness_serial', 'correctness_pipeline'):
                session(store, boot, configs[name], mode, label + '_qualification', topology, telemetry=telemetry)
        for block in range(2):
            order = [selection['baseline'], selection['intervention']]
            if block:
                order.reverse()
            arms = {}
            for name in order:
                arms[name] = session(store, boot, configs[name], 'pipeline', label + str(block), topology, telemetry=telemetry)
            output.append({'boot_id': boot['boot_id'], 'block': block, 'topology': topology,
                           'stage_index': selection['stage_index'], 'control': arms[selection['baseline']],
                           'contrast': arms[selection['intervention']]})
    new_json(store.directory / 'analysis' / f'paired_boot_{boot["boot_index"]}.json', output)
    return output


def pilot(store):
    if (store.directory / 'analysis/pilot.json').exists():
        raise ValueError('Pilot already completed; use replicate after a physical kernel reboot')
    binding = build_runner(store)
    boot = preflight(store)
    boot['boot_index'] = 1
    if (store.directory / 'analysis/boot_1.json').exists():
        raise ValueError('Partial pilot exists; start a new protocol, no silent retry')
    deploy(store, boot, binding)
    try:
        configs = {c['condition']: c for c in matrix()}
        for enabled in (False, True):
            for mode in ('correctness_serial', 'correctness_pipeline'):
                session(store, boot, configs['P0'], mode, 'initial_qualification',
                        telemetry=enabled, label=mode + ('_on' if enabled else '_off'))
        # Same executable and configuration, alternating order to expose drift.
        telemetry_sessions = []
        for block, order in enumerate(((False, True), (True, False))):
            for enabled in order:
                telemetry_sessions.append(session(store, boot, configs['P0'], 'pipeline',
                    'telemetry' + str(block), telemetry=enabled, label='on' if enabled else 'off'))
        ratios = []
        for block in range(2):
            pair = [s for s in telemetry_sessions if s['block'] == 'telemetry' + str(block)]
            off = next(s for s in pair if not s['telemetry'])['metrics']['ii_ms']
            on = next(s for s in pair if s['telemetry'])['metrics']['ii_ms']
            ratios.append((on - off) / off)
        use_telemetry = all(abs(r) <= 0.02 for r in ratios)
        telemetry = {'paired_relative_changes': ratios, 'max_abs_change': max(map(abs, ratios)),
                     'primary_telemetry': use_telemetry, 'threshold': 0.02}
        new_json(store.directory / 'analysis/telemetry_gate.json', telemetry)
        # Full outputs are qualified separately, never taken from performance runs.
        for name in ('P1', 'P2', 'P3', 'P4', 'P5'):
            for mode in ('correctness_serial', 'correctness_pipeline'):
                session(store, boot, configs[name], mode, 'qualification', telemetry=use_telemetry)
        missing = [('cpu_00_02', 3, [0, 1, 2]), ('cpu_18_20', 2, [0, 1]), ('cpu_18_20', 1, [3])]
        for component, count, cpus in missing:
            condition = {'condition': component + '_t' + str(count) + '_c' + ''.join(map(str, cpus)),
                         'threads': [count], 'affinity': [cpus], 'spin_count': 0, 'queue_depth': 2}
            session(store, boot, condition, 'component', 'matched_missing', component=component)
        sessions = []
        for block, order in enumerate(store.contract['block_orders']):
            for name in order:
                # Short serial observations check drift and bind intrinsic service to each block.
                serial = session(store, boot, configs[name], 'serial', block, telemetry=use_telemetry)
                pipe = session(store, boot, configs[name], 'pipeline', block, telemetry=use_telemetry)
                sessions.extend((serial, pipe))
                new_json(store.directory / 'analysis' / f'services_b1_{block}_{name}.json', pair_services(serial, pipe))
        selection = select_cpu_contrast(sessions)
        diagnostics = []
        if selection is None:
            for block, depths in enumerate(((1, 2, 4), (4, 2, 1))):
                for depth in depths:
                    config = copy.deepcopy(configs['P0']); config['queue_depth'] = depth
                    diagnostics.append(session(store, boot, config, 'pipeline', 'qdiag' + str(block),
                                               label='q' + str(depth), telemetry=use_telemetry))
        vta_trigger = any(pipe['metrics']['stages'][1]['run_ms']['mean'] >
                          1.05 * serial['metrics']['stages'][1]['run_ms']['mean']
                          for serial, pipe in zip(sessions[::2], sessions[1::2]))
        if vta_trigger:
            for topology in ('03:15', '03:17'):
                stage = 'vta_' + topology.replace(':', '_')
                config = {'condition': 'VTA1', 'threads': [1], 'affinity': [[0, 1, 2, 3]],
                          'spin_count': 0, 'queue_depth': 2}
                diagnostics.append(session(store, boot, config, 'component', 'vta_profile', topology,
                                           component=stage, profile=True))
                diagnostics.append(session(store, boot, configs['P0'], 'pipeline', 'vta_profile', topology,
                                           telemetry=True, profile=True))
        result = {'scope': 'single_boot_exploratory', 'boot_id': boot['boot_id'],
                  'telemetry_gate': telemetry, 'sessions': sessions, 'selection': selection,
                  'diagnostics': diagnostics, 'vta_profile_triggered': vta_trigger,
                  'ddr_attribution': 'unknown', 'historical_34ms': 'diagnostic_only',
                  'formal_static_score_changed': False}
        if selection:
            new_json(store.directory / 'analysis/selected_contrast.json', selection)
            collect_pairs(store, boot, selection, 'replicate')
        new_json(store.directory / 'analysis/pilot.json', result)
        return result
    finally:
        cleanup_rpc(store, boot)


def replicate(store):
    if not (store.directory / 'analysis/pilot.json').exists():
        raise ValueError('A successful pilot is required')
    selection = read_json(store.directory / 'analysis/selected_contrast.json')
    existing = [read_json(p) for p in sorted((store.directory / 'analysis').glob('boot_*.json'))]
    if len(existing) >= 3:
        raise ValueError('Three boots already collected; use analyze')
    boot = preflight(store)
    if boot['boot_id'] in {b['boot_id'] for b in existing}:
        raise ValueError('Independent kernel boot required; restarting RPC/reloading bitstream does not count')
    boot['boot_index'] = len(existing) + 1
    deploy(store, boot, build_runner(store))
    try:
        return collect_pairs(store, boot, selection, 'replicate')
    finally:
        cleanup_rpc(store, boot)


def analyze(store):
    output = store.directory / 'analysis' / ('report_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6])
    output.mkdir(exist_ok=False)
    result = {'scope': 'offline_only', 'ddr_attribution': 'unknown', 'hardware_internal_dma': 'unknown'}
    result['frozen_proxy'] = read_json(store.directory / store.contract['paths']['proxy'])
    result['frozen_proxy']['runner'] = store.contract['source_identity']['proxy_runner']
    result['protocol_contract'] = str(store.directory / 'contract.json')
    pilot_path = store.directory / 'analysis/pilot.json'
    if pilot_path.exists():
        result.update(read_json(pilot_path))
    else:
        result['sessions'] = [read_json(p) for p in sorted((store.directory / 'raw/sessions').glob('*/result_summary.json'))
                              if read_json(p)['mode'] in ('pipeline', 'serial')]
        if result['sessions']:
            result['scope'] = 'partial_single_boot_exploratory'
    invalid = store.directory / 'invalid_run.json'
    if invalid.exists():
        result['validation_failure'] = read_json(invalid)
    pairs = [r for p in sorted((store.directory / 'analysis').glob('paired_boot_*.json')) for r in read_json(p)]
    result['independent_boot_count'] = len({r['boot_id'] for r in pairs})
    if result['independent_boot_count'] == 3:
        result['replication'] = {topology: analyze_replicates([r for r in pairs if r['topology'] == topology])
                                 for topology in ('03:17', '03:15')}
    else:
        result['causal_conclusion'] = '探索性结果；需用户完成另外两个独立kernel boot，暂不宣称因果闭合。'
    offline_path = store.directory / 'analysis/offline.json'
    if offline_path.exists():
        result['historical_e0'] = read_json(offline_path)
    new_json(output / 'analysis.json', result)
    write_report(output / 'CONCLUSION_ZH.md', result)
    # Reproducible static SVG timeline; no plotting dependency is needed on the board.
    for path in sorted((store.directory / 'raw/sessions').glob('*/result_summary.json')):
        summary = read_json(path)
        if summary['mode'] not in ('pipeline', 'serial'):
            continue
        rows = [json.loads(line) for line in (store.directory / summary['raw_jsonl']).read_text().splitlines()][7:27]
        if not rows:
            continue
        start = min(r['stage0_start_ms'] for r in rows)
        end = max(r[f"stage{len(summary['program_identity'])-1}_end_ms"] for r in rows)
        count = len(summary['program_identity'])
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{count*65+60}" viewBox="0 0 1200 {count*65+60}">',
               '<rect width="100%" height="100%" fill="white"/>',
               f'<text x="12" y="20" font-size="14">{summary["session_id"]}; wall monotonic, ms</text>']
        for i, stage in enumerate(summary['program_identity']):
            svg.append(f'<text x="8" y="{55+i*65}" font-size="12">{stage["name"]}</text>')
            for row in rows:
                x = 130 + (row[f'stage{i}_start_ms'] - start) / (end-start) * 1040
                width = (row[f'stage{i}_end_ms'] - row[f'stage{i}_start_ms']) / (end-start) * 1040
                svg.append(f'<rect x="{x:.3f}" y="{35+i*65}" width="{width:.3f}" height="30" fill="{("#2674a5", "#dc8c20", "#2b9563")[i%3]}"><title>frame {row["frame_id"]}; {row[f"stage{i}_ms"]:.3f} ms</title></rect>')
        svg.append('</svg>')
        (output / (summary['session_id'] + '.svg')).write_text('\n'.join(svg))
    store.event('analysis_written', report_directory=str(output.relative_to(store.directory)))
    print(str(output), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'offline', 'pilot', 'replicate', 'analyze'))
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--source-dir', default=str(DEFAULT_SOURCE))
    parser.add_argument('--execute-board', action='store_true', help='Explicit board execution opt-in; required for pilot/replicate')
    args = parser.parse_args(argv)
    if args.phase in ('pilot', 'replicate') and not args.execute_board:
        parser.error('pilot/replicate require explicit --execute-board')
    if args.phase == 'freeze':
        store = freeze_protocol(args.source_dir, args.run_dir)
        freeze_controls(store)
        print(str(store.directory), flush=True)
        return 0
    store = load_protocol(args.run_dir)
    if args.phase in ('pilot', 'replicate'):
        verify_controls(store)
    if (store.directory / 'invalid_run.json').exists() and args.phase in ('pilot', 'replicate'):
        raise ValueError('Failed protocol cannot collect more samples; create a new protocol')
    event_count_before = len(store.events())
    try:
        if args.phase == 'offline':
            result = analyze_existing(store.contract['source_identity']['directory'])
            new_json(store.directory / 'analysis/offline.json', result)
            write_report(store.directory / 'analysis/OFFLINE_ZH.md', result)
            print('E0 offline audit PASS', flush=True)
        elif args.phase == 'pilot':
            pilot(store)
            analyze(store)
        elif args.phase == 'replicate':
            replicate(store)
            analyze(store)
        else:
            analyze(store)
    except Exception as error:
        new_events = store.events()[event_count_before:]
        affected_execution = any(e['event'] == 'board_deployed' or
            (e['event'] == 'action_start' and e.get('action', '').startswith('native_')) for e in new_events)
        if args.phase in ('pilot', 'replicate') and affected_execution:
            invalid = store.directory / 'invalid_run.json'
            if not invalid.exists():
                new_json(invalid, {'phase': args.phase, 'error': str(error),
                                   'raw_evidence_preserved': True, 'retry_requires_new_protocol': True})
            store.event('validation_stopped', phase=args.phase, error=str(error))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
