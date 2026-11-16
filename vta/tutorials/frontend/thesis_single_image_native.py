#!/usr/bin/env python3
"""Shared reset-gated native execution for single-image candidates and ablations."""
import json
from pathlib import Path
import shlex
import tarfile
import numpy as np
from run_thesis_single_image_smoke import SSH, checked
from thesis_completion_context_reset import reset
from thesis_supplement_run import sha256, write_new_json
from thesis_single_image_board_check import numeric
from thesis_e2e_ablation_protocol import window_metrics


def runner_name(r):
    selection = r / 'analysis/native_runner_selection.json'
    if not selection.exists():
        return 'vta_stage_pipeline_runner_raw_integer'
    record = json.loads(selection.read_text())
    if sha256(r / 'raw' / record['name']) != record['sha256']:
        raise ValueError('Selected native runner drifted')
    return record['name']


def native(store, identity, stages, threads, boot_index, boot_id, mode, runs,
           managed=False, anchor_plan=None, capacity=None, profile=False):
    r = store.directory
    board = json.loads((r / 'analysis/board_directory.json').read_text())['path']
    reset(store, boot_index, boot_id, identity, board)
    destination = r / 'raw/native_sessions' / identity
    destination.mkdir(parents=True, exist_ok=False)
    remote_output = board + '/native_sessions/' + identity
    argv = [board + '/' + runner_name(r)]
    for index, (stage, count) in enumerate(zip(stages, threads)):
        prefix = '--stage' + str(index)
        argv += [prefix + '-name', stage['name'], prefix + '-device', stage['device'],
                 prefix + '-graph', board + '/' + stage['directory'] + '/graph.json',
                 prefix + '-lib', board + '/' + stage['directory'] + '/module.so',
                 prefix + '-params', board + '/' + stage['directory'] + '/params.bin',
                 prefix + '-input-names', ','.join(stage['input_names']),
                 prefix + '-runtime-num-threads', str(count)]
        if stage['device'] == 'cpu':
            argv += [prefix + '-cpu-affinity', ','.join(str(i) for i in range(count))]
    argv += ['--input', board + '/inputs/input.bin', '--runs', str(runs), '--queue-depth', '2',
             '--runtime-num-threads', str(max(threads)), '--output-jsonl', remote_output + '/result.jsonl']
    if mode in ('correctness', 'serial', 'gate_serial'):
        argv += ['--serial']
    if managed:
        argv += ['--p8-managed-slots', '1' if mode in ('correctness', 'serial', 'gate_serial') and not anchor_plan else '2', '--p8-slot-timeout-s', '60']
    if anchor_plan:
        argv += ['--shared-buffer-plan', board + '/' + anchor_plan]
    if capacity:
        argv += ['--command-capacity-certificate', board + '/' + capacity, '--bitstream-artifact', board + '/vta_hpc.bit']
    argv += ['--output-mode', 'raw' if managed else 'raw_all_stages'] if mode in ('correctness', 'pipeline_correctness') else ['--output-mode', 'classification']
    if mode in ('correctness', 'pipeline_correctness'):
        argv += ['--output-dump-dir', remote_output + '/outputs']
    if mode in ('correctness', 'pipeline_correctness') or profile:
        argv += ['--allocation-snapshot', remote_output + '/allocations.jsonl']
    script = destination / 'execute.sh'
    stability_env = ('export THESIS_OUTPUT_STABILITY_DIR=' + remote_output + '/stability\n') if mode == 'correctness' else 'unset THESIS_OUTPUT_STABILITY_DIR\n'
    diagnostics = '1' if mode in ('correctness', 'pipeline_correctness') or profile else '0'
    capacity_env = ''
    if capacity:
        certificate_path = r / 'raw' / capacity
        certificate = json.loads(certificate_path.read_text())
        values = certificate['runtime_contract']['environment']
        capacity_env = 'export VTA_INSN_BUFFER_BYTES=' + shlex.quote(values['VTA_INSN_BUFFER_BYTES']) + ' VTA_UOP_BUFFER_BYTES=' + shlex.quote(values['VTA_UOP_BUFFER_BYTES']) + ' VTA_COMMAND_MANIFEST_ID=' + sha256(certificate_path) + '\n'
    script.write_text(f"""set -eu
mkdir -p {remote_output}
export LD_LIBRARY_PATH={board} LD_PRELOAD={board}/libvta.so
export TVM_NUM_THREADS={max(threads)} TVM_THREAD_POOL_SPIN_COUNT=0 VTA_REPLAY_POLICY=disabled
export VTA_QUEUE_DIAGNOSTICS={diagnostics} C3_COMPLETION_TRACE={diagnostics}
unset VTA_INSN_BUFFER_BYTES VTA_UOP_BUFFER_BYTES VTA_INSN_SUBMIT_THRESHOLD_BYTES VTA_COMMAND_MANIFEST_ID
{capacity_env}{stability_env}{shlex.join(argv)} > {remote_output}/native.stdout 2> {remote_output}/native.stderr &
thesis_native_pid=$!
printf '%s\\n' "$thesis_native_pid" > {remote_output}/native.pid
if wait "$thesis_native_pid"; then thesis_native_status=0; else thesis_native_status=$?; fi
cat {remote_output}/native.stdout {remote_output}/native.stderr > {remote_output}/native.log
# Preserve partial rows and all diagnostic files even when native exits nonzero.
tar -cf - -C {remote_output} .
exit "$thesis_native_status"
""")
    write_new_json(destination / 'session_contract.json', {
        'boot_id': boot_id, 'boot_index': boot_index, 'stages': stages, 'threads': threads,
        'actual_policy': 'managed_K2' if managed else 'ordinary_copy', 'anchor_plan': anchor_plan,
        'capacity_certificate': capacity, 'capacity_environment': capacity_env, 'mode': mode, 'runs': runs, 'argv': argv,
        'warmup_rows': 0 if mode in ('correctness', 'pipeline_correctness') else 5, 'numeric_input_sha256': sha256(r / 'raw/inputs/input.npy'),
        'window_scoring': 'discard first five warmup rows, then first two of 302 window rows',
    })
    action = store.execute(SSH + ['sh -s'], 'native_' + identity,
                     'functional_qualification' if mode in ('correctness', 'pipeline_correctness') else ('deployment' if mode == 'gate_serial' else 'performance'),
                     600 if mode == 'window' else 60, script)
    if action['status'] == 'timeout':
        cleanup = destination / 'timeout_recovery.sh'
        cleanup.write_text('set -eu\npid=$(cat ' + remote_output + '/native.pid)\n'
            'if test -e /proc/\"$pid\"/exe; then\n'
            'test \"$(readlink /proc/\"$pid\"/exe)\" = ' + board + '/' + runner_name(r) + '\n'
            'kill -TERM \"$pid\"\nsleep 1\n'
            'if test -e /proc/\"$pid\"/exe; then kill -KILL \"$pid\"; fi\nfi\n'
            'tar -cf - -C ' + remote_output + ' .\n')
        recovery = store.execute(SSH + ['sh -s'], 'native_timeout_recovery_' + identity, 'failed_recovery', 30, cleanup)
        if recovery['status'] == 'ok':
            archive_source = r / recovery['stdout']
        else:
            raise RuntimeError('Native timeout; recovery evidence: ' + recovery['stderr'])
    else:
        archive_source = r / action['stdout']
    with tarfile.open(archive_source) as archive:
        for item in archive.getmembers():
            if item.name.startswith('/') or '..' in Path(item.name).parts or item.issym() or item.islnk():
                raise ValueError('Unsafe native result archive')
        archive.extractall(destination)
    if action['status'] != 'ok':
        raise RuntimeError('Native session failed: ' + action['status'] + '; raw files preserved at ' + str(destination))
    return analyze(store, destination, identity, stages, boot_id, mode, runs, managed, remote_output)


def analyze(store, destination, identity, stages, boot_id, mode, runs, managed, remote_output):
    r = store.directory
    rows = [json.loads(line) for line in (destination / 'result.jsonl').read_text().splitlines()]
    if len(rows) != runs or [row['frame_id'] for row in rows] != list(range(runs)):
        raise ValueError('Incomplete or reordered native frames')
    expected = np.load(r / 'raw/host_reference/reference_logits.npy', allow_pickle=False)
    top1 = int(np.argsort(-expected.reshape(-1), kind='stable')[0])
    if any(row['top1'] != top1 for row in rows):
        raise ValueError('Native frame prediction changed')
    metrics = {'identity': identity, 'boot_id': boot_id, 'mode': mode, 'actual_completed_calls': runs,
               'raw_rows_sha256': sha256(destination / 'result.jsonl')}
    if mode in ('correctness', 'pipeline_correctness'):
        with np.load(r / 'raw/host_reference/reference_boundaries.npz', allow_pickle=False) as data:
            references = {key: data[key] for key in data.files}
        checks = []
        for row in rows:
            descriptors = [(row['raw_output_files'], [stages[-1]['output_node_ids'][0]])]
            if not managed:
                descriptors += [(stage['output_files'], spec['output_node_ids'])
                                for stage, spec in zip(row['stage_raw_outputs'], stages)]
            for items, nodes in descriptors:
                if len(items) != len(nodes):
                    raise ValueError('Incomplete native output frontier')
                for item, node in zip(items, nodes):
                    local = destination / Path(item['path']).relative_to(remote_output)
                    actual = np.frombuffer(local.read_bytes(), dtype=item['dtype']).reshape(item['shape'])
                    checks.append(numeric(actual, references[f'node_{node}'], store.contract['numeric']))
        metrics['numeric_checks'] = checks
        if mode == 'correctness':
            log = destination / 'native.stdout'
            source = log.read_text() if log.exists() else (destination / 'native.log').read_text()
            if not log.exists():
                import re
                # Reconstruct old combined stdout/stderr without changing raw files.
                source = re.sub(r'\[(?:C3_COMPLETION|VTA_QUEUE)\] [^\n]*\n', '', source)
                write_new_json(destination / 'stability_log_reconstruction.json', {
                    'raw_log_sha256': sha256(destination / 'native.log'),
                    'method': 'remove complete known stderr JSON records interleaved in buffered stdout',
                    'raw_files_preserved': True})
            observations = [json.loads(line.split('[THESIS_STABILITY] ', 1)[1])
                            for line in source.splitlines() if '[THESIS_STABILITY] ' in line]
            expected_count = runs * sum(len(stage['output_node_ids']) for stage in stages)
            if len(observations) != expected_count:
                raise ValueError('Missing immediate/delayed complete frontier observations')
            stage_lookup = {stage['name']: stage for stage in stages}
            for item in observations:
                node = stage_lookup[item['stage']]['output_node_ids'][item['output_index']]
                pair = []
                for suffix in ('immediate', 'delayed'):
                    filename = destination / 'stability' / (item['stem'] + '_' + suffix + '.bin')
                    value = np.frombuffer(filename.read_bytes(), dtype=item['dtype']).reshape(item['shape'])
                    numeric(value, references[f'node_{node}'], store.contract['numeric'])
                    pair.append(value.tobytes())
                if pair[0] != pair[1] or not item['stable'] or item['delayed_ms'] - item['immediate_ms'] < 9.999:
                    raise ValueError('Unstable native frontier')
            metrics['stability_observations'] = observations
        metrics['full_output_passed'] = True
    elif mode == 'window':
        if runs != 307:
            raise ValueError('Smoke window requires 5 warmup +302 scored-prefix frames')
        completed = [row['completion_ms'] / 1000 for row in rows[5:]]
        metrics['window'] = window_metrics(completed)
        metrics['warmup_calls'] = 5
    elif mode == 'gate_serial':
        if runs != 6:
            raise ValueError('Gate serial requires 5 warmup +1 paired observation')
        metrics['gate_latency_ms'] = rows[-1]['total_latency_ms']
        metrics['warmup_calls'] = 5
    else:
        if runs != 20:
            raise ValueError('Serial smoke requires 5 warmup +15 scored calls')
        key = 'profile_pipeline_latency_ms' if mode == 'profile' else 'serial_latency_ms'
        metrics[key] = [row['total_latency_ms'] for row in rows[5:]]
        metrics['warmup_calls'] = 5
    write_new_json(destination / 'metrics.json', metrics)
    return metrics
