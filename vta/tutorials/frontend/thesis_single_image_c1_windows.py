#!/usr/bin/env python3
"""Three distinct kernel-boot candidate windows for the frozen single-input pool."""
import argparse
import json
from pathlib import Path
import shlex
import statistics
import tarfile
import time
import uuid
from run_thesis_single_image_smoke import SSH, checked, PYTHON, FRONTEND
from thesis_supplement_run import RunStore, sha256, write_new_json
from thesis_single_image_native import native, runner_name
from thesis_completion_context_reset import reset


def stages_for(r, candidate):
    manifest = json.loads((r / 'raw/c1_components_int8/component_manifest.json').read_text())
    lookup = {(p['begin'], p['end_exclusive'], p['device']): p for p in manifest['programs']}
    result = []
    for spec in candidate['ranges']:
        stage = dict(lookup[tuple(spec)])
        stage['directory'] = 'c1_components_int8/' + stage['name']
        result.append(stage)
    return result


def prepare(store, redeployment=False):
    r = store.directory
    frozen = r / 'raw/c1_frozen/fair_support_contract.json'
    source = json.loads((r / 'analysis/board_artifact_bindings.json').read_text())
    files = {name: Path(item['host_path']) for name, item in source.items()}
    files[runner_name(r)] = r / 'raw' / runner_name(r)
    files['thesis_graph_snapshot.so'] = r / 'raw/thesis_graph_snapshot.so'
    files['inputs/input.bin'] = r / 'raw/inputs/input.bin'
    for directory in ('board_programs', 'c1_components_int8'):
        for p in (r / 'raw' / directory).rglob('*'):
            if p.is_file() and p.name != 'lowering_qir.json':
                files[str(p.relative_to(r / 'raw'))] = p
    archive = r / 'raw/c1_boot_restore.tgz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, path in files.items():
            tar.add(path, arcname=name)
    write_new_json(r / 'analysis/c1_boot_restore_bindings.json', {
        name: {'host_path': str(path), 'sha256': sha256(path)} for name, path in files.items()})
    measurement = r / 'raw/c1_measurement_contract.json'
    if redeployment:
        if not store.contract.get('snapshot_fix_redeployment') or not measurement.exists():
            raise ValueError('Redeployment requires the original immutable C1 measurement contract')
        store.event('snapshot_redeployment_restore_bundle_prepared',
            source_selection_contract_sha256=sha256(measurement), runner=runner_name(r))
        return
    write_new_json(measurement, {
        'scope': 'single-image engineering selection, not formal phase04 64+3 input qualification',
        'support_sha256': sha256(frozen), 'boot_count': 3, 'kernel_boot_ids_must_be_distinct': True,
        'candidate_qualification_calls': 10, 'window': {'warmup_rows': 5, 'rows': 302, 'discard_front_rows': 2},
        'selection': 'O2 frozen first five; median II over three boots, tie candidate_id',
        'static_knownness': 'unknown internal-DMA dimensions trigger original semantic-band M1 fallback',
        'pool_domain': 'one VTA island only; integer packed entry; no free operator statistics or old-model outputs',
        'invalid_candidate_policy': 'keep identity/paid costs; no replacement point; stop campaign on hardware/numeric fault',
    })


def collect(store, index):
    r = store.directory
    board_state = r / ('analysis/board_boot.json' if index == 1 else f'analysis/c1_boot{index}_board.json')
    boot = json.loads(board_state.read_text())['boot_id']
    current = checked(store, SSH + ['cat /proc/sys/kernel/random/boot_id'], f'c1_boot{index}_identity', 'one_time_preparation', 30)
    if (r / current['stdout']).read_text().strip() != boot:
        raise ValueError('Kernel boot changed before candidate collection')
    frozen = json.loads((r / 'raw/c1_frozen/fair_support_contract.json').read_text())
    lookup = {c['candidate_id']: c for c in frozen['support']}
    results = []
    for candidate_id in frozen['measurement_orders'][str(index - 1)]:
        candidate = lookup[candidate_id]
        stages = stages_for(r, candidate)
        if index == 1:
            native(store, f'b{index}_C1_{candidate_id}_correctness', stages, candidate['thread_vector'],
                   index, boot, 'correctness', 10)
        metric = native(store, f'b{index}_C1_{candidate_id}_window', stages, candidate['thread_vector'],
                        index, boot, 'window', 307)
        results.append({'candidate_id': candidate_id, 'ii_seconds': metric['window']['ii_seconds'], 'boot_id': boot})
        print(f'boot{index}: {candidate_id} complete, II={metric["window"]["ii_seconds"]:.6f}s', flush=True)
    write_new_json(r / f'analysis/c1_boot{index}_windows.json', {'boot_id': boot, 'results': results})
    if index == 3:
        boots = [json.loads((r / f'analysis/c1_boot{i}_windows.json').read_text()) for i in range(1, 4)]
        if len({b['boot_id'] for b in boots}) != 3:
            raise ValueError('C1 requires three distinct kernel boots')
        values = {candidate: [] for candidate in lookup}
        for b in boots:
            if {row['candidate_id'] for row in b['results']} != set(values):
                raise ValueError('Incomplete frozen-pool coverage')
            for row in b['results']:
                values[row['candidate_id']].append(row['ii_seconds'])
        labels = {key: statistics.median(v) for key, v in values.items()}
        prefix = frozen['orders']['O2_semantic_band'][:5]
        selected = min(prefix, key=lambda key: (labels[key], key))
        write_new_json(r / 'analysis/B_selection.json', {'candidate_id': selected, 'candidate': lookup[selected],
            'stages': stages_for(r, lookup[selected]), 'boot_ids': [b['boot_id'] for b in boots],
            'frozen_O2_prefix': prefix, 'candidate_median_ii_seconds': labels, 'candidate_boot_points': values,
            'selection_scope': 'single-image, three-boot frozen support; not formal64+3 qualification',
            'oracle_candidate_id': min(labels, key=lambda key: (labels[key], key)),
            'oracle_used_to_select_B': False})
        print('B frozen from O2 first5: ' + selected, flush=True)


def reboot_restore(store, index, already_rebooted=False, reuse_qualified_boot=False):
    r = store.directory
    board = json.loads((r / 'analysis/board_directory.json').read_text())['path']
    previous = json.loads((r / f'analysis/c1_boot{index-1}_windows.json').read_text())['boot_id']
    reboot = r / f'raw/c1_reboot{index}.sh'
    if reuse_qualified_boot:
        if not store.contract.get('snapshot_fix_redeployment'):
            raise ValueError('Qualified-boot redeployment is not declared')
        state = json.loads((r / f'analysis/c1_boot{index}_board.json').read_text())
        old = Path(store.contract['reused_evidence_run'])
        old_board = json.loads((old / 'analysis/board_directory.json').read_text())['path']
        stop = r / 'raw/stop_identified_idle_source_rpc.sh'
        from thesis_completion_context_reset import RPC_CHILDREN_CHECK
        stop.write_text('set -eu\ntest "$(cat /proc/sys/kernel/random/boot_id)" = ' + state['boot_id'] + '\n'
            'pid=$(cat ' + old_board + '/rpc.pid)\ntest "$(readlink /proc/$pid/exe)" = ' + old_board + '/tvm_rpc\n'
            + RPC_CHILDREN_CHECK + '\ntest -z "$(rpc_children)"\nkill "$pid"\n')
        checked(store, SSH + ['sh -s'], 'stop_owned_idle_source_rpc_for_redeployment', 'failed_recovery', 30, stop)
        already_rebooted = True
    if not already_rebooted:
        reboot.write_text('set -eu\ntest "$(cat /proc/sys/kernel/random/boot_id)" = ' + previous + '\n'
        'if grep -q " /media/sd-mmcblk1p1 " /proc/mounts; then mount -o remount,ro /media/sd-mmcblk1p1; fi\n'
        'if grep -q " /media/sd-mmcblk1p2 " /proc/mounts; then mount -o remount,ro /media/sd-mmcblk1p2; fi\n'
        'sync\n(sleep 1; /sbin/reboot) </dev/null > /var/volatile/thesis_reboot.log 2>&1 &\necho THESIS_GRACEFUL_REBOOT_REQUESTED\n')
        checked(store, SSH + ['sh -s'], f'c1_reboot{index}_request', 'one_time_preparation', 30, reboot)
    else:
        store.event('user_cold_reboot_attach', boot_index=index, previous_boot_id=previous,
                    rule='new identity must differ; same frozen artifacts; completion fully recollected')
    # Short individual probes allow progress reporting while Linux boots.
    deadline = time.monotonic() + 180
    boot = None
    attempt = 0
    while time.monotonic() < deadline:
        time.sleep(3)
        attempt += 1
        result = store.execute(SSH + ['cat /proc/sys/kernel/random/boot_id'],
                               f'c1_boot{index}_recovery_probe{attempt}', 'failed_recovery', 12)
        if result['status'] == 'ok':
            value = (r / result['stdout']).read_text().strip()
            uuid.UUID(value)
            if value != previous:
                if reuse_qualified_boot and value != state['boot_id']:
                    raise ValueError('Qualified kernel identity drifted during redeployment')
                boot = value
                break
        print(f'boot{index} recovery probe{attempt}: waiting for new kernel identity', flush=True)
    if boot is None:
        raise TimeoutError('Board did not return with a new boot_id within180s')
    if not reuse_qualified_boot:
        write_new_json(r / f'analysis/c1_boot{index}_board.json', {'boot_id': boot, 'previous_boot_id': previous})
    else:
        write_new_json(r / 'analysis/redeployment_kernel_binding.json', {'boot_id': boot,
            'source_run': store.contract['reused_evidence_run'], 'new_kernel_boot': False,
            'completion_and_output_qualification': 'fully recollected before new model sessions'})
    checked(store, SSH + ['test ! -d ' + board + ' && mkdir ' + board], f'c1_boot{index}_ram_directory', 'one_time_preparation', 30)
    checked(store, SSH + ['cat > ' + board + '/c1_boot_restore.tgz'], f'c1_boot{index}_restore_transfer',
            'one_time_preparation', 120, r / 'raw/c1_boot_restore.tgz')
    bindings = json.loads((r / 'analysis/c1_boot_restore_bindings.json').read_text())
    script = r / f'raw/c1_boot{index}_restore.sh'
    lines = ['set -eu',
        'if grep -q " /media/sd-mmcblk1p1 " /proc/mounts; then mount -o remount,ro /media/sd-mmcblk1p1; fi',
        'if grep -q " /media/sd-mmcblk1p2 " /proc/mounts; then mount -o remount,ro /media/sd-mmcblk1p2; fi',
        'cd ' + board, 'tar -xzf c1_boot_restore.tgz']
    for name, item in bindings.items():
        lines.append('test "$(sha256sum ' + shlex.quote(name) + ' | cut -d \' \' -f 1)" = ' + item['sha256'])
    lines += ['chmod755_PLACEHOLDER', 'if ! test -e /dev/udmabuf0; then /sbin/insmod ./u-dma-buf.ko udmabuf0=201326592; fi',
        'test "$(cat /sys/class/u-dma-buf/udmabuf0/size)" = 201326592',
        'ln -sf ' + board + '/vta_hpc.bit /lib/firmware/vta_hpc.bit',
        "printf '%s\n' vta_hpc.bit > /sys/class/fpga_manager/fpga0/firmware",
        'test "$(cat /sys/class/fpga_manager/fpga0/state)" = operating',
        'export LD_LIBRARY_PATH=' + board + ' LD_PRELOAD=' + board + '/libvta.so',
        'export TVM_NUM_THREADS=1 VTA_REPLAY_POLICY=disabled VTA_QUEUE_DIAGNOSTICS=1 C3_COMPLETION_TRACE=1',
        'export AXU5EVB_DRIVER_INFO=1 AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000 AXU5EVB_DRIVER_POLL_SLEEP_NS=1000',
        'unset VTA_INSN_SUBMIT_THRESHOLD_BYTES VTA_INSN_BUFFER_BYTES VTA_UOP_BUFFER_BYTES',
        './tvm_rpc server --host=0.0.0.0 --port=9090 --port-end=9091 > rpc.log 2>&1 </dev/null &',
        'echo $! > rpc.pid', 'sleep 1', 'kill -0 "$(cat rpc.pid)"']
    script.write_text('\n'.join(lines).replace('chmod755_PLACEHOLDER', 'chmod 755 tvm_rpc ' + runner_name(r)) + '\n')
    checked(store, SSH + ['sh -s'], f'c1_boot{index}_restore', 'one_time_preparation', 60, script)
    # A new kernel boot has no inherited completion/output qualification.
    for mode in ('O', 'I', 'W', 'IW'):
        reset(store, index, boot, mode, board)
        checked(store, [PYTHON, str(FRONTEND / 'thesis_completion_board.py'), str(r),
            '--boot-index', str(index), '--boot-id', boot, '--mode', mode,
            '--board-directory', board], f'c1_boot{index}_completion_{mode}', 'functional_qualification', 60)
    log = checked(store, SSH + ['cat ' + board + '/rpc.log'],
        f'c1_boot{index}_completion_log', 'functional_qualification', 30)
    checked(store, [PYTHON, str(FRONTEND / 'thesis_completion_trace_audit.py'), str(r),
        '--boot-index', str(index), '--log', str(r / log['stdout'])],
        f'c1_boot{index}_completion_trace_audit', 'functional_qualification', 60)
    checked(store, [PYTHON, str(FRONTEND / 'thesis_completion_dependency_audit.py'), '--run-dir', str(r),
        '--boot-index', str(index), '--log', str(r / log['stdout']),
        '--output', str(r / f'analysis/boot{index}_dependency_audit.json')],
        f'c1_boot{index}_completion_dependency_audit', 'functional_qualification', 60)
    print(f'boot{index} restored: {boot}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('prepare', 'prepare-redeployment', 'collect', 'reboot-restore'))
    p.add_argument('run_dir', type=Path)
    p.add_argument('--boot-index', type=int, choices=(1, 2, 3), default=1)
    p.add_argument('--already-rebooted', action='store_true')
    p.add_argument('--reuse-qualified-boot', action='store_true')
    args = p.parse_args()
    s = RunStore(args.run_dir)
    if (s.directory / 'invalid_session.json').exists():
        raise ValueError('Invalid run')
    try:
        if args.operation == 'prepare': prepare(s)
        elif args.operation == 'prepare-redeployment': prepare(s, True)
        elif args.operation == 'collect': collect(s, args.boot_index)
        else: reboot_restore(s, args.boot_index, args.already_rebooted, args.reuse_qualified_boot)
    except BaseException as error:
        write_new_json(s.directory / 'invalid_session.json', {'operation': 'c1_' + args.operation,
                       'reason': str(error), 'all_paid_costs_retained': True})
        raise
