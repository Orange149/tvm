#!/usr/bin/env python3
"""Reuse sealed build artifacts, preserving costs and collecting new samples."""
import argparse
import json
from pathlib import Path
import shutil
import uuid
from run_thesis_single_image_smoke import SSH, PYTHON, FRONTEND, checked
from thesis_completion_context_reset import reset, RPC_CHILDREN_CHECK
from thesis_supplement_run import RunStore, sha256, write_new_json


def retry(store, previous, programs_dir='board_programs'):
    r = store.directory
    manifest = json.loads((previous / 'artifact_hashes.json').read_text())
    paths = list((previous / 'raw' / programs_dir).rglob('*'))
    paths += list((previous / 'raw').glob('*.npy'))
    paths += [previous / 'raw/thesis_graph_snapshot.so', previous / 'raw/vta_stage_pipeline_runner']
    if (previous / 'raw/vta_stage_pipeline_runner_raw_integer').exists():
        paths.append(previous / 'raw/vta_stage_pipeline_runner_raw_integer')
    reused = []
    for p in paths:
        if not p.is_file():
            continue
        relative = p.relative_to(previous)
        if sha256(p) != manifest[str(relative)]['sha256']:
            raise ValueError('Sealed artifact changed: ' + str(p))
        target_relative = Path('raw/board_programs') / p.relative_to(previous / 'raw' / programs_dir) if p.is_relative_to(previous / 'raw' / programs_dir) else relative
        target = r / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        reused.append({'source': str(p), 'target': str(relative), 'sha256': sha256(target)})
    write_new_json(r / 'analysis/reused_build_artifacts.json', {
        'source_run': str(previous), 'source_manifest_sha256': sha256(previous / 'artifact_hashes.json'),
        'files': reused, 'costs': 'original build costs retained in source run, not claimed zero cold preparation',
        'samples': 'all host and board observations recollected in current run'})
    bindings = json.loads((previous / 'analysis/board_artifact_bindings.json').read_text())
    write_new_json(r / 'analysis/board_artifact_bindings.json', bindings)
    old_board = json.loads((previous / 'analysis/board_directory.json').read_text())['path']
    board = '/var/volatile/thesis_' + r.name
    write_new_json(r / 'analysis/board_directory.json', {'path': board})
    result = checked(store, SSH + ['cat /proc/sys/kernel/random/boot_id'], 'retry_boot_identity', 'one_time_preparation', 30)
    boot = (r / result['stdout']).read_text().strip()
    uuid.UUID(boot)
    write_new_json(r / 'analysis/board_boot.json', {'boot_id': boot})
    # Verify the old parent has no children; stop only this identified idle server.
    start = r / 'raw/retry_restore_ram.sh'
    original = (previous / 'raw/board_restore.sh').read_text()
    prefix = 'set -eu\nold=' + old_board + '\nnew=' + board + '\npid=$(cat "$old/rpc.pid")\ntest "$(readlink /proc/$pid/exe)" = "$old/tvm_rpc"\n'
    prefix += RPC_CHILDREN_CHECK + '\ntest -z "$(rpc_children)"\nkill "$pid"\ntest ! -d "$new"\nmkdir "$new"\ncp "$old/board_bundle.tar" "$new/board_bundle.tar"\n'
    start.write_text(prefix + original.replace(old_board, board))
    checked(store, SSH + ['sh -s'], 'retry_restore_ram_and_runtime', 'one_time_preparation', 60, start)
    for mode in ('O', 'I', 'W', 'IW'):
        reset(store, 1, boot, mode, board)
        checked(store, [PYTHON, str(FRONTEND / 'thesis_completion_board.py'), str(r), '--boot-index', '1',
                        '--boot-id', boot, '--mode', mode, '--board-directory', board],
                'retry_completion_' + mode, 'functional_qualification', 60)
    result = checked(store, SSH + ['cat ' + board + '/rpc.log'], 'completion_log_fetch', 'functional_qualification', 30)
    log = r / result['stdout']
    checked(store, [PYTHON, str(FRONTEND / 'thesis_completion_trace_audit.py'), str(r), '--boot-index', '1', '--log', str(log)],
            'retry_completion_trace_audit', 'functional_qualification', 60)
    checked(store, [PYTHON, str(FRONTEND / 'thesis_completion_dependency_audit.py'), '--run-dir', str(r), '--boot-index', '1',
                    '--log', str(log), '--output', str(r / 'analysis/boot1_dependency_audit.json')],
            'retry_completion_dependency_audit', 'functional_qualification', 60)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    p.add_argument('previous_run', type=Path)
    p.add_argument('--programs-dir', default='board_programs')
    args = p.parse_args()
    store = RunStore(args.run_dir)
    try:
        retry(store, args.previous_run, args.programs_dir)
    except BaseException as error:
        write_new_json(store.directory / 'invalid_session.json', {'operation': 'retry', 'reason': str(error), 'all_paid_costs_retained': True})
        raise
