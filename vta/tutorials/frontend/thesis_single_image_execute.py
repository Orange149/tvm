#!/usr/bin/env python3
"""Transfer frozen smoke packages and execute reset-gated model checks."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from run_thesis_single_image_smoke import SSH, PYTHON, FRONTEND, checked
from thesis_supplement_run import RunStore, sha256, write_new_json
from thesis_completion_context_reset import reset


def prepare(store, reuse_board_from=None):
    r = store.directory
    manifest = r / 'raw/board_programs/program_manifest.json'
    if not (r / 'analysis/boot1_trace_audit.json').exists():
        raise ValueError('Missing completion trace audit')
    dependency = json.loads((r / 'analysis/boot1_dependency_audit.json').read_text())
    if dependency['status'] != 'STORE_ACK_BOUND_PASS':
        raise ValueError('Completion dependency gate failed')
    files = [r / 'raw/thesis_graph_snapshot.so', r / 'raw/vta_stage_pipeline_runner',
             r / 'raw/inputs/input.bin', manifest]
    files += sorted(p for p in (r / 'raw/board_programs').rglob('*') if p.is_file())
    files = sorted(set(files))
    bindings = {str(p.relative_to(r / 'raw')): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in files}
    write_new_json(r / 'analysis/model_deployment_bindings.json', bindings)
    sums = r / 'raw/model_artifacts.sha256'
    sums.write_text(''.join(v['sha256'] + '  ' + name + '\n' for name, v in bindings.items()))
    archive = r / 'raw/model_bundle.tar'
    with tarfile.open(archive, 'w') as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(r / 'raw')))
        tar.add(sums, arcname=sums.name)
    board = json.loads((r / 'analysis/board_directory.json').read_text())['path']
    if reuse_board_from:
        old_sums = reuse_board_from / 'raw/model_artifacts.sha256'
        old_manifest = json.loads((reuse_board_from / 'artifact_hashes.json').read_text())
        if sha256(old_sums) != old_manifest['raw/model_artifacts.sha256']['sha256'] or old_sums.read_bytes() != sums.read_bytes():
            raise ValueError('Cannot reuse a different model bundle')
        old_board = json.loads((reuse_board_from / 'analysis/board_directory.json').read_text())['path']
        checked(store, SSH + ['cp ' + old_board + '/model_bundle.tar ' + board + '/model_bundle.tar'],
                'model_bundle_ram_reuse', 'one_time_preparation', 60)
    else:
        checked(store, SSH + ['cat > ' + board + '/model_bundle.tar'], 'model_bundle_transfer', 'one_time_preparation', 120, archive)
    script = r / 'raw/model_unpack.sh'
    script.write_text('set -eu\ncd ' + board + '\ntar -xf model_bundle.tar\nsha256sum -c model_artifacts.sha256\nchmod 755 vta_stage_pipeline_runner\n')
    checked(store, SSH + ['sh -s'], 'model_bundle_unpack_and_verify', 'one_time_preparation', 60, script)


def rpc_check(store, condition):
    r = store.directory
    boot = json.loads((r / 'analysis/board_boot.json').read_text())['boot_id']
    board = json.loads((r / 'analysis/board_directory.json').read_text())['path']
    reset(store, 1, boot, condition, board)
    result = store.execute([PYTHON, str(FRONTEND / 'thesis_single_image_board_check.py'), str(r),
                            '--condition', condition], 'model_numeric_' + condition, 'functional_qualification', 60)
    # Fetch logs even on failure, preserving the failing actual submission.
    checked(store, SSH + ['cat ' + board + '/rpc.log'], 'model_log_' + condition, 'functional_qualification', 30)
    if result['status'] != 'ok':
        raise RuntimeError(condition + ': ' + result['status'] + '; stderr=' + result['stderr'])
    if condition == 'A':
        import numpy as np
        q = r / 'raw/board_numeric/Qboard'
        a = r / 'raw/board_numeric/A'
        expected = np.load(q / 'b1_Qboard_r00_s0_immediate.npy', allow_pickle=False)
        checks = []
        for i in range(10):
            actual = np.load(a / f'b1_A_r{i:02}_s2_immediate.npy', allow_pickle=False)
            from thesis_single_image_board_check import numeric
            checks.append(numeric(actual, expected, store.contract['numeric']))
        write_new_json(r / 'analysis/A_vs_Qboard_checks.json', {'checks': checks, 'extra_quantize_calls': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('prepare', 'Qboard', 'A'))
    p.add_argument('run_dir', type=Path)
    p.add_argument('--reuse-board-from', type=Path)
    args = p.parse_args()
    store = RunStore(args.run_dir)
    if (store.directory / 'invalid_session.json').exists():
        raise ValueError('Invalid run cannot resume')
    try:
        if args.operation == 'prepare':
            prepare(store, args.reuse_board_from)
        else:
            rpc_check(store, args.operation)
    except BaseException as error:
        write_new_json(store.directory / 'invalid_session.json', {'operation': args.operation,
                       'reason': str(error), 'all_paid_costs_retained': True, 'stop_later_conditions': True})
        store.event('single_image_run_invalid', operation=args.operation, reason=str(error))
        raise
