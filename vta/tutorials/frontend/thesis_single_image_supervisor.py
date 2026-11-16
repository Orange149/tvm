#!/usr/bin/env python3
"""Continue the approved serial campaign after its active first C1 boot."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from thesis_supplement_run import RunStore, sha256, write_new_json
from thesis_single_image_report import checkpoint

FRONTEND = Path(__file__).resolve().parent


def run(store, script, *arguments):
    source = FRONTEND / script
    store.event('supervisor_phase_start', script=script, source_sha256=sha256(source), arguments=list(arguments))
    print('START ' + script + ' ' + ' '.join(arguments), flush=True)
    subprocess.run([sys.executable, str(source), *arguments, str(store.directory)], check=True)
    store.event('supervisor_phase_end', script=script)
    checkpoint(store)


def main(directory, after_c1=False):
    store = RunStore(directory)
    r = store.directory
    if (r / 'invalid_session.json').exists() or (r / 'artifact_hashes.json').exists():
        raise ValueError('Cannot continue invalid or sealed run')
    waited_index = 3 if after_c1 else 1
    print(f'Waiting for active boot{waited_index} frozen-support collection; no board operation in parallel.', flush=True)
    deadline = time.monotonic() + 10800
    while True:
        if (r / 'invalid_session.json').exists():
            raise ValueError('Active first C1 collection failed; stop before reboot')
        try:
            first = json.loads((r / f'analysis/c1_boot{waited_index}_windows.json').read_text())
            if len(first['results']) == 16:
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError('No completed boot1 collection within the per-call derived collection budget')
        time.sleep(5)
    for index in (() if after_c1 else (2, 3)):
        source = FRONTEND / 'thesis_single_image_c1_windows.py'
        for operation in ('reboot-restore', 'collect'):
            store.event('supervisor_phase_start', script=source.name, source_sha256=sha256(source),
                        operation=operation, boot_index=index)
            print(f'START C1 {operation} boot{index}', flush=True)
            subprocess.run([sys.executable, str(source), operation, str(r), '--boot-index', str(index)], check=True)
            store.event('supervisor_phase_end', script=source.name, operation=operation, boot_index=index)
            checkpoint(store)
    run(store, 'thesis_single_image_integrated.py', 'integrate')
    # SDK environment is needed only for cross compilation, never board execution.
    shell = r / 'raw/build_current_c3_proposal.sh'
    with shell.open('x') as stream:
        stream.write('set -e\nunset LD_LIBRARY_PATH\nsource /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux\n'
            'export PYTHONPATH=/home/orange/code/tvm/python:/home/orange/code/tvm/vta/python\nunset LD_LIBRARY_PATH\n'
            + sys.executable + ' ' + str(FRONTEND / 'thesis_single_image_c3.py') + ' build "$1"\n')
    action = store.execute(['bash', str(shell), str(r)], 'build_current_B_c3_residency_proposal', 'search', 1200)
    if action['status'] != 'ok':
        raise RuntimeError('C3 contract/extraction failed: ' + action['stderr'])
    run(store, 'thesis_single_image_c3.py', 'integrate')
    run(store, 'thesis_single_image_integrated.py', 'performance')
    # Final reporter seals the timeline. Nothing may append to this run afterward.
    subprocess.run([sys.executable, str(FRONTEND / 'thesis_single_image_report.py'), 'final', str(r)], check=True)
    print('SINGLE_IMAGE_SMOKE_PASS; all raw evidence sealed; formal phases unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--after-c1', action='store_true', help='Wait for active boot3 collection, then integrate; do not request another reboot')
    args = parser.parse_args()
    try:
        main(args.run_dir, args.after_c1)
    except BaseException as error:
        r = args.run_dir.resolve()
        if not (r / 'artifact_hashes.json').exists():
            if not (r / 'invalid_session.json').exists():
                write_new_json(r / 'invalid_session.json', {'operation': 'serial_supervisor',
                    'reason': str(error), 'all_paid_costs_retained': True, 'stop_later_conditions': True})
            store = RunStore(r)
            store.event('supervisor_stopped', reason=str(error))
            checkpoint(store)
        raise
