#!/usr/bin/env python3
"""Reload frozen firmware before a fresh stage01 RPC child, with a boot-bound receipt."""
import argparse
import json
from pathlib import Path
import shlex
from thesis_supplement_run import RunStore, sha256, write_new_json


RPC_CHILDREN_CHECK = '''rpc_children() {
  for thesis_status in /proc/[0-9]*/status; do
    if ! thesis_ppid=$(awk '/^PPid:/ {print $2; exit}' "$thesis_status" 2>/dev/null); then
      test ! -e "$thesis_status" || return 2
      continue
    fi
    test -n "$thesis_ppid" || return 2
    if test "$thesis_ppid" = "$pid"; then
      thesis_child=${thesis_status#/proc/}
      printf '%s\\n' "${thesis_child%/status}"
    fi
  done
}
'''


def reset(run, index, boot_id, condition, board):
    required = run.contract['completion'].get('execution_context_reset', {})
    if not required.get('required_before_each_new_command_queue'):
        raise ValueError('Use a new contract declaring the context reset protocol')
    session = f'b{index}_{condition}'
    if any(e['event'] == 'context_reset_ready' and e.get('session_id') == session for e in run.events()):
        raise ValueError('Context reset receipt already exists; do not overwrite or silently retry')
    bindings = json.loads((run.directory / 'analysis/board_artifact_bindings.json').read_text())
    for name in ('vta_hpc.bit', 'tvm_rpc'):
        if sha256(bindings[name]['host_path']) != bindings[name]['sha256']:
            raise ValueError('Deployment artifact changed')
    script = '''#!/bin/sh
set -eu
test "$(cat /proc/sys/kernel/random/boot_id)" = ''' + shlex.quote(boot_id) + '''
remote=''' + shlex.quote(board) + '''
test -f "$remote/rpc.pid"
pid=$(cat "$remote/rpc.pid")
test "$(readlink "/proc/$pid/exe")" = "$remote/tvm_rpc"
# The parent remains idle; a child owns any RPC condition. Do not reprogram with
# a live child or assume that host process exit alone cleared FPGA stream state.
# /proc/.../children depends on CONFIG_CHECKPOINT_RESTORE. Use the universally
# available status PPid field instead, and fail closed on non-transient read errors.
''' + RPC_CHILDREN_CHECK + '''
attempt=0
thesis_children=$(rpc_children)
while test -n "$thesis_children"; do
  attempt=$((attempt + 1))
  test "$attempt" -le 5
  sleep 1
  thesis_children=$(rpc_children)
done
test "$(readlink "/proc/$pid/exe")" = "$remote/tvm_rpc"
test "$(sha256sum "$remote/vta_hpc.bit" | cut -d ' ' -f 1)" = ''' + bindings['vta_hpc.bit']['sha256'] + '''
ln -sf "$remote/vta_hpc.bit" /lib/firmware/vta_hpc.bit
printf 'vta_hpc.bit\n' > /sys/class/fpga_manager/fpga0/firmware
test "$(cat /sys/class/fpga_manager/fpga0/state)" = operating
test "$(cat /proc/sys/kernel/random/boot_id)" = ''' + shlex.quote(boot_id) + '''
printf 'THESIS_CONTEXT_RESET_OK ''' + session + ' ' + boot_id + ' ' + bindings['vta_hpc.bit']['sha256'] + ''' operating\n'
'''
    path = run.directory / f'raw/context_reset_{session}.sh'
    with path.open('x') as stream:
        stream.write(script)
    argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=/home/orange/.ssh/known_hosts',
            '-o', 'HostKeyAlgorithms=+ssh-rsa', '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa',
            'root@192.168.1.247', 'sh -s']
    result = run.execute(argv, f'context_reset_{session}', 'one_time_preparation', 60, path)
    expected = f'THESIS_CONTEXT_RESET_OK {session} {boot_id} {bindings["vta_hpc.bit"]["sha256"]} operating'
    if result['status'] != 'ok' or expected not in (run.directory / result['stdout']).read_text().splitlines():
        run.event('context_reset_failed', session_id=session, boot_id=boot_id, action_id=result['action_id'])
        raise ValueError('Reset failed or boot-bound receipt missing; no RPC calls allowed')
    receipt = {'session_id': session, 'boot_id': boot_id, 'board_directory': board,
               'bitstream_sha256': bindings['vta_hpc.bit']['sha256'],
               'action_id': result['action_id'], 'reset_script_sha256': sha256(path),
               'action_stdout': result['stdout'], 'action_stdout_sha256': result['stdout_sha256'],
               'hardware_state': 'operating', 'preceding_rpc_children_absent': True,
               'scope': 'Successful reprogramming operation, not direct FPGA payload/FIFO occupancy readback'}
    relative = f'analysis/context_reset_{session}.json'
    write_new_json(run.directory / relative, receipt)
    run.event('context_reset_ready', session_id=session, boot_id=boot_id,
              receipt=relative, receipt_sha256=sha256(run.directory / relative))
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir')
    p.add_argument('--boot-index', type=int, required=True)
    p.add_argument('--boot-id', required=True)
    p.add_argument('--mode', choices=('O', 'I', 'W', 'IW'), required=True)
    p.add_argument('--board-directory', required=True)
    args = p.parse_args()
    print(json.dumps(reset(RunStore(args.run_dir), args.boot_index, args.boot_id, args.mode, args.board_directory)))
