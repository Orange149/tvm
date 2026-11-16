#!/usr/bin/env python3
"""Prepare a flat RAM bundle with explicit source and binary bindings."""
import json
from pathlib import Path
import shutil
import tarfile
from thesis_supplement_run import RunStore, sha256, write_new_json

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent / 'report_out/stage_tile_cotuning/thesis_supplement_experiments'


def prepare(run):
    r = run.directory
    board = '/var/volatile/thesis_' + r.name
    files = {'vta_hpc.bit': ROOT / 'apps/vta_rpc/firmware/vta_hpc.bit',
             'u-dma-buf.ko': ROOT / 'u-dma-buf.ko',
             'thesis_completion_harness_shared.so': r / 'raw/thesis_completion_harness_shared.so'}
    for name in ('libtvm_runtime.so', 'libvta.so', 'tvm_rpc'):
        files[name] = r / 'raw/aarch64_build' / name
    for mode in (0, 1, 4, 5):
        files[f'mode{mode}.so'] = r / f'raw/mode{mode}.so'
    bindings = {n: {'host_path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size}
                for n, p in files.items()}
    write_new_json(r / 'analysis/board_artifact_bindings.json', bindings)
    sums = r / 'raw/board_artifacts.sha256'
    sums.open('x').write(''.join(v['sha256'] + '  ' + n + '\n' for n, v in bindings.items()))
    with tarfile.open(r / 'raw/board_bundle.tar', 'w') as archive:
        for name, path in files.items():
            archive.add(path, arcname=name)
        archive.add(sums, arcname=sums.name)
    previous = '/var/volatile/thesis_' + run.contract['retry_of'] if run.contract.get('retry_of') else '/var/volatile/thesis_20260916T004042Z_completion_93e3d8bd'
    (r / 'raw/board_prepare.sh').open('x').write('''#!/bin/sh
set -eu
expected_boot_id=${1:?Pass the boot_id obtained through authenticated SSH}
test "$(cat /proc/sys/kernel/random/boot_id)" = "$expected_boot_id"
previous=''' + previous + '''
if test -f "$previous/rpc.pid"; then
  pid=$(cat "$previous/rpc.pid")
  if test -e "/proc/$pid/exe"; then
    test "$(readlink "/proc/$pid/exe")" = "$previous/tvm_rpc"
    kill "$pid"
  fi
fi
test ! -d "''' + board + '''"
mkdir "''' + board + '''"
cat /proc/sys/kernel/random/boot_id
uname -a
cat /proc/mounts
ps | awk '/[t]vm_rpc/ {print}'
''')
    (r / 'raw/board_start.sh').open('x').write('''#!/bin/sh
set -eu
remote=''' + board + '''
cd "$remote"
tar -xf board_bundle.tar
sha256sum -c board_artifacts.sha256
chmod 755 tvm_rpc
if ! test -e /dev/udmabuf0; then
  /sbin/insmod "$remote/u-dma-buf.ko" udmabuf0=201326592
fi
test "$(cat /sys/class/u-dma-buf/udmabuf0/size)" = 201326592
ln -sf "$remote/vta_hpc.bit" /lib/firmware/vta_hpc.bit
printf 'vta_hpc.bit\n' > /sys/class/fpga_manager/fpga0/firmware
test "$(cat /sys/class/fpga_manager/fpga0/state)" = operating
cat /proc/sys/kernel/random/boot_id
cat /sys/class/u-dma-buf/udmabuf0/size /sys/class/u-dma-buf/udmabuf0/phys_addr /sys/class/u-dma-buf/udmabuf0/sync_mode
cat /sys/kernel/debug/clk/clk_summary
export LD_LIBRARY_PATH="$remote"
export LD_PRELOAD="$remote/libvta.so"
export C3_COMPLETION_TRACE=1
export TVM_NUM_THREADS=1
export VTA_REPLAY_POLICY=disabled
export VTA_QUEUE_DIAGNOSTICS=1
export VTA_RUNTIME_TRACE=0
export AXU5EVB_DRIVER_INFO=1
export AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000
export AXU5EVB_DRIVER_POLL_SLEEP_NS=1000
unset VTA_INSN_SUBMIT_THRESHOLD_BYTES VTA_INSN_BUFFER_BYTES VTA_UOP_BUFFER_BYTES
./tvm_rpc server --host=0.0.0.0 --port=9090 --port-end=9091 > "$remote/rpc.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$remote/rpc.pid"
cat "$remote/rpc.pid"
sleep 1
kill -0 "$(cat "$remote/rpc.pid")"
''')
    source_bindings = {}
    for name in ('thesis_completion_prepare.py', 'thesis_completion_operators.py',
                 'thesis_completion_board.py', 'thesis_completion_context_reset.py',
                 'thesis_completion_dependency_audit.py', 'thesis_completion_harness.cc', 'thesis_supplement_run.py'):
        path = Path(__file__).with_name(name)
        shutil.copyfile(path, r / 'raw' / ('executed_' + name))
        source_bindings[str(path)] = {'sha256': sha256(path), 'snapshot': 'raw/executed_' + name}
    write_new_json(r / 'analysis/executed_source_bindings.json', source_bindings)
    write_new_json(r / 'analysis/board_directory.json', {'path': board})
    run.event('stage01_board_artifacts_frozen',
              binding_sha256=sha256(r / 'analysis/board_artifact_bindings.json'),
              bundle_sha256=sha256(r / 'raw/board_bundle.tar'),
              source_bindings_sha256=sha256(r / 'analysis/executed_source_bindings.json'))
    print(board)


if __name__ == '__main__':
    pointer = json.loads((BASE / 'results/ACTIVE_CAMPAIGN.json').read_text())
    prepare(RunStore(pointer['run_dir']))
