"""One zero-DNN process checks clock scope and scheduler-observer overhead."""
from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,now,sync_tree
from cpu_vta_partitioning.transport import SSH
from finite_service.common import freeze
from yolo_scheduler_trace.events import decode_stream,clip

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/clock_scope_audit_01'
CONFIG=ROOT/'artifacts/cross_network_model_01/board.local.json'


def prepare():
    build=OUT/'build';build.mkdir(parents=True,exist_ok=True)
    original=ROOT/'scheduler_event_audit/switch_probe.cc'
    extension=ROOT/'clock_scope_workload.cc'
    # Preserve the existing qualified event ABI. No changes to old sources.
    prefix=original.read_text().split('struct Segment {',1)[0]
    source=build/'clock_scope_probe.cc';source.write_text(prefix+extension.read_text())
    toolchain=read(ROOT/'artifacts/cross_network_model_01/phase_profile/build/BUILD.json')['toolchain']
    if sha256(toolchain['sdk_setup'])!=toolchain['sdk_setup_sha256']:
        raise ValueError('SDK identity changed')
    for tool in toolchain['tools'].values():
        if sha256(tool['path'])!=tool['sha256']:raise ValueError('toolchain changed')
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    command=['bash','-c','source "$1" || exit $?; shift; exec "$@"','clock-scope-sdk',toolchain['sdk_setup'],
             'bash','-c','exec $CXX -std=c++17 -O2 "$1" -o "$2" -pthread','clock-scope-compile',
             str(source),str(build/'clock_scope_probe')]
    run=subprocess.run(command,env=env,text=True,capture_output=True,timeout=120)
    (build/'compile.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError('clock scope self-test compilation failed')
    binary=build/'clock_scope_probe'
    sysroot=str(Path(toolchain['sdk_setup']).parent/'sysroots/aarch64-xilinx-linux')
    qemu=subprocess.run(['qemu-aarch64-static','-L',sysroot,str(binary),'--describe'],text=True,capture_output=True,timeout=15)
    if qemu.returncode or json.loads(qemu.stdout)['DNN_inferences']!=0:raise ValueError('QEMU loader check failed')
    freeze(OUT/'PLAN.json',dict(binary=str(binary),binary_sha256=sha256(binary),
        source_files={str(p):sha256(p) for p in (original,extension,source,Path(__file__))},
        toolchain=toolchain,description=json.loads(qemu.stdout),max_native_processes=1,
        original_waiting_and_affinity_preserved=True,no_DNN_or_VTA=True,
        no_runtime_or_module_changes=True,no_automatic_repeat=True,
        hypothesis='CPU-clock versus switch-span boundary differences and tracing perturbation; no fitted predictor correction',
        sources=['https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/sched/core.c',
                 'https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/events/core.c',
                 'https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/time/posix-cpu-timers.c']))
    return dict(binary=str(binary),qemu_loader='PASSED',max_native_processes=1,DNN=0)


def checked_plan():
    p=read(OUT/'PLAN.json')
    for path,h in p['source_files'].items():
        if sha256(path)!=h:raise ValueError('frozen self-test source changed')
    if sha256(p['binary'])!=p['binary_sha256']:raise ValueError('self-test binary changed')
    return p


def analyze_rows(rows):
    if [(r['case'],r['trace']) for r in rows]!=[('busy',False),('busy',True),('poll',False),('poll',True)]:
        raise ValueError('clock self-test cases differ')
    summaries=[]
    for r in rows:
        a,b=r['wall_start_ns'],r['wall_end_ns'];CPU=r['cpu_end_ns']-r['cpu_start_ns']
        if not 0<a<b or not 0<=CPU<=b-a:raise ValueError('work clock scope invalid')
        if not (r['read0_start_ns']<=r['read0_end_ns']<=a<=b<=r['read1_start_ns']<=r['read1_end_ns']):
            raise ValueError('counter reading envelope invalid')
        before,after=r['counter_before'],r['counter_after']
        if any(after[k]<before[k] for k in ('value','enabled','running')):raise ValueError('counter moved backwards')
        task=after['value']-before['value']
        row=dict(case=r['case'],trace=r['trace'],wall_ms=(b-a)/1e6,thread_CPU_ms=CPU/1e6,
            task_clock_ms=task/1e6,counter_external_wall_envelope_ns=(a-r['read0_start_ns'])+(r['read1_end_ns']-b),
            same_work_requested=True,not_a_DNN_model_calibration=True)
        if r['trace']:
            decoded=decode_stream(r,r['pid']);body=clip(decoded,a,b)
            if not body['fully_covered']:raise ValueError('self-test window not fully captured')
            outs=sum(e['direction']=='out' and a<=e['time_ns']<b for e in decoded['events'])
            row.update(switch_span_CPU_ms=body['on_CPU_ns']/1e6,switch_outs=outs,
                thread_clock_minus_switch_ms=(CPU-body['on_CPU_ns'])/1e6,
                difference_per_switch_us=(CPU-body['on_CPU_ns'])/outs/1000 if outs else None,
                record_loss=False)
        elif r['records_hex'] or r['head'] or r['overflow']:
            raise ValueError('disabled trace produced records')
        summaries.append(row)
    return summaries


def board():
    p=checked_plan()
    if (OUT/'RECEIPT.json').exists():
        receipt=read(OUT/'RECEIPT.json')
        if sha256(receipt['raw_path'])!=receipt['raw_sha256']:raise ValueError('saved self-test raw changed')
        return dict(status='ALREADY_EXECUTED',new_native_processes=0,receipt=str(OUT/'RECEIPT.json'))
    if (OUT/'STARTED.json').exists():raise ValueError('earlier native start has no receipt; inspect it, never auto-repeat')
    ssh=SSH(read(CONFIG)['board'],execute_board=True);boot=ssh.boot_id()
    remote=ssh.config['remote_base']+'/diagnostics/clock_scope_audit_01/'+boot
    ssh.idle_command('mkdir -p '+shlex.quote(remote),boot=boot)
    ssh.upload(p['binary'],remote+'/clock_scope_probe')
    command='cd '+shlex.quote(remote)+' && test "$(sha256sum clock_scope_probe | cut -d " " -f 1)" = '+shlex.quote(p['binary_sha256'])+' && chmod 755 clock_scope_probe && ./clock_scope_probe > result.jsonl 2> stderr.txt; code=$?; cat result.jsonl; cat stderr.txt >&2; exit "$code"'
    freeze(OUT/'STARTED.json',dict(boot_id=boot,remote=remote,plan_sha256=sha256(OUT/'PLAN.json'),started_at=now()))
    started=time.monotonic()
    try:
        raw=ssh.idle_command(command,boot=boot,timeout=20)
    except Exception as e:
        write(OUT/'FAILURE.json',dict(error=str(e),boot_id=boot,no_automatic_repeat=True))
        raise
    raw_path=OUT/'native.jsonl';raw_path.write_text(raw)
    summaries=analyze_rows([json.loads(l) for l in raw.splitlines()])
    receipt=dict(boot_id=boot,raw_path=str(raw_path),raw_sha256=sha256(raw_path),binary_sha256=p['binary_sha256'],
        summaries=summaries,seconds=time.monotonic()-started,DNN_inferences=0,VTA_calls=0,
        native_processes=1,system_changes=False,rebooted=False)
    freeze(OUT/'RECEIPT.json',receipt);sync_tree(OUT)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','board'))
    parser.add_argument('--execute-board',action='store_true');a=parser.parse_args()
    if a.command=='board' and not a.execute_board:parser.error('--execute-board is required')
    print(json.dumps(prepare() if a.command=='prepare' else board(),ensure_ascii=False,indent=2))
