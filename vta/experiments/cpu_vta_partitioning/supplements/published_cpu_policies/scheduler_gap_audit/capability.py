"""Zero-inference native TASKSTATS qualification, boot bound and replayable."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from cpu_vta_partitioning.common import inventory, now, read, sha256, sync_tree, write
from cpu_vta_partitioning.transport import SSH
from .common import CONFIG, PARENT, freeze


def build_probe(output):
    output=Path(output); build=output/'build'; build.mkdir(parents=True, exist_ok=True)
    src=Path(__file__).with_name('counter_probe.cc')
    tc=read(PARENT/'phase_profile/build/BUILD.json')['toolchain']
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']: raise ValueError('SDK changed')
    for v in tc['tools'].values():
        if sha256(v['path'])!=v['sha256']: raise ValueError('compiler changed')
    env=dict(os.environ); env.pop('LD_LIBRARY_PATH',None)
    command=['bash','-c','source "$1" || exit $?; shift; exec "$@"','scheduler-sdk',tc['sdk_setup'],
             'bash','-c','exec $CXX -std=c++17 -O2 "$1" -o "$2" -pthread','scheduler-compile',
             str(src), str(build/'counter_probe')]
    started=time.monotonic(); checked=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
    (build/'probe_compile.log').write_text(checked.stdout+checked.stderr)
    if checked.returncode: raise RuntimeError('counter self-test build failed')
    record=dict(binary_sha256=sha256(build/'counter_probe'), toolchain=tc,
                sources={str(src):sha256(src),str(src.with_name('taskstats_client.h')):sha256(src.with_name('taskstats_client.h'))},
                seconds=time.monotonic()-started, runtime_rebuilt=False, modules_rebuilt=False)
    write(build/'COUNTER_BUILD.json',record,replace=True)
    return record


def audit(output, config=CONFIG, *, execute_board=False):
    ssh=SSH(read(config)['board'],execute_board=execute_board); boot=ssh.boot_id()
    raw=ssh.idle_command('uname -r; uname -m; cat /proc/cmdline; '
        'zcat /proc/config.gz | grep -E "CONFIG_(SCHED_INFO|SCHEDSTATS|TASKSTATS|TASK_DELAY_ACCT|PERF_EVENTS|FTRACE)"; '
        'if test -r /proc/sys/kernel/task_delayacct; then printf "runtime_delayacct="; cat /proc/sys/kernel/task_delayacct; fi',boot=boot)
    result=dict(boot_id=boot,raw=raw,read_at=now(),CPU_queue_available_not_yet_proven=True,
                TASKSTATS='CONFIG_TASKSTATS=y' in raw,TASK_DELAY_ACCT='CONFIG_TASK_DELAY_ACCT=y' in raw,
                runtime_knob_changed=False, rebooted=False)
    path=Path(output)/'capability'/boot/'AUDIT.json'
    if path.exists():
        previous=read(path)
        if any(previous[k]!=result[k] for k in ('boot_id','raw','TASKSTATS','TASK_DELAY_ACCT')):
            raise ValueError('kernel accounting capability changed within batch')
        return previous
    write(path,result); return result


def check_self_test(rows):
    if len(rows)!=4 or [r['case'] for r in rows]!=['alone','contended','contended','sleep']:
        raise ValueError('self-test missing native scenarios')
    for r in rows:
        if any(type(r[k]) is not int or r[k]<0 for k in ('tid','wall_ns','cpu_ns','queue_ns','read_max_ns')):
            raise ValueError('invalid self-test counter')
        if r['cpu_ns']>r['wall_ns']+100000: raise ValueError('CPU clock scope wrong')
    a,b,c,s=rows
    # One second of two pinned runnable tasks should give much more than
    # timer/read noise. This is an interface self-test, not a DNN drift gate.
    noise=max(a['read_max_ns'],b['read_max_ns'],c['read_max_ns'],s['read_max_ns'],1000000)
    positive=all(r['queue_ns']>max(100000000,5*a['queue_ns'],10*noise) for r in (b,c))
    sleep_ok=s['queue_ns']<s['wall_ns']//5 and s['cpu_ns']<s['wall_ns']//10
    return dict(passed=positive and sleep_ok, contention_detected=positive,
                sleeping_is_not_CPU_queue=sleep_ok, negative_counters=False,
                zeros_are_not_no_wait=True, noise_envelope_ns=noise)


def self_test(output, config=CONFIG, *, execute_board=False):
    output=Path(output); cap=audit(output,config,execute_board=execute_board)
    if not cap['TASKSTATS'] or not cap['TASK_DELAY_ACCT']:
        return dict(status='COUNTERS_UNAVAILABLE',DNN_inferences=0,audit=cap)
    ssh=SSH(read(config)['board'],execute_board=execute_board);boot=cap['boot_id']
    result_path=output/'capability'/boot/'SELF_TEST.json'
    if result_path.exists():
        value=read(result_path)
        if sha256(value['binary'])!=value['binary_sha256']: raise ValueError('self-test binary changed')
        return value
    record=build_probe(output);binary=(output/'build/counter_probe').resolve()
    remote=ssh.config['remote_base']+'/diagnostics/scheduler_gap_audit_01/'+boot
    ssh.idle_command('mkdir -p '+shlex.quote(remote),boot=boot)
    ssh.upload(binary,remote+'/counter_probe')
    runner=PARENT/'phase_profile/resnet18/capsule/runtime'
    # The board's system loader/libstdc++ match the authenticated SDK ABI.
    script='cd '+shlex.quote(remote)+' && test "$(sha256sum counter_probe | cut -d " " -f 1)" = '+shlex.quote(record['binary_sha256'])+' && chmod 755 counter_probe && ./counter_probe'
    started=time.monotonic()
    try:
        raw=ssh.idle_command(script,boot=boot,timeout=30)
        (output/'capability'/boot/'self_test.jsonl').write_text(raw)
        rows=[json.loads(s) for s in raw.splitlines()]; validation=check_self_test(rows)
        result=dict(boot_id=boot,binary=str(binary),binary_sha256=record['binary_sha256'],
                    rows=rows,validation=validation,passed=validation['passed'],
                    seconds=time.monotonic()-started,DNN_inferences=0,runtime_knob_changed=False,
                    source_audit_sha256=sha256(output/'capability'/boot/'AUDIT.json'))
    except Exception as error:
        result=dict(boot_id=boot,passed=False,error=str(error),DNN_inferences=0,
                    binary=str(binary),binary_sha256=record['binary_sha256'])
    write(result_path,result); sync_tree(result_path.parent); return result
