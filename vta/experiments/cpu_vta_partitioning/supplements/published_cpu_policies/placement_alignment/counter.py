"""Prepare locally; native capability test is a separate explicit operation."""
from pathlib import Path
import json
import os
import shutil
import shlex
import subprocess
from cpu_vta_partitioning.common import read,sha256,write,now
from .common import ROOT,PARENT


def check(document):
    if document.get('status')!='MEASURED':return dict(passed=False,reason='interface unavailable; not zero CPU residency')
    cpus=document['cpus'];rows=document['rows']
    if not 2<=len(cpus)<=4 or len(set(cpus))!=len(cpus) or [r['case'] for r in rows]!=['busy_first','busy_migrated','sleep']:
        raise ValueError('counter scenarios incomplete')
    checks=[]
    for row,requested_core in zip(rows,(cpus[0],cpus[1],cpus[1])):
        if row['requested_core']!=requested_core or any(type(row[k]) is not int or row[k]<0 for k in ('CPU_ns','wall_ns','read_envelope_ns')):
            raise ValueError('invalid CPU location or timing envelope')
        if len(row['before'])!=len(cpus)+1 or len(row['after'])!=len(cpus)+1:raise ValueError('per-CPU counter set incomplete')
        delta=[]
        for a,b in zip(row['before'],row['after']):
            if any(type(a[k]) is not int or type(b[k]) is not int or not 0<=a[k]<=b[k] for k in ('value','enabled','running')):
                raise ValueError('counter moved backwards')
            if b['running']-a['running']>b['enabled']-a['enabled']:
                raise ValueError('counter running time exceeds enabled time')
            delta.append(b['value']-a['value'])
        tol=max(2000000,4*row['read_envelope_ns'],int(.03*max(row['CPU_ns'],delta[0])))
        conservation=abs(sum(delta[1:])-delta[0])<=tol and abs(delta[0]-row['CPU_ns'])<=tol
        target=cpus.index(row['requested_core'])+1
        location=sum(delta[1:])-delta[target]<=tol and (delta[target]>100000000 if row['case']!='sleep' else True)
        sleep=row['case']!='sleep' or delta[0]<max(5000000,4*row['read_envelope_ns'])
        checks.append(dict(case=row['case'],passed=conservation and location and sleep,CPU_ns=row['CPU_ns'],all_CPUs_ns=delta[0],
            by_CPU_ns=dict(zip(map(str,cpus),delta[1:])),conservation=conservation,location=location,sleep_not_CPU=sleep,
            no_enabled_running_rescaling=True,tolerance_ns=tol))
    return dict(passed=all(r['passed'] for r in checks),cases=checks,scaling_forbidden='per-CPU running fraction is intentional filtering, not missing samples')


def prepare(output):
    output=Path(output).resolve();build=output/'counter';build.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('per_core_probe.cc');tc=read(ROOT/'artifacts/cross_network_model_01/phase_profile/build/BUILD.json')['toolchain']
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']:raise ValueError('SDK changed')
    for t in tc['tools'].values():
        if sha256(t['path'])!=t['sha256']:raise ValueError('compiler changed')
    for t in tc.get('setup_files',{}).values():
        if sha256(t['path'])!=t['sha256']:raise ValueError('SDK environment changed')
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    cmd=['bash','-c','source "$1" || exit $?; shift; exec "$@"','counter-sdk',tc['sdk_setup'],
         'bash','-c','exec $CXX -std=c++17 -O2 "$1" -o "$2"','counter-compile',str(source),str(build/'per_core_probe')]
    p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120)
    (build/'compile.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('counter cross compilation failed')
    runtime=PARENT/'resnet18/capsule/runtime'
    qemu=shutil.which('qemu-aarch64-static')
    if not qemu:raise ValueError('QEMU required for ABI loading')
    q=subprocess.run([qemu,str(runtime/'ld-linux-aarch64.so.1'),'--library-path',str(runtime),str(build/'per_core_probe'),'--describe'],capture_output=True,text=True,timeout=20)
    if q.returncode:raise ValueError('ARM ABI describe failed')
    described=json.loads(q.stdout)
    if described['sampling'] or described['DNN_inferences']!=0:raise ValueError('unexpected probe work')
    result=dict(prepared=True,binary_sha256=sha256(build/'per_core_probe'),source_sha256=sha256(source),
        toolchain=tc,QEMU_load_passed=True,QEMU_does_not_verify_counters=True,description=described,
        native_board_counter_validated=False,new_board_sessions=0,new_inferences=0,
        reference='https://man7.org/linux/man-pages/man2/perf_event_open.2.html')
    write(build/'PREPARED.json',result,replace=True);return result


def board_self_test(output,*,execute_board=False):
    if not execute_board:raise ValueError('new board capability test requires explicit --execute-board authorization')
    from cpu_vta_partitioning.transport import SSH
    from scheduler_gap_audit.common import CONFIG
    output=Path(output);build=output/'counter';proof=read(build/'PREPARED.json');binary=build/'per_core_probe'
    if sha256(binary)!=proof['binary_sha256']:raise ValueError('prepared binary changed')
    if sha256(Path(__file__).with_name('per_core_probe.cc'))!=proof['source_sha256']:raise ValueError('counter source changed')
    ssh=SSH(read(CONFIG)['board'],execute_board=True);boot=ssh.boot_id()
    destination=ssh.config['remote_base']+'/diagnostics/per_core_clock_01/'+boot
    ssh.idle_command('mkdir -p '+shlex.quote(destination),boot=boot);ssh.upload(binary,destination+'/probe')
    command='cd '+shlex.quote(destination)+' && test "$(sha256sum probe | cut -d " " -f 1)" = '+shlex.quote(proof['binary_sha256'])+' && chmod 755 probe && { if ./probe > counter_result.json; then :; else :; fi; cat counter_result.json; }'
    raw_path=None
    try:
        raw=ssh.idle_command(command,boot=boot,timeout=20)
        raw_path=build/'native'/boot/(now().replace(':','-')+'.json')
        raw_path.parent.mkdir(parents=True,exist_ok=True);raw_path.write_text(raw)
        doc=json.loads(raw);validation=check(doc)
    except Exception as e:
        doc=dict(status='UNAVAILABLE',error=str(e));validation=dict(passed=False,reason='interface unverified, no fallback measurement')
    value=dict(boot_id=boot,binary_sha256=proof['binary_sha256'],document=doc,validation=validation,
        DNN_inferences=0,FPGA_access=False,rebooted=False,system_configuration_changed=False,recorded_at=now())
    if raw_path is not None:value.update(raw_path=str(raw_path),raw_sha256=sha256(raw_path))
    write(build/'BOARD_SELF_TEST.json',value,replace=True);return value
