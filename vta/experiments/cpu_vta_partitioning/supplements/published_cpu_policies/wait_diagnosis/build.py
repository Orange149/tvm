"""Compile the sole add-on and exercise the actual ARM runtime under QEMU."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
from cpu_vta_partitioning.common import read,sha256,write
from .common import ROOT,BUILD_RECORD,PARENT


def compile_native(work):
    work=Path(work);work.mkdir(parents=True,exist_ok=True);source=Path(__file__).parent
    record=read(BUILD_RECORD);tc=record['toolchain'];setup=Path(tc['sdk_setup'])
    if sha256(setup)!=tc['sdk_setup_sha256']:raise ValueError('SDK setup changed')
    for tool in tc['tools'].values():
        if sha256(tool['path'])!=tool['sha256']:raise ValueError('SDK compiler identity changed')
    for item in tc['setup_files'].values():
        if sha256(item['path'])!=item['sha256']:raise ValueError('SDK helper changed')
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    commands=[('adapter',source/'adapter.cc',work/'libwait_adapter.so',True),
              ('harness',source/'harness.cc',work/'harness',False)]
    for name,src,dest,shared in commands:
        recipe='exec $CXX -std=c++17 -O2 '+('-shared -fPIC ' if shared else '')+'"$1" -o "$2" -ldl -pthread'
        args=[str(src),str(dest)]
        if not shared:
            recipe+=' -L"$3" -Wl,-rpath,\'$ORIGIN/runtime\' -ltvm_runtime';args.append(str(PARENT/'runtime'))
        command=['bash','-c','source "$1" || exit $?; shift; exec "$@"','wait-sdk',str(setup),
                 'bash','-c',recipe,'wait-compile',*args]
        checked=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
        (work/(name+'_compile.log')).write_text(checked.stdout+checked.stderr)
        if checked.returncode:raise RuntimeError('wait '+name+' compilation failed')
    return dict(adapter_sha256=sha256(work/'libwait_adapter.so'),harness_sha256=sha256(work/'harness'),
                source_sha256={p.name:sha256(p) for p in commands_sources(source)},toolchain=tc)


def commands_sources(source):return [source/'adapter.cc',source/'harness.cc']


def self_test(work,*,native=True):
    work=Path(work).resolve();results=[]
    loader=PARENT/'runtime/ld-linux-aarch64.so.1';qemu=shutil.which('qemu-aarch64-static')
    if not qemu:raise RuntimeError('QEMU required for true-runtime interposition qualification')
    for policy,probe,error in [('original',True,False),('blocking',True,False),('blocking',False,False),('blocking',True,True)]:
        key=policy+'_'+str(int(probe))+'_'+str(int(error));status=work/(key+'.json');trace=work/(key+'.jsonl')
        envs=dict(TVM_NUM_THREADS='4',TVM_BIND_THREADS='0',TVM_EXCLUDE_WORKER0='1',TVM_THREAD_POOL_SPIN_COUNT='0',
                  LD_PRELOAD=str(work/'libwait_adapter.so'),VTA_WAIT_POLICY=policy,VTA_WAIT_PROBE=str(int(probe)),
                  VTA_WAIT_TRACE=str(trace),VTA_WAIT_STATUS=str(status))
        # QEMU -E affects only the guest; never preload an ARM library into QEMU itself.
        args=[qemu,*[item for k,v in envs.items() for item in ('-E',k+'='+v)],str(loader),'--library-path',
              str(PARENT/'runtime'),str(work/'harness'),str(int(error))]
        started=time.monotonic();p=subprocess.run(args,capture_output=True,text=True,timeout=60)
        (work/(key+'.log')).write_text(p.stdout+p.stderr)
        if p.returncode or not status.exists():raise ValueError('ARM runtime qualification failed: '+key)
        st=read(status)
        if (st['errors'] or st['overflow'] or st['callbacks']<4 or
            (policy=='original' and not st['yield_calls']) or
            (policy=='blocking' and not st['futex_waits'])):
            raise ValueError('ARM internal interposition or futex coverage missing')
        if probe:
            rows=[json.loads(s) for s in trace.read_text().splitlines()]
            if len(rows)!=st['records'] or sum(r['kind']==1 for r in rows)!=st['callbacks']:
                raise ValueError('ARM callback records incomplete')
        results.append(dict(mode=key,passed=True,status=st,seconds=time.monotonic()-started))
    write(work/'ARM_SELF_TEST.json',dict(passed=True,cases=results,
        runtime_sha256=sha256(PARENT/'runtime/libtvm_runtime.so'),original_runtime_used=True,
        source_binary_revision_claimed=False),replace=True)
    return results
