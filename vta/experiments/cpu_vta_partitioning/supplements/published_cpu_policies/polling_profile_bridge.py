"""Supply the existing optional executor's disabled queue-counter ABI.

The first batch failed before its first phase: phase recording requires
SchedulerGapSnapshot. Reuse its authenticated adapter with original waiting,
worker probing OFF and queue counters explicitly OFF. No queue values from
these sessions are treated as observations.
"""
from pathlib import Path
from types import FunctionType
import json,os,shutil,subprocess

from cpu_vta_partitioning.common import digest,inventory,read,sha256,write
from finite_service.common import freeze
from polling_profile import workflow as base
from device_phase import collect as legacy
from phase_profile.prepare import link_checked

ROOT=base.ROOT
OUTPUT=ROOT/'artifacts/polling_profile_02'
RUN=ROOT/'runs/polling_profile_02'


def sources():
    result=base.sources()
    for path in [Path(__file__),ROOT/'run_polling_profile_bridge.py',ROOT/'tests/test_polling_profile_bridge.py']:
        result[str(path)]=sha256(path)
    return result


def validate(bundle):
    result=base.validate(bundle);bundle=Path(bundle)
    bridge=read(bundle/'plans/queue_counter_support.json')
    if bridge['sources']!=sources():raise ValueError('poll bridge source changed')
    m=read(bundle/'manifest.json')
    if sha256(bundle/'runner/libwait_adapter.so')!=m['adapter_sha256']:
        raise ValueError('original queue-support adapter changed')
    return result


def prepare(output=OUTPUT):
    output=Path(output);old=read(base.OUTPUT/'PLAN.json')
    if (output/'PLAN.json').exists():
        if read(output/'PLAN.json')['sources']!=sources():raise ValueError('bridge plan changed')
        return {n:validate(output/n/'capsule') for n in old['bundle_ids']}
    ids={}
    for network in old['bundle_ids']:
        source=base.OUTPUT/network/'capsule';base.validate(source)
        target=output/network/'capsule';m=read(source/'manifest.json')
        for leaf,value in m['files'].items():link_checked(source/leaf,target/leaf,value['sha256'])
        write(target/'plans/queue_counter_support.json',dict(sources=sources(),
            original_adapter_sha256=m['adapter_sha256'],waiting='original',worker_probe=False,
            queue_counters_enabled=False,queue_fields_are_not_observations=True,
            reason='existing executor phase Span requires SchedulerGapSnapshot even when queue statistics are not being studied',
            failed_parent_plan_sha256=sha256(base.OUTPUT/'PLAN.json')))
        new={k:v for k,v in m.items() if k not in ('bundle_id','files')}
        new.update(experiment='polling_profile_02',files=inventory(target))
        new['bundle_id']=digest(new)[:24];write(target/'manifest.json',new);ids[network]=new['bundle_id']
    # Zero-DNN ARM qualification of the missing ABI with its counters OFF.
    build=output/'abi';build.mkdir(parents=True,exist_ok=True)
    source=build/'support.cc'
    source.write_text('''#include <cstdint>
#include <dlfcn.h>
int main(){auto f=reinterpret_cast<int(*)(uint64_t*)>(dlsym(RTLD_DEFAULT,"SchedulerGapSnapshot"));
 if(!f)return 1;uint64_t q[5]{1,2,3,4,5};if(f(q))return 2;
 for(auto v:q)if(v)return 3;return 0;}
''')
    tc=read(base.OUTPUT/'build/BUILD.json')['toolchain'];env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    args=['bash','-c','source "$1" || exit $?; shift; exec "$@"','poll-support-sdk',tc['sdk_setup'],
        'bash','-c','exec $CXX -std=c++17 -O2 "$1" -o "$2" -ldl','poll-support-build',str(source),str(build/'support')]
    p=subprocess.run(args,env=env,capture_output=True,text=True,timeout=120)
    (build/'compile.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('queue support ABI harness build failed')
    capsule=output/'yolov3_tiny416/capsule';runtime=capsule/'runtime';status=build/'status.json'
    preload=':'.join(map(str,[capsule/'runner/libwait_adapter.so',runtime/'libtvm_runtime.so',runtime/'libvta.so']))
    args=[shutil.which('qemu-aarch64-static'),'-E','LD_PRELOAD='+preload,
        '-E','VTA_QUEUE_COUNTERS=0','-E','VTA_QUEUE_STATUS='+str(status),
        '-E','VTA_WAIT_POLICY=original','-E','VTA_WAIT_PROBE=0',
        '-E','VTA_WAIT_STATUS='+str(build/'waiting_status.json'),
        str(runtime/'ld-linux-aarch64.so.1'),'--library-path',str(runtime),str(build/'support')]
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    (build/'qemu.log').write_text(p.stdout+p.stderr)
    if p.returncode or read(status)!=dict(enabled=False,errors=0,reads=0):
        raise ValueError('original disabled queue-support ABI qualification failed')
    waiting=read(build/'waiting_status.json')
    if waiting['policy']!='original' or waiting['probe'] or waiting['errors']:
        raise ValueError('queue support changed waiting/probing')
    freeze(output/'SUPPORT_ABI.json',dict(passed=True,QEMU_only=True,DNN_inferences=0,
        status=read(status),waiting_status=waiting,binary_sha256=sha256(build/'support'),source_sha256=sha256(source),
        adapter_sha256=read(capsule/'manifest.json')['adapter_sha256']))
    freeze(output/'PLAN.json',dict(**{k:v for k,v in old.items() if k not in ('bundle_ids','sources')},
        sources=sources(),bundle_ids=ids,experiment='polling_profile_02',
        queue_support_qualification_sha256=sha256(output/'SUPPORT_ABI.json'),
        failed_parent='polling_profile_01',parent_accepted_sessions=0,
        parent_native_startup_failures=1,parent_VTA_phase_calls=0,
        queue_fields_are_not_observations=True))
    return {n:validate(output/n/'capsule') for n in ids}


def task(bundle,spec):
    doc=base.task(bundle,spec);doc.pop('task_id');m=read(Path(bundle)/'manifest.json')
    doc.update(experiment='polling_profile_02',queue_support=dict(adapter_sha256=m['adapter_sha256'],
        enabled=False,worker_probe=False,waiting='original',queue_fields_are_not_observations=True))
    doc['task_id']=digest(doc)[:24];return doc


def command(doc,bundle,remote,job):
    fn=FunctionType(legacy.command.__code__,{**legacy.command.__globals__,'task':task})
    argv,inputs=fn(doc,bundle,remote,job)
    if doc['device_observer']:
        index=next(i for i,x in enumerate(argv) if x.startswith('LD_PRELOAD='))
        argv[index]='LD_PRELOAD='+':'.join([remote+'/runner/libdevice_phase.so',
            remote+'/runner/libwait_adapter.so',remote+'/runtime/libvta.so'])
        at=argv.index(remote+'/runtime/ld-linux-aarch64.so.1')
        argv[at:at]=['VTA_WAIT_POLICY=original','VTA_WAIT_PROBE=0','VTA_QUEUE_COUNTERS=0',
                     'VTA_WAIT_STATUS='+job+'/wait_support_status.json',
                     'VTA_QUEUE_STATUS='+job+'/queue_support_status.json']
    return argv,inputs


def summarize(raw,doc,bundle):
    value=base.summarize(raw,doc,bundle)
    if doc['device_observer']:
        status=read(Path(raw)/'queue_support_status.json')
        if status!=dict(enabled=False,errors=0,reads=0):
            raise ValueError('queue counters unexpectedly enabled or support failed')
        waiting=read(Path(raw)/'wait_support_status.json')
        if waiting['policy']!='original' or waiting['probe'] or waiting['errors'] or waiting['overflow']:
            raise ValueError('original waiting support changed behavior')
        value['queue_fields_are_not_observations']=True
    return value


def board(output=OUTPUT,run_dir=RUN,*,execute_board=False):
    fn=FunctionType(base.board.__code__,{**base.board.__globals__,'sources':sources,'validate':validate,
        'task':task,'command':command,'summarize':summarize})
    return fn(output,run_dir,execute_board=execute_board)
