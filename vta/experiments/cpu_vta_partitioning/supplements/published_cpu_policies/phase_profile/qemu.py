"""Real ARM CPU modules and full frontier outputs, with optional observers."""
import os
from pathlib import Path
import shutil
import subprocess
import time

from cpu_vta_partitioning.common import read,write
from .runtime import make_task,native_command,phase_summary


def qemu_summary(raw,task,bundle):
    from cpu_vta_partitioning.measurement import frames,correctness
    from .worker import validate_workers
    affinity=read(raw/'affinity.json')
    ignored=[]
    for cfg,stage in zip(task['stages'],affinity['stages']):
        slots=sorted(stage['worker_slots'],key=lambda s:s['slot'])
        if [s['slot'] for s in slots]!=list(range(cfg['threads'])):
            raise ValueError('QEMU active logical worker coverage differs')
        if len({s['tid'] for s in slots})!=cfg['threads'] or slots[0]['tid']!=stage['caller_tid']:
            raise ValueError('QEMU logical worker TIDs differ')
        for slot,core in zip(slots,cfg['worker_core_map']):
            if slot['cpus']!=[core]:raise ValueError('QEMU active worker affinity differs')
        active={s['tid'] for s in slots}
        for helper in stage['effective_affinities']:
            if helper['tid'] in active:continue
            # QEMU itself has an emulation helper, reported by /proc but not
            # participating in any guest logical callback. This is CI-only;
            # actual board acceptance still uses the unmodified strict ledger.
            if helper['cpu_time_delta_ms']!=0:raise ValueError('nonlogical QEMU helper consumed measured CPU')
            ignored.append(helper)
    rows=frames(raw/'result.jsonl',1);correctness(rows,task,bundle,raw)
    phase=phase_summary(raw,task,affinity)
    worker=validate_workers(raw,task,bundle,affinity)
    return dict(correct=True,phase=phase,worker=worker,CI_only_idle_nonlogical_helpers=ignored)


def check(output):
    output=Path(output).resolve();qemu=shutil.which('qemu-aarch64-static')
    if not qemu:raise RuntimeError('QEMU required')
    required=[('resnet18','cpu_00_02_t3'),('resnet18','cpu_20_20_t1'),
              ('yolov3_tiny416','cpu_07_08_t3'),('yolov3_tiny416','cpu_13_13_t1')]
    results=[]
    for network,case in required:
        bundle=output/network/'capsule';raw=output/'qemu'/(network+'_'+case);raw.mkdir(parents=True,exist_ok=True)
        task=make_task(bundle,dict(network=network,scope='component',case=case,phase='qualify',probe=True,runner='phase'))
        argv,inputs=native_command(task,bundle,str(bundle),str(raw));(raw/'inputs.txt').write_text(inputs)
        at=argv.index(str(bundle)+'/runtime/ld-linux-aarch64.so.1')
        variables=argv[2:at]
        # Guest loader variables must never be preloaded into host QEMU.
        command=[qemu,*[v for item in variables for v in ('-E',item)],*argv[at:]]
        started=time.monotonic();result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        (raw/'native.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('ARM component qualification failed: '+network+' '+case+'; '+str(raw/'native.log'))
        calculated=qemu_summary(raw,task,bundle)
        if not calculated['correct'] or calculated['phase']['events']!=3:raise ValueError('ARM phase/output qualification failed')
        results.append(dict(network=network,case=case,passed=True,summary=calculated,seconds=time.monotonic()-started))
        print('ARM阶段资格通过',network,case,flush=True)
    write(output/'QEMU_QUALIFICATION.json',dict(passed=True,cases=results,board_access=False,
        capsule_ids={network:read(output/network/'capsule/manifest.json')['bundle_id']
                     for network in ('resnet18','yolov3_tiny416')}),replace=True)
    return dict(passed=True,cases=len(results))
