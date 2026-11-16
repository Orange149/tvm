"""One small optional observer, with real ABI and ARM forwarding qualification."""
import os
from pathlib import Path
import shutil
import subprocess
import time

from cpu_vta_partitioning.common import read,sha256,write
from workload_model.common import ROOT,require_space
from phase_profile.build import RECORD,RUNTIME


def build(output):
    require_space();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    record=read(RECORD);tc=record['toolchain'];repo=ROOT.parents[4]
    for tool in tc['tools'].values():
        if sha256(tool['path'])!=tool['sha256']:raise ValueError('SDK tool changed')
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']:raise ValueError('SDK setup changed')
    runtime=RUNTIME/'libvta.so'
    symbols=subprocess.run(['readelf','-Ws',str(runtime)],capture_output=True,text=True,check=True).stdout
    reloc=subprocess.run(['readelf','-rW',str(runtime)],capture_output=True,text=True,check=True).stdout
    if '_ZN9VTADevice3RunEjjj' not in symbols or not any(
            'JUMP_SLOT' in line and ' VTADeviceRun ' in line for line in reloc.splitlines()):
        raise ValueError('driver ABI or interposable device call differs')
    source=Path(__file__).with_name('observer.cc');header=repo/'3rdparty/vta-hw/include/vta/driver.h'
    fake=output/'fake.cc';harness=output/'harness.cc'
    fake.write_text('#include <vta/driver.h>\nextern "C" int VTADeviceRun(VTADeviceHandle h,vta_phy_addr_t a,uint32_t n,uint32_t w) '
                    '{return h==(void*)0x1234&&a==0xfedcba98U&&n==17&&w==19 ? 7 : 99;}\n')
    harness.write_text('#include <vta/driver.h>\nint main(){for(int i=0;i<3;++i) '
                       'if(VTADeviceRun((void*)0x1234,0xfedcba98U,17,19)!=7)return 1;return 0;}\n')
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);started=time.monotonic()
    commands=[f'exec $CXX -std=c++17 -O2 -fPIC -shared "$1" -I"$3/3rdparty/vta-hw/include" -o "$2" -ldl -pthread',
              f'exec $CXX -std=c++17 -O2 -fPIC -shared "$1" -I"$3/3rdparty/vta-hw/include" -o "$2"',
              'exec $CXX -std=c++17 -O2 "$1" -I"$3/3rdparty/vta-hw/include" -L"$4" -lfake -o "$2"']
    outputs=[output/'libdevice_phase.so',output/'libfake.so',output/'harness']
    for recipe,src,dest in zip(commands,[source,fake,harness],outputs):
        cmd=['bash','-c','source "$1" || exit $?; shift; exec "$@"','device-sdk',tc['sdk_setup'],
             'bash','-c',recipe,'device-build',str(src),str(dest),str(repo),str(output)]
        result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120)
        (output/(dest.name+'.build.log')).write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('device observer/harness build failed')
    qemu=shutil.which('qemu-aarch64-static')
    if not qemu:raise RuntimeError('QEMU required for ABI forwarding qualification')
    trace=output/'qemu_device_trace.jsonl'
    cmd=[qemu,'-E','LD_PRELOAD='+str(outputs[0]),'-E','VTA_DEVICE_PHASE_TRACE='+str(trace),
         str(RUNTIME/'ld-linux-aarch64.so.1'),'--library-path',str(output)+':'+str(RUNTIME),str(outputs[2])]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    (output/'qemu.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise ValueError('ARM forwarding qualification failed')
    import json
    rows=[json.loads(line) for line in trace.read_text().splitlines()]
    if rows[0]['events']!=3 or rows[0]['overflow'] or len(rows)!=4 or any(
            r['result']!=7 or r['instructions']!=17 or r['wall_end_ns']<r['wall_start_ns'] for r in rows[1:]):
        raise ValueError('ARM observer changed arguments, results or call counts')
    proof=dict(observer_sha256=sha256(outputs[0]),source_sha256=sha256(source),
        driver_header_sha256=sha256(header),runtime_sha256=sha256(runtime),
        ABI='VTADeviceRun(void*,uint32_t,uint32_t,uint32_t); actual VTADevice::Run Ejjj and PLT audited',
        qemu_argument_return_forwarding_passed=True,qemu_device_calls=3,
        new_runtime_builds=0,new_stage_module_builds=0,seconds=time.monotonic()-started,
        source_files={str(p):sha256(p) for p in (source,Path(__file__))})
    write(output/'BUILD.json',proof,replace=True)
    return proof
