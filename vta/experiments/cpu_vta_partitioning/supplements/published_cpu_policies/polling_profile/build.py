from pathlib import Path
import json,os,shutil,subprocess,time
from cpu_vta_partitioning.common import read,write,sha256

ROOT=Path(__file__).resolve().parents[1]


def build(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('observer.cc')
    record=read(ROOT/'artifacts/cross_network_model_01/phase_profile/build/BUILD.json')
    tc=record['toolchain'];repo=ROOT.parents[4]
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']:raise ValueError('SDK setup changed')
    for tool in tc['tools'].values():
        if sha256(tool['path'])!=tool['sha256']:raise ValueError('SDK tool changed')
    runtime=ROOT/'artifacts/finite_service_01/prospective/yolov3_tiny416/capsule/runtime'
    library=runtime/'libvta.so';header=repo/'3rdparty/vta-hw/include/vta/driver.h'
    dis=subprocess.run(['aarch64-linux-gnu-objdump','-d','--disassemble=VTADriverProfilerStatus',str(library)],
        capture_output=True,text=True,check=True)
    (output/'status_disassembly.txt').write_text(dis.stdout)
    if '<VTADriverProfilerStatus>:' not in dis.stdout or '[x19, #48]' not in dis.stdout:
        raise ValueError('existing 64-byte driver stats ABI not established')
    fake=output/'fake.cc';harness=output/'harness.cc'
    fake.write_text('''#include <vta/driver.h>
static VTADriverProfilerStats s{};
extern "C" void VTADriverProfilerStatus(VTADriverProfilerStats* out){*out=s;}
extern "C" int VTADeviceRun(VTADeviceHandle h,vta_phy_addr_t a,uint32_t n,uint32_t w){
 if(h!=(void*)0x1234||a!=0xfedcba98U||n!=17||w!=19)return 99;
 ++s.run_calls;s.run_insns+=n;s.poll_iters+=11;s.run_total_us+=10;
 s.submit_mmio_us+=1;s.post_start_sleep_us+=2;s.poll_wait_us+=7;return 7;
}
''')
    harness.write_text('''#include <vta/driver.h>
int main(){for(int i=0;i<3;++i)if(VTADeviceRun((void*)0x1234,0xfedcba98U,17,19)!=7)return 1;
 VTADriverProfilerStats s{};VTADriverProfilerStatus(&s);
 return s.run_calls==3&&s.run_insns==51&&s.poll_iters==33?0:2;}
''')
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    recipes=['exec $CXX -std=c++17 -O2 -fPIC -shared "$1" -I"$3/3rdparty/vta-hw/include" -o "$2" -ldl -pthread',
             'exec $CXX -std=c++17 -O2 -fPIC -shared "$1" -I"$3/3rdparty/vta-hw/include" -o "$2"',
             'exec $CXX -std=c++17 -O2 "$1" -I"$3/3rdparty/vta-hw/include" -L"$4" -lfake -o "$2"']
    outputs=[output/'libdevice_phase.so',output/'libfake.so',output/'harness'];start=time.monotonic()
    for recipe,src,dest in zip(recipes,[source,fake,harness],outputs):
        args=['bash','-c','source "$1" || exit $?; shift; exec "$@"','poll-sdk',tc['sdk_setup'],
            'bash','-c',recipe,'poll-build',str(src),str(dest),str(repo),str(output)]
        p=subprocess.run(args,env=env,capture_output=True,text=True,timeout=120)
        (output/(dest.name+'.log')).write_text(p.stdout+p.stderr)
        if p.returncode:raise ValueError('poll observer build failed')
    qemu=shutil.which('qemu-aarch64-static')
    if not qemu:raise ValueError('QEMU required')
    trace=output/'qemu_trace.jsonl'
    args=[qemu,'-E','LD_PRELOAD='+str(outputs[0]),'-E','VTA_DEVICE_PHASE_TRACE='+str(trace),
        str(runtime/'ld-linux-aarch64.so.1'),'--library-path',str(output)+':'+str(runtime),str(outputs[2])]
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    (output/'qemu.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('ARM driver observer forwarding failed')
    rows=[json.loads(s) for s in trace.read_text().splitlines()]
    if rows[0]['events']!=3 or rows[0]['overflow'] or len(rows)!=4 or any(
        r['result']!=7 or r['instructions']!=17 or r['driver_run_calls']!=1 or r['driver_poll_iters']!=11
        or r['driver_run_insns']!=17 or not r['counters_monotonic'] for r in rows[1:]):
        raise ValueError('ARM driver stats forwarding/count difference failed')
    proof=dict(source_sha256=sha256(source),observer_sha256=sha256(outputs[0]),
        header_sha256=sha256(header),runtime_sha256=sha256(library),
        status_disassembly_sha256=sha256(output/'status_disassembly.txt'),toolchain=tc,
        ABI='Ejjj 32-bit address; DriverProfilerStats 64 bytes, field offsets 24/32/56',
        QEMU_forwarding_passed=True,QEMU_not_FPGA_validation=True,
        runtime_rebuilt=False,executor_rebuilt=False,stage_modules_recompiled=False,
        seconds=time.monotonic()-start)
    write(output/'BUILD.json',proof);return proof
