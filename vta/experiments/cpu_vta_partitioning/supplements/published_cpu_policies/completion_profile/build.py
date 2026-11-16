from pathlib import Path
import json,os,shutil,subprocess,time
from cpu_vta_partitioning.common import read,write,sha256

ROOT=Path(__file__).resolve().parents[1]


def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('observer.cc');repo=ROOT.parents[4]
    parent=read(ROOT/'artifacts/polling_profile_01/build/BUILD.json');tc=parent['toolchain']
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']:raise ValueError('SDK setup changed')
    for tool in tc['tools'].values():
        if sha256(tool['path'])!=tool['sha256']:raise ValueError('SDK tool changed')
    runtime=ROOT/'artifacts/polling_profile_02/yolov3_tiny416/capsule/runtime'
    library=runtime/'libvta.so'
    relocation=subprocess.run(['readelf','-rW',str(library)],capture_output=True,text=True,check=True).stdout
    if not all(' '+name+' ' in relocation for name in ('VTAReadMappedReg','VTAWriteMappedReg')):
        raise ValueError('deployed MMIO calls are not interposable')
    dis=subprocess.run(['aarch64-linux-gnu-objdump','-d','--demangle',str(library)],capture_output=True,text=True,check=True).stdout
    for name in ('VTAReadMappedReg@plt','VTAWriteMappedReg@plt'):
        if '<'+name+'>' not in dis:raise ValueError('deployed driver MMIO call path unavailable')
    (output/'relocations.txt').write_text(relocation);(output/'driver_disassembly.txt').write_text(dis)
    fake=output/'fake.cc';harness=output/'harness.cc'
    fake.write_text('''#include <vta/driver.h>
#include <cstdint>
extern "C" void VTAWriteMappedReg(void*,uint32_t,uint32_t);
extern "C" uint32_t VTAReadMappedReg(void*,uint32_t);
static VTADriverProfilerStats stats{};static unsigned reads=0,target=0;static int errors=0;
extern "C" void VTADriverProfilerStatus(VTADriverProfilerStats* p){*p=stats;}
extern "C" void VTAWriteMappedReg(void* h,uint32_t o,uint32_t v){if(o!=0||!h||(v!=1&&v!=129))++errors;}
extern "C" uint32_t VTAReadMappedReg(void* h,uint32_t o){if(h!=(void*)3||o!=24)++errors;return ++reads>=target?1:0;}
extern "C" int VTADeviceRun(VTADeviceHandle h,vta_phy_addr_t a,uint32_t n,uint32_t w){
 if(h!=(void*)0x1234||a!=0xfedcba98U||w!=19)return 99;
 reads=0;target=n;VTAWriteMappedReg((void*)1,0,1);
 for(uintptr_t k=2;k<5;++k)VTAWriteMappedReg((void*)k,0,129);
 unsigned limit=n==9?3:n;for(unsigned k=0;k<limit;++k)if(VTAReadMappedReg((void*)3,24)==1)break;
 ++stats.run_calls;stats.run_insns+=n;stats.poll_iters+=reads;stats.timeout_calls+=n==9;
 stats.run_total_us+=10;stats.submit_mmio_us+=1;stats.post_start_sleep_us+=2;stats.poll_wait_us+=7;
 return errors?99:n==9?1:0;
}
''')
    harness.write_text('''#include <vta/driver.h>
int main(){for(unsigned n:{5U,1U,9U})if(VTADeviceRun((void*)0x1234,0xfedcba98U,n,19)!=(n==9?1:0))return 1;return 0;}
'''.replace('#include <vta/driver.h>','#include <vta/driver.h>\n#include <initializer_list>'))
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);start=time.monotonic()
    recipes=['exec $CXX -std=c++17 -O2 -fPIC -shared "$1" -I"$3/3rdparty/vta-hw/include" -o "$2" -ldl -pthread',
             'exec $CXX -std=c++17 -O2 -fPIC -shared "$1" -I"$3/3rdparty/vta-hw/include" -o "$2"',
             'exec $CXX -std=c++17 -O2 "$1" -I"$3/3rdparty/vta-hw/include" -L"$4" -lfake -o "$2"']
    outputs=[output/'libdevice_phase.so',output/'libfake.so',output/'harness']
    for recipe,src,dest in zip(recipes,[source,fake,harness],outputs):
        args=['bash','-c','source "$1" || exit $?; shift; exec "$@"','completion-sdk',tc['sdk_setup'],
            'bash','-c',recipe,'completion-build',str(src),str(dest),str(repo),str(output)]
        p=subprocess.run(args,env=env,capture_output=True,text=True,timeout=120)
        (output/(dest.name+'.log')).write_text(p.stdout+p.stderr)
        if p.returncode:raise ValueError('completion observer build failed: '+p.stderr[-1500:])
    trace=output/'qemu_trace.jsonl'
    args=[shutil.which('qemu-aarch64-static'),'-E','LD_PRELOAD='+str(outputs[0]),
        '-E','VTA_DEVICE_PHASE_TRACE='+str(trace),str(runtime/'ld-linux-aarch64.so.1'),
        '--library-path',str(output)+':'+str(runtime),str(outputs[2])]
    p=subprocess.run(args,capture_output=True,text=True,timeout=30);(output/'qemu.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('ARM MMIO forwarding harness failed')
    rows=[json.loads(s) for s in trace.read_text().splitlines()]
    if rows[0]['events']!=3 or rows[0]['overflow'] or len(rows)!=4:raise ValueError('ARM completion event count failed')
    for e,n,result in zip(rows[1:],[5,1,3],[0,0,1]):
        if e['observed_poll_reads']!=n or e['driver_poll_iters']!=n or e['result']!=result or e['MMIO_error']:
            raise ValueError('ARM original read/write count/return forwarding failed')
        if e['start_writes']!=1 or e['auto_writes']!=3 or e['true_reads']!=int(result==0):
            raise ValueError('ARM launch/timeout/first-poll semantics failed')
    proof=dict(toolchain=tc,source_sha256=sha256(source),observer_sha256=sha256(outputs[0]),
        runtime_sha256=sha256(library),driver_header_sha256=sha256(repo/'3rdparty/vta-hw/include/vta/driver.h'),
        actual_PLT_read_write_verified=True,read_offset=24,start_value=1,auto_value=129,
        QEMU_forwarding_passed=True,QEMU_first_poll_and_timeout_passed=True,
        QEMU_not_FPGA_validation=True,extra_MMIO_reads=0,runtime_rebuilt=False,
        executor_rebuilt=False,stage_modules_recompiled=False,seconds=time.monotonic()-start)
    write(output/'BUILD.json',proof);return proof
