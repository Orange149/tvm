"""Decode a verified sleep placement rule; do not fit DNN measurements."""
from pathlib import Path
import json
import re
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,sha256
from finite_service.common import freeze

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/compiled_sleeper_01'
KERNEL=ROOT/'artifacts/three_thread_worker_gap_01/kernel_parameter_sources'


def audit():
    d=read(OUT/'DISASSEMBLY.json');i=read(KERNEL/'IMAGE_IDENTITY.json')
    w=read(KERNEL/'COMPILED_WAKEUP.json');symbols=read(OUT/'FUNCTION_SYMBOLS.json')
    if (d['Image_sha256']!=i['Image_sha256'] or sha256(KERNEL/'Image')!=i['Image_sha256'] or
            symbols['boot_id']!=i['boot_id'] or i['running_notes_sha256'] not in symbols['raw']):
        raise ValueError('running kernel identity differs')
    row=d['functions']['enqueue_task_fair'];path=Path(row['path'])
    if sha256(path)!=row['sha256']:raise ValueError('enqueue machine code audit changed')
    text=path.read_text()
    pattern=(r'tbz\tw20, #0, 0x[0-9a-f]+\n'
        r'[0-9a-f]+:\s+[0-9a-f]+\s+adrp\tx0, (0x[0-9a-f]+)\n'
        r'[0-9a-f]+:\s+[0-9a-f]+\s+ldr\tx2, \[x26, #48\]\n'
        r'[0-9a-f]+:\s+[0-9a-f]+\s+ldr\tw1, \[x0, #(\d+)\]\n'
        r'[0-9a-f]+:\s+[0-9a-f]+\s+ldr\tx0, \[x25, #88\]\n'
        r'[0-9a-f]+:\s+[0-9a-f]+\s+sub\tx1, x2, x1, lsr #1\n'
        r'[0-9a-f]+:\s+[0-9a-f]+\s+sub\tx2, x1, x0\n'
        r'[0-9a-f]+:\s+[0-9a-f]+\s+cmp\tx2, #0x0\n'
        r'[0-9a-f]+:\s+[0-9a-f]+\s+csel\tx0, x0, x1, le\n'
        r'[0-9a-f]+:\s+[0-9a-f]+\s+str\tx0, \[x25, #88\]')
    found=list(re.finditer(pattern,text))
    if len(found)!=1:raise ValueError('sleep placement machine-code pattern not uniquely verified')
    m=found[0];address=int(m[1],16)+int(m[2]);offset=address-int(d['base'],16)
    if offset!=w['wake_load_file_offset']-4:raise ValueError('sleep rule does not load audited latency field')
    normalized=struct.unpack_from('<I',(KERNEL/'Image').read_bytes(),offset)[0]
    if normalized!=6000000:raise ValueError('compiled normalized latency changed')
    latency=normalized*w['factor'];credit=latency//2
    q=read(ROOT/'artifacts/current_model_gap_01/INPUTS.json')['quanta_ns']
    out=dict(sleeper_credit_ns=credit,compiled_normalized_latency_ns=normalized,
        configured_latency_ns=latency,factor=w['factor'],online_cores=w['online_cores'],
        wakeup_granularity_ns=w['wakeup_granularity_ns'],
        old_sleeper_credit_ns={str(n):int(v)*int(n)/2 for n,v in q.items()},
        emitted_rule='max(previous_vruntime, minimum_vruntime - configured_latency/2)',
        applies_on_ENQUEUE_WAKEUP=True,GENTLE_FAIR_SLEEPERS_shift_confirmed=True,
        exact_machine_code=m[0],load_file_offset=offset,
        derivation='compiled initializer and emitted placement arithmetic; four online CPUs as recorded',
        current_kernel_memory_value_not_directly_read=True,assumption_no_unrecorded_kernel_memory_change=True,
        Image_sha256=i['Image_sha256'],build_id=i['build_id'],boot_id=i['boot_id'],
        DNN_FPS_used=False,new_native_workload_sessions=0,source_sha256=sha256(__file__),
        source='https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/sched/fair.c',
        source_function='place_entity, inlined into enqueue_task_fair',
        provenance={str(OUT/'DISASSEMBLY.json'):sha256(OUT/'DISASSEMBLY.json'),str(KERNEL/'COMPILED_WAKEUP.json'):sha256(KERNEL/'COMPILED_WAKEUP.json')})
    freeze(OUT/'COMPILED_SLEEPER.json',out);return out


if __name__=='__main__':
    d=audit();print(json.dumps({k:v for k,v in d.items() if k!='exact_machine_code'},ensure_ascii=False,indent=2))
