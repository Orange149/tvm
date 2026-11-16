#!/usr/bin/env python3
"""Reconcile actual driver allocation addresses, independently of slot views."""
import argparse,json,re
from pathlib import Path
from c2_yolo224_prepare import write,sha

ALLOC=re.compile(r'Alloc size=(\d+) aligned=(\d+) vaddr=(0x[0-9a-f]+) paddr=(0x[0-9a-f]+) used=(\d+)/(\d+)')
H2D=re.compile(r'CopyDataFromTo H2D dst=(0x[0-9a-f]+) dst_phy=(0x[0-9a-f]+) src=(0x[0-9a-f]+) size=(\d+)')
D2H=re.compile(r'CopyDataFromTo D2H dst=(0x[0-9a-f]+) src=(0x[0-9a-f]+) src_phy=(0x[0-9a-f]+) size=(\d+)')


def read(p):return json.loads(p.read_text())


def reconcile(out, prefix='allocation_', selected_modes=None):
    modes={}
    for mode in selected_modes or ('M0','M1','M2','M3','manual','L1','L2','CP','E1'):
        directory=out/'runs'/(prefix+mode)
        if not read(directory/'verified.json')['all_outputs_byte_exact']:raise ValueError('unverified allocation run')
        contract=read(directory/'contract.json');result=read(directory/'result.json')
        if contract['environment']['AXU5EVB_DRIVER_INFO']!='1':raise ValueError('allocation logging disabled')
        text=(directory/'stderr').read_text();snap={s['phase']:s for s in result['allocation_snapshots']}
        loaded=[m[1] for line in text.splitlines() if (m:=re.search(r'calling init:\s+(/\S+)',line))]
        root=read(out/'board_state.json')['remote']+'/package/'
        if not all(root+name in loaded for name in ('libtvm_runtime.so','libvta.so')):
            raise ValueError('dynamic loader did not record the actual frozen runtime paths')
        after=snap['after_steady_state'];graph_count=snap['graphs_loaded']['allocation_count']
        slot_count=snap['slots_allocated']['allocation_count'];rows=[];offset=0
        for index,match in enumerate(ALLOC.finditer(text)):
            size,aligned,cpu,physical,used,capacity=match.groups()
            size=int(size);aligned=int(aligned);used=int(used);physical=int(physical,16);cpu=int(cpu,16)
            if aligned!=(size+255)//256*256 or used!=offset+aligned or int(capacity)!=after['capacity_bytes']:
                raise ValueError('driver allocation size/offset conservation failed')
            if cpu!=after['virtual_base']+offset or physical!=after['physical_base']+offset:
                raise ValueError('actual allocation virtual/physical translation differs')
            phase='graph_pool' if index<graph_count else 'external_slot' if index<slot_count else 'runtime'
            rows.append({'allocation':index,'phase':phase,'requested_bytes':size,'aligned_bytes':aligned,
                'virtual':hex(cpu),'physical':hex(physical),'offset':offset,'pool_high_water_after':used})
            offset=used
        if len(rows)!=after['allocation_count'] or offset!=after['high_water_bytes']:
            raise ValueError('driver trace does not cover the complete actual allocation ledger')
        runtime=rows[slot_count:]
        env=contract['environment'];insn=int(env['VTA_INSN_BUFFER_BYTES']);uop=int(env['VTA_UOP_BUFFER_BYTES'])
        # Frozen CommandQueue::InitSpace initializes UOP then instruction backing.
        if len(runtime)!=4 or [x['requested_bytes'] for x in runtime[:2]]!=[uop,insn]:
            raise ValueError('unexpected command backing allocation order/count')
        runtime[0]['phase']='UOP_backing';runtime[1]['phase']='instruction_backing'
        for row in runtime[2:]:row['phase']='runtime_workspace'
        for slot in result['slots']:
            if slot['copy_staging']:
                address=int(slot['virtual'],16)
                if after['virtual_base']<=address<after['virtual_base']+after['capacity_bytes']:
                    raise ValueError('ordinary CPU staging unexpectedly aliases the coherent pool')
                continue
            matches=[x for x in rows if int(x['physical'],16)==int(slot['physical'],16)]
            if len(matches)!=1:raise ValueError('slot view does not resolve to exactly one physical allocation')
            owner=matches[0]
            if slot['bytes']>owner['requested_bytes'] or owner['phase']!=('graph_pool' if slot['borrowed'] else 'external_slot'):
                raise ValueError('slot ownership/extent classification differs')
        # The initial VTA graph allocation is input SID0, as frozen SetupStorage
        # allocates pools in SID order. Plain CPU graph pools do not use this ABI.
        payloads=result.get('boundary_payload_bytes',[200704,62475])
        if rows[0]['requested_bytes']!=payloads[0]:raise ValueError('input pool SID0 identity changed')
        host_to_input=[m for m in H2D.finditer(text) if int(m[2],16)==int(rows[0]['physical'],16)]
        to_host=list(D2H.finditer(text));n=len(result['frames'])
        expected=n if mode in ('M0','CP') else 0
        if len(host_to_input)!=expected or any(int(m[4])!=payloads[0] for m in host_to_input):
            raise ValueError('actual driver input-copy ledger differs from boundary policy')
        if mode=='M0':
            staging={int(x['virtual'],16):x for x in result['slots'] if x['edge']==1}
            if len(to_host)!=2*n or sum(int(m[4]) for m in to_host)!=payloads[1]*n:
                raise ValueError('actual dual-head get_output copies are incomplete')
            if any(int(m[1],16) not in staging or int(m[4])!=staging[int(m[1],16)]['bytes'] for m in to_host):
                raise ValueError('driver get_output did not target the frozen CPU staging slots')
        elif to_host:raise ValueError('shared outputs unexpectedly copied through the driver')
        modes[mode]={'all_outputs_byte_exact':True,'frames':n,'allocations':rows,
            'physical_allocations_deduplicated':len(rows),'high_water_bytes':offset,
            'command_backing_bytes':insn+uop,'workspace_bytes':sum(x['aligned_bytes'] for x in runtime[2:]),
            'actual_driver_boundary_H2D_payload_bytes':sum(int(m[4]) for m in host_to_input),
            'actual_driver_boundary_D2H_payload_bytes':sum(int(m[4]) for m in to_host),
            'loaded_init_objects':loaded,
            'stderr_sha256':sha(directory/'stderr'),'result_sha256':sha(directory/'result.json')}
    target=out/'allocation_trace_reconciliation.json'
    value={'status':'passed','boot_id':read(out/'board_state.json')['boot_id'],'modes':modes,
        'scope':'actual driver allocation/physical ranges; diagnostics excluded from formal timing',
        'CPU_heap_scope':'ordinary CPU staging payload/views recorded; libc metadata and total RSS not measured',
        'generator_sha256':sha(Path(__file__))}
    if not target.exists():write(target,value)
    elif read(target)!=value:raise ValueError('physical allocation reconciliation changed')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=reconcile(a.out);print(json.dumps({'status':r['status'],'modes':len(r['modes'])}))
