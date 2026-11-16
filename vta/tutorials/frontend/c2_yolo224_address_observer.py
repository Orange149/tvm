#!/usr/bin/env python3
"""Build an observation-only helper and independently reconcile actual ranges."""
import argparse,json,os,re,shutil,subprocess
from pathlib import Path
from c2_yolo224_prepare import HERE,write,sha
from c3_input_early_release_audit import instructions
from audit_shared_buffer_storage import tensor_bytes


def build(out):
    evidence=out/'reader_evidence/compute_arm.txt'
    code=instructions(evidence.read_text())
    calls=[x['address']+4 for x in code if x['mnemonic']=='bl' and 'VTABufferCPUPtr' in x['operands']]
    if len(calls)!=2:raise ValueError('expected two CPU pointer resolutions')
    compiler=shutil.which('aarch64-xilinx-linux-g++')
    target=out/'c2_yolo224_address_observer.so'
    command=[compiler,'--sysroot='+os.environ['SDKTARGETSYSROOT'],'-std=c++17','-O2','-fPIC','-shared',
        str(HERE/'c2_yolo224_address_observer.cc'),'-ldl','-pthread','-o',str(target)]
    result=subprocess.run(command,check=True,capture_output=True,text=True)
    write(out/'address_observer_config.json',{'return_pcs':[hex(c) for c in calls],
        'helper_sha256':sha(target),'source_sha256':sha(HERE/'c2_yolo224_address_observer.cc'),
        'compute_evidence_sha256':sha(evidence),'command':command,
        'scope':'qualification only; transparent call-through, file logging, no performance claim'})


def verify(out,label):
    directory=out/'runs'/label
    result=json.loads((directory/'result.json').read_text())
    captures=[json.loads(x) for x in (directory/'reader_addresses.jsonl').read_text().splitlines()]
    if len(captures)!=2*len(result['frames']):raise ValueError('not all actual reader calls captured')
    slots=[x for x in result['slots'] if x['edge']==0]
    starts={x['frame']:x['time_ms'] for x in result['events'] if x['event']=='last_reader_start'}
    ends={x['frame']:x['time_ms'] for x in result['events'] if x['event']=='last_reader_return'}
    proof=json.loads((out/'last_reader_certificate.json').read_text())
    output_bytes=tensor_bytes(proof['reader_output_shape'],proof['reader_output_dtype'])
    input_bytes=tensor_bytes(proof['input_shape'],proof['input_dtype'])
    pool=next(x for x in result['allocation_snapshots'] if x['phase']=='after_steady_state')
    rows=[]
    for i,frame in enumerate(result['frames']):
        pair=captures[2*i:2*i+2]
        if [x['pair'] for x in pair]!=[i,i] or [x['site'] for x in pair]!=[0,1]:raise ValueError('callback pair order mismatch')
        expected=next(x for x in slots if x['slot']==frame['tokens'][0]['slot'])
        match=[x for x in pair if int(x['physical'],16)==int(expected['physical'],16)]
        if len(match)!=1:raise ValueError('actual CPU reader did not access the shared slot')
        source=match[0];destination=next(x for x in pair if x is not source)
        if int(source['cpu'],16)!=int(expected['virtual'],16):raise ValueError('virtual address differs from slot')
        for kind in ('cpu','physical'):
            a=int(source[kind],16);b=int(destination[kind],16)
            if a<b+output_bytes and b<a+input_bytes:raise ValueError('actual input/output ranges alias')
        for capture,size in ((source,input_bytes),(destination,output_bytes)):
            cpu=int(capture['cpu'],16);physical=int(capture['physical'],16)
            offset=cpu-pool['virtual_base']
            if offset<0 or offset+size>pool['high_water_bytes'] or physical!=pool['physical_base']+offset:
                raise ValueError('actual reader range is outside allocated physical pool or translation differs')
            if not physical or cpu%256 or physical%256:raise ValueError('actual reader range alignment invalid')
        for x in pair:
            if x['tid']!=result['affinity'][1]['caller_tid']:raise ValueError('reader ran on an unexpected actor')
            if not x['real_resolver'].endswith('/package/libvta.so'):raise ValueError('observer resolved an unknown runtime')
            if not starts[frame['frame_id']]-.01<=x['time_ns']/1e6<=ends[frame['frame_id']]+.01:
                raise ValueError('address capture was outside actual CPU reader invocation')
        rows.append({'frame':frame['frame_id'],'generation':frame['tokens'][0]['generation'],
            'input':source,'packed_output':destination,'input_bytes':input_bytes,'output_bytes':output_bytes,
            'later_argument_checks':frame['reader_argument_checks'],'virtual_and_physical_disjoint':True})
    value={'status':'passed','qualification_label':label,
        'frames':len(rows),'captures':len(captures),'rows':rows,'raw_capture_sha256':sha(directory/'reader_addresses.jsonl'),
        'observer_sha256':sha(out/'c2_yolo224_address_observer.so'),
        'scope':'observation-only qualification; excluded from every performance comparison'}
    target=out/'dynamic_address_certificate.json'
    if not target.exists():write(target,value)
    elif json.loads(target.read_text())!=value:raise ValueError('address certificate reconciliation changed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('build','verify'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--label',default='address_observed_E1')
    a=p.parse_args();build(a.out) if a.phase=='build' else verify(a.out,a.label)
