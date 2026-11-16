#!/usr/bin/env python3
"""Independent raw-byte, generation, submission and allocation reconciliation."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
from c2_yolo224_prepare import write,sha


def read(path):return json.loads(Path(path).read_text())


def outputs(directory):
    result=read(directory/'result.json')
    if int((directory/'exitcode').read_text()) or result['status']!='passed':
        raise ValueError('invalid target execution: '+str(directory))
    raw=(directory/'outputs.bin').read_bytes();offset=0
    for row in result['frames']:
        if not row['complete'] or len(row['outputs'])!=8:raise ValueError('incomplete frame/output bundle')
        values=[]
        for index,entry in enumerate(row['outputs']):
            if entry['index']!=index or entry['offset']!=offset:raise ValueError('output ordering/offset mismatch')
            size=entry['bytes'];value=raw[offset:offset+size]
            if len(value)!=size:raise ValueError('truncated output')
            values.append(value);offset+=size
        yield row,values
    if offset!=len(raw):raise ValueError('unaccounted output bytes')


def reference(directory, input_count=64):
    result={};counts=defaultdict(int)
    for row,values in outputs(directory):
        index=row['input_index'];counts[index]+=1
        if index in result and result[index]!=values:raise ValueError('M0 is not repeatable: '+str(index))
        for meta,value in zip(row['outputs'],values):
            if meta['dtype']=='float32' and any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f',value)):
                raise ValueError('nonfinite M0 output')
        result[index]=values
    if set(result)!=set(range(input_count)) or min(counts.values())<2:raise ValueError('two complete cycles of frozen inputs required')
    return result


def lease_audit(result):
    slots={};acquired=published=consumed=released=0
    for event in result['events']:
        name=event['event']
        if name not in ('producer_acquired','producer_published','consumer_acquired','input_lease_released'):continue
        key=(event['edge'],event['slot']);token=(event['frame'],event['generation'])
        previous=slots.get(key,('free',None))
        if name=='producer_acquired':
            if previous[0]!='free' or (previous[1] and token[1]!=previous[1][1]+1):raise ValueError('overwrite/generation violation')
            slots[key]=('writing',token);acquired+=1
        else:
            expected={'producer_published':'writing','consumer_acquired':'ready','input_lease_released':'using'}[name]
            if previous!=(expected,token):raise ValueError('stale token or invalid bundle transition')
            state={'producer_published':'ready','consumer_acquired':'using','input_lease_released':'free'}[name]
            slots[key]=(state,token)
            if name=='producer_published':published+=1
            elif name=='consumer_acquired':consumed+=1
            else:released+=1
    n=len(result['frames'])*2
    if (acquired,published,consumed,released)!=(n,n,n,n) or any(x[0]!='free' for x in slots.values()):
        raise ValueError('not all frame bundles drained exactly once')
    for row in result['frames']:
        for edge,token in enumerate(row['tokens']):
            if token['edge']!=edge or token['frame']!=row['frame_id']:raise ValueError('frame/bundle token mismatch')
    return {'acquires':acquired,'publishes':published,'consumes':consumed,'releases':released,'all_free':True}


def metrics(directory,ref):
    result=read(directory/'result.json');frames=result['frames']
    checked_bytes=0;outputs_checked=0
    for row,values in outputs(directory):
        if values!=ref[row['input_index']]:
            mismatches=[i for i,(a,b) in enumerate(zip(values,ref[row['input_index']])) if a!=b]
            raise ValueError(f"raw mismatch in {directory.name}, frame {row['frame_id']}, outputs {mismatches}")
        checked_bytes+=sum(map(len,values));outputs_checked+=8
    lifecycle=lease_audit(result)
    measured=[r for r in frames if r['measured']]
    if len(measured)<2:raise ValueError('completion window needs two frames')
    ii=(measured[-1]['completion_ms']-measured[0]['completion_ms'])/(len(measured)-1)
    if ii<=0:raise ValueError('nonpositive completion interval')
    payloads=result.get('boundary_payload_bytes',[200704,62475])
    if len(payloads)!=2 or any(x<=0 for x in payloads):raise ValueError('invalid boundary payload contract')
    slot_payloads=[sum(s['bytes'] for s in result['slots'] if s['edge']==edge and s['slot']==0) for edge in range(2)]
    if payloads!=slot_payloads:raise ValueError('boundary payload differs from actual slot members')
    mode=result['mode'];expected=[2*x for x in payloads] if mode=='M0' else [payloads[0],0] if mode=='CP' else [0,0]
    if any(r['boundary_copied_bytes']!=expected for r in frames):raise ValueError('copy byte conservation mismatch')
    snapshots={s['phase']:s for s in result['allocation_snapshots']}
    after=snapshots['after_steady_state'];slots=result['slots']
    physical={};external=borrowed=host_staging=0
    for slot in slots:
        if slot['copy_staging']:host_staging+=slot['bytes'];continue
        start=int(slot['physical'],16);size=slot['bytes']
        for address,n in physical.items():
            if start<address+n and address<start+size:raise ValueError('slot physical allocations overlap')
        physical[start]=size
        if slot['borrowed']:borrowed+=size
        else:external+=((size+255)//256)*256
    delta=snapshots['slots_allocated']['high_water_bytes']-snapshots['graphs_loaded']['high_water_bytes']
    if delta!=external:raise ValueError('new slot allocation does not match real pool delta')
    if 'after_warmup' in snapshots and snapshots['after_warmup']['high_water_bytes']!=after['high_water_bytes']:
        raise ValueError('DMA pool grows after warmup')
    waits=defaultdict(float);pending={}
    for e in result['events']:
        key=(e['edge'],e['frame'])
        if e['event']=='producer_wait':pending[key]=e['time_ms']
        elif e['event']=='producer_acquired' and key in pending:waits[e['edge']]+=e['time_ms']-pending.pop(key)
    # Full production ends at whole-bundle publication. The old consumer ends
    # when GraphExecutor.run returns, excluding later protocol/log cleanup.
    published={(e['edge'],e['frame']):e['time_ms'] for e in result['events'] if e['event']=='producer_published'}
    reader_returns={e['frame']:e['time_ms'] for e in result['events'] if e['event']=='last_reader_return'}
    reader_starts={e['frame']:e['time_ms'] for e in result['events'] if e['event']=='last_reader_start'}
    input_returns={e['frame']:e['time_ms'] for e in result['events'] if e['event']=='input_lease_released' and e['edge']==0}
    early={e['frame']:e['time_ms'] for e in result['events'] if e['event']=='validated_early_release'}
    if mode=='E1':
        if not result['proxy_enabled'] or len(early)!=len(frames):raise ValueError('E1 did not use its reader hook exactly once per frame')
        for r in frames:
            i=r['frame_id']
            if not r['reader_done'] or not (reader_returns[i]<=input_returns[i]<=early[i]<=r['stage_end_ms'][1]):
                raise ValueError('lease was not returned after the actual last read and before consumer completion')
    if result['detailed_argument_checks']:
        if any(not r['reader_done'] or r['reader_argument_checks']<=0 for r in frames):
            raise ValueError('actual later-node argument checks are incomplete')
    if mode=='CP':
        bounds=reader_starts if result['proxy_enabled'] else {r['frame_id']:r['stage_begin_ms'][1] for r in frames}
        if any(input_returns[r['frame_id']]>bounds[r['frame_id']] for r in frames):
            raise ValueError('private-copy lease was held until after consumer entry')
    byid={r['frame_id']:r for r in frames};overlap_start=overlap_complete=0;eligible=0
    for frame in measured:
        following=byid.get(frame['frame_id']+1)
        if not following:continue
        eligible+=1
        end=frame['stage_end_ms'][1]
        overlap_start+=following['stage_begin_ms'][0]<end
        overlap_complete+=published[(0,following['frame_id'])]<end
    profile=result.get('runtime_profile',{})
    submissions=[]
    for line in (directory/'stderr').read_text().splitlines():
        if line.startswith('[VTA_QUEUE] '):submissions.append(json.loads(line[len('[VTA_QUEUE] '):]))
    for row in submissions:
        if row['timeout']!=0:raise ValueError('device timeout')
    # Status is populated by a global profiler; never query/create a main-thread queue.
    if submissions and int(profile['synchronize_calls'])!=len(submissions):
        raise ValueError('submission log does not cover all runtime synchronizations')
    return {'mode':mode,'boot_id':result['boot_id'],'frames':len(frames),'measured_frames':len(measured),
        'raw_outputs_checked':outputs_checked,'raw_bytes_checked':checked_bytes,'all_outputs_byte_exact':True,
        'lifecycle':lifecycle,'ii_ms':ii,'fps':1000/ii,
        'completion_window_ms':measured[-1]['completion_ms']-measured[0]['completion_ms'],
        'boundary_copied_bytes_per_frame':expected,
        'handoff_ms_median':statistics.median(sum(r['boundary_handoff_ms']) for r in measured),
        'input_queue_wait_ms_mean':[statistics.mean(r['input_queue_wait_ms'][i] for r in measured) for i in range(3)],
        'producer_slot_wait_ms_total':dict(waits),'pool_high_water_bytes':after['high_water_bytes'],
        'slot_external_actual_bytes':external,'borrowed_pool_payload_bytes':borrowed,
        'host_copy_staging_bytes':host_staging,'fixed_reserved_pool_bytes':after['capacity_bytes'],
        'retained_output_bytes':result['retained_output_bytes'],'steady_pool_growth_bytes':0 if 'after_warmup' in snapshots else None,
        'overlap':{'eligible_pairs':eligible,'next_production_starts_before_old_consumer_end':overlap_start,
                   'next_production_finishes_before_old_consumer_end':overlap_complete},
        'submission_records':len(submissions),'instruction_peak_bytes':max((r['insn_bytes'] for r in submissions),default=None),
        'uop_peak_bytes':max((r['uop_bytes'] for r in submissions),default=None),
        'synchronize_calls':profile.get('synchronize_calls'),'proxy_enabled':result['proxy_enabled'],
        'detailed_argument_checks':result['detailed_argument_checks'],
        'result_sha256':sha(directory/'result.json'),'outputs_sha256':sha(directory/'outputs.bin')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',default='reference_M0_01');p.add_argument('--labels',nargs='*')
    args=p.parse_args();runs=args.out/'runs';ref=reference(runs/args.reference)
    labels=args.labels or [x.name for x in runs.iterdir() if x.is_dir() and (x/'exitcode').exists() and (x/'exitcode').read_text().strip()=='0']
    results={}
    for label in labels:
        value=metrics(runs/label,ref);results[label]=value
        target=runs/label/'verified.json'
        if not target.exists():write(target,value)
        elif read(target)!=json.loads(json.dumps(value)):raise ValueError('verification changed')
        print(json.dumps({'label':label,**{k:value[k] for k in ('mode','frames','fps','boundary_copied_bytes_per_frame','pool_high_water_bytes')}}),flush=True)
    target=args.out/'analysis.json'
    if target.exists():
        number=2
        while (args.out/f'analysis_{number:02d}.json').exists():number+=1
        target=args.out/f'analysis_{number:02d}.json'
    write(target,results)


if __name__=='__main__':main()
