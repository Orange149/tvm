"""Exact native-to-TIR reachability by DAG payload propagation.

This preserves the previous instruction FIFO/token graph and material/token
binding. It propagates mapped TIR bits directly instead of building native
ancestor bits and projecting every material source for every target.
"""
from collections import defaultdict, deque

from c3_five_tradeoff_commands import decode_submissions
from c3_valid_window_tps_audit import bind_dependencies, bind_material, require


def material_ancestor_payloads(records, instruction_payloads):
    """Return exact strict ancestor TIR payloads for every native instruction."""
    edges,last = defaultdict(set),{}
    producers,consumers = defaultdict(list),defaultdict(list)
    for index,record in enumerate(records):
        require(record['index']==index,'NATIVE_INSTRUCTION_IDS_NOT_CONTIGUOUS')
        stage=record['stage']
        if stage in last:
            edges[last[stage]].add(index)
        last[stage]=index
        pop_prev,pop_next,push_prev,push_next=record['bits']
        for flag,lane,push in ((pop_prev,(stage-1,stage),False),
                               (pop_next,(stage+1,stage),False),
                               (push_prev,(stage,stage-1),True),
                               (push_next,(stage,stage+1),True)):
            if flag:
                require(min(lane)>=1 and max(lane)<=3,'invalid adjacent token lane')
                (producers if push else consumers)[lane].append(index)
    for lane in set(producers)|set(consumers):
        require(len(producers[lane])==len(consumers[lane]),'unmatched native tokens on '+str(lane))
        for source,target in zip(producers[lane],consumers[lane]):
            edges[source].add(target)
    indegree=[0]*len(records)
    for targets in edges.values():
        for target in targets:
            indegree[target]+=1
    queue=deque(index for index,degree in enumerate(indegree) if not degree)
    ancestors,visited=[0]*len(records),0
    while queue:
        index=queue.popleft();visited+=1
        payload=ancestors[index]|instruction_payloads[index]
        for target in edges[index]:
            ancestors[target]|=payload
            indegree[target]-=1
            if not indegree[target]:queue.append(target)
    require(visited==len(records),'native queue/token circular wait')
    return ancestors


def native_binding_details(events,text):
    """Same evidence as the sealed single-submit binder, with faster projection."""
    blocks=decode_submissions(text)
    require(len(blocks)==1,'NATIVE_BINDING_REQUIRES_ONE_SUBMISSION')
    records=blocks[0]
    converted,kernels=[],[]
    for eid,event in enumerate(events):
        kind=event['kind']
        value={'event':eid,'kind':kind}
        if kind=='kernel':
            uops=event['uops'];require(uops,'NATIVE_BINDING_EMPTY_KERNEL')
            loops=uops[0]['loops']
            require(all(uop['loops']==loops for uop in uops),'NATIVE_BINDING_MIXED_HARDWARE_LOOPS')
            require(0<=len(loops)<=2,'NATIVE_BINDING_HARDWARE_LOOP_ARITY')
            gemm_loops=[list(loop) for loop in loops]
            while len(gemm_loops)<2:gemm_loops.append([1,0,0,0])
            mode,reset=event['mode_reset']
            value.update(kernel=len(kernels),mode=mode,reset=reset,uops=len(uops),loops=gemm_loops)
            kernels.append(uops)
        elif kind in ('load','store'):
            if kind=='load':
                values=[event['offset'],event['x'],event['y'],event['stride'],
                        *event['pad'],event['sram'],event['domain']]
            else:
                values=[event['sram'],event['domain'],event['offset'],event['x'],event['y'],event['stride']]
            value.update(object=event['tensor'],values=values)
        elif kind in ('push','pop'):value['lane']=event['pair']
        elif kind!='sync':raise ValueError('NATIVE_BINDING_UNKNOWN_EVENT: '+kind)
        converted.append(value)
    material=bind_material(converted,kernels,records)
    dependencies,drain,pairs=bind_dependencies(converted,records)
    payloads=[0]*len(records)
    for item in material:
        payloads[item['instruction']]|=1<<item['tir_event']
    native_payloads=material_ancestor_payloads(records,payloads)
    ancestors=[0]*len(events)
    for item in material:
        ancestors[item['tir_event']]=native_payloads[item['instruction']]
    return dict(status='ACTUAL_NATIVE_FIFO_TOKEN_BINDING_PASSED',submission_count=1,
                instruction_count=len(records),ancestors=ancestors,
                material_map=material,dependency_map=dependencies,
                runtime_drain_events=drain,token_pairs=pairs,
                limits=['Control events have no synthesized ancestor entries',
                        'DumpInsn does not expose serialized UOP values or kernel signatures'])


def bind_native(events,text):
    return native_binding_details(events,text)['ancestors']
