"""Bounded progress certificates for encoded VTA instructions.

The forward-edge theorem accounts for finite fetch FIFOs and token queues:
take the earliest incomplete global instruction. Its token predecessors have
completed, no token write can saturate, and either it is queued/active or fetch
can enqueue it (all earlier instructions in that same FIFO have completed).
Consequently every reachable unfinished state has a progress transition.
Transfers/actions are assumed to finish; this is not a DDR timeout proof.

Non-forward graphs use finite-state exploration, never the theorem. Actions
retain read, pop-prev, pop-next, complete, push-prev, push-next order; an
instruction can hold a consumed token while waiting for a second token.
"""
from collections import Counter, deque


LANES=((1,2),(2,1),(2,3),(3,2))


def certify_progress(records, command_depths, token_depths, max_states=1_000_000):
    queues={s:[] for s in (1,2,3)}
    producers={lane:[] for lane in LANES}
    consumers={lane:[] for lane in LANES}
    prefix=[(0,0,0)]
    for ordinal,rec in enumerate(records):
        if rec['index']!=ordinal or rec['stage'] not in queues or len(rec['bits'])!=4:
            raise ValueError('MALFORMED_NATIVE_INSTRUCTION')
        stage=rec['stage']
        queues[stage].append(rec)
        old=list(prefix[-1]);old[stage-1]+=1;prefix.append(tuple(old))
        lanes=((stage-1,stage),(stage+1,stage),(stage,stage-1),(stage,stage+1))
        for k,(flag,lane) in enumerate(zip(rec['bits'],lanes)):
            if flag not in (0,1) or flag and lane not in LANES:
                raise ValueError('INVALID_NATIVE_TOKEN_BIT')
            if flag:
                (consumers if k<2 else producers)[lane].append(ordinal)
    if any(command_depths.get(s,0)<=0 for s in queues) or any(token_depths.get(l,0)<=0 for l in LANES):
        raise ValueError('MISSING_FINITE_QUEUE_CAPACITY')
    pairs=[]
    for lane in LANES:
        if len(producers[lane])!=len(consumers[lane]):
            raise ValueError('UNMATCHED_NATIVE_TOKENS')
        pairs += [{'lane':list(lane),'producer':p,'consumer':c}
                  for p,c in zip(producers[lane],consumers[lane])]
    forward=all(p['producer']<p['consumer'] for p in pairs)
    writes_bound=all(len(producers[l])<=token_depths[l] for l in LANES)
    common={'initial_tokens':[0]*4, 'command_depths':{str(k):v for k,v in command_depths.items()},
            'token_depths':{str(k):v for k,v in token_depths.items()},
            'token_pair_count':len(pairs), 'all_token_edges_forward':forward,
            'token_write_counts':{str(k):len(v) for k,v in producers.items()},
            'finite_action_completion_assumed':True, 'proves_cycle_timing':False}
    if forward and writes_bound:
        return {**common,'status':'FINITE_FIFO_PROGRESS_PROVED',
                'method':'earliest_incomplete_global_instruction_and_no_token_saturation',
                'states_enumerated':0}
    # State: fetch count, (completed-count, phase) for each module, tokens.
    initial=(0,((0,0),(0,0),(0,0)),(0,0,0,0))
    todo=deque([initial]);seen={initial};terminals=0
    while todo:
        fetch,actors,tokens=state=todo.popleft()
        if fetch==len(records) and all(k==len(queues[s]) and phase==0
                for s,(k,phase) in enumerate(actors,1)):
            if any(tokens):
                raise ValueError('TOKENS_LEFT_AFTER_COMPLETION')
            terminals+=1;continue
        successors=[]
        if fetch<len(records):
            stage=records[fetch]['stage'];k,phase=actors[stage-1]
            occupancy=prefix[fetch][stage-1]-k-int(phase>0)
            if occupancy<command_depths[stage]:
                successors.append((fetch+1,actors,tokens))
        for stage,(k,phase) in enumerate(actors,1):
            if k==len(queues[stage]):continue
            rec=queues[stage][k]
            new_tokens=list(tokens)
            if phase==0:
                if rec['index']>=fetch:continue
                next_phase=1
            elif phase in (1,2,4,5):
                bit={1:0,2:1,4:2,5:3}[phase]
                if rec['bits'][bit]:
                    lane=((stage-1,stage),(stage+1,stage),(stage,stage-1),(stage,stage+1))[bit]
                    idx=LANES.index(lane)
                    if phase<3:
                        if not new_tokens[idx]:continue
                        new_tokens[idx]-=1
                    else:
                        if new_tokens[idx]>=token_depths[lane]:continue
                        new_tokens[idx]+=1
                next_phase=phase+1
            else:next_phase=phase+1
            aa=list(actors)
            aa[stage-1]=(k+1,0) if next_phase==6 else (k,next_phase)
            successors.append((fetch,tuple(aa),tuple(new_tokens)))
        if not successors:
            return {**common,'status':'REACHABLE_FINITE_FIFO_DEADLOCK',
                    'method':'bounded_state_exploration','states_enumerated':len(seen),
                    'deadlock_state':state}
        for succ in successors:
            if succ not in seen:
                seen.add(succ)
                if len(seen)>max_states:
                    return {**common,'status':'ANALYSIS_STATE_LIMIT',
                            'states_enumerated':len(seen),'max_states':max_states}
                todo.append(succ)
    return {**common,'status':'FINITE_FIFO_PROGRESS_PROVED',
            'method':'exhaustive_finite_state_exploration',
            'states_enumerated':len(seen),'terminal_states':terminals}

