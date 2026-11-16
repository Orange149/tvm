"""V4 bounded native proof: zero-UOP NOPs terminate greedy LOAD search.

No target is executed here. Host cache definitions persist within one caller-
bound module/session; successful Sync clears per-submit SRAM residency and
DRAM serialization. Unknown AutoSync cuts are rejected, never guessed.
"""
import hashlib
from pathlib import Path
import re

from c3_valid_window_uop_values import (_normalize, _initializers, _instructions,
                                        _expected_loops, _require)
from c3_official_window_native_binding import native_binding_details
from c3_five_tradeoff_commands import decode_submissions

HEADER = Path(__file__).with_name('c3_official_window_runtime_uop_queue.h')
DEPTH = 8192
LIMIT = 16
SCHEMA = 'c3_official_window_uop_cache_definitions_v1'


def _simulate(kernels, definitions, nop_barriers=None):
    """Exact supported 32 MiB queue: values, wrap, eviction, and safe greedy."""
    objects = {key:{'values':value,'begin':0,'end':0} for key,value in definitions.items()}
    cache, dram, instructions = [], [], []
    cache_idx, sram_end, wraps = 0, 0, 0
    guarded = 0
    nop_barriers = nop_barriers or {}
    for kernel_index, kernel in enumerate(kernels):
        for _ in range(nop_barriers.get(kernel_index, 0)):
            instructions.append({'kind':'nop'})
        obj = objects[kernel['identity']]
        if obj['begin'] == obj['end']:
            size = len(obj['values'])
            _require(0 < size <= DEPTH, 'SINGLE_KERNEL_UOP_CAPACITY')
            prev = cache[cache_idx-1] if cache_idx else None
            if prev and prev['begin'] != prev['end'] and prev['values'] == obj['values']:
                obj['begin'],obj['end'] = prev['begin'],prev['end']
                cache.insert(cache_idx,obj)
                cache_idx += 1
            else:
                if not any(item is obj for item in dram):
                    dram.append(obj)
                begin = 0
                if sram_end + size > DEPTH:
                    cache_idx, sram_end = 0, size
                    wraps += 1
                else:
                    begin = sram_end
                    sram_end += size
                evict_begin = cache_idx
                while cache_idx < len(cache):
                    old = cache[cache_idx]
                    if old['begin'] >= sram_end:
                        break
                    old['begin'] = old['end'] = 0
                    cache_idx += 1
                obj['begin'],obj['end'] = begin,sram_end
                cache.insert(cache_idx,obj)
                del cache[evict_begin:cache_idx]
                cache_idx = evict_begin+1
                source = 0
                for item in dram:
                    if item is obj:
                        break
                    source += len(item['values'])
                prior_idx = next((i for i in range(len(instructions)-1,-1,-1)
                                  if instructions[i]['kind'] in ('load','nop')),None)
                prior = instructions[prior_idx] if prior_idx is not None and instructions[prior_idx]['kind'] == 'load' else None
                conflict = prior is not None and any(
                    item['kind'] == 'compute' and item['begin'] < sram_end and begin < item['end']
                    for item in instructions[prior_idx+1:])
                extend = prior and prior['begin']+prior['size'] == begin and \
                    prior['source']+prior['size'] == source and cache_idx > 1 and \
                    prior['begin']+prior['size']+size < DEPTH
                if extend and conflict:
                    guarded += 1
                if extend and not conflict:
                    prior['size'] += size
                else:
                    instructions.append({'kind':'load','source':source,'begin':begin,'size':size})
        instructions.append({'kind':'compute','begin':obj['begin'],'end':obj['end']})
    packed = [value for item in dram for value in item['values']]
    _require(len(packed)*4 <= (1 << 25),'UNSUPPORTED_AUTOSYNC_UOP_SERIALIZATION')
    return [item for item in instructions if item['kind'] != 'nop'],packed,wraps,guarded


def _fragments(events, text, ledger):
    _require(ledger.get('schema') == 'c3_official_window_submit_ledger_v1','SUBMIT_SCHEMA')
    _require(ledger.get('mode') in ('detailed','minimal'),'SUBMIT_COLLECTION_MODE')
    _require(ledger.get('overflow') is False,'SUBMIT_LEDGER_OVERFLOW')
    rows = ledger.get('submits',[])
    _require(0 < len(rows) <= LIMIT and ledger.get('recorded_submits') ==
             ledger.get('attempted_submits') == len(rows),'SUBMIT_CENSUS')
    blocks = list(re.finditer(r'There are (\d+) instructions\n(.*?)(?=There are \d+ instructions|\Z)',text,re.S))
    _require(len(blocks) == len(rows),'ACTUAL_SUBMIT_LEDGER_CENSUS')
    fragments, pending = [], []
    for eid,event in enumerate(events):
        pending.append((eid,event))
        if event['kind'] == 'sync':
            _require(any(e['kind'] in ('kernel','load','store') for _,e in pending),'EMPTY_SYNC_UNSUPPORTED')
            fragments.append(pending)
            pending = []
    _require(not pending,'FINAL_EXPLICIT_SYNC_REQUIRED')
    _require(len(fragments) == len(blocks),'UNSUPPORTED_AUTOSYNC_FRAGMENT_MAPPING')
    for ordinal,(row,block) in enumerate(zip(rows,blocks),1):
        _require(row.get('ordinal') == ordinal and row.get('insns') == int(block[1]),'SUBMIT_ORDER_OR_INSN_COUNT')
        _require(row.get('reason') == 'explicit_sync','UNSUPPORTED_AUTOSYNC_REASON')
        _require(row.get('completion') is True and type(row.get('timeout')) is int and row['timeout'] == 0,
                 'SUBMIT_NOT_SUCCESSFULLY_COMPLETE')
        _require(type(row.get('serialized_uop_bytes')) is int and row['serialized_uop_bytes'] >= 0 and
                 row['serialized_uop_bytes'] % 4 == 0,'SUBMIT_UOP_BYTE_UNIT')
        if 'insn_bytes' in row:
            _require(row['insn_bytes'] == row['insns']*16,'SUBMIT_INSN_BYTE_UNIT')
    return fragments,[block.group(0) for block in blocks],rows


def certify_native(events, dump_text, submit_ledger, cache_definitions=None):
    """Full finite native/FIFO certificate for 1..16 explicit completed Syncs.

    `cache_definitions` must belong to the SAME loaded module/session cold call.
    The caller binds module, runtime, helper, session, source and replay status.
    Partial dumps, overflow, unknown cuts, missing tokens, and identity mutation
    fail closed. This function does not prove hardware pipeline cycle latency.
    """
    header_sha = hashlib.sha256(HEADER.read_bytes()).hexdigest()
    fragments,blocks,rows = _fragments(events,dump_text,submit_ledger)
    all_kernels = []
    unique,contracts = {},{}
    for event in events:
        if event['kind'] != 'kernel':
            continue
        signature = event.get('signature')
        if signature is not None:
            _require(isinstance(signature,(tuple,list)),'UNSUPPORTED_CACHE_SIGNATURE_TYPE')
            signature = list(signature)
            _require(signature == [] or len(signature) == 1 and type(signature[0]) is int and
                     0 <= signature[0] <= 98,'UNSUPPORTED_CACHE_SIGNATURE_KEY')
        kernel = _normalize(event)
        _require(0 < len(kernel['values']) <= DEPTH,'SINGLE_KERNEL_UOP_CAPACITY')
        contract = {'args':kernel['args'],'loops':kernel['loops']}
        key = kernel['identity']
        _require(key not in contracts or contracts[key] == contract,'CACHE_IDENTITY_CONTENT_MUTATION')
        unique.setdefault(key,kernel['values'])
        contracts.setdefault(key,contract)
        all_kernels.append(kernel)
    _require(bool(all_kernels),'EMPTY_NATIVE_KERNEL_TRACE')
    initial = _initializers(dump_text)
    dump_sha = hashlib.sha256(dump_text.encode()).hexdigest()
    if cache_definitions is None:
        _require(len(initial) == len(unique) and initial == list(unique.values()),'COLD_INITIALIZER_CENSUS_OR_VALUE')
        frozen = {'schema':SCHEMA,'header_sha256':header_sha,'origin_dump_sha256':dump_sha,
                  'definitions':unique,'contracts':contracts}
    else:
        _require(cache_definitions.get('schema') == SCHEMA and cache_definitions.get('header_sha256') == header_sha,
                 'CACHE_DEFINITION_SOURCE_BINDING')
        _require(not initial,'WARM_CALL_CREATES_UNKNOWN_KERNEL')
        _require(cache_definitions.get('definitions') == unique and cache_definitions.get('contracts') == contracts,
                 'WARM_FULL_CACHE_CONTRACT_MUTATION')
        frozen = cache_definitions
    ancestors = [0]*len(events)
    material_map,dependency_map,submissions,checks = [],[],[],[]
    tensor_domain_dram_bases = {}
    completed_material = 0
    total_wraps,total_guarded,total_packed,total_loads,total_computes = 0,0,0,0,0
    for ordinal,(fragment,text,row) in enumerate(zip(fragments,blocks,rows),1):
        fragment_events = [event for _,event in fragment]
        native = native_binding_details(fragment_events,text)
        offset = fragment[0][0]
        _require(all(eid == offset+local for local,(eid,_) in enumerate(fragment)),
                 'NATIVE_FRAGMENT_TIR_IDS_NOT_CONTIGUOUS')
        for local,mask in enumerate(native['ancestors']):
            eid = offset+local
            ancestors[eid] = mask << offset
            if fragment_events[local]['kind'] in ('kernel','load','store'):
                ancestors[eid] |= completed_material
        for output,field in ((material_map,'material_map'),(dependency_map,'dependency_map')):
            for entry in native[field]:
                output.append({**entry,'tir_event':offset+entry['tir_event'],'submission':ordinal})
        for entry in native['material_map']:
            if entry['kind'] not in ('load','store'):
                continue
            event = fragment_events[entry['tir_event']]
            key = (event['tensor'],event['domain'])
            dram_base = entry['native']['dram']-event['offset']
            _require(key not in tensor_domain_dram_bases or tensor_domain_dram_bases[key] == dram_base,
                     'CROSS_SUBMIT_TENSOR_DOMAIN_DRAM_BASE_CHANGED')
            tensor_domain_dram_bases.setdefault(key,dram_base)
        kernels = [_normalize(event) for event in fragment_events if event['kind'] == 'kernel']
        nop_barriers = compute_nop_barriers(text)
        simulated,packed,wraps,guarded = _simulate(kernels,frozen['definitions'],nop_barriers)
        actual = _instructions(text)
        _require(len(actual) == len(simulated),'ACTUAL_SAFE_QUEUE_REPLAY_CENSUS')
        sources = [item['source'] for item in actual if item['kind'] == 'load']
        _require(bool(sources),'PER_SUBMIT_SRAM_LOAD_ABSENT')
        base = min(sources)
        _require(row['serialized_uop_bytes'] == len(packed)*4,'ACTUAL_SERIALIZED_UOP_LEDGER_MISMATCH')
        live = [None]*DEPTH
        compute_idx = 0
        for fifo,(actual_row,predicted) in enumerate(zip(actual,simulated)):
            _require(actual_row['kind'] == predicted['kind'],'ACTUAL_SAFE_QUEUE_REPLAY_ORDER')
            if actual_row['kind'] == 'load':
                source = actual_row['source']-base
                start,size = actual_row['begin'],actual_row['size']
                _require({'kind':'load','source':source,'begin':start,'size':size} == predicted,
                         'ACTUAL_SAFE_QUEUE_LOAD_DESCRIPTOR')
                _require(0 <= source <= source+size <= len(packed) and 0 <= start <= start+size <= DEPTH,
                         'ACTUAL_UOP_RANGE')
                live[start:start+size] = packed[source:source+size]
            else:
                kernel = kernels[compute_idx]
                _require(actual_row['begin'] == predicted['begin'] and actual_row['end'] == predicted['end'],
                         'ACTUAL_UOP_KERNEL_REFERENCE')
                _require(actual_row['mode'] == kernel['args'][0][0] and actual_row['reset'] == kernel['args'][0][1] and
                         actual_row['loops'] == _expected_loops(kernel),'ACTUAL_KERNEL_MODE_LOOP')
                _require(live[actual_row['begin']:actual_row['end']] == kernel['values'],'ACTUAL_UOP_WRONG_OR_OVERWRITTEN')
                checks.append({'submission':ordinal,'compute_index':total_computes+compute_idx,'fifo_index':fifo,
                               'uop_begin':actual_row['begin'],'uop_end':actual_row['end'],
                               'kernel_identity':kernel['identity'],'values_match':True})
                compute_idx += 1
        _require(compute_idx == len(kernels),'ACTUAL_COMPUTE_CENSUS')
        completed_material |= sum(1 << eid for eid,event in fragment if event['kind'] in ('kernel','load','store'))
        submissions.append({'ordinal':ordinal,'instruction_count':native['instruction_count'],
                            'runtime_drain_events':native['runtime_drain_events'],'token_pairs':native['token_pairs'],
                            'serialized_uop_bytes':len(packed)*4,'uop_loads':len(sources),
                            'uop_sram_wraps':wraps,'greedy_read_conflicts_guarded':guarded,'completion':True,
                            'compute_nop_greedy_barriers':nop_barriers})
        total_wraps += wraps;total_guarded += guarded;total_packed += len(packed)
        total_loads += len(sources);total_computes += compute_idx
    return {'status':'ACTUAL_BOUNDED_NATIVE_FIFO_TOKEN_UOP_WITH_NOP_BARRIERS_CERTIFIED','dump_sha256':dump_sha,
            'submission_count':len(submissions),'instruction_count':sum(item['instruction_count'] for item in submissions),
            'ancestors':ancestors,'material_map':material_map,'dependency_map':dependency_map,'submissions':submissions,
            'cache_definitions':frozen,'uop_values':{'compute_references':total_computes,'uop_loads':total_loads,
              'serialized_uop_bytes':total_packed*4,'uop_sram_wraps':total_wraps,
              'greedy_read_conflicts_guarded':total_guarded,'checks':checks},
            'tensor_domain_dram_bases':[{'tensor':tensor,'domain':domain,'dram_base':base}
                                       for (tensor,domain),base in tensor_domain_dram_bases.items()],
            'support':'1..16 completed explicit Sync fragments, fixed 32 MiB queues; unknown AutoSync cuts rejected',
            'limits':['Caller binds same module/session cold definitions and frozen compiled TIR/runtime identity',
                      'Dump does not directly expose ALU immediate; full args/loops are frozen across calls',
                      'No cycle latency proof or bitstream-bound ACC feedback interval is inferred']}


def compute_nop_barriers(text):
    """NOPs are not payload LOADs, but ARE the runtime's last LOAD/UOP candidate.

    Frozen PushNoop initializes x/y/DRAM/SRAM to zero. DumpInsn names it NOP;
    its token bits are independently bound to TIR events by native_binding.
    These markers never become DRAM base samples or serialized UOP content.
    """
    blocks=decode_submissions(text)
    _require(len(blocks)==1,'NOP_BARRIERS_REQUIRE_ONE_SUBMIT_FRAGMENT')
    kernel_index=0
    barriers={}
    for record in blocks[0]:
        if record['description']=='NOP-COMPUTE-STAGE':
            _require(any(record['bits']),'UNEXPLAINED_EMPTY_COMPUTE_NOP')
            barriers[kernel_index]=barriers.get(kernel_index,0)+1
        elif record['description']=='GEMM' or record['description'].startswith('ALU -'):
            kernel_index+=1
    return barriers
