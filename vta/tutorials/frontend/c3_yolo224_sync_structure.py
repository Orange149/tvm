"""Read-only bounded TIR census of existing YOLO224 graphs; no target calls.

No native FIFO/capacity or transformed-program certification is claimed.
Runtime mode calls are recorded separately; the analysis-only TIR projection
omits their setters without changing or compiling any executable artifact.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
HOLD=HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
SOURCE=HOLD/'20261008_c3_yolo224_incremental_prepare_run03'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def require(v,s):
    if not v:raise ValueError(s)

def stage(e):
    k=e['kind']
    if k=='kernel':return 2
    if k=='load':return 1 if e['domain'] in (1,2) else 2
    if k=='store':return 3
    if k=='push':return e['pair'][0]
    if k=='pop':return e['pair'][1]
    return None

def paired_graph(events):
    pred=[[] for _ in events];last={};queues=defaultdict(deque);tokens=[];last_compute=None
    for i,e in enumerate(events):
        st=stage(e)
        if st:
            if st in last:pred[i].append((last[st],'fifo'))
            last[st]=i
        if e['kind']=='kernel':last_compute=i
        if e['kind']=='push':queues[tuple(e['pair'])].append((i,last_compute if st==2 else None))
        if e['kind']=='pop':
            lane=tuple(e['pair']);require(queues[lane],'TOKEN_UNDERFLOW')
            push,kernel=queues[lane].popleft()
            pred[i].append((push,'token'))
            tokens.append({'lane':list(lane),'push':push,'pop':i,'producer_kernel':kernel})
        if e['kind']=='sync':
            require(not any(queues.values()),'UNBALANCED_TOKENS_AT_SYNC')
            pred[i]=[(p,'completion') for p in last.values()]
            last={1:i,2:i,3:i}
    require(not any(queues.values()),'UNBALANCED_TOKENS_AT_END')
    return pred,tokens

def reachable(pred,source,target,cut):
    stack=[target];seen=set()
    while stack:
        i=stack.pop()
        if i==source:return True
        if i<source or i in seen:continue
        seen.add(i)
        for p,cause in pred[i]:
            if (p,i,cause)!=cut:stack.append(p)
    return False

def input_cells(e):
    left,top,right,bottom=e['pad']
    return set(range(e['sram'],e['sram']+(left+e['x']+right)*(top+e['y']+bottom)))

def classify_events(events):
    from c3_valid_window import expanded_uops
    pred,tokens=paired_graph(events)
    load_fifo=[i for i,e in enumerate(events) if stage(e)==1 or e['kind']=='sync']
    positions={i:j for j,i in enumerate(load_fifo)}
    models={};rows=[]
    def reads(i):
        e=events[i];key=json.dumps(e['uops'],sort_keys=True)
        if key not in models:models[key]={u[3] for u in expanded_uops(e)}
        return models[key]
    for t in tokens:
        if t['lane']!=[2,1]:continue
        k=t['producer_kernel'];pe=events[k] if k is not None else None
        label='UNRESOLVED' if pe is None else ('RESET' if pe['mode_reset']==[0,1] else 'MAC' if pe['mode_reset']==[0,0] else 'ALU')
        row={**t,'producer_kind':label}
        tail=load_fifo[positions[t['pop']]+1:]
        first=tail[0] if tail else None
        if first is None or events[first]['kind']!='load':
            row['classification']='CONTROL_ONLY_OR_NO_FOLLOWING_LOAD';rows.append(row);continue
        row['first_load_event']=first;row['first_load_domain']=events[first]['domain']
        if events[first]['domain']!=2:
            row['classification']='WAIT_BEFORE_WEIGHT_OR_OTHER_LOAD';rows.append(row);continue
        prefix=[];weight=None;boundary=None
        for i in tail:
            e=events[i]
            if e['kind']=='load' and e['domain']==2:prefix.append(i)
            elif e['kind']=='load' and e['domain']==1:weight=i;break
            else:boundary={'event':i,'kind':e['kind']};break
        row.update(input_prefix=prefix,following_weight=weight,boundary=boundary,
                   prefix_DDR_bytes=sum(events[i]['x']*events[i]['y']*16 for i in prefix))
        if label not in ('MAC','RESET'):
            row['classification']='PRODUCER_NOT_GEMM';rows.append(row);continue
        readset=reads(k);cut=(t['push'],t['pop'],'token')
        collisions=[];released=[]
        for i in prefix:
            overlap=readset & input_cells(events[i]);retained=reachable(pred,k,i,cut)
            if not retained:released.append(i)
            if overlap and not retained:collisions.append({'load_event':i,'INP_address':min(overlap)})
        row.update(newly_released_inputs=released,collision_witnesses=collisions[:2])
        if collisions:
            row['classification']='MAC_INPUT_OVERWRITE_PROTECTION' if label=='MAC' else 'RESET_PHYSICAL_READ_UNRESOLVED'
        elif not released:row['classification']='OTHER_DEPENDENCIES_RETAIN_PRODUCER_ORDER'
        elif weight is None:row['classification']='NO_WEIGHT_TARGET_BEFORE_CONTROL_BOUNDARY'
        else:row['classification']='ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE'
        rows.append(row)
    return {'status':'TIR_STRUCTURE_AND_DIRECT_OVERWRITE_SCREEN_ONLY',
            'event_count':len(events),'token_lanes':dict(Counter(str(t['lane']) for t in tokens)),
            'g2l_count':len(rows),'categories':dict(Counter(r['classification'] for r in rows)),
            'MAC_categories':dict(Counter(r['classification'] for r in rows if r['producer_kind']=='MAC')),
            'RESET_categories':dict(Counter(r['classification'] for r in rows if r['producer_kind']=='RESET')),
            'paired_tokens_balanced':True,'waits':rows,
            'full_native_safety_certified':False,'performance_bottleneck_measured':False}

def project(function):
    import tvm
    from tvm import tir
    from c3_valid_window import call_info,integer
    from c3_official_window_method import _trace,_limits
    settings=[]
    def visit(node):
        if isinstance(node,tir.Evaluate):
            info=call_info(node)
            if info and info[0]=='VTASetDebugMode':
                flag=integer(info[1][-1]);require(flag in (0,32),'UNKNOWN_RUNTIME_FLAG')
                settings.append(flag)
                return tir.Evaluate(0)
        return None
    body=tir.stmt_functor.ir_transform(function.body,None,visit,['tir.Evaluate'])
    if settings:require(settings==[32,0],'RUNTIME_MODE_STRUCTURE_UNRESOLVED')
    events,counts=_trace(function.with_body(body),_limits(None))
    return events,counts,settings

def worker(row,output):
    import tvm
    import vta  # Register VTA intrinsics before deserializing archived TIR.
    path=Path(row['tir']);require(sha(path)==row['sha256'],'TIR_IDENTITY_CHANGED')
    function=tvm.ir.load_json(path.read_text())
    events,counts,settings=project(function)
    d=classify_events(events)
    d.update(source=row,counts=counts,runtime_flag_calls=settings,force_serial_requested=32 in settings,
             scope='existing fused function TIR; runtime commands and automatic submissions not reconstructed')
    write(output,d)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--row',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:worker(read(a.row),a.output)
    except Exception as e:
        write(a.output,{'status':'ANALYSIS_UNRESOLVED','error':str(e),'type':type(e).__name__})
        raise
