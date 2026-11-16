"""Read-only finite-domain initializer equivalence census over archived TIR.

No lowering, target compilation, FSim, RPC or executable transformation. Each
scope stays a separate handle. Full mode, loop and uop arguments are compared.
Only unconditional, constant serial host domains and empty/single-int closures
are supported. Unknown domains remain unresolved rather than zero opportunity.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
HOLD=HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def require(v,s):
    if not v:raise ValueError(s)


def full_content(event):
    require(event['uops'] and all(row['loops']==event['uops'][0]['loops'] for row in event['uops']),
            'UNSUPPORTED_PER_UOP_LOOP_VARIATION')
    return (event['mode'],tuple(tuple(row['args']) for row in event['uops']),
            tuple(tuple(loop) for loop in event['uops'][0]['loops']))


def classify(events):
    """Exact tuple equality, with representatives taken from original signatures."""
    scopes={}
    for event in events:
        scope=scopes.setdefault(event['scope'],{})
        sig=tuple(event['signature']);content=full_content(event)
        require(len(sig)<=1 and (not sig or 0<=sig[0]<=98),'SIGNATURE_CONTRACT')
        require(sig not in scope or scope[sig]==content,'IDENTITY_CONTENT_MUTATION')
        scope[sig]=content
    rows=[]
    for scope,identities in scopes.items():
        groups={}
        for sig,content in sorted(identities.items()):groups.setdefault(content,[]).append(sig)
        classes=[{'representative':list(v[0]),'members':[list(x) for x in v],
                  'base_uops':len(k[1])} for k,v in groups.items()]
        signatures=sorted(identities)
        period=None
        if all(len(s)==1 for s in signatures):
            values=[s[0] for s in signatures]
            if values==list(range(values[0],values[-1]+1)):
                for p in range(1,len(values)):
                    if all(identities[(k,)]==identities[(values[0]+(k-values[0])%p,)] for k in values):
                        period=p;break
        modes={(c[0],c[1][0][:2]) for c in identities.values()}
        require(len(modes)==1,'SCOPE_MODE_CHANGED')
        mode,reset=next(iter(modes))
        rows.append({'scope':scope,'mode':mode,'mode_reset':list(reset),'signatures':len(identities),
                     'content_classes':len(classes),'redundant_identities':len(identities)-len(classes),
                     'smallest_period_on_complete_frozen_domain':period,'classes':classes,
                     'identity_base_uops':sum(len(c[1]) for c in identities.values()),
                     'class_base_uops':sum(len(c[1]) for c in groups)})
    return rows


def inspect_function(function):
    import tvm
    from tvm import tir
    from c3_valid_window import integer
    from c3_official_window_method import _trace,_limits
    rows=[];unresolved=[];scope_count=0
    def walk(node,hosts=(),conditional=False):
        nonlocal scope_count
        if isinstance(node,tir.AttrStmt) and node.attr_key=='coproc_uop_scope':
            ordinal=scope_count;scope_count+=1
            try:
                require(not conditional,'CONDITIONAL_OR_NONCONSTANT_HOST_DOMAIN')
                free=list(tir.analysis.undefined_vars(node.body))
                require(len(free)<=1 and all(str(v.dtype)=='int32' for v in free),'CLOSURE_ARITY_OR_DTYPE')
                body=tir.AttrStmt(node.node,'coproc_scope',tir.IntImm('int32',2),node)
                domain=[]
                if free:
                    owners=[f for f in hosts if f.loop_var.same_as(free[0])]
                    require(len(owners)==1,'FREE_INTEGER_NOT_A_DIRECT_HOST_COUNTER')
                    owner=owners[0];lo=integer(owner.min);n=integer(owner.extent)
                    require(n>0 and 0<=lo and lo+n<=99,'SIGNATURE_DOMAIN_OR_KEY_RANGE')
                    domain=list(range(lo,lo+n))
                    body=tir.For(owner.loop_var,owner.min,owner.extent,owner.kind,body)
                # Analysis-only projection: enumerate each reachable closure
                # value once, excluding DMA and repeated outer iterations.
                events,counts=_trace(function.with_body(body),_limits(None))
                result=classify(events)
                require(len(result)==1,'SCOPE_PROJECTION_CENSUS')
                result[0].update(scope=ordinal,domain=domain,closure_arity=len(free),
                                 initializer_projection_steps=counts['expanded_uop_steps'])
                rows.extend(result)
            except (ValueError,TypeError) as error:
                unresolved.append({'scope':ordinal,'reason':str(error)})
            return
        if isinstance(node,tir.For):
            bad=conditional
            try:
                lo,n=integer(node.min),integer(node.extent)
                require(n>=0,'NEGATIVE_HOST_DOMAIN')
                if n==0:return
                bad |= node.kind not in (tir.ForKind.SERIAL,tir.ForKind.UNROLLED) or node.thread_binding is not None or bool(node.annotations)
            except ValueError:bad=True
            walk(node.body,hosts+(node,),bad)
        elif isinstance(node,tir.SeqStmt):
            for child in node.seq:walk(child,hosts,conditional)
        elif isinstance(node,tir.IfThenElse):
            walk(node.then_case,hosts,True)
            if node.else_case is not None:walk(node.else_case,hosts,True)
        elif isinstance(node,(tir.AttrStmt,tir.LetStmt,tir.Allocate,tir.AssertStmt)):
            walk(node.body,hosts,conditional)
        elif not isinstance(node,tir.Evaluate):
            unresolved.append({'scope':None,'reason':'UNKNOWN_STATEMENT:'+type(node).__name__})
    walk(function.body)
    mac=[r for r in rows if r['mode_reset']==[0,0]]
    return {'status':'FINITE_DOMAIN_CONTENT_SCREEN_COMPLETE' if not unresolved else 'SUPPORT_UNRESOLVED',
            'scopes':rows,'unresolved':unresolved,'static_scopes':scope_count,
            'MAC_identities':sum(r['signatures'] for r in mac),
            'MAC_content_classes':sum(r['content_classes'] for r in mac),
            'MAC_redundant_identities':sum(r['redundant_identities'] for r in mac),
            'identity_base_uops':sum(r['identity_base_uops'] for r in rows),
            'class_base_uops':sum(r['class_base_uops'] for r in rows),
            'transformation_implemented':False,'native_lifetime_recertification_required':True}


def inventory():
    objects=[]
    for label,name,node in [('TPS24','20261016_c3_tps_sync_release_local_run04',24),
                            ('TPS34','20261017_c3_node34_tps_local_run02',34)]:
        root=HOLD/name
        objects.append({'id':label,'family':'qualified_TPS','network':'yolo224','nodes':[node],
                        'path':str(root/'prepare/R/parent.json'),'root':str(root),
                        'native':str(root/'R_detailed/native_certificate_01.json')})
    official=HOLD/'20261014_c3_official_valid_window_run01'
    call_index=read(official/'closure_final02/local_call_index.json')
    for path in sorted((official/'local_final01').glob('*_descriptor.json')):
        d=read(path)
        calls=[r for r in call_index if r['object_id']==d['object_id'] and r['mode']=='detailed' and r['call_index']==1]
        require(len(calls)==1,'FINAL_QUALIFICATION_INDEX_NOT_UNIQUE')
        native=Path(calls[0]['native_certificate'])
        objects.append({'id':d['object_id'],'family':'official_reinstantiated','network':d['network'],
                        'nodes':d['graph_callsite_aliases'],'path':str(REPO/d['parent_module']),
                        'symbol':d['global_symbol'],'root':str(official),'native':str(native)})
    previous=HOLD/'20261016_c3_yolo224_sync_structure_readonly03'
    for d in read(previous/'inventory.json'):
        if d['variant']!='G2':continue
        objects.append({'id':'yolo224_'+d['object_id'],'family':'existing_graph_not_TPS','network':'yolo224',
                        'nodes':[d['node']],'path':d['tir'],'root':str(HOLD/'20261008_c3_yolo224_incremental_prepare_run03'),
                        'expected_sha256':d['sha256'],'force_serial_requested':read(previous/'objects'/(d['object_id']+'.json'))['force_serial_requested']})
    require(Counter(r['family'] for r in objects)=={'qualified_TPS':2,'official_reinstantiated':26,'existing_graph_not_TPS':11},'ARCHIVED_POOL_CENSUS_CHANGED')
    return objects


def worker(row):
    import tvm,vta
    from c3_tps_sync_release_prepare import complete_function
    path=Path(row['path']);digest=sha(path)
    root=Path(row['root'])
    seal=root/'completion_manifest.json'
    if seal.exists():require(read(seal)['artifact_sha256'][str(path.relative_to(root))]==digest,'SEALED_INPUT_CHANGED')
    else:require(row.get('expected_sha256')==digest,'UNSEALED_INPUT_IDENTITY')
    module=tvm.ir.load_json(path.read_text())
    if isinstance(module,tvm.IRModule):
        if 'symbol' in row:_,function=complete_function(module,row['symbol'])
        else:
            require(len(module.functions)==1,'SINGLE_FUNCTION_REQUIRED')
            function=module[list(module.functions)[0]]
    else:function=module
    result=inspect_function(function)
    if 'native' in row:
        native_path=Path(row['native'])
        require(read(seal)['artifact_sha256'][str(native_path.relative_to(root))]==sha(native_path),'SEALED_NATIVE_CHANGED')
        cert=read(native_path)
        contracts=cert['cache_definitions']['contracts']
        mode_count=Counter(tuple(c['args'][0][:2]) for c in contracts.values())
        if result['status']=='FINITE_DOMAIN_CONTENT_SCREEN_COMPLETE':
            require(mode_count[(0,0)]==result['MAC_identities'],'MAC_SOURCE_NATIVE_IDENTITY_CENSUS_CHANGED')
            require(sum(len(c['args']) for c in contracts.values())==result['identity_base_uops'],'FULL_SOURCE_NATIVE_IDENTITY_CENSUS_CHANGED')
        result['adopted_native_uop_values']={k:v for k,v in cert['uop_values'].items() if k!='checks'}
    return {**row,'source_sha256':digest,**result,'new_target_calls':0,'new_compilations':0,'board_contacted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--row',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    try:write(a.output,worker(read(a.row)))
    except Exception as error:
        write(a.output,{'status':'AUDIT_TOOL_OR_INPUT_FAILURE','error':str(error)})
        raise
