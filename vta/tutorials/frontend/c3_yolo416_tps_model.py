"""Appendix-A TPS reconstruction for this graph's 3x3 and 1x1 convolutions.

No timing or compiler query is used to select tiles. Logical allocations and
the bias proxy are not native traffic or a prediction of complete-call time.
Only the already agreed batch1/16x16/stride1/dilation1 family is supported.
"""
from dataclasses import asdict
from itertools import product
from c3_banerjee_baseline import Tile,divisors


def geometry(workload):
    if len(workload)!=8 or workload[0]!='conv2d_packed.vta':raise ValueError('TPS_WORKLOAD')
    x,w=workload[1][1],workload[2][1]
    if not (len(x)==len(w)==6 and x[0]==x[4]==1 and x[5]==w[4]==w[5]==16 and x[1]==w[1]
            and workload[1][2]==workload[2][2]=='int8' and workload[7]=='int32'
            and workload[3]==workload[5]==[1,1] and workload[6]=='NCHW1n16c'):
        raise ValueError('TPS_PACKED_GEOMETRY_SUPPORT')
    kh,kw=w[2:4];padding=workload[4]
    if not (([kh,kw]==[3,3] and padding==[1,1,1,1]) or
            ([kh,kw]==[1,1] and padding==[0,0,0,0])):raise ValueError('TPS_KERNEL_PADDING_SUPPORT')
    return (1,x[2],x[3],x[1],w[0]),(kh,kw),padding[:2]


def cost(tile,workload,capacity):
    dims,(kh,kw),(ph,pw)=geometry(workload)
    inner=(tile.b,tile.h,tile.w,tile.ci,tile.co)
    if any(type(v) is not int or v<=0 or n%v for n,v in zip(dims,inner)):
        raise ValueError('TPS_NONINTEGRAL_TILE')
    if tile.oc_nthread not in (1,2) or tile.h_nthread not in (1,2):raise ValueError('TPS_THREADS')
    bo,ho,wo,cio,coo=(n//v for n,v in zip(dims,inner))
    threads=tile.oc_nthread*tile.h_nthread
    usage={'INP':tile.b*tile.ci*16*(tile.h+2*ph)*(tile.w+2*pw)*threads,
           'WGT':tile.co*tile.ci*256*kh*kw*threads,
           'ACC':(tile.b*tile.co*16*tile.h*tile.w+tile.co*16)*4*threads}
    reasons=['paper_both_threads_two'] if threads==4 else []
    exact=coo%tile.oc_nthread==0 and ho%tile.h_nthread==0
    if not exact:reasons.append('local_exact_outer_thread_divisibility')
    reasons += [k+'_paper_capacity' for k,v in usage.items() if v>capacity[k]]
    mul=bo*(ho//tile.h_nthread)*(coo//tile.oc_nthread)*wo*cio if exact else None
    loads=None if mul is None else {'INP':mul*usage['INP'],'WGT':mul*usage['WGT'],
                                   'ACC_bias_proxy':bo*ho*wo*(dims[4]*16)*4}
    return {'id':tile.identity,'tile':asdict(tile),'paper_allocation_bytes':usage,
            'paper_payload_bytes':loads,'paper_score_bytes':None if loads is None else sum(loads.values()),
            'necessary_conditions_passed':not reasons,'rejections':reasons,'full_fusion_qualified':False}


def enumerate_workload(workload,capacity):
    dims,_,_=geometry(workload)
    return [cost(Tile(*values),workload,capacity) for values in product(
        *(divisors(n) for n in dims),(1,2),(1,2))]


def select(workload,capacity):
    rows=enumerate_workload(workload,capacity)
    ranked=sorted((r for r in rows if r['necessary_conditions_passed']),
                  key=lambda r:(r['paper_score_bytes'],r['id']))
    if not ranked:raise ValueError('TPS_EMPTY_LEGAL_ANALYTICAL_DOMAIN')
    first=ranked[0]
    t=first['tile']
    config={'index':-1,'code_hash':None,'entity':[
        [name,'sp',[-1,t[key]]] for name,key in zip(
            ('tile_b','tile_h','tile_w','tile_ci','tile_co'),('b','h','w','ci','co'))]+
        [[name,'ot',t[name]] for name in ('oc_nthread','h_nthread')]}
    return {'selected':first,'complete_config_entity':config,'raw_candidates':len(rows),
            'necessary_domain':len(ranked),'minimum_ties':sum(r['paper_score_bytes']==first['paper_score_bytes'] for r in ranked),
            'rule':'minimum paper_score_bytes, then Tile.identity lexical order; all supported thread families',
            'hardware_qualified':False,'performance_read':False}
