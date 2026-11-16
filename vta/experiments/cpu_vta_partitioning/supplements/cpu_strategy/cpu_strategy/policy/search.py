"""Complete interval label-setting DP with an independent DFS enumerator.

This extends the historical coverage/device/island recurrence with fixed-device
loads. Labels are complete resource vectors and back-pointers; no Top-1 Pareto
rule or scalar best-prefix pruning is applied to Top-K paths.
"""
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappush, heappop
from itertools import count, product
import math
from cpu_vta_partitioning.common import digest

from cpu_vta_partitioning.domain import Stage
from .domain import (CORES, CUTS, BASE_THREADS, FAMILIES, candidate, component,
                     component_key, cut_candidate, partitions)


class Costs:
    def __init__(self, records, homogeneous=True):
        self.records=records
        self.homogeneous=homogeneous
        self.aliases={}

    @lru_cache(None)
    def get(self, device, begin, end, threads, mask, mode, host):
        row=Stage(device,begin,end,threads,mask).to_dict()
        c=component(row,mode,host)
        key=component_key(c)
        original=key
        if key not in self.records and device=="cpu" and self.homogeneous and len(mask)==threads:
            c=component(Stage(device,begin,end,threads,CORES[:threads]).to_dict(),mode,host)
            key=component_key(c)
            if key!=original:self.aliases[original]={"source_key":key,"assumption":"homogeneous_core_permutation"}
        if key not in self.records: raise ValueError("missing matched component cost: "+original)
        value=self.records[key]["cost"]
        fields=tuple(float(value[k]) for k in ("service_ms","run_core_ms","boundary_core_ms"))
        if any(not math.isfinite(x) or x<0 for x in fields):raise ValueError("invalid component cost")
        return fields

    def stage(self,s,mode,host):
        if isinstance(s,dict):s=Stage.from_dict(s)
        return self.get(s.device,s.begin,s.end,s.threads,s.mask,mode,tuple(host))


def evaluate(c, costs, method="enhanced"):
    groups={tuple(g):0. for g in c["core_partition"]}
    cpu_max=vta=wc=wh=wb=0.
    confined=[0.]*15;prefix=[0.]*4
    for row in c["stages"]:
        s=Stage.from_dict(row);service,work,boundary=costs.stage(s,c["affinity_mode"],c["host_mask"])
        if s.device=="cpu":
            cpu_max=max(cpu_max,service);wc+=work
            if s.mask in groups:groups[s.mask]+=service
            for n in range(s.threads,5):prefix[n-1]+=work/n
        else:vta+=service;wh+=work
        wb+=boundary;mask=sum(1<<i for i in s.mask)
        boundary_mask=(1<<s.mask[0]) if s.device=='cpu' and c['affinity_mode']=='pinned' else mask
        for subset in range(1,16):
            if mask&subset==mask:confined[subset-1]+=work/subset.bit_count()
            if boundary_mask&subset==boundary_mask:confined[subset-1]+=boundary/subset.bit_count()
    score=(max(cpu_max,vta,(wc+wh+wb)/4,*prefix) if method=="prefix"
           else max(vta,*groups.values(),*confined,cpu_max))
    return {"score_ms":round(score,9),"cpu_service_max_ms":cpu_max,"vta_service_sum_ms":vta,
            "group_services_ms":{','.join(map(str,k)):v for k,v in groups.items()},
            "cpu_core_ms":wc,"host_core_ms":wh,"boundary_core_ms":wb,
            "subset_capacity_ms":confined,"objective":method}


def _fixed_candidates(cut, mode, reserved, homogeneous):
    cores=CORES[:3] if reserved else CORES
    a,b=CUTS[cut]
    seen=set()
    for part in partitions(cores,not homogeneous):
        for first,last in product(part,repeat=2):
            c=cut_candidate(cut,(len(first),len(last)),(first,last),mode,"resource_group",reserved,part)
            # Unused physical groups cannot affect execution identity.
            used=tuple(sorted(set((first,last))))
            c=cut_candidate(cut,(len(first),len(last)),(first,last),mode,"resource_group",reserved,used)
            if c["id"] not in seen:seen.add(c["id"]);yield c


def best_fixed(cut,costs,mode="group",reserved=False):
    rows=((evaluate(c,costs),c) for c in _fixed_candidates(cut,mode,reserved,costs.homogeneous))
    return min(rows,key=lambda x:(x[0]["score_ms"],x[1]["id"]))[1]


def mirror_check(costs):
    comparisons=[]
    for begin,end in sorted({(0,3),*( (b,21) for a,b in CUTS.values())}):
        for mode in ("group","pinned"):
            for n in range(1,4):
                a=costs.get("cpu",begin,end,n,CORES[:n],mode,CORES)[0]
                b=costs.get("cpu",begin,end,n,CORES[4-n:],mode,CORES)[0]
                comparisons.append({"begin":begin,"end":end,"mode":mode,"threads":n,
                                    "relative_difference":abs(a-b)/max(min(a,b),1e-12)})
    return {"homogeneous":all(v["relative_difference"]<=.05 for v in comparisons),"comparisons":comparisons,
            "interpretation":"representative permutation assumption, not measurements of every mask"}


def policy_selection(records):
    exact=Costs(records,False)
    mirror=mirror_check(exact)
    costs=Costs(records,mirror["homogeneous"])
    selected={};pool={};pairs={}
    for cut in CUTS:
        selected[cut]={}
        def add(name,c):selected[cut][name]=c["id"];pool[c["id"]]=c;return c
        base=BASE_THREADS[cut]
        add("current",cut_candidate(cut,base,(CORES,CORES[:base[1]])))
        a,b=CUTS[cut];ns=[]
        for begin,end in ((0,a),(b,21)):
            ns.append(min(range(1,5),key=lambda n:(costs.get("cpu",begin,end,n,CORES,"group",CORES)[0],n)))
        add("local_fastest",cut_candidate(cut,ns,(CORES,CORES)))
        grouped=add("fixed_groups",best_fixed(cut,costs))
        pinned=add("same_groups_pinned",candidate(grouped["stages"],"pinned","resource_group",CORES,CORES,grouped["core_partition"]))
        add("pinned_reallocated",best_fixed(cut,costs,"pinned"))
        reserved=add("reserved_host",best_fixed(cut,costs,"pinned",True))
        shared_stages=[dict(s,mask=list(CORES),signature=Stage("vta",s["begin"],s["end"],1,CORES).signature)
                       if s["device"]=="vta" else s for s in reserved["stages"]]
        shared=add("reserved_cpu_shared_host",candidate(shared_stages,"pinned","resource_group",CORES,CORES,reserved["core_partition"]))
        pairs[cut]={"pinning":[pinned["id"],grouped["id"]],"host_placement":[reserved["id"],shared["id"]]}
        cpu=[s for s in grouped["stages"] if s["device"]=="cpu"]
        if cpu[0]["mask"]==cpu[1]["mask"]:
            off=add("same_group_gate_off",candidate(grouped["stages"],"group","off",CORES,CORES,grouped["core_partition"]))
            pairs[cut]["gate"]=[grouped["id"],off["id"]]
    for name,masks in (("overlap",((0,1),(0,1))),("separate",((0,1),(2,3)))):
        c=cut_candidate("current",(2,2),masks);pool[c["id"]]=c
        selected["current"][name]=c["id"]
    pairs["current"]["placement"]=[selected["current"]["separate"],selected["current"]["overlap"]]
    if len(pool)>26:raise ValueError("policy pool exceeds preregistered 26 configurations")
    return {"source":"component_profiles_before_pipeline", "families":selected,"pool":pool,"pairs":pairs,
            "mirror_check":mirror,"cost_aliases":costs.aliases}


def freeze_rule(selection, discovery, check):
    """Both inputs are two-round means; confirmation never calls this function."""
    families=selection["families"]
    summaries={}
    for family in FAMILIES:
        gains={}
        for cut in CUTS:
            key=families[cut][family];base=families[cut]["current"]
            if key not in discovery or key not in check or base not in check:raise ValueError("policy check coverage incomplete")
            gains[cut]=check[base]/check[key]-1
        geometric=math.exp(sum(math.log1p(g) for g in gains.values())/len(gains))-1
        summaries[family]={"per_cut_gain":gains,"geometric_gain":geometric,
                           "supported":geometric>=.05 and min(gains.values())>=-.05}
    simple="fixed_groups"
    supported=[f for f in FAMILIES[2:] if summaries[f]["supported"]]
    order={f:i for i,f in enumerate(FAMILIES)}
    chosen=min(supported,key=lambda f:(-summaries[f]["geometric_gain"],order[f])) if supported else simple
    # Require a material extra benefit before adopting more complex binding or
    # host reservation. If the simple rule itself is unsupported, record that.
    if chosen!=simple:
        extra=math.exp(sum(math.log(check[families[c][simple]]/check[families[c][chosen]]) for c in CUTS)/3)-1
        if extra<.05:chosen=simple
    return {"family":chosen,"affinity_mode":"pinned" if chosen in ("same_groups_pinned","pinned_reallocated","reserved_host") else "group",
            "reserved_host":chosen=="reserved_host","homogeneous":selection["mirror_check"]["homogeneous"],
            "preliminary_support":summaries[chosen]["supported"],"family_results":summaries,
            "scope":"provisional_after_independent_check; final three boots decide support"}


@dataclass(frozen=True)
class Label:
    position:int
    device:str
    islands:int
    group_loads:tuple
    cpu_max:float
    vta:float
    work:float
    prefix:tuple
    subsets:tuple
    path:tuple


def _score(label,method):
    return (max(label.cpu_max,label.vta,label.work/4,*label.prefix) if method=="prefix"
            else max(label.cpu_max,label.vta,*label.group_loads,*label.subsets))


def rank_dp(programs, records, rule, method="enhanced",k=20,unit_count=21,max_islands=3):
    return _rank(programs,records,rule,method,k,unit_count,max_islands,False)


def enumerate_reference(programs, records, rule, method="enhanced",k=20,unit_count=21,max_islands=3):
    from cpu_vta_partitioning.domain import topologies
    costs=Costs(records,rule['homogeneous']);mode=rule['affinity_mode']
    cores=CORES[:3] if rule['reserved_host'] else CORES;host=(3,) if rule['reserved_host'] else CORES
    parts=(list(partitions(cores,not rule['homogeneous'])) if method=='enhanced' else
           [tuple((c,) for c in cores)] if method=='single_core' else [()])
    best=[];seen=set();visited=0
    for topology in topologies(programs,unit_count,max_islands):
        cpu_count=sum(p['device']=='cpu' for p in topology)
        for part in parts:
            choices=list(part) if method!='prefix' else [cores[:n] for n in range(1,len(cores)+1)]
            for allocation in product(choices,repeat=cpu_count):
                masks=iter(allocation);stages=[]
                for p in topology:
                    mask=next(masks) if p['device']=='cpu' else host
                    stages.append(Stage(p['device'],p['begin'],p['end'],len(mask) if p['device']=='cpu' else 1,mask).to_dict())
                used=tuple(sorted(set(allocation)));cfg={'stages':stages,'affinity_mode':mode,'core_partition':used if method!='prefix' else (), 'host_mask':host}
                value=evaluate(cfg,costs,method)['score_ms'];visited+=1
                if len(best)==k and value>best[-1][0]:continue
                if unit_count==21:
                    row=candidate(stages,mode,'off' if method=='prefix' else 'resource_group',host,host,used)
                    row.update(evaluate(row,costs,method));ident=row['id']
                else:
                    ident=digest((stages,used,method));row={'id':ident,'stages':stages,'score_ms':value}
                if ident in seen:continue
                seen.add(ident);best.append((value,ident,row));best.sort(key=lambda x:(x[0],x[1]));del best[k:]
    return {'algorithm':'independent_topology_mapping_enumeration','method':method,'rankings':[r[2] for r in best],
            'visited_terminals':visited,'cost_aliases':costs.aliases}


def _rank(programs,records,rule,method,k,unit_count,max_islands,reference):
    if method not in ("enhanced","single_core","prefix") or type(k)is not int or k<1:raise ValueError("invalid search mode/budget")
    costs=Costs(records,rule["homogeneous"]);mode=rule["affinity_mode"]
    cores=CORES[:3] if rule["reserved_host"] else CORES;host=(3,) if rule["reserved_host"] else CORES
    parts=(list(partitions(cores,not rule["homogeneous"])) if method=="enhanced" else
           [tuple((c,) for c in cores)] if method=="single_core" else [()])
    edges={}
    for p in programs.values():
        if p["end"]<=unit_count:edges.setdefault((p["begin"],p["device"]),[]).append(p)
    for values in edges.values():values.sort(key=lambda x:(x["end"],x["name"] if "name"in x else ""))
    best=[];seen=set();expanded=completed=pruned=0;serial=count();pending=[]

    def terminal(label,part):
        nonlocal completed
        completed+=1
        value=round(_score(label,method),9)
        if len(best)==k and value>best[-1][0]:return
        stages=[Stage(*s).to_dict() for s in label.path]
        # Unit-test mini-graphs need not cover 21 units; full public candidates do.
        if unit_count!=21:
            used=tuple(sorted({s[4] for s in label.path if s[0]=='cpu'}))
            ident=digest((stages,used,method))
            row={"id":ident,"stages":stages,"score_ms":value}
        else:
            used=tuple(sorted({s.mask for s in (Stage.from_dict(x) for x in stages) if s.device=="cpu"}))
            c=candidate(stages,mode,"off" if method=="prefix" else "resource_group",host,host,used if method!="prefix" else ())
            ident=c["id"];row={**c,**evaluate(c,costs,method)}
        if ident in seen:return
        seen.add(ident);best.append((value,ident,row));best.sort(key=lambda x:(x[0],x[1]));del best[k:]

    def successors(label,part):
        for p in edges.get((label.position,label.device),[]):
            islands=label.islands+(p["device"]=="vta")
            if islands>max_islands:continue
            masks=([host] if p["device"]=="vta" else list(part) if method!="prefix" else [cores[:n] for n in range(1,len(cores)+1)])
            for mask in masks:
                t=len(mask) if p["device"]=="cpu" else 1
                service,work,boundary=costs.get(p["device"],p["begin"],p["end"],t,mask,mode,host)
                loads=list(label.group_loads);prefix=list(label.prefix);subsets=list(label.subsets)
                cpu=label.cpu_max;vta=label.vta
                if p["device"]=="cpu":
                    cpu=max(cpu,service)
                    if part:loads[part.index(mask)]+=service
                    for n in range(t,5):prefix[n-1]+=work/n
                else:vta+=service
                bitmask=sum(1<<i for i in mask)
                boundary_mask=(1<<mask[0]) if p['device']=='cpu' and mode=='pinned' else bitmask
                for subset in range(1,16):
                    if bitmask&subset==bitmask:subsets[subset-1]+=work/subset.bit_count()
                    if boundary_mask&subset==boundary_mask:subsets[subset-1]+=boundary/subset.bit_count()
                yield Label(p["end"],"cpu" if p["device"]=="vta" else "vta",islands,
                    tuple(loads),cpu,vta,label.work+work+boundary,tuple(prefix),tuple(subsets),
                    (*label.path,(p["device"],p["begin"],p["end"],t,mask)))

    for part in parts:
        pending=[]
        start=Label(0,"cpu",0,(0.,)*len(part),0.,0.,0.,(0.,)*4,(0.,)*15,())
        heappush(pending,(0.,next(serial),part,start))
        while pending:
            bound,_,part,label=heappop(pending)
            if len(best)==k and round(bound,9)>best[-1][0]:
                pruned+=len(pending)+1;break
            if label.position==unit_count:
                if label.device=="vta" and 1<=label.islands<=max_islands:terminal(label,part)
                continue
            expanded+=1
            for nxt in successors(label,part):
                heappush(pending,(_score(nxt,method),next(serial),part,nxt))
    return {"algorithm":"independent_depth_first_enumeration" if reference else "complete_interval_resource_label_setting_dp",
            "method":method,"rankings":[x[2] for x in best],"expanded_labels":expanded,
            "visited_terminals":completed,"bound_pruned_labels":pruned,"top_k":k,
            "cost_aliases":costs.aliases,"rule":rule,"unit_count":unit_count,
            "ranking_uses_pipeline_feedback":False}
