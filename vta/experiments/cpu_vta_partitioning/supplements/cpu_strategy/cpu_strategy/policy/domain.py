"""Explicit execution identities, qualified intervals and fixed CPU devices."""
from copy import deepcopy
from itertools import product

from cpu_vta_partitioning.common import digest
from cpu_vta_partitioning.domain import Stage
from ..domain import ROOT, PROJECT

CORES = (0, 1, 2, 3)
CUTS = {"current": (3, 18), "historical_cut": (3, 17), "light_tail": (3, 20)}
BASE_THREADS = {"current": (4, 2), "historical_cut": (4, 3), "light_tail": (4, 1)}
FAMILIES = ("current", "local_fastest", "fixed_groups", "same_groups_pinned", "pinned_reallocated", "reserved_host")
METHODS = ("single_core", "prefix", "enhanced")
ENV = {"cores": 4, "frequency_khz": 1066666, "poll_sleep_ns": 1000, "post_start_sleep_ns": 1000,
       "spin_count": 0, "exclude_worker0": 1, "bind_threads": 1}


def protocol():
    return {"experiment": "resnet18_resource_policy_dp", "schema_version": 1, "seed": 20261104,
            "unit_count": 21, "maximum_vta_islands": 3, "environment": deepcopy(ENV),
            "cuts": {k:list(v) for k,v in CUTS.items()}, "families": list(FAMILIES),
            "boot_phases": ["policy_discovery", "policy_check", "top20_screening", "confirmation", "confirmation", "confirmation"],
            "rounds": 2, "calibration": {"indices": [0], "warmup": 5, "scored": 32, "tail": 0},
            "performance": {"indices": [16], "warmup": 8, "extended_warmup": 24, "scored": 32, "tail": 4},
            "qualification_indices": [0,16,17,18], "anchor_drift": .05, "max_attempts": 2,
            "practical_gain": .05, "maximum_cut_regression": .05, "mirror_tolerance": .05,
            "top_k": 20, "comparison_budget": 5, "max_policy_pool": 26, "max_search_pool": 30,
            "cpu_gate_modes": ["off", "resource_group"], "copy": "ordinary", "queue_depth": 2,
            "numeric": {"integer":"exact", "rtol":1e-4, "atol":1e-5},
            "boot_policy": "manual_reboot_only", "formal_significance": False,
            "historical_11511": "different_float_boundaries_and_input_not_a_reproduction_target"}


def partitions(cores, contiguous=False):
    """Canonical set partitions; four cores have 15 and three cores have 5."""
    cores = tuple(cores)
    def visit(index, groups):
        if index == len(cores):
            result = tuple(tuple(g) for g in groups)
            if not contiguous or all(tuple(range(g[0],g[-1]+1)) == g for g in result):
                yield result
            return
        core = cores[index]
        for i in range(len(groups)):
            yield from visit(index+1, [g+[core] if j==i else g for j,g in enumerate(groups)])
        yield from visit(index+1, groups+[[core]])
    yield from visit(0, [])


def stage(device, begin, end, threads, mask):
    return Stage(device,begin,end,threads,tuple(mask)).to_dict()


def candidate(stages, mode="group", gate="off", host=CORES, control=CORES, partition=()):
    partition=tuple(sorted({tuple(r["mask"]) for r in stages if r["device"]=="cpu"})) if gate=="resource_group" else ()
    value = {"stages":deepcopy(stages), "affinity_mode":mode, "execution_mode":gate,
             "host_mask":list(host), "control_mask":list(control),
             "core_partition":[list(g) for g in partition]}
    for row in value["stages"]:
        row["worker_core_map"] = list(row["mask"]) if row["device"]=="cpu" and mode=="pinned" else []
    value["id"] = "cfg_"+digest(value)[:24]
    validate_candidate(value)
    return value


def validate_candidate(value, programs=None, component=False):
    mode, gate = value["affinity_mode"], value["execution_mode"]
    if mode not in ("group","pinned") or gate not in ("off","resource_group"):
        raise ValueError("unsupported affinity/gate mode")
    for key in ("host_mask","control_mask"):
        mask=value[key]
        if not mask or any(type(x)is not int for x in mask) or mask!=sorted(set(mask)) or not set(mask)<=set(CORES):
            raise ValueError("invalid host/control affinity")
    rows = value["stages"]
    if not rows: raise ValueError("empty candidate")
    cpumasks=[]
    for i,row in enumerate(rows):
        s=Stage.from_dict(row)
        if (s.device not in ("cpu","vta") or type(s.begin) is not int or type(s.end) is not int
                or not 0<=s.begin<s.end<=21 or type(s.threads) is not int or s.threads<1
                or not s.mask or any(type(x)is not int for x in s.mask) or s.mask!=tuple(sorted(set(s.mask))) or not set(s.mask)<=set(CORES)
                or row.get("program")!=s.program or row.get("signature")!=s.signature):
            raise ValueError("invalid stage or stage identity")
        if programs is not None:
            if s.program not in programs or any(programs[s.program][k]!=getattr(s,k) for k in ("device","begin","end")):
                raise ValueError("unqualified interval")
        if s.device=="cpu":
            if s.threads>len(s.mask) or (mode=="pinned" and (s.threads!=len(s.mask) or row.get("worker_core_map")!=list(s.mask))):
                raise ValueError("CPU slot/core mapping differs")
            if mode=="group" and row.get("worker_core_map",[]): raise ValueError("group mode has a worker map")
            if gate=="resource_group" and s.threads!=len(s.mask): raise ValueError("fixed group needs one thread per core")
            cpumasks.append(s.mask)
        elif s.threads!=1 or list(s.mask)!=value["host_mask"]:
            raise ValueError("VTA host placement differs")
        if not component and i and (rows[i-1]["end"]!=s.begin or rows[i-1]["device"]==s.device):
            raise ValueError("noncontiguous/nonalternating stages")
    if gate=="resource_group" and any(set(a)&set(b) and a!=b for a in cpumasks for b in cpumasks):
        raise ValueError("fixed CPU groups partially overlap")
    if value["host_mask"]==[3] and any(3 in m for m in cpumasks):
        raise ValueError("reserved host core used by CPU computation")
    if not component and (rows[0]["device"]!="cpu" or rows[0]["begin"]!=0 or rows[-1]["device"]!="cpu" or rows[-1]["end"]!=21
                          or not 1<=sum(r["device"]=="vta" for r in rows)<=3):
        raise ValueError("candidate does not cover the complete qualified graph")
    if "id" in value and value["id"]!="cfg_"+digest({k:v for k,v in value.items() if k!="id"})[:24]:
        raise ValueError("candidate identity differs")
    return tuple(Stage.from_dict(r) for r in rows)


def cut_candidate(cut, threads, masks, mode="group", gate="off", reserved=False, partition=()):
    a,b=CUTS[cut];host=(3,) if reserved else CORES
    return candidate([stage("cpu",0,a,threads[0],masks[0]), stage("vta",a,b,1,host),
                      stage("cpu",b,21,threads[1],masks[1])],mode,gate,host,host,partition)


def anchor():
    return cut_candidate("current",(1,1),((1,),(0,)))


def component(row, mode="group", control=CORES):
    row=deepcopy(row)
    host=tuple(row["mask"]) if row["device"]=="vta" else CORES
    actual_mode=mode if row["device"]=="cpu" else "group"
    if row["device"]=="cpu": control=CORES
    return candidate_component(row,actual_mode,host,control)


def candidate_component(row,mode,host,control):
    row["worker_core_map"]=list(row["mask"]) if row["device"]=="cpu" and mode=="pinned" else []
    value={"stages":[row],"affinity_mode":mode,"execution_mode":"off", "host_mask":list(host),
           "control_mask":list(control),"core_partition":[]}
    validate_candidate(value,component=True)
    value["id"]="cfg_"+digest(value)[:24]
    return value


def component_key(value):
    validate_candidate(value,component=True)
    row=value["stages"][0]
    # A serial CPU caller performs computation, not pipeline control. Its
    # control-mask choice cannot change the isolated operation.
    return "cmp_"+digest({"stage":row,"affinity_mode":value["affinity_mode"],
        "control_mask":value["control_mask"] if row["device"]=="vta" else None})[:24]


def policy_components():
    values={}
    for name in CUTS:
        a,b=CUTS[name]
        for begin,end in ((0,a),(b,21)):
            for mode in ("group","pinned"):
                for n in range(1,5):
                    for mask in (tuple(range(n)),tuple(range(4-n,4))):
                        c=component(stage("cpu",begin,end,n,mask),mode)
                        values[component_key(c)]=c
            # Local-fastest and legacy shared controls have wider masks.
            for n in range(1,5):
                c=component(stage("cpu",begin,end,n,CORES));values[component_key(c)]=c
        for host in (CORES,(3,)):
            c=component(stage("vta",a,b,1,host),control=host);values[component_key(c)]=c
    return dict(sorted(values.items()))


def complete_components(programs, mode, reserved=False, homogeneous=True):
    cores=CORES[:3] if reserved else CORES;host=(3,) if reserved else CORES
    values={}
    for p in programs.values():
        if p["device"]=="vta":
            masks=[host]
        elif homogeneous:
            masks=[cores[:n] for n in range(1,len(cores)+1)]
        else:
            masks=sorted({g for part in partitions(cores,True) for g in part})
        for mask in masks:
            n=len(mask) if p["device"]=="cpu" else 1
            c=component(stage(p["device"],p["begin"],p["end"],n,mask),mode,host)
            values[component_key(c)]=c
    return dict(sorted(values.items()))
