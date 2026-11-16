"""Exhaustive mask search; at most 50,625 allocations in memory per topology."""
import bisect
import itertools
import random
from functools import lru_cache

import numpy as np
from cpu_vta_partitioning.domain import Stage, topologies
from .domain import METHODS, masks, configuration, fixed_compatible
from .models import score


@lru_cache(maxsize=8)
def allocations(n, choices):
    return np.indices((choices,) * n, dtype=np.int16).reshape(n, -1).T


def first_k(values, eligible, k):
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return indices
    values = np.round(values[indices], 9)
    if len(indices) > k:
        cutoff = np.partition(values, k-1)[k-1]
        keep = np.flatnonzero(values <= cutoff)
        # Cartesian allocations are already in candidate-ID lexicographic order.
        strict = keep[values[keep] < cutoff]
        tied = keep[values[keep] == cutoff][:k-len(strict)]
        chosen = np.concatenate((strict, tied))
        order = np.lexsort((indices[chosen], values[chosen]))
        return indices[chosen[order]]
    return indices[np.lexsort((indices, values))]


def topology_values(topology, costs, environment):
    options = masks(environment["cpu_cores"])
    bit_options = np.array([sum(1 << c for c in m) for m in options])
    widths = np.array([len(m) for m in options])
    cpus = [p for p in topology if p["device"] == "cpu"]
    index = allocations(len(cpus), len(options))
    selected_masks = bit_options[index]
    selected_widths = widths[index]
    services, runs, works = [], [], []
    for i,p in enumerate(cpus):
        cs = [costs[Stage("cpu",p["begin"],p["end"],len(m),m).signature] for m in options]
        services.append(np.array([c.service_ms for c in cs])[index[:,i]])
        runs.append(np.array([c.run_core_ms for c in cs])[index[:,i]])
        works.append(np.array([c.run_core_ms+c.boundary_core_ms for c in cs])[index[:,i]])
    services, runs, works = map(lambda x:np.stack(x,axis=1), (services,runs,works))
    vta_service = host_work = 0.
    for p in topology:
        if p["device"] == "vta":
            c = costs[Stage("vta",p["begin"],p["end"],1,tuple(environment["host_cores"])).signature]
            vta_service += c.service_ms
            host_work += c.run_core_ms+c.boundary_core_ms
    service = np.maximum(services.max(axis=1),vta_service)
    total = np.maximum(service,(works.sum(axis=1)+host_work)/4)
    if environment["name"] == "dedicated3":
        total = np.maximum(np.maximum(total,works.sum(axis=1)/3),host_work)
    generic = service.copy()
    host_mask = sum(1 << c for c in environment["host_cores"])
    for subset in range(1,16):
        demand = np.where((selected_masks & subset)==selected_masks, works, 0.).sum(axis=1)
        if host_mask & subset == host_mask:
            demand += host_work
        generic = np.maximum(generic,demand/subset.bit_count())
    fixed = np.ones(len(index),dtype=bool)
    for i,j in itertools.combinations(range(len(cpus)),2):
        a,b = selected_masks[:,i],selected_masks[:,j]
        fixed &= (a==b)|((a&b)==0)
    device = total.copy()
    fixed_indices = np.flatnonzero(fixed)
    for mask in bit_options:
        device[fixed_indices] = np.maximum(device[fixed_indices],np.where(
            selected_masks[fixed_indices]==mask,services[fixed_indices],0.).sum(axis=1))
    prefix_bits = [(1 << t)-1 for t in range(1,len(environment["cpu_cores"])+1)]
    prefix = np.isin(selected_masks,prefix_bits).all(axis=1)
    existing = generic.copy()
    if environment["name"] == "shared4":
        # Preserve the frozen complete model: total ALL work / 4 plus the old
        # local constraints for CPU run only; boundary CPU is not added twice.
        existing = np.maximum(service,(works.sum(axis=1)+host_work)/4)
        for k in range(1,5):
            existing = np.maximum(existing,np.where(selected_widths<=k,runs,0.).sum(axis=1)/k)
    all_ = np.ones(len(index),dtype=bool)
    return options,index,{"single_core":(device,(selected_widths==1).all(axis=1)),
        "fixed_groups_serial":(device,fixed),"fixed_groups_open":(device,fixed),
        "total_capacity":(total,all_),"existing":(existing,prefix),
        "subset_capacity":(generic,all_),"legacy_independent":(service,all_)}


def rank(programs, costs, environment, *, k=20, seed=20261102, topology_filter=None):
    best = {m:[] for m in (*METHODS,"legacy_independent")}
    rows, random_pool = {}, {i:[] for i in (1,2,3)}
    rng = np.random.default_rng(seed)
    counts = {"topologies":0, **{m:0 for m in best}}
    for topology in topologies(programs):
        if topology_filter is not None and tuple(p["name"] for p in topology) != tuple(topology_filter):
            continue
        counts["topologies"] += 1
        options,index,values = topology_values(topology,costs,environment)
        def make(i,method):
            mode = "resource_group" if method == "fixed_groups_serial" else "off"
            return configuration(topology,[options[j] for j in index[i]],environment,mode)
        for method,(scores,eligible) in values.items():
            counts[method] += int(eligible.sum())
            for i in first_k(scores,eligible,k):
                value = round(float(scores[i]),9)
                if len(best[method]) == k and value > best[method][-1][0]:
                    continue
                r = make(i,method)
                item = (value,r["id"])
                if len(best[method]) < k or item < best[method][-1]:
                    bisect.insort(best[method],item)
                    del best[method][k:]
                    rows[r["id"]] = {**r,"score_ms":value}
        if topology_filter is None:
            islands = sum(p["device"]=="vta" for p in topology)
            priorities = rng.random(len(index))
            sampled = np.argpartition(priorities,min(41,len(index)-1))[:42]
            for i in sampled:
                priority = float(priorities[i])
                pool = random_pool[islands]
                if len(pool)==42 and priority > pool[-1][0]:
                    continue
                r = make(i,"subset_capacity")
                bisect.insort(pool,(priority,r["id"],r))
                del pool[42:]
        # Do not retain the candidates displaced from every bounded list.
        live = {key for items in best.values() for _,key in items}
        rows = {key:rows[key] for key in live}
    rankings = {m:[{**rows[key],"score_ms":value} for value,key in items] for m,items in best.items()}
    excluded = {r["id"] for m in METHODS for r in rankings[m][:5]}
    controls = []
    shuffle = random.Random(seed)
    for pool in random_pool.values():
        candidates = [r for _,key,r in pool if key not in excluded]
        shuffle.shuffle(candidates)
        controls.extend(candidates[:4])
    return {"environment":environment,"counts":counts,"rankings":rankings,
            "stratified_random":controls,"seed":seed,"source":"exhaustive_calibrated_search"}


def reference(programs,costs,environment,k=20):
    """Independent scalar Cartesian enumeration for adversarial small domains."""
    result = {m:[] for m in (*METHODS,"legacy_independent")}
    options = masks(environment["cpu_cores"])
    prefix = [tuple(range(t)) for t in range(1,len(environment["cpu_cores"])+1)]
    for topology in topologies(programs):
        for allocation in itertools.product(options,repeat=sum(p["device"]=="cpu" for p in topology)):
            r = configuration(topology,allocation,environment)
            values = score(r,costs,environment)
            for m in result:
                if m=="single_core" and any(len(a)!=1 for a in allocation): continue
                if m.startswith("fixed_groups") and not fixed_compatible(allocation): continue
                if m=="existing" and any(a not in prefix for a in allocation): continue
                rr = configuration(topology,allocation,environment,
                                   "resource_group" if m=="fixed_groups_serial" else "off")
                bisect.insort(result[m],(round(values[m],9),rr["id"]))
                del result[m][k:]
    return result


def screening_pool(ranking):
    values = [r for m in METHODS for r in ranking["rankings"][m][:5]] + ranking["stratified_random"]
    return {r["id"]:r for r in values}


def select(ranking, measurements):
    import statistics
    selection = {}
    for method in METHODS:
        rows = ranking["rankings"][method][:5]
        for r in rows:
            if len(measurements[r["id"]])!=2:
                raise ValueError("selection requires two independent screening boots")
        winner = min(rows,key=lambda r:(statistics.mean(measurements[r["id"]].values()),r["id"]))
        selection[method] = {"1":rows[0]["id"],"5":winner["id"]}
    return selection
