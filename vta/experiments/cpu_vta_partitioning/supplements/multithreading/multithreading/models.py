"""Explicit adaptations of physical-device and CPU core-time models.

Fixed-group scores accumulate isolated wall service per device. In open mode
this is an approximation, not a claim that TVM enforces the paper's schedule.
No model is advertised as an exact prediction of concurrent pipeline II.
"""
from collections import defaultdict
from cpu_vta_partitioning.domain import Stage
from cpu_vta_partitioning.models import Cost, score as original_score
from .domain import masks


def score(candidate, costs, environment):
    stages = [Stage.from_dict(s) for s in candidate["stages"]]
    cpu_service, groups, work = [], defaultdict(float), []
    vta_service = 0.
    cpu_work = host_work = 0.
    for s in stages:
        c = costs[s.signature]
        w = c.run_core_ms + c.boundary_core_ms
        if s.device == "cpu":
            cpu_service.append(c.service_ms)
            groups[s.mask] += c.service_ms
            cpu_work += w
        else:
            vta_service += c.service_ms
            host_work += w
        work.append((set(s.mask), w))
    service = max([vta_service, *cpu_service])
    total = max(service, (cpu_work + host_work) / 4)
    if environment["name"] == "dedicated3":
        total = max(total, cpu_work / 3, host_work)
    subsets = {str(a):sum(w for m,w in work if m <= set(a)) / len(a)
               for a in masks(range(4))}
    fixed = max(total, *groups.values())
    original = original_score(stages, costs, 4)
    return {"single_core":fixed, "fixed_groups_serial":fixed, "fixed_groups_open":fixed,
            "total_capacity":total, "existing":original["O" if environment["name"] == "shared4" else "D"],
            "subset_capacity":max(service, *subsets.values()), "legacy_independent":service,
            "components":{"service_ms":service,"cpu_core_ms":cpu_work,"host_core_ms":host_work,
                          "fixed_group_services":{str(k):v for k,v in groups.items()},"subset_bounds":subsets}}


def load_costs(document, bundle_id):
    from .domain import signatures
    if (document.get("source") != "board_calibration" or document.get("bundle_id") != bundle_id
            or len(document.get("boot_ids",[])) != 2 or len(set(document["boot_ids"])) != 2
            or document.get("experiment") != "resnet18_multithread_mapping"):
        raise ValueError("new matching two-boot calibration is required")
    return {k:Cost.from_dict(v) for k,v in document["costs"].items()}
