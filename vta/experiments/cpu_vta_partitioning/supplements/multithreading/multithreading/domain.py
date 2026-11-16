"""Physical CPU mappings, fixed partitions, and canonical executable identities."""
import itertools
from pathlib import Path

from cpu_vta_partitioning.common import digest
from cpu_vta_partitioning.domain import Stage, candidate_id, topologies

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
METHODS = ("single_core", "fixed_groups_serial", "fixed_groups_open", "total_capacity",
           "existing", "subset_capacity")
NAMES = dict(zip(METHODS, ("单线程物理核心映射", "固定核心组多线程，组内串行",
    "固定核心组多线程，无门控", "重叠核心组＋总核时容量", "现有方法",
    "重叠核心组＋全部子集容量")))
NAMES["legacy_independent"] = "旧简化对照（忽略共享容量）"
ENV_NAMES = {"shared4":"四核共享","dedicated3":"三核计算、一核专用主机"}
ENVIRONMENTS = (
    {"name":"shared4", "cores":4, "cpu_cores":[0,1,2,3], "host_cores":[0,1,2,3],
     "poll_sleep_ns":1000, "post_start_sleep_ns":1000},
    {"name":"dedicated3", "cores":4, "cpu_cores":[0,1,2], "host_cores":[3],
     "poll_sleep_ns":1000, "post_start_sleep_ns":1000})


def masks(cores):
    return sorted((tuple(c for c in cores if bits & (1 << c))
                   for bits in range(1, 1 << 4) if bits & ~sum(1 << c for c in cores) == 0),
                  key=lambda m:(len(m), m))


def partitions(items):
    """All set partitions, once each; unused blocks do not affect candidate IDs."""
    items = tuple(items)
    if not items:
        yield ()
        return
    head, *tail = items
    for part in partitions(tail):
        yield ((head,),) + part
        for i in range(len(part)):
            yield part[:i] + ((head,) + part[i],) + part[i+1:]


def fixed_compatible(cpu_masks):
    return all(a == b or not set(a) & set(b)
               for a,b in itertools.combinations(cpu_masks, 2))


def configuration(topology, allocation, environment, mode="off"):
    it = iter(allocation)
    stages = []
    for p in topology:
        mask = tuple(next(it)) if p["device"] == "cpu" else tuple(environment["host_cores"])
        stages.append(Stage(p["device"], p["begin"], p["end"],
                            len(mask) if p["device"] == "cpu" else 1, mask))
    return row(stages, mode)


def row(stages, mode="off"):
    stages = tuple(s if isinstance(s, Stage) else Stage.from_dict(s) for s in stages)
    if mode not in ("off", "resource_group"):
        raise ValueError("unsupported CPU execution mode")
    if mode == "resource_group" and not fixed_compatible([s.mask for s in stages if s.device == "cpu"]):
        raise ValueError("fixed resource groups partially overlap")
    return {"id":candidate_id(stages) + "__gate_" + mode,
            "stages":[s.to_dict() for s in stages], "execution_mode":mode}


def validate_row(value, programs, environment):
    from cpu_vta_partitioning.domain import validate_candidate
    stages = tuple(Stage.from_dict(s) for s in value["stages"])
    validate_candidate(stages, programs, 4)
    if row(stages, value["execution_mode"])["id"] != value["id"]:
        raise ValueError("candidate identity mismatch")
    for s in stages:
        if s.device == "cpu" and not set(s.mask) <= set(environment["cpu_cores"]):
            raise ValueError("CPU mask escapes environment")
        if s.device == "vta" and s.mask != tuple(environment["host_cores"]):
            raise ValueError("VTA host mask mismatch")
    return stages


def signatures(programs):
    result = {}
    for p in programs.values():
        choices = masks(range(4)) if p["device"] == "cpu" else [(0,1,2,3), (3,)]
        for mask in choices:
            s = Stage(p["device"], p["begin"], p["end"],
                      len(mask) if p["device"] == "cpu" else 1, mask)
            result[s.signature] = s
    return dict(sorted(result.items()))


def protocol():
    return {"experiment":"resnet18_multithread_mapping", "schema_version":1,
        "seed":20261102, "methods":list(METHODS), "environments":list(ENVIRONMENTS),
        "unit_count":21, "maximum_vta_islands":3, "top_k":20, "screen_top_k":5,
        "random_per_island":4, "calibration":{"boots":2,"rounds":2,"warmup":5,"scored":32},
        "screening_boots":2, "confirmation_boots":6, "rounds":2, "budgets":[1,5],
        "performance":{"warmup":20,"extended_warmup":100,"scored":300,"tail":20,
                       "warmup_tolerance":.02,"anchor_drift":.05,"replacement_rounds":2},
        "qualification_indices":list(range(16))+[64,65,66],
        "evaluation_indices":list(range(16,64)), "numeric":{"integer":"exact","rtol":1e-4,"atol":1e-5},
        "statistics":{"alpha":.05,"practical_gain":.05,"equivalence_margin":.05,
                      "primary_comparators":["single_core","fixed_groups_open","subset_capacity"]},
        "expected":{"cpu_programs":85,"vta_programs":87,"signatures":1449,"topologies":4623,
                    "shared4_configurations":182877075,"dedicated3_configurations":8857779},
        "runtime":{"queue_depth":2,"spin_count":0,"replay":"disabled","copy":"ordinary"},
        "boot_policy":"manual_reboot_only", "score_rounding_ms":1e-9}


def task(kind, stages, environment, *, execution_mode="off", **kwargs):
    from cpu_vta_partitioning.measurement.tasks import task as base_task
    if kind in ("calibration", "qualify_component") and execution_mode != "off":
        raise ValueError("isolated component tasks cannot use pipeline gates")
    return base_task(kind, stages, environment, execution_mode=execution_mode,
                     experiment="resnet18_multithread_mapping", **kwargs)


def task_for(kind, candidate, environment, **kwargs):
    return task(kind, candidate["stages"], environment,
                execution_mode=candidate["execution_mode"], candidate=candidate["id"], **kwargs)
