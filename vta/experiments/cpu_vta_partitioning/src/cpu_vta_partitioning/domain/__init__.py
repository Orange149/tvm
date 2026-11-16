"""Finite, compiler-qualified contiguous partition domain."""
from dataclasses import dataclass
import itertools


def validate_protocol(config):
    """Changing preregistered rules needs an implementation/protocol version bump."""
    fixed = {"unit_count":21,"maximum_vta_islands":3,"groups":["R","C","H","A","O","G","D"],
             "top_k":20,"seed":20261023,"session_timeout_s":600,
             "runtime":{"queue_depth":2,"spin_count":0,"nice":0,"cpu_gate":"off","boundary_policy":"ordinary_copy","replay":"disabled"},
             "calibration":{"boots":2,"rounds":2,"warmup":5,"scored":32},
             "screening":{"boots":2,"rounds":2,"top_k":5,"random_per_island":4},
             "confirmation":{"boots":6,"rounds":2,"budgets":[1,5]},
             "performance":{"warmup":20,"extended_warmup":100,"scored":300,"tail":20,"warmup_tolerance":.02,"anchor_drift":.05,"max_replacement_rounds":2},
             "numeric":{"integer":"exact","float_atol":.00001,"float_rtol":.0001},
             "inputs":{"calibration_real":16,"evaluation_real":48},
             "statistics":{"alpha":.05,"practical_gain":.05,"equivalence_margin":.05,"comparators":["H","G","D"]}}
    for key,value in fixed.items():
        if config.get(key) != value:
            raise ValueError("unsupported protocol change: " + key + "; update implementation and protocol version together")
    if config['environment'] != {'cores':4,'poll_sleep_ns':1000,'post_start_sleep_ns':1000}:
        raise ValueError("main environment is frozen; use the prepared sensitivity matrix")


@dataclass(frozen=True, order=True)
class Stage:
    device: str
    begin: int
    end: int  # exclusive
    threads: int
    mask: tuple[int, ...]

    @property
    def program(self):
        return f"{self.device}_{self.begin:02}_{self.end - 1:02}"

    @property
    def signature(self):
        return f"{self.program}_t{self.threads}_a{'-'.join(map(str, self.mask))}"

    def to_dict(self):
        return {"device": self.device, "begin": self.begin, "end": self.end,
                "threads": self.threads, "mask": list(self.mask), "program": self.program,
                "signature": self.signature}

    @classmethod
    def from_dict(cls, row):
        return cls(row["device"], row["begin"], row["end"], row["threads"], tuple(row["mask"]))


def candidate_id(stages):
    return "__".join(s.signature for s in stages)


def topology_id(stages):
    return "__".join(s.program for s in stages)


def topologies(programs, unit_count=21, max_islands=3):
    edges = {}
    for p in programs.values():
        edges.setdefault((p["begin"], p["device"]), []).append(p)
    for values in edges.values():
        values.sort(key=lambda x: x["end"])

    def visit(position, device, islands, path):
        if position == unit_count:
            if device == "vta" and 1 <= islands <= max_islands:
                yield tuple(path)
            return
        for item in edges.get((position, device), []):
            count = islands + (device == "vta")
            if count <= max_islands:
                yield from visit(item["end"], "vta" if device == "cpu" else "cpu",
                                 count, path + [item])
    yield from visit(0, "cpu", 0, [])


def configurations(topology, cores=4, dedicated=False):
    cpu_count = sum(p["device"] == "cpu" for p in topology)
    choices = range(1, cores if dedicated else cores + 1)
    for allocation in itertools.product(choices, repeat=cpu_count):
        counts = iter(allocation)
        stages = []
        for p in topology:
            t = next(counts) if p["device"] == "cpu" else 1
            mask = tuple(range(t)) if p["device"] == "cpu" else (
                (cores - 1,) if dedicated else tuple(range(cores)))
            stages.append(Stage(p["device"], p["begin"], p["end"], t, mask))
        yield tuple(stages)


def signatures(programs, cores=4):
    result = {}
    for p in programs.values():
        options = [(t, tuple(range(t))) for t in range(1, cores + 1)] if p["device"] == "cpu" else [
            (1, tuple(range(cores))), (1, (cores - 1,))]
        for t, mask in options:
            s = Stage(p["device"], p["begin"], p["end"], t, mask)
            result[s.signature] = s
    return dict(sorted(result.items()))


def validate_candidate(stages, programs, cores, max_islands=3):
    if not stages or stages[0].begin != 0 or stages[-1].end != 21:
        raise ValueError("candidate does not cover all 21 units")
    if stages[0].device != "cpu" or stages[-1].device != "cpu":
        raise ValueError("CPU endpoints required")
    for i, s in enumerate(stages):
        if s.program not in programs or not s.mask or not set(s.mask) <= set(range(cores)):
            raise ValueError("unqualified program or invalid affinity")
        if s.device == "cpu" and len(s.mask) != s.threads:
            raise ValueError("CPU mask/thread mismatch")
        if i and (stages[i - 1].end != s.begin or stages[i - 1].device == s.device):
            raise ValueError("noncontiguous or noncanonical candidate")
    if not 1 <= sum(s.device == "vta" for s in stages) <= max_islands:
        raise ValueError("invalid VTA island count")
