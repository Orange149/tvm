"""Frozen R/C/H/A/O/G/D formulas. All time units are milliseconds.

Boundary ownership: each stage owns its set_input and get_output exactly once.
VTA service is serialized across ALL VTA stages by the native executor mutex.
Core-time is measured process CPU time in isolated serial sessions, not wall time.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Cost:
    service_ms: float
    run_core_ms: float
    boundary_core_ms: float

    @classmethod
    def from_dict(cls, row):
        result = cls(*(float(row[k]) for k in ("service_ms", "run_core_ms", "boundary_core_ms")))
        if any(not math.isfinite(v) or v < 0 for v in result.__dict__.values()):
            raise ValueError("negative or nonfinite cost")
        return result


def score(stages, costs, cores=4):
    """Deliberately simple reference implementation, also used for explanations."""
    cpu, vta, wc, wh, wb = [], [], 0., 0., 0.
    old = [0.] * cores
    work = []
    for stage in stages:
        c = costs[stage.signature]
        if stage.device == "cpu":
            cpu.append(c.service_ms)
            wc += c.run_core_ms
            for k in range(stage.threads, cores + 1):
                old[k - 1] += c.run_core_ms / k
        else:
            vta.append(c.service_ms)
            wh += c.run_core_ms
        wb += c.boundary_core_ms
        work.append((sum(1 << x for x in stage.mask), c.run_core_ms + c.boundary_core_ms))
    service = max([sum(vta), *cpu])
    capacity = {str(subset): sum(w for mask, w in work if mask & subset == mask) / subset.bit_count()
                for subset in range(1, 1 << cores)}
    c_score = max(service, (wc + wb) / cores)
    h_score = max(service, (wc + wh + wb) / cores)
    generic = max(service, *capacity.values())
    return {"R": service, "C": c_score, "H": h_score,
            "A": max(c_score, *old), "O": max(h_score, *old), "G": generic, "D": generic,
            "components": {"service": service, "cpu_core_ms": wc, "host_core_ms": wh,
                           "boundary_core_ms": wb, "local_cpu": max(old), "capacity": capacity}}


def load_costs(document, *, environment=None, bundle_id=None):
    if document.get("source") != "board_calibration":
        raise ValueError("rankings require fresh board calibration, not historical/synthetic costs")
    if environment is not None and document["environment"] != environment:
        raise ValueError("cost environment mismatch")
    if bundle_id is not None and document["bundle_id"] != bundle_id:
        raise ValueError("cost program identity mismatch")
    if len(set(document["boot_ids"])) != 2:
        raise ValueError("calibration requires two distinct kernel boot IDs")
    return {k: Cost.from_dict(v) for k, v in document["costs"].items()}
