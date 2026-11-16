"""Reuse the existing bounded collector actor with independently profiled work.

The frozen finite predictor rejects observed pipeline controls. This adapter
keeps that restriction and accepts only an exact isolated serialization-cost
projection. Input production, collection bookkeeping and buffer lifetime work
remain unmodeled; no whole-collector-cost claim is made.
"""
from collections import deque
from fractions import Fraction
from compiled_sleep_model import CompiledFinite


FIELDS = {"schema", "origin", "network", "ring_count", "CPU_ns", "warmup", "scored",
          "output_bytes", "output_count", "receipt_sha256", "plan_sha256", "runtime_sha256",
          "serializer_source_sha256", "contains_pipeline_data", "measured_scope"}


def validate_cost(cost):
    if set(cost) != FIELDS:
        raise ValueError("only an isolated output-serialization cost projection is accepted")
    if (cost["schema"] != "isolated-output-control-v1" or cost["origin"] != "isolated_output_serialization"
        or cost["contains_pipeline_data"] is not False or cost["measured_scope"] != "ResultJSON_and_deferred_string_insertion"):
        raise ValueError("observed pipeline control costs cannot enter this predictor")
    if cost["network"] not in ("resnet18", "yolov3_tiny416") or cost["ring_count"] not in (1, 4):
        raise ValueError("unknown independently measured reference signature")
    if type(cost["CPU_ns"]) is not int or cost["CPU_ns"] < 0 or cost["warmup"] != 2 or cost["scored"] != 8:
        raise ValueError("invalid isolated measurement scope")
    if any(not isinstance(cost[k], str) or len(cost[k]) != 64 or
           any(c not in "0123456789abcdef" for c in cost[k]) for k in
           ("receipt_sha256", "plan_sha256", "runtime_sha256", "serializer_source_sha256")):
        raise ValueError("isolated source identity missing")
    return cost


class IsolatedOutputFinite(CompiledFinite):
    def __init__(self, *args, isolated_output_cost, collector_mask, **kwargs):
        self.isolated_output_cost = dict(validate_cost(isolated_output_cost))
        if kwargs.get("controls"):
            raise ValueError("pipeline control observations remain forbidden")
        if (not collector_mask or sorted(set(collector_mask)) != collector_mask
            or any(type(c) is not int or c not in range(4) for c in collector_mask)):
            raise ValueError("invalid collector allowed cores")
        super().__init__(*args, **kwargs)
        # Initialization is complete but no tasks have started. Reuse the
        # existing AuditReplay producer/collector queues and actors unchanged.
        if self.source or self.active or any(self.queues):
            raise ValueError("collector must be configured before execution")
        cpu = Fraction(self.isolated_output_cost["CPU_ns"])
        if cpu:
            self.controls = {
                "_producer": dict(CPU_ns=0, mask=collector_mask),
                "_collector": dict(CPU_ns=cpu, mask=collector_mask),
            }
            self.control_work = {"_producer": Fraction(0), "_collector": cpu}
            self.control_enabled = True
            self.queues.append(deque())

    def run(self):
        r = super().run()
        r.update(isolated_output_cost=self.isolated_output_cost,
            collector_scope="independently measured ResultJSON and string insertion only",
            input_production_CPU_unmodeled=True,
            frame_lifetime_and_other_collector_CPU_unmodeled=True,
            output_cost_not_added_directly_to_II=True,
            no_pipeline_control_values_used=True)
        r["replay"]["control_CPU_unmodeled"] = True  # Other control work remains unknown.
        r["replay"]["output_serialization_CPU_modeled"] = bool(self.isolated_output_cost["CPU_ns"])
        return r
