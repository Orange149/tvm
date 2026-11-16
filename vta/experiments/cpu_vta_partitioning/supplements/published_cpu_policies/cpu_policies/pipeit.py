# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Paper transcription of Pipe-it's additive-cost pipeline heuristics.

Source: Wang et al., *High-Throughput CNN Inference on Embedded ARM
big.LITTLE Multi-Core Processors*, arXiv:1903.05898v3, Section VI, printed
pages 8--10, Algorithms 1--3 and Equations (9), (11), (12).

This module independently transcribes the paper; it does not copy the ARM
Compute Library implementation. The author-linked runtime is Elliezza/CLCode,
commit ffa5d82c188398805dcbdfc50ffb7b23a41c57cf. Its fixed workers and stage
locks are separate execution requirements, not implemented by this scheduler.

Times are per-layer costs indexed by (core type, core count). Stage costs are
strictly their Eq. (9) sum. There is no fused-interval callback, no FPGA-host
contention term, and no replacement of the heuristic with an exact minimax
solver. In particular Algorithm 1's strict inequality can reject an optimal
crossover split; that behavior is intentionally preserved.

The printed Algorithm 3 does not fully specify traversal after mutating P.
The caller must choose an explicit ``variant``. Both implemented traversals
retry the newly merged stage, following the explanatory text's B2 -> B3 -> B4
example. ``paper_break_retry_merged`` stops a core-type loop on the first
rejected merge, as its printed ``break`` states. The alternative
``continue_adjacent_retry_merged`` continues with the next pair on rejection
and is explicitly an adaptation, not an assertion about unpublished code.
"""
from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Real
from typing import Mapping, Sequence


PAPER_URL = "https://arxiv.org/pdf/1903.05898v3"
RUNTIME_COMMIT = "ffa5d82c188398805dcbdfc50ffb7b23a41c57cf"
MERGE_VARIANTS = ("paper_break_retry_merged", "continue_adjacent_retry_merged")
TimeColumns = Mapping[tuple[str, int], Sequence[float]]


@dataclass(frozen=True)
class CoreGroup:
    """A stage's fixed, disjoint physical compute cores (not caller threads)."""

    core_type: str
    cores: tuple[int, ...]

    def __post_init__(self):
        values = tuple(self.cores)
        if not isinstance(self.core_type, str) or not self.core_type:
            raise ValueError("core type must be a nonempty string")
        if (not values or any(type(i) is not int or i < 0 for i in values)
                or values != tuple(sorted(set(values)))):
            raise ValueError("core IDs must be nonempty, unique, sorted nonnegative integers")
        object.__setattr__(self, "cores", values)

    @property
    def configuration(self):
        return self.core_type, len(self.cores)

    def to_dict(self):
        return {"core_type": self.core_type, "core_count": len(self.cores), "cores": list(self.cores)}


def _times(values):
    if not values:
        raise ValueError("time column must contain at least one layer")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) or v < 0 for v in values):
        raise ValueError("layer times must be finite nonnegative numbers")
    return tuple(float(v) for v in values)


def _columns(time_columns):
    if not time_columns:
        raise ValueError("empty time matrix")
    answer = {}
    for key, value in time_columns.items():
        if (not isinstance(key, tuple) or len(key) != 2 or not isinstance(key[0], str)
                or not key[0] or type(key[1]) is not int or key[1] < 1):
            raise ValueError("time columns must use (core_type, positive_core_count) keys")
        answer[key] = _times(value)
    if len({len(v) for v in answer.values()}) != 1:
        raise ValueError("time columns have different layer counts")
    return answer


def _groups(groups):
    groups = tuple(groups)
    if not groups or any(not isinstance(g, CoreGroup) for g in groups):
        raise ValueError("pipeline must contain CoreGroup objects")
    cpus = [cpu for group in groups for cpu in group.cores]
    if len(cpus) != len(set(cpus)):
        raise ValueError("Pipe-it pipeline compute groups must not share physical cores")
    return groups


def find_split(layers, first_times, second_times):
    """Algorithm 1: transfer a suffix while new upstream > new downstream.

    Layer IDs are zero based and must be consecutive in original order.
    Equality and crossover are both rejected, exactly as in the pseudocode.
    This is deliberately not a scan over all split points for the best maximum.
    """
    first_times, second_times = _times(first_times), _times(second_times)
    if len(first_times) != len(second_times):
        raise ValueError("time columns have different layer counts")
    layers = tuple(layers)
    if any(type(i) is not int or not 0 <= i < len(first_times) for i in layers):
        raise ValueError("layer ID outside the time matrix")
    if layers and layers != tuple(range(layers[0], layers[-1] + 1)):
        raise ValueError("workload layers must be consecutive and in original order")
    left, right = list(layers), []
    for layer in reversed(layers):
        # Evaluate the mathematical remaining/transferred sums directly.
        # Subtracting a floating total can turn equal 0.1/0.1 into a spurious
        # strict inequality (e.g. fsum([0.1, 0.2]) - 0.2 > 0.1).
        new_first = fsum(first_times[i] for i in left[:-1])
        new_second = fsum(second_times[i] for i in [layer, *right])
        if new_first > new_second:
            left.pop()
            right.insert(0, layer)
        else:
            break
    return left, right


def _workflow(columns, groups):
    missing = [g.configuration for g in groups if g.configuration not in columns]
    if missing:
        raise ValueError("missing matched layer time configuration: " + str(missing[0]))
    count = len(next(iter(columns.values())))
    layers = [list(range(count)), *[[] for _ in groups[1:]]]
    seen = set()
    while True:
        identity = tuple(map(tuple, layers))
        if identity in seen:
            raise RuntimeError("Pipe-it workload stabilization failed: repeated allocation cycle")
        seen.add(identity)
        old = [list(group) for group in layers]
        for i in range(len(groups)-1):
            workload = layers[i] + layers[i+1]
            layers[i], layers[i+1] = find_split(
                workload, columns[groups[i].configuration], columns[groups[i+1].configuration])
        if layers == old:
            break
    stage_times = [fsum(columns[g.configuration][i] for i in assigned)
                   for g, assigned in zip(groups, layers)]
    interval = max(stage_times)
    # Eq. (11) assumes positive processing time; a zero-work synthetic graph
    # does not have a finite throughput prediction.
    return {"groups": [g.to_dict() for g in groups], "layers": layers,
            "stage_times": stage_times, "interval": interval,
            "throughput": 1 / interval if interval else None, "passes": len(seen),
            "cost_model": "paper_eq9_additive_layers",
            "objective": "paper_eq11_inverse_max_stage_time"}


def work_flow(time_columns: TimeColumns, groups: Sequence[CoreGroup]):
    """Algorithm 2: repeatedly balance adjacent stages until unchanged.

    Groups are supplied in the paper's proposed faster-to-slower order. No
    sorting is inferred from physical core IDs and no layer is reordered.
    Empty stages are retained, as allowed by the paper. Costs and inputs are
    not mutated. A homogeneous-only inventory is a one-core-type specialization
    of these equations, not a reproduction of the paper's two-cluster board.
    """
    return _workflow(_columns(time_columns), _groups(groups))


def merge_stage(time_columns: TimeColumns, core_inventory: Mapping[str, Sequence[int]], *, variant: str):
    """Algorithm 3 with a mandatory, recorded traversal interpretation.

    ``core_inventory`` preserves cluster/type order: faster cluster first.
    Start with one stage per physical core; only same-type neighbors may
    merge. Eq. (12) is strict. Every accepted merge reruns Algorithm 2.
    All counts 1..H of each type require actual/supplied cost columns.
    """
    if variant not in MERGE_VARIANTS:
        raise ValueError("choose an explicit supported merge traversal variant")
    columns = _columns(time_columns)
    if not core_inventory:
        raise ValueError("empty core inventory")
    groups = []
    for kind, cores in core_inventory.items():
        whole = CoreGroup(kind, tuple(cores))
        for n in range(1, len(whole.cores)+1):
            if (kind, n) not in columns:
                raise ValueError("missing matched layer time configuration: " + str((kind, n)))
        groups.extend(CoreGroup(kind, (cpu,)) for cpu in whole.cores)
    groups = list(_groups(groups))
    original_inventory = {(g.core_type, cpu) for g in groups for cpu in g.cores}
    current = _workflow(columns, groups)
    history = []
    for kind in core_inventory:
        index = next(i for i, group in enumerate(groups) if group.core_type == kind)
        while index+1 < len(groups) and groups[index+1].core_type == kind:
            left, right = groups[index:index+2]
            merged = CoreGroup(kind, tuple(sorted(left.cores + right.cores)))
            workload = current["layers"][index] + current["layers"][index+1]
            new_time = fsum(columns[merged.configuration][i] for i in workload)
            old_bound = max(current["stage_times"][index:index+2])
            accepted = new_time < old_bound
            event = {"pair": [left.to_dict(), right.to_dict()], "layers": list(workload),
                     "merged_time": new_time, "old_pair_max": old_bound,
                     "accepted": accepted, "interval_before": current["interval"]}
            history.append(event)
            if accepted:
                groups[index:index+2] = [merged]
                current = _workflow(columns, groups)
                event["interval_after"] = current["interval"]
                # Retry this new stage with its next neighbor (explicit variant).
            elif variant == "paper_break_retry_merged":
                break
            else:
                index += 1
    if {(g.core_type, cpu) for g in groups for cpu in g.cores} != original_inventory:
        raise RuntimeError("physical core inventory changed during stage merging")
    return {**current, "algorithm": "pipeit_algorithms_1_2_3",
            "merge_variant": variant, "merge_history": history,
            "paper_source": PAPER_URL,
            "implementation_origin": "independent_paper_pseudocode_transcription",
            "merge_traversal_is_declared_interpretation": True,
            "runtime_contract": "fixed compute workers; disjoint core groups; one job per group at a time"}
