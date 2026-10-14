"""Isolated adapters to the hash-bound node24 experimental passes.

Only the allowed R domain and slot constant are specialized.  Original
factories, scheduling interfaces, runtime, and historical files stay intact.
R13 must structurally reproduce the frozen anchor before board use.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from c3_residency_tradeoff import digest


BASE_PASS_SHA = "78c55941d630e81966c6ae9fcf2a7b3676ea09d5e68a179fee56deb53c774bb3"
BASE_BUILDER_SHA = "d21d2941bfa0371c5b5bac6080dea266145eecf7940d033fa475e99a606bd769"
ALLOWED_R = (0, 8, 12, 13)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("frozen adapter source pattern changed: " + old)
    return source.replace(old, new, 1)


def compile_factory(function, replacements, namespace):
    source = inspect.getsource(function)
    for old, new in replacements:
        source = replace_once(source, old, new)
    local = {}
    exec(compile(source, __file__ + ":" + function.__name__, "exec"), namespace, local)
    return local[function.__name__]


def make_probe(resident_groups):
    if resident_groups not in ALLOWED_R:
        raise ValueError("R outside frozen experiment")
    import run_vta_c3_node24_partial_local as base
    from vta.top import vta_node24_partial as passes
    if digest(base.__file__) != BASE_BUILDER_SHA or digest(passes.__file__) != BASE_PASS_SHA:
        raise ValueError("frozen node24 source binding changed")
    slots = compile_factory(passes.Node24WeightSlots, [
        ("resident_groups not in (12, 13)", "resident_groups not in (0, 8, 12, 13)"),
        ("group_count not in (14, 15, 16)", "group_count != 16"),
    ], dict(passes.__dict__))
    dma = compile_factory(passes.Node24GroupWeightDMA, [
        ("slot = tir.Select(group.loop_var < 13, group.loop_var, tir.IntImm(\"int32\", 13))",
         "slot = tir.Select(group.loop_var < %d, group.loop_var, tir.IntImm(\"int32\", %d))" %
         (resident_groups, resident_groups)),
    ], dict(passes.__dict__))
    namespace = dict(base.__dict__)
    namespace.update(Node24WeightSlots=slots, Node24GroupWeightDMA=dma)
    # Read back actual graph parameter buffers, not unchanged host source arrays.
    probe = compile_factory(base.probe, [
        ("resident_groups != 13", "resident_groups not in (0, 8, 12, 13)"),
        ("actual = runner.get_output(0).numpy()",
         "actual = runner.get_output(0).numpy()\n"
         "                device_unchanged = all(np.array_equal(runner.get_input(n).numpy(), v) "
         "for n, v in zip(('p015', 'p17', 'p26'), (x, w, add)))"),
        ("\"arguments_unchanged\": before == [sha(value.tobytes()) for value in (x, w, add)]",
         "\"arguments_unchanged\": device_unchanged and "
         "before == [sha(value.tobytes()) for value in (x, w, add)]"),
    ], namespace)
    return probe


def run_worker(output, resident_groups, board_compile=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    return make_probe(resident_groups)(
        output, "node24_partial_input_experimental", resident_groups=resident_groups,
        keep_resident=True, fsim=not board_compile, board_compile=board_compile,
        batch_group_load=True, group_gemm=True)
