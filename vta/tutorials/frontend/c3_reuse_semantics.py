"""Exact static LOAD/STORE regions, conditional paths, and storage arenas.

This is a request/liveness-scope audit, not an asynchronous consumer/overwrite
proof. Intervals are in packed elements; DDR bytes exclude padding zero-fill.
"""
import math
from collections import defaultdict
from itertools import product

import tvm
from tvm import tir

MEMORY_NAMES = {0: "uop", 1: "wgt", 2: "inp", 3: "acc", 4: "out", 5: "acc8"}


def union_length(intervals):
    end, length = None, 0
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("negative interval length")
        length += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return length


def extract_requests(module, env, max_expansions=200000):
    analyzer = tvm.arith.Analyzer()
    elem_bytes = {0: 4, 1: env.WGT_ELEM_BYTES, 2: env.INP_ELEM_BYTES,
                  3: env.ACC_ELEM_BYTES, 4: env.OUT_ELEM_BYTES, 5: env.ACC_ELEM_BYTES // 4}
    sites, groups, compute_sites, expanded = [], [], [], 0
    buffer_capacities = {}
    for buffer in module["main"].buffer_map.values():
        if all(isinstance(extent, tir.IntImm) for extent in buffer.shape):
            dtype = tvm.DataType(str(buffer.dtype))
            buffer_capacities[buffer.data] = math.prod(int(extent) for extent in buffer.shape) * dtype.bits * dtype.lanes // 8

    def simplify(expr, bindings):
        return analyzer.simplify(tir.stmt_functor.substitute(expr, bindings))

    def integer(expr, bindings):
        value = simplify(expr, bindings)
        if not isinstance(value, tir.IntImm):
            raise ValueError("non-static descriptor or conditional path: " + str(value))
        return int(value)

    def call_site(call, loops, conditions, lets):
        nonlocal expanded
        if isinstance(call.op, tvm.ir.Op) and (
            call.op.name == "tir.vta.uop_push" or
            (call.op.name == "tir.call_extern" and isinstance(call.args[0], tir.StringImm)
             and call.args[0].value == "VTAUopPush")
        ):
            compute_sites.append({"op": call.op.name,
                                  "loop_scope": [{"var": str(var), "min": lo, "extent": size} for var, lo, size in loops],
                                  "conditions": [{"expr": str(expr), "expected": expected} for expr, expected in conditions]})
        if not (isinstance(call.op, tvm.ir.Op) and call.op.name == "tir.call_extern"
                and isinstance(call.args[0], tir.StringImm)):
            return
        name = call.args[0].value
        if name not in ("VTALoadBuffer2D", "VTAStoreBuffer2D"):
            return
        args = [simplify(arg, lets) for arg in call.args[1:]]
        conditions = [(simplify(expr, lets), expected) for expr, expected in conditions]
        used = set()

        def collect(node):
            if isinstance(node, tir.Var):
                used.add(node)
        for expr in args + [item[0] for item in conditions]:
            tir.stmt_functor.post_order_visit(expr, collect)
        relevant = [loop for loop in loops if loop[0] in used]
        multiplicity = math.prod(loop[2] for loop in loops if loop[0] not in used)
        combinations = math.prod(loop[2] for loop in relevant)
        expanded += combinations
        if expanded > max_expansions:
            raise ValueError("request expansion bound exceeded")
        site_id = len(sites)
        site = {"site": site_id, "extern": name,
                "loop_scope": [{"var": str(var), "min": lo, "extent": size} for var, lo, size in loops],
                "conditions": [{"expr": str(expr), "expected": expected} for expr, expected in conditions],
                "symbolic_arguments": ["<command_handle>" if index == 0 else str(arg) for index, arg in enumerate(args)], "executed_calls": 0}
        sites.append(site)
        for values in product(*(range(lo, lo + size) for _, lo, size in relevant)):
            bindings = {var: tir.IntImm(var.dtype, value) for (var, _, _), value in zip(relevant, values)}
            if not all(bool(integer(expr, bindings)) == expected for expr, expected in conditions):
                continue
            load = name == "VTALoadBuffer2D"
            memory = integer(args[11] if load else args[2], bindings)
            x = integer(args[3] if load else args[5], bindings)
            y = integer(args[4] if load else args[6], bindings)
            stride = integer(args[5] if load else args[7], bindings)
            offset = integer(args[2] if load else args[4], bindings)
            target = integer(args[10] if load else args[1], bindings)
            padding = [integer(expr, bindings) for expr in args[6:10]] if load else [0] * 4
            if memory not in elem_bytes or min(x, y, stride, offset, target, *padding) < 0:
                raise ValueError("unsupported or negative DMA descriptor")
            pointer = args[1] if load else args[3]
            capacity = buffer_capacities.get(pointer) if isinstance(pointer, tir.Var) else None
            if capacity is not None and y and (offset + (y - 1) * stride + x) * elem_bytes[memory] > capacity:
                raise ValueError("DMA rectangle exceeds frozen DDR tensor")
            row = {"site": site_id, "direction": "load" if load else "store",
                   "memory_name": MEMORY_NAMES[memory], "memory_type": memory,
                   "source_buffer": str(args[1] if load else "SRAM:" + MEMORY_NAMES[memory]),
                   "destination_buffer": "SRAM:" + MEMORY_NAMES[memory] if load else str(args[3]),
                   "ddr_offset_elements": offset, "sram_offset_elements": target,
                   "x_size": x, "y_size": y, "x_stride": stride, "padding": padding,
                   "element_bytes": elem_bytes[memory], "bytes_per_request": x * y * elem_bytes[memory],
                   "ddr_buffer_capacity_bytes": capacity, "ddr_bounds_checked": capacity is not None,
                   "multiplicity": multiplicity}
            groups.append(row)
            site["executed_calls"] += multiplicity

    def walk(stmt, loops=(), conditions=(), lets=None):
        lets = {} if lets is None else lets
        if isinstance(stmt, tir.For):
            lo, extent = integer(stmt.min, lets), integer(stmt.extent, lets)
            if extent < 0:
                raise ValueError("negative loop extent")
            walk(stmt.body, loops + ((stmt.loop_var, lo, extent),), conditions, lets)
        elif isinstance(stmt, tir.SeqStmt):
            for child in stmt.seq:
                walk(child, loops, conditions, lets)
        elif isinstance(stmt, tir.IfThenElse):
            walk(stmt.then_case, loops, conditions + ((stmt.condition, True),), lets)
            if stmt.else_case is not None:
                walk(stmt.else_case, loops, conditions + ((stmt.condition, False),), lets)
        elif isinstance(stmt, tir.LetStmt):
            bound = dict(lets)
            bound[stmt.var] = simplify(stmt.value, lets)
            walk(stmt.body, loops, conditions, bound)
        elif isinstance(stmt, (tir.AttrStmt, tir.Allocate, tir.AssertStmt)):
            walk(stmt.body, loops, conditions, lets)
        elif isinstance(stmt, tir.Evaluate):
            tir.stmt_functor.post_order_visit(stmt.value, lambda node: call_site(node, loops, conditions, lets)
                                             if isinstance(node, tir.Call) else None)
        elif not isinstance(stmt, tir.BufferStore):
            raise ValueError("unsupported lowered statement " + type(stmt).__name__)
    walk(module["main"].body)
    totals, regions = defaultdict(int), defaultdict(list)
    for row in groups:
        prefix = row["direction"] + "_buffer_2d"
        count, size = row["multiplicity"], row["multiplicity"] * row["bytes_per_request"]
        totals[prefix + "_calls"] += count
        totals[prefix + "_bytes"] += size
        totals[prefix + "_" + row["memory_name"] + "_calls"] += count
        totals[prefix + "_" + row["memory_name"] + "_bytes"] += size
        key = (row["direction"], row["memory_name"], row["source_buffer"], row["element_bytes"])
        for i in range(row["y_size"]):
            start = row["ddr_offset_elements"] + i * row["x_stride"]
            regions[key].append((start, start + row["x_size"]))
    unique, ratios = {}, {}
    for (direction, memory, _, byte_width), intervals in regions.items():
        key = direction + "_" + memory
        unique[key] = unique.get(key, 0) + union_length(intervals) * byte_width
    for key, size in unique.items():
        direction, memory = key.split("_", 1)
        transferred = totals[direction + "_buffer_2d_" + memory + "_bytes"]
        ratios[key] = transferred / size if size else None
    return {"totals": dict(totals), "unique_source_region_bytes": unique, "reload_ratios": ratios,
            "load_store_sites": sites, "request_groups": groups, "compute_sites": compute_sites,
            "scope": "logical source intervals and SRAM LOAD/STORE destinations; no GEMM consumer/overwrite proof"}


def capture_lower(schedule, tensors, build_config_factory=None):
    """Observe storage-rewritten arenas immediately before device-address lowering."""
    import vta
    captured = []

    @tir.transform.prim_func_pass(opt_level=0, name="c3.ReuseCaptureStorage")
    def capture(function, *_):
        captured.append(tvm.IRModule({"main": function}))
        return function

    config_factory = build_config_factory or vta.build_config
    config = config_factory(disabled_pass={"tir.CommonSubexprElimTIR"})
    passes = list(config.config["tir.add_lower_pass"])
    index = next(i for i, item in enumerate(passes) if str(item[1].info.name).endswith("LowerDeviceStorageAccessInfo"))
    passes.insert(index, (3, capture))
    with tvm.transform.PassContext(config={"tir.add_lower_pass": passes},
                                   disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = tvm.lower(schedule, tensors, name="main")
    if len(captured) != 1:
        raise ValueError("missing unambiguous storage-rewritten capture")
    allocations = []

    def visit(node):
        if isinstance(node, tir.Allocate):
            if any(not isinstance(extent, tir.IntImm) for extent in node.extents):
                raise ValueError("dynamic SRAM allocation span")
            annotation = node.buffer_var.type_annotation
            scope = str(getattr(annotation, "storage_scope", "global") or "global")
            dtype = tvm.DataType(str(node.dtype))
            allocations.append({"buffer": str(node.buffer_var), "scope": scope,
                                "bytes": math.prod(int(x) for x in node.extents) * dtype.bits * dtype.lanes // 8,
                                "extents": [int(x) for x in node.extents]})
    tir.stmt_functor.post_order_visit(captured[0]["main"].body, visit)
    spans = defaultdict(int)
    for allocation in allocations:
        spans[allocation["scope"]] += allocation["bytes"]
    return module, captured[0], {"allocations": allocations,
                                "scope_arena_span_sum_upper_bound_bytes": dict(spans),
                                "capture": "after StorageRewrite, before LowerDeviceStorageAccessInfo",
                                "simultaneous_live_peak": None}
