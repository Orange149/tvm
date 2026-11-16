"""Experiment-only, bounded VTA window analysis and compact GEMM connection.

The pass consumes numeric post-FoldUopLoop TIR.  Its analysis uses parameter
positions, static scope identities, concrete LOAD versions and ACC stages;
no node number, tensor name, variable name or eight-slice pattern is used.
The projected dependency proof must additionally be checked against native
runtime instructions before this experiment can be qualified for hardware.
"""

from collections import Counter, defaultdict
import hashlib
from itertools import product
import json

import tvm
from tvm import tir
import vta  # Registers the TIR VTA intrinsics used by saved functions.

from c3_five_tradeoff_commands import completion_dag
from c3_thread_sync_audit import CAPACITY, ELEMENT_BYTES


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def integer(expr, bindings=None):
    value = tvm.arith.Analyzer().simplify(tir.stmt_functor.substitute(expr, bindings or {}))
    if not isinstance(value, tir.IntImm):
        raise ValueError("UNKNOWN_STATIC_RELATION: " + str(value))
    return int(value)


def call_info(stmt):
    if not isinstance(stmt, tir.Evaluate) or not isinstance(stmt.value, tir.Call):
        return None
    call = stmt.value
    if call.op.name == "tir.call_extern":
        return call.args[0].value, list(call.args[1:])
    return call.op.name, list(call.args)


def expanded_uops(kernel):
    loops = kernel["uops"][0]["loops"]
    for indices in product(*(range(loop[0]) for loop in loops)):
        for uop in kernel["uops"]:
            values = list(uop["args"])
            for operand in range(3):
                values[operand + 2] += sum(i * loop[operand + 1] for i, loop in zip(indices, loops))
            yield values


def trace_function(function, max_steps=2000000):
    """Finite TIR projection retaining logical buffer identity and tokens."""
    if function.attrs is None or "tir.noalias" not in function.attrs or not function.attrs["tir.noalias"]:
        raise ValueError("UNKNOWN_ALIAS: requires explicit tir.noalias")
    buffers, buffer_bytes = {}, {}
    for ordinal, param in enumerate(function.params):
        if param not in function.buffer_map:
            raise ValueError("UNKNOWN_ABI: non-buffer parameter")
        buffer = function.buffer_map[param]
        buffers[buffer.data] = ordinal
        count = 1
        for extent in buffer.shape:
            count *= integer(extent)
        dtype = tvm.DataType(buffer.dtype)
        buffer_bytes[ordinal] = count * dtype.bits * dtype.lanes // 8
    events, kernel, loops, steps = [], None, [], 0
    total_hardware_loops = 0
    scope_ids, identities = {}, {}

    def walk(node, bindings):
        nonlocal kernel, steps, total_hardware_loops
        steps += 1
        if steps > max_steps:
            raise ValueError("FINITE_EXPANSION_LIMIT")
        if isinstance(node, tir.For):
            lo, count = integer(node.min, bindings), integer(node.extent, bindings)
            if not 0 <= count <= max_steps:
                raise ValueError("UNSUPPORTED_LOOP_EXTENT")
            for value in range(lo, lo + count):
                walk(node.body, {**bindings, node.loop_var: tir.IntImm(node.loop_var.dtype, value)})
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                walk(child, bindings)
        elif isinstance(node, tir.LetStmt):
            value = integer(node.value, bindings)
            walk(node.body, {**bindings, node.var: tir.IntImm(node.var.dtype, value)})
        elif isinstance(node, tir.IfThenElse):
            branch = node.then_case if integer(node.condition, bindings) else node.else_case
            if branch is not None:
                walk(branch, bindings)
        elif isinstance(node, tir.AttrStmt):
            if node.attr_key != "coproc_uop_scope":
                walk(node.body, bindings)
                return
            if kernel is not None or loops:
                raise ValueError("NESTED_KERNEL")
            if not isinstance(node.value, tir.StringImm) or node.value.value not in ("VTAPushGEMMOp", "VTAPushALUOp"):
                raise ValueError("UNKNOWN_UOP_DISPATCH")
            site = scope_ids.setdefault(node, len(scope_ids))
            free = list(tir.analysis.undefined_vars(node.body))
            if len(free) > 1 or any(str(var.dtype) != "int32" for var in free):
                raise ValueError("CACHE_SIGNATURE_ARITY")
            signature = tuple(integer(var, bindings) for var in free)
            if signature and not 0 <= signature[0] <= 98:
                raise ValueError("CACHE_SIGNATURE_RANGE")
            kernel = {"kind": "kernel", "scope": site, "signature": signature,
                      "cache_identity": {"scope": site, "signature": signature},
                      "uops": [], "mode": node.value.value}
            total_hardware_loops = 0
            walk(node.body, bindings)
            if loops or not kernel["uops"] or len(kernel["uops"]) > CAPACITY["UOP"]:
                raise ValueError("UOP_KERNEL_CAPACITY_OR_BALANCE")
            modes = {tuple(op["args"][:2]) for op in kernel["uops"]}
            if len(modes) != 1:
                raise ValueError("MIXED_KERNEL_MODE")
            mode, reset = next(iter(modes))
            if mode not in (0, 1) or reset not in (0, 1):
                raise ValueError("INVALID_KERNEL_MODE")
            if kernel["mode"] != ("VTAPushGEMMOp" if mode == 0 else "VTAPushALUOp"):
                raise ValueError("UOP_DISPATCH_MODE_MISMATCH")
            kernel["mode_reset"] = [mode, reset]
            if mode == 1:
                settings = {tuple(op["args"][5:8]) for op in kernel["uops"]}
                if len(settings) != 1:
                    raise ValueError("INCONSISTENT_ALU_KERNEL_SETTINGS")
                opcode, use_imm, imm = next(iter(settings))
                if reset or opcode not in (0, 1, 2, 3) or use_imm not in (0, 1) or not -32768 <= imm <= 32767:
                    raise ValueError("UNSUPPORTED_ALU_ENCODING")
            if any(op["loops"] != kernel["uops"][0]["loops"] for op in kernel["uops"]):
                raise ValueError("INCONSISTENT_KERNEL_HARDWARE_LOOPS")
            # This is the runtime's two preceding base-uop destination check.
            for i, op in enumerate(kernel["uops"]):
                if any(op["args"][2] == prior["args"][2] for prior in kernel["uops"][max(0, i - 2):i]):
                    raise ValueError("UOP_BASE_DESTINATION_HAZARD")
            content = digest(kernel["uops"])
            key = (site, signature)
            if key in identities and identities[key] != content:
                raise ValueError("CACHE_SEGMENT_IDENTITY_COLLISION")
            identities[key] = content
            events.append(kernel)
            kernel = None
        elif isinstance(node, (tir.Allocate, tir.AssertStmt)):
            if not integer(node.condition, bindings):
                raise ValueError("UNSUPPORTED_FALSE_ALLOCATION_OR_ASSERT")
            if isinstance(node, tir.Allocate):
                for extent in node.extents:
                    if integer(extent, bindings) < 0:
                        raise ValueError("UNSUPPORTED_ALLOCATION_EXTENT")
            walk(node.body, bindings)
        elif isinstance(node, tir.Evaluate):
            info = call_info(node)
            if info is None:
                if integer(node.value, bindings) != 0:
                    raise ValueError("UNKNOWN_NONCALL_EFFECT")
                return
            name, args = info
            if name == "VTAUopLoopBegin":
                values = [integer(arg, bindings) for arg in args]
                if kernel is None or kernel["uops"] or total_hardware_loops >= 2 or len(values) != 4:
                    raise ValueError("HARDWARE_LOOP_ARITY")
                if not 1 <= values[0] < 2**14 or any(not 0 <= v < limit for v, limit in zip(values[1:], (2048, 2048, 1024))):
                    raise ValueError("HARDWARE_LOOP_FIELD_RANGE")
                loops.append(values)
                total_hardware_loops += 1
            elif name == "VTAUopLoopEnd":
                if not loops:
                    raise ValueError("HARDWARE_LOOP_BALANCE")
                loops.pop()
            elif name in ("tir.vta.uop_push", "VTAUopPush"):
                if kernel is None or len(args) != 8:
                    raise ValueError("UNSCOPED_UOP")
                values = [integer(arg, bindings) for arg in args]
                kernel["uops"].append({"args": values, "loops": [list(loop) for loop in loops]})
            elif name in ("VTALoadBuffer2D", "VTAStoreBuffer2D"):
                if kernel is not None:
                    raise ValueError("INTERNAL_TENSOR_EVENT")
                load = name == "VTALoadBuffer2D"
                pointer = args[1] if load else args[3]
                if pointer not in buffers:
                    raise ValueError("UNKNOWN_LOGICAL_BUFFER")
                event = {"kind": "load" if load else "store", "tensor": buffers[pointer],
                         "domain": integer(args[11] if load else args[2], bindings),
                         "offset": integer(args[2] if load else args[4], bindings),
                         "sram": integer(args[10] if load else args[1], bindings),
                         "x": integer(args[3] if load else args[5], bindings),
                         "y": integer(args[4] if load else args[6], bindings),
                         "stride": integer(args[5] if load else args[7], bindings),
                         "pad": [integer(arg, bindings) for arg in args[6:10]] if load else [0, 0, 0, 0]}
                if min(event["offset"], event["sram"], event["x"], event["y"], event["stride"], *event["pad"]) < 0:
                    raise ValueError("NEGATIVE_DMA_DESCRIPTOR")
                domain = {1: "WGT", 2: "INP", 3: "ACC", 4: "OUT"}.get(event["domain"])
                if domain is None:
                    raise ValueError("UNSUPPORTED_TENSOR_DMA_DOMAIN")
                end = event["offset"] + max(0, event["y"] - 1) * event["stride"] + event["x"]
                if end * ELEMENT_BYTES[domain] > buffer_bytes[event["tensor"]]:
                    raise ValueError("LOGICAL_DDR_DESCRIPTOR_BOUNDS")
                if any(v >= 65536 for v in (event["x"], event["y"], event["stride"])) or any(p >= 16 for p in event["pad"]):
                    raise ValueError("DMA_INSTRUCTION_FIELD_RANGE")
                events.append(event)
            elif name in ("tir.vta.coproc_dep_push", "tir.vta.coproc_dep_pop"):
                if kernel is not None:
                    raise ValueError("INTERNAL_NECESSARY_NOTIFICATION")
                pair = [integer(arg, bindings) for arg in args]
                if len(pair) != 2 or abs(pair[0] - pair[1]) != 1 or min(pair) < 1 or max(pair) > 3:
                    raise ValueError("UNSUPPORTED_TOKEN_RELATION")
                events.append({"kind": "push" if name.endswith("push") else "pop", "pair": pair})
            elif name in ("tir.vta.coproc_sync", "VTASynchronize"):
                events.append({"kind": "sync"})
            else:
                raise ValueError("UNKNOWN_CALL: " + name)
        else:
            raise ValueError("UNKNOWN_STATEMENT: " + type(node).__name__)

    walk(function.body, {})
    return events


def projected_dag(events):
    """Completion edges on tensor commands and explicit token event nodes.

    Token nodes denote completion points, not extra emitted instructions. Native
    binding below verifies these edges against the actual runtime encoding.
    """
    records = []
    for event in events:
        kind = event["kind"]
        stage = (1 if event["domain"] in (1, 2) else 2) if kind == "load" else 3 if kind == "store" else 2
        bits = [0, 0, 0, 0]
        if kind in ("push", "pop"):
            src, dst = event["pair"]
            stage = src if kind == "push" else dst
            index = (2 if src > dst else 3) if kind == "push" else (0 if src < dst else 1)
            bits[index] = 1
        records.append({"stage": stage, "bits": bits})
    return completion_dag(records)


def dma_cells(event):
    left, top, right, bottom = event["pad"]
    width = event["x"] + left + right
    for y in range(event["y"] + top + bottom):
        for x in range(width):
            addr = event["sram"] + y * width + x
            padding = x < left or x >= left + event["x"] or y < top or y >= top + event["y"]
            logical = ("zero",) if padding else (event["tensor"], event["offset"] + (y - top) * event["stride"] + x - left)
            yield addr, logical


def certify_trace(events, ancestors=None):
    """Prove versions, asynchronous overwrites, ACC stages and product multisets."""
    ancestors = projected_dag(events) if ancestors is None else ancestors
    before = lambda a, b: a != b and bool(ancestors[b] & (1 << a))
    versions, readers, acc, outputs = {}, defaultdict(set), {}, {}
    products, phases, observed = {}, {}, set()
    consumer_bindings, overwrite_count, terms_count = [], 0, 0

    def bounded(addr, domain):
        if not 0 <= addr < CAPACITY[domain]:
            raise ValueError("FINAL_EFFECTIVE_ADDRESS: %s %d" % (domain, addr))

    def overwrite(domain, addr, event_id):
        nonlocal overwrite_count
        old = versions.get((domain, addr))
        if old is not None:
            if any(not before(reader, event_id) for reader in readers[(domain, addr, old[0])]):
                raise ValueError("OVERWRITE_BEFORE_LAST_READ")
            overwrite_count += 1

    for eid, event in enumerate(events):
        kind = event["kind"]
        if kind == "load":
            domain = {1: "WGT", 2: "INP", 3: "ACC"}.get(event["domain"])
            if domain is None:
                raise ValueError("UNSUPPORTED_TENSOR_LOAD_DOMAIN")
            for addr, logical in dma_cells(event):
                bounded(addr, domain)
                overwrite(domain, addr, eid)
                versions[domain, addr] = (eid, logical)
                if domain == "ACC":
                    acc[addr] = ("tensor", logical, eid)
        elif kind == "kernel":
            mode, reset = event["mode_reset"]
            uses = set()
            for values in expanded_uops(event):
                _, _, dst, src, wgt, opcode, immediate, imm = values
                bounded(dst, "ACC")
                if reset:
                    overwrite("ACC", dst, eid)
                    stage = (eid, dst)
                    products[stage] = Counter()
                    phases[stage] = []
                    acc[dst] = ("reduction", stage, eid)
                    versions["ACC", dst] = (eid, ("stage", stage))
                    continue
                if dst not in acc or acc[dst][0] != "reduction":
                    raise ValueError("ACC_NOT_INITIALIZED_REDUCTION")
                stage = acc[dst][1]
                if stage in observed:
                    raise ValueError("COMPUTE_AFTER_OUTPUT_OBSERVATION")
                if not before(acc[dst][2], eid) and acc[dst][2] != eid:
                    raise ValueError("ACC_STAGE_NOT_ORDERED")
                if mode == 0:
                    if phases[stage]:
                        raise ValueError("MAC_AFTER_POSTPROCESS")
                    logical = []
                    for domain, addr in (("INP", src), ("WGT", wgt)):
                        bounded(addr, domain)
                        if (domain, addr) not in versions:
                            raise ValueError("MISSING_FIRST_USE_LOAD")
                        version, origin = versions[domain, addr]
                        if not before(version, eid):
                            raise ValueError("READ_BEFORE_LOAD_COMPLETION")
                        readers[domain, addr, version].add(eid)
                        uses.add((domain, addr, version, origin))
                        logical.append(origin)
                    products[stage][tuple(logical)] += 1
                    terms_count += 1
                else:
                    operand = ("imm", imm)
                    if not immediate:
                        bounded(src, "ACC")
                        if src not in acc or (acc[src][2] != eid and not before(acc[src][2], eid)):
                            raise ValueError("ALU_SOURCE_NOT_ORDERED")
                        if src != dst and acc[src][0] != "tensor":
                            raise ValueError("UNSUPPORTED_CROSS_REDUCTION_ALU_READ")
                        operand = acc[src][:2] if src != dst else ("self",)
                        version = versions["ACC", src][0]
                        readers["ACC", src, version].add(eid)
                    phases[stage].append((opcode, immediate, operand))
                acc[dst] = ("reduction", stage, eid)
                versions["ACC", dst] = (eid, ("stage", stage))
            if uses:
                consumer_bindings.append({"event": eid, "scope": event["scope"],
                    "versions": sorted(uses, key=repr)})
        elif kind == "store":
            if event["domain"] != 4:
                raise ValueError("UNSUPPORTED_STORE_DOMAIN")
            for addr, logical in dma_cells(event):
                bounded(addr, "ACC")
                if addr not in acc or acc[addr][0] != "reduction" or not before(acc[addr][2], eid):
                    raise ValueError("STORE_BEFORE_REDUCTION_POSTPROCESS")
                stage = acc[addr][1]
                version = versions["ACC", addr][0]
                readers["ACC", addr, version].add(eid)
                if logical in outputs:
                    raise ValueError("MULTIPLE_OUTPUT_OBSERVATIONS")
                observed.add(stage)
                outputs[logical] = {"terms": products[stage].copy(), "postprocess": list(phases[stage])}
    serial = [[list(key), sorted(((list(a), list(b), count) for (a, b), count in value["terms"].items()), key=repr),
               value["postprocess"]] for key, value in sorted(outputs.items())]
    if not serial:
        raise ValueError("NO_OBSERVABLE_OUTPUT_IN_SUPPORT_DOMAIN")
    maximum_terms = max((sum(value["terms"].values()) for value in outputs.values()), default=0)
    bound = maximum_terms * 16 * 128 * 128
    if bound >= 2**31:
        raise ValueError("INT8_REDUCTION_PARTIAL_SUM_BOUND")
    requests = [event for event in events if event["kind"] in ("load", "store")]
    control = [event for event in events if event["kind"] in ("push", "pop", "sync")]
    return {"status": "PROJECTED_VERSIONS_ACC_PRODUCTS_PASSED_NATIVE_BINDING_REQUIRED",
            "packed_product_count": terms_count, "packed_outputs": len(outputs),
            "maximum_packed_terms_per_output": maximum_terms, "int8_partial_sum_bound": bound,
            "tensor_requests_sha256": digest(requests), "control_trace_sha256": digest(control),
            "output_product_postprocess_sha256": digest(serial),
            "overwrite_cells_proved": overwrite_count,
            "consumer_versions_sha256": digest(consumer_bindings),
            "consumer_versions": consumer_bindings, "output_terms": serial}


def _parts(node):
    return list(node.seq) if isinstance(node, tir.SeqStmt) else [node]


def statement_sequence(items):
    return tir.SeqStmt(items) if len(items) > 1 else items[0] if items else tir.Evaluate(0)


def _kernel_block(block):
    if not isinstance(block, tir.AttrStmt) or block.attr_key != "coproc_scope" or integer(block.value) != 2:
        return None, "NON_COMPUTE_OR_UNKNOWN_EVENT"
    parts = _parts(block.body)
    found = [i for i, item in enumerate(parts) if isinstance(item, tir.AttrStmt) and item.attr_key == "coproc_uop_scope"]
    if len(found) != 1:
        return None, "NO_UNIQUE_KERNEL"
    index = found[0]
    kernel = parts[index]
    if not isinstance(kernel.value, tir.StringImm) or kernel.value.value != "VTAPushGEMMOp":
        return None, "POSTPROCESS_OR_MODE"
    content = _parts(kernel.body)
    begins, ends = [], []
    while content and call_info(content[0]) is not None and call_info(content[0])[0] == "VTAUopLoopBegin":
        begins.append(content.pop(0))
    while content and call_info(content[-1]) is not None and call_info(content[-1])[0] == "VTAUopLoopEnd":
        ends.insert(0, content.pop())
    if len(begins) > 2 or len(begins) != len(ends) or not content:
        return None, "INCOMPATIBLE_HARDWARE_LOOP"
    pushes, count = [], 0

    def visit(node, multiplicity=1):
        nonlocal count
        if isinstance(node, tir.For):
            visit(node.body, multiplicity * integer(node.extent))
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child, multiplicity)
        else:
            info = call_info(node)
            if info is None or info[0] != "tir.vta.uop_push":
                raise ValueError("UNKNOWN_INTERNAL_EFFECT")
            if [integer(arg) for arg in info[1][:2]] != [0, 0]:
                raise ValueError("RESET_OR_MODE")
            pushes.append(info[1])
            count += multiplicity
    try:
        for child in content:
            visit(child)
    except ValueError as error:
        return None, str(error)
    if not pushes or count <= 0 or count > CAPACITY["UOP"]:
        return None, "UOP_CAPACITY"
    return {"block": block, "kernel": kernel, "prefix": parts[:index], "suffix": parts[index + 1:],
            "begins": begins, "ends": ends, "content": content, "count": count}, None


def AggregateWindows(q, decisions=None):
    """Connect at most q compatible kernels after a full bounded safety audit."""
    if not isinstance(q, int) or q < 1:
        raise ValueError("q must be a positive upper bound")
    decisions = decisions if decisions is not None else []

    @tir.transform.prim_func_pass(opt_level=0, name="tir.c3.AggregateWindows")
    def transform(function, _module, _context):
        original_trace = trace_function(function)
        original = certify_trace(original_trace)

        def post(node):
            if not isinstance(node, tir.SeqStmt):
                return None
            result, pending = [], []

            def flush(reason):
                if not pending:
                    return
                decisions.append({"segments": len(pending), "base_uops": sum(p["count"] for p in pending), "stop": reason})
                if len(pending) == 1:
                    result.append(pending[0]["block"])
                else:
                    first, last = pending[0], pending[-1]
                    k = first["kernel"]
                    body = statement_sequence([*first["begins"], *[child for p in pending for child in p["content"]], *first["ends"]])
                    merged = tir.AttrStmt(k.node, k.attr_key, k.value, body, k.span)
                    outer = first["block"]
                    result.append(tir.AttrStmt(outer.node, outer.attr_key, outer.value,
                        statement_sequence([*first["prefix"], merged, *last["suffix"]]), outer.span))
                pending.clear()

            for block in node.seq:
                candidate, reason = _kernel_block(block)
                if candidate is None:
                    flush(reason)
                    result.append(block)
                    continue
                if pending:
                    first, previous = pending[0], pending[-1]
                    if previous["suffix"] or candidate["prefix"]:
                        flush("INTERNAL_NECESSARY_EVENT")
                    elif len(first["begins"]) != len(candidate["begins"]) or any(
                            not tvm.ir.structural_equal(a, b) for a, b in zip(
                                first["begins"] + first["ends"], candidate["begins"] + candidate["ends"])):
                        flush("INCOMPATIBLE_HARDWARE_LOOP")
                    elif sum(p["count"] for p in pending) + candidate["count"] > CAPACITY["UOP"]:
                        flush("UOP_CAPACITY")
                pending.append(candidate)
                if len(pending) >= q:
                    flush("Q_UPPER_BOUND")
            flush("REGION_END")
            return statement_sequence(result)

        body = tir.stmt_functor.ir_transform(function.body, None, post, ["tir.SeqStmt"])
        transformed = function.with_body(body)
        new = certify_trace(trace_function(transformed))
        for field in ("tensor_requests_sha256", "control_trace_sha256", "output_product_postprocess_sha256"):
            if original[field] != new[field]:
                raise ValueError("AGGREGATION_EQUIVALENCE_FAILED: " + field)
        return transformed
    return transform
