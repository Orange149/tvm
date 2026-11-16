"""Bounded official-baseline window analysis; no tuning or node-name matching.

This module never moves work across a host loop iteration.  Static proofs are
explicitly provisional until the separate actual FIFO/UOP certificate passes.
An unbound hardware dependency interval preserves every proposed join seam.
"""

from collections import Counter
from pathlib import Path
import hashlib
import math

import tvm
from tvm import tir

from c3_thread_sync_audit import CAPACITY, ELEMENT_BYTES
from c3_valid_window import call_info, certify_trace, digest, integer, statement_sequence


EXPANDED_UOP_LIMIT = 5_000_000
FROZEN_BITSTREAM_SHA256 = "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6"
TRANSPARENT_ATTRIBUTES = frozenset(("storage_scope", "storage_alignment"))


class AuditFailure(ValueError):
    """A classified support, resource, or program-legality failure."""
    def __init__(self, category, reason, details=None):
        super().__init__(category + ": " + reason)
        self.category, self.reason, self.details = category, reason, details or {}


def _fail(category, reason, **details):
    raise AuditFailure(category, reason, details)


def _parts(node):
    return list(node.seq) if isinstance(node, tir.SeqStmt) else [node]


def _limits(limits):
    result = {"expanded_uop_steps": EXPANDED_UOP_LIMIT, "tir_walk_visits": 50_000_000}
    if limits:
        unknown = set(limits) - set(result)
        if unknown:
            _fail("UNSUPPORTED_CONTRACT", "UNKNOWN_LIMIT_FIELD", fields=sorted(unknown))
        result.update(limits)
    if result["expanded_uop_steps"] != EXPANDED_UOP_LIMIT:
        _fail("UNSUPPORTED_CONTRACT", "EXPANDED_UOP_LIMIT_MUST_BE_EXACTLY_5000000")
    if not isinstance(result["tir_walk_visits"], int) or result["tir_walk_visits"] <= 0:
        _fail("UNSUPPORTED_CONTRACT", "INVALID_TIR_WALK_LIMIT")
    return result


def structure_statistics(function):
    """Cheap syntactic census, available even when full auditing is unsupported."""
    stats = Counter()
    attrs, for_kinds, reasons = Counter(), Counter(), Counter()

    def walk(node, stage=None):
        stats["statement_nodes"] += 1
        if isinstance(node, tir.AttrStmt):
            attrs[node.attr_key] += 1
            if node.attr_key == "coproc_scope":
                try:
                    stage = integer(node.value)
                except ValueError:
                    reasons["DYNAMIC_COMPUTE_SCOPE"] += 1
                    stage = None
                if stage == 2:
                    stats["compute_scopes"] += 1
                    direct = [x for x in _parts(node.body) if isinstance(x, tir.AttrStmt)
                              and x.attr_key == "coproc_uop_scope"]
                    if len(direct) > 1:
                        stats["compute_scopes_with_multiple_direct_kernels"] += 1
            elif node.attr_key == "coproc_uop_scope":
                stats["static_kernel_scopes"] += 1
            elif node.attr_key not in TRANSPARENT_ATTRIBUTES:
                reasons["UNKNOWN_ATTRIBUTE:" + node.attr_key] += 1
            walk(node.body, stage)
        elif isinstance(node, tir.For):
            for_kinds[str(node.kind)] += 1
            stats["host_for_nodes"] += 1
            if node.kind not in (tir.ForKind.SERIAL, tir.ForKind.UNROLLED) or node.thread_binding is not None:
                reasons["NON_SERIAL_OR_BOUND_THREAD_LOOP"] += 1
            walk(node.body, stage)
        elif isinstance(node, tir.SeqStmt):
            candidates = [_candidate(x, stage) for x in node.seq]
            units = [entry[0] for entry in candidates]
            for unit, reason in candidates:
                if unit is None and reason and reason.startswith("SUPPORT_UNRESOLVED"):
                    reasons[reason] += 1
            runs, current = [], 0
            for unit in units:
                if unit is None:
                    if current:
                        runs.append(current)
                    current = 0
                else:
                    current += 1
            if current:
                runs.append(current)
            stats["natural_adjacent_runs"] += sum(n >= 2 for n in runs)
            stats["natural_adjacent_seams"] += sum(max(0, n-1) for n in runs)
            stats["largest_syntactic_run"] = max([stats["largest_syntactic_run"], *runs])
            for child in node.seq:
                walk(child, stage)
        elif isinstance(node, tir.IfThenElse):
            walk(node.then_case, stage)
            if node.else_case is not None:
                walk(node.else_case, stage)
        elif isinstance(node, (tir.LetStmt, tir.Allocate, tir.AssertStmt)):
            walk(node.body, stage)
        elif not isinstance(node, tir.Evaluate):
            reasons["UNKNOWN_STATEMENT:" + type(node).__name__] += 1

    walk(function.body)
    return {"status": "SYNTACTIC_CENSUS_ONLY_NOT_A_SAFETY_PROOF", "counts": dict(stats),
            "attributes": dict(attrs), "for_kinds": dict(for_kinds), "unresolved_shapes": dict(reasons),
            "host_loop_iterations_not_crossed": True}


def _trace(function, limits):
    """Full finite TIR projection with exact per-dynamic-kernel work accounting."""
    if function.attrs is None or "tir.noalias" not in function.attrs or not function.attrs["tir.noalias"]:
        _fail("UNSUPPORTED_SEMANTICS", "EXPLICIT_NOALIAS_REQUIRED")
    buffers, buffer_bytes, buffer_dtypes = {}, {}, {}
    for ordinal, param in enumerate(function.params):
        if param not in function.buffer_map:
            _fail("UNSUPPORTED_SEMANTICS", "NON_BUFFER_ABI_PARAMETER")
        buffer = function.buffer_map[param]
        if buffer.data in buffers:
            _fail("UNSUPPORTED_SEMANTICS", "ALIASED_ABI_BUFFER_DATA")
        buffers[buffer.data] = ordinal
        if integer(buffer.elem_offset) != 0:
            _fail("UNSUPPORTED_SEMANTICS", "NONZERO_ABI_ELEMENT_OFFSET")
        if buffer.strides:
            expected = 1
            for extent, stride in reversed(list(zip(buffer.shape, buffer.strides))):
                if integer(stride) != expected:
                    _fail("UNSUPPORTED_SEMANTICS", "NONCONTIGUOUS_ABI_STRIDES")
                expected *= integer(extent)
        count = math.prod(integer(extent) for extent in buffer.shape)
        if count <= 0:
            _fail("UNSUPPORTED_SEMANTICS", "NONPOSITIVE_ABI_BUFFER_EXTENT")
        dtype = tvm.DataType(buffer.dtype)
        buffer_bytes[ordinal] = count * dtype.bits * dtype.lanes // 8
        buffer_dtypes[ordinal] = str(buffer.dtype)
    events, scopes, identities, loops = [], {}, {}, []
    kernel, hardware_begins = None, 0
    visits, expanded = 0, 0

    def walk(node, bindings, stage=None):
        nonlocal kernel, hardware_begins, visits, expanded
        visits += 1
        if visits > limits["tir_walk_visits"]:
            _fail("AUDIT_AST_VISIT_LIMIT", "TIR_WALK_LIMIT", visits=visits)
        if isinstance(node, tir.For):
            if node.kind not in (tir.ForKind.SERIAL, tir.ForKind.UNROLLED) or node.thread_binding is not None:
                _fail("UNSUPPORTED_SEMANTICS", "NON_SERIAL_OR_BOUND_THREAD_LOOP")
            if node.annotations:
                _fail("UNSUPPORTED_SEMANTICS", "UNRECOGNIZED_LOOP_ANNOTATIONS")
            lo, count = integer(node.min, bindings), integer(node.extent, bindings)
            if count < 0:
                _fail("PROGRAM_ILLEGAL", "NEGATIVE_LOOP_EXTENT")
            if count > limits["tir_walk_visits"]:
                _fail("AUDIT_AST_VISIT_LIMIT", "HOST_TRIP_COUNT_EXCEEDS_WALK_BUDGET", count=count)
            for value in range(lo, lo + count):
                walk(node.body, {**bindings, node.loop_var: tir.IntImm(node.loop_var.dtype, value)}, stage)
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                walk(child, bindings, stage)
        elif isinstance(node, tir.LetStmt):
            value = integer(node.value, bindings)
            walk(node.body, {**bindings, node.var: tir.IntImm(node.var.dtype, value)}, stage)
        elif isinstance(node, tir.IfThenElse):
            branch = node.then_case if integer(node.condition, bindings) else node.else_case
            if branch is not None:
                walk(branch, bindings, stage)
        elif isinstance(node, tir.AttrStmt):
            if node.attr_key == "coproc_scope":
                new_stage = integer(node.value, bindings)
                if new_stage not in (1, 2, 3):
                    _fail("UNSUPPORTED_SEMANTICS", "UNKNOWN_COPROCESSOR_STAGE")
                walk(node.body, bindings, new_stage)
                return
            if node.attr_key in TRANSPARENT_ATTRIBUTES:
                walk(node.body, bindings, stage)
                return
            if node.attr_key != "coproc_uop_scope":
                _fail("UNSUPPORTED_SEMANTICS", "UNKNOWN_ATTRIBUTE", attr=node.attr_key)
            if stage != 2 or kernel is not None or loops:
                _fail("UNSUPPORTED_SEMANTICS", "KERNEL_SCOPE_STAGE_OR_NESTING")
            if not isinstance(node.value, tir.StringImm) or node.value.value not in ("VTAPushGEMMOp", "VTAPushALUOp"):
                _fail("UNSUPPORTED_SEMANTICS", "UNKNOWN_UOP_DISPATCH")
            site = scopes.setdefault(node, len(scopes))
            free = list(tir.analysis.undefined_vars(node.body))
            if len(free) > 1 or any(str(var.dtype) != "int32" for var in free):
                _fail("UNSUPPORTED_SEMANTICS", "CACHE_SIGNATURE_ARITY_OR_DTYPE")
            signature = tuple(integer(var, bindings) for var in free)
            if signature and not 0 <= signature[0] <= 98:
                _fail("PROGRAM_ILLEGAL", "CACHE_SIGNATURE_ENCODING_RANGE", signature=signature)
            kernel = {"kind": "kernel", "scope": site, "signature": signature,
                      "cache_identity": {"scope": site, "signature": signature},
                      "uops": [], "mode": node.value.value,
                      "source_scope_sha256": hashlib.sha256(tvm.ir.save_json(node.body).encode()).hexdigest()}
            hardware_begins = 0
            walk(node.body, bindings, stage)
            if loops or not kernel["uops"]:
                _fail("PROGRAM_ILLEGAL", "EMPTY_KERNEL_OR_UNBALANCED_LOOPS")
            if len(kernel["uops"]) > CAPACITY["UOP"]:
                _fail("PROGRAM_ILLEGAL", "SINGLE_KERNEL_UOP_CAPACITY", base_uops=len(kernel["uops"]))
            mode_reset = {tuple(op["args"][:2]) for op in kernel["uops"]}
            if len(mode_reset) != 1:
                _fail("PROGRAM_ILLEGAL", "MIXED_MODE_OR_RESET_IN_KERNEL")
            mode, reset = next(iter(mode_reset))
            if mode not in (0, 1) or reset not in (0, 1) or kernel["mode"] != ("VTAPushGEMMOp" if mode == 0 else "VTAPushALUOp"):
                _fail("PROGRAM_ILLEGAL", "DISPATCH_MODE_RESET_MISMATCH")
            kernel["mode_reset"] = [mode, reset]
            for op in kernel["uops"]:
                if any(not 0 <= value < limit for value, limit in zip(op["args"][2:5], (2048, 2048, 1024))):
                    _fail("PROGRAM_ILLEGAL", "BASE_UOP_ADDRESS_ENCODING")
                if mode == 0 and op["args"][5:8] != [0, 0, 0]:
                    _fail("UNSUPPORTED_SEMANTICS", "GEMM_AUXILIARY_FIELDS")
            if any(op["loops"] != kernel["uops"][0]["loops"] for op in kernel["uops"]):
                _fail("PROGRAM_ILLEGAL", "INCONSISTENT_HARDWARE_LOOPS")
            if mode == 1:
                settings = {tuple(op["args"][5:8]) for op in kernel["uops"]}
                if len(settings) != 1:
                    _fail("PROGRAM_ILLEGAL", "MIXED_ALU_SETTINGS")
                opcode, use_imm, imm = next(iter(settings))
                if reset or opcode not in (0, 1, 2, 3) or use_imm not in (0, 1) or not -32768 <= imm <= 32767:
                    _fail("PROGRAM_ILLEGAL", "ALU_ENCODING")
            for i, op in enumerate(kernel["uops"]):
                if any(op["args"][2] == prior["args"][2] for prior in kernel["uops"][max(0, i-2):i]):
                    _fail("PROGRAM_ILLEGAL", "RUNTIME_VERIFYDEP_BASE_DESTINATION_HAZARD")
            content = digest(kernel["uops"])
            key = (site, signature)
            if key in identities and identities[key] != content:
                _fail("PROGRAM_ILLEGAL", "SAME_CACHE_IDENTITY_DIFFERENT_CONTENT")
            identities[key] = content
            count = len(kernel["uops"]) * math.prod(loop[0] for loop in kernel["uops"][0]["loops"])
            expanded += count
            if expanded > EXPANDED_UOP_LIMIT:
                _fail("AUDIT_STEP_LIMIT", "EXPANDED_UOP_STEPS_EXCEED_5000000", expanded_uop_steps=expanded)
            kernel["expanded_uop_steps"] = count
            events.append(kernel)
            kernel = None
        elif isinstance(node, (tir.Allocate, tir.AssertStmt)):
            if not integer(node.condition, bindings):
                _fail("UNSUPPORTED_SEMANTICS", "FALSE_ALLOCATION_OR_ASSERT")
            if isinstance(node, tir.Allocate) and any(integer(n, bindings) < 0 for n in node.extents):
                _fail("PROGRAM_ILLEGAL", "NEGATIVE_ALLOCATION")
            walk(node.body, bindings, stage)
        elif isinstance(node, tir.Evaluate):
            info = call_info(node)
            if info is None:
                if integer(node.value, bindings) != 0:
                    _fail("UNSUPPORTED_SEMANTICS", "UNKNOWN_NONCALL_EFFECT")
                return
            name, args = info
            if name == "VTAUopLoopBegin":
                values = [integer(arg, bindings) for arg in args]
                if kernel is None or kernel["uops"] or hardware_begins >= 2 or len(values) != 4:
                    _fail("PROGRAM_ILLEGAL", "HARDWARE_LOOP_ARITY_OR_PLACEMENT")
                if not 1 <= values[0] < 2**14 or any(not 0 <= v < limit for v, limit in zip(values[1:], (2048, 2048, 1024))):
                    _fail("PROGRAM_ILLEGAL", "HARDWARE_LOOP_FIELD_RANGE")
                loops.append(values)
                hardware_begins += 1
            elif name == "VTAUopLoopEnd":
                if kernel is None or not loops:
                    _fail("PROGRAM_ILLEGAL", "HARDWARE_LOOP_END_WITHOUT_BEGIN")
                loops.pop()
            elif name in ("tir.vta.uop_push", "VTAUopPush"):
                if kernel is None or len(args) != 8:
                    _fail("PROGRAM_ILLEGAL", "UNSCOPED_OR_WRONG_ARITY_UOP")
                values = [integer(arg, bindings) for arg in args]
                kernel["uops"].append({"args": values, "loops": [list(loop) for loop in loops]})
                if len(kernel["uops"]) > CAPACITY["UOP"]:
                    _fail("PROGRAM_ILLEGAL", "SINGLE_KERNEL_UOP_CAPACITY", base_uops=len(kernel["uops"]))
            elif name in ("VTALoadBuffer2D", "VTAStoreBuffer2D"):
                if kernel is not None:
                    _fail("UNSUPPORTED_SEMANTICS", "TENSOR_EVENT_INSIDE_KERNEL")
                load = name == "VTALoadBuffer2D"
                if len(args) != (12 if load else 8):
                    _fail("PROGRAM_ILLEGAL", "DMA_CALL_ARITY")
                pointer = args[1] if load else args[3]
                if pointer not in buffers:
                    _fail("UNSUPPORTED_SEMANTICS", "UNKNOWN_LOGICAL_BUFFER")
                event = {"kind": "load" if load else "store", "tensor": buffers[pointer],
                         "domain": integer(args[11] if load else args[2], bindings),
                         "offset": integer(args[2] if load else args[4], bindings),
                         "sram": integer(args[10] if load else args[1], bindings),
                         "x": integer(args[3] if load else args[5], bindings),
                         "y": integer(args[4] if load else args[6], bindings),
                         "stride": integer(args[5] if load else args[7], bindings),
                         "pad": [integer(arg, bindings) for arg in args[6:10]] if load else [0]*4}
                if min(event["offset"], event["sram"], event["x"], event["y"], event["stride"], *event["pad"]) < 0:
                    _fail("PROGRAM_ILLEGAL", "NEGATIVE_DMA_DESCRIPTOR")
                domain = {1: "WGT", 2: "INP", 3: "ACC", 4: "OUT"}.get(event["domain"])
                if domain is None:
                    _fail("UNSUPPORTED_SEMANTICS", "DMA_DOMAIN")
                if buffer_dtypes[event["tensor"]] != ("int32" if domain == "ACC" else "int8"):
                    _fail("UNSUPPORTED_SEMANTICS", "OPERAND_DTYPE_DIFFERS_FROM_FROZEN_HARDWARE")
                expected_stage = (1 if event["domain"] in (1, 2) else 2) if load else 3
                if stage != expected_stage:
                    _fail("UNSUPPORTED_SEMANTICS", "DMA_COPROCESSOR_STAGE_MISMATCH")
                end = event["offset"] + max(0, event["y"]-1)*event["stride"] + event["x"]
                if end*ELEMENT_BYTES[domain] > buffer_bytes[event["tensor"]]:
                    _fail("PROGRAM_ILLEGAL", "LOGICAL_DDR_BOUNDS")
                if any(v >= 65536 for v in (event["x"], event["y"], event["stride"])) or any(p >= 16 for p in event["pad"]):
                    _fail("PROGRAM_ILLEGAL", "DMA_INSTRUCTION_FIELD_RANGE")
                events.append(event)
            elif name in ("tir.vta.coproc_dep_push", "tir.vta.coproc_dep_pop"):
                if kernel is not None:
                    _fail("UNSUPPORTED_SEMANTICS", "INTERNAL_NECESSARY_NOTIFICATION")
                pair = [integer(arg, bindings) for arg in args]
                if len(pair) != 2 or abs(pair[0]-pair[1]) != 1 or min(pair) < 1 or max(pair) > 3:
                    _fail("UNSUPPORTED_SEMANTICS", "TOKEN_RELATION")
                events.append({"kind": "push" if name.endswith("push") else "pop", "pair": pair})
            elif name in ("tir.vta.coproc_sync", "VTASynchronize"):
                if kernel is not None:
                    _fail("UNSUPPORTED_SEMANTICS", "SYNC_INSIDE_KERNEL")
                events.append({"kind": "sync"})
            else:
                _fail("UNSUPPORTED_SEMANTICS", "UNKNOWN_CALL", name=name)
        else:
            _fail("UNSUPPORTED_SEMANTICS", "UNKNOWN_STATEMENT", type=type(node).__name__)

    try:
        walk(function.body, {})
    except AuditFailure:
        raise
    except ValueError as error:
        _fail("UNSUPPORTED_SEMANTICS", str(error))
    return events, {"expanded_uop_steps": expanded, "tir_walk_visits": visits,
                    "static_kernel_scopes": len(scopes), "distinct_cache_identities": len(identities)}


def _executed_count(events):
    """Independent generator pass; checks exact expansion count, not scalar MACs."""
    from c3_valid_window import expanded_uops
    count = 0
    for event in events:
        if event["kind"] == "kernel":
            for _ in expanded_uops(event):
                count += 1
                if count > EXPANDED_UOP_LIMIT:
                    _fail("AUDIT_STEP_LIMIT", "EXECUTED_EXPANDED_UOP_LIMIT")
    return count


def audit_function(function, limits=None, hw_contract=None):
    """Audit an unchanged full function. Unknown join hardware never blocks B."""
    census = structure_statistics(function)
    events, count = _trace(function, _limits(limits))
    try:
        semantic = certify_trace(events)
    except ValueError as error:
        _fail("PROGRAM_ILLEGAL_OR_UNSUPPORTED_SEMANTICS", str(error))
    executed = _executed_count(events)
    if executed != count["expanded_uop_steps"]:
        _fail("AUDIT_INTERNAL_ERROR", "ESTIMATED_EXECUTED_STEP_MISMATCH")
    proof = {k: v for k, v in semantic.items() if k not in ("consumer_versions", "output_terms")}
    return {"status": "PROJECTED_FULL_FUNCTION_PASSED_NATIVE_FIFO_OPERAND_PENDING",
            "statistics": census, "counts": {**count, "executed_expanded_uop_steps": executed},
            "semantic": proof, "events": events, "native_fifo_operand_certificate_required": True,
            "hw_contract_status": "VERIFIED_DEP_INTERVAL" if _hardware_interval(hw_contract) is not None else "UNRESOLVED_BITSTREAM_BUILD_BINDING",
            "declared_hw_contract_status": (hw_contract or {}).get("status", "UNRESOLVED_BITSTREAM_BUILD_BINDING")}


def _kernel_info(kernel):
    if not isinstance(kernel, tir.AttrStmt) or kernel.attr_key != "coproc_uop_scope":
        return None, "NOT_DIRECT_KERNEL"
    if not isinstance(kernel.value, tir.StringImm) or kernel.value.value != "VTAPushGEMMOp":
        return None, "ALU_OR_MODE_BOUNDARY"
    content, begins, ends = _parts(kernel.body), [], []
    while content and call_info(content[0]) and call_info(content[0])[0] == "VTAUopLoopBegin":
        begins.append(content.pop(0))
    while content and call_info(content[-1]) and call_info(content[-1])[0] == "VTAUopLoopEnd":
        ends.insert(0, content.pop())
    if len(begins) > 2 or len(begins) != len(ends) or not content:
        return None, "HARDWARE_LOOP_STRUCTURE"
    count = 0
    def visit(node, multiplier=1):
        nonlocal count
        if isinstance(node, tir.For):
            if node.kind not in (tir.ForKind.SERIAL, tir.ForKind.UNROLLED) or node.thread_binding is not None or node.annotations:
                raise ValueError("KERNEL_LOOP_SUPPORT")
            visit(node.body, multiplier*integer(node.extent))
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child, multiplier)
        else:
            info = call_info(node)
            if info is None or info[0] != "tir.vta.uop_push" or len(info[1]) != 8:
                raise ValueError("UNKNOWN_KERNEL_EFFECT")
            if [integer(v) for v in info[1][:2]] != [0, 0]:
                raise ValueError("RESET_OR_MODE_BOUNDARY")
            count += multiplier
    try:
        for child in content:
            visit(child)
    except (ValueError, TypeError) as error:
        reason = str(error)
        # B's finite trace can be valid while candidate parsing cannot count a
        # symbolic initializer or interpret a supported trace-call alias.
        # Preserve that distinction from an established RESET/mode boundary.
        if reason != "RESET_OR_MODE_BOUNDARY":
            reason = "SUPPORT_UNRESOLVED_KERNEL_CANDIDATE:" + reason
        return None, reason
    if not 0 < count <= CAPACITY["UOP"]:
        return None, "BASE_UOP_CAPACITY"
    return {"kernel": kernel, "content": content, "begins": begins, "ends": ends, "count": count,
            "source_scope_sha256": hashlib.sha256(tvm.ir.save_json(kernel.body).encode()).hexdigest()}, None


def _candidate(node, stage):
    if stage == 2 and isinstance(node, tir.AttrStmt) and node.attr_key == "coproc_uop_scope":
        result, reason = _kernel_info(node)
        if result is not None:
            result.update(node=node, outer=None, prefix=[], suffix=[])
        return result, reason
    if isinstance(node, tir.AttrStmt) and node.attr_key == "coproc_scope":
        try:
            if integer(node.value) != 2:
                return None, "NON_COMPUTE_BOUNDARY"
        except ValueError:
            return None, "UNKNOWN_COMPUTE_SCOPE"
        parts = _parts(node.body)
        found = [i for i, child in enumerate(parts) if isinstance(child, tir.AttrStmt)
                 and child.attr_key == "coproc_uop_scope"]
        if len(found) != 1:
            return None, "MULTI_KERNEL_COMPUTE_SCOPE_RECURSED" if found else "NO_DIRECT_KERNEL"
        at = found[0]
        result, reason = _kernel_info(parts[at])
        if result is not None:
            result.update(node=node, outer=node, prefix=parts[:at], suffix=parts[at+1:])
        return result, reason
    return None, "HOST_LOOP_BOUNDARY" if isinstance(node, tir.For) else "NECESSARY_OR_UNKNOWN_EVENT_BOUNDARY"


def _hardware_interval(contract):
    """A source hash alone is not a bitstream build certificate."""
    if not contract or contract.get("status") != "VERIFIED_DEP_INTERVAL":
        return None
    interval = contract.get("verify_dep_interval")
    source = contract.get("source_path")
    if not isinstance(interval, int) or interval <= 0 or not source:
        return None
    path = Path(source)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != contract.get("source_sha256"):
        return None
    if contract.get("bitstream_sha256") != FROZEN_BITSTREAM_SHA256:
        return None
    # The issuer must attest an actual source->bitstream build binding, not just
    # independently hash two unrelated artifacts. Unit fixtures are explicit.
    binding_path = contract.get("verified_build_binding_path")
    if not binding_path or not Path(binding_path).is_file():
        return None
    raw = Path(binding_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != contract.get("verified_build_binding_sha256"):
        return None
    import json
    try:
        binding = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if binding.get("status") != "BUILD_AND_INTERVAL_VERIFIED" or any(binding.get(key) != contract.get(key)
            for key in ("source_sha256", "bitstream_sha256", "verify_dep_interval")):
        return None
    return interval


def _merge(units):
    first, last = units[0], units[-1]
    body = statement_sequence([*first["begins"], *[child for unit in units for child in unit["content"]], *first["ends"]])
    kernel = first["kernel"]
    merged = tir.AttrStmt(kernel.node, kernel.attr_key, kernel.value, body, kernel.span)
    # Signature shape is checked before publishing a concatenated scope. Its
    # values/content for every invocation are subsequently checked by _trace.
    free = list(tir.analysis.undefined_vars(merged.body))
    if len(free) > 1 or any(str(var.dtype) != "int32" for var in free):
        _fail("UNSUPPORTED_SEMANTICS", "MERGED_CACHE_SIGNATURE_ARITY_OR_DTYPE")
    if first["outer"] is None:
        return merged
    outer = first["outer"]
    return tir.AttrStmt(outer.node, outer.attr_key, outer.value,
                        statement_sequence([*first["prefix"], merged, *last["suffix"]]), outer.span)


def _preserved_host_loops(function):
    """Host loops outside uop initialization retain their ordered descriptors."""
    result = []
    def visit(node, in_kernel=False):
        if isinstance(node, tir.AttrStmt):
            visit(node.body, in_kernel or node.attr_key == "coproc_uop_scope")
        elif isinstance(node, tir.For):
            if not in_kernel:
                result.append(node)
            visit(node.body, in_kernel)
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child, in_kernel)
        elif isinstance(node, tir.IfThenElse):
            visit(node.then_case, in_kernel)
            if node.else_case is not None:
                visit(node.else_case, in_kernel)
        elif isinstance(node, (tir.LetStmt, tir.Allocate, tir.AssertStmt)):
            visit(node.body, in_kernel)
    visit(function.body)
    return result


def aggregate_function(function, hw_contract=None, limits=None):
    """Longest safe syntactic prefix, without q search or host-loop movement."""
    baseline = audit_function(function, limits, hw_contract)
    decisions, interval = [], _hardware_interval(hw_contract)
    # Every source scope's concrete dynamic invocations are already enumerated
    # by the full B audit, so seam guards evaluate real signature values and
    # real effective addresses rather than a symbolic endpoint guess.
    witnesses = {}
    for event in baseline["events"]:
        if event["kind"] == "kernel":
            witnesses.setdefault(event["source_scope_sha256"], []).append(event)

    def guard_prefix(units):
        from collections import deque
        from c3_valid_window import expanded_uops
        _merge(units)
        streams = [witnesses.get(unit["source_scope_sha256"]) for unit in units]
        if any(stream is None for stream in streams) or len({len(stream) for stream in streams}) != 1:
            _fail("UNSUPPORTED_SEMANTICS", "JOIN_DYNAMIC_SCOPE_PROVENANCE_UNRESOLVED")
        for group in zip(*streams):
            loop_rows = group[0]["uops"][0]["loops"]
            if any(event["uops"][0]["loops"] != loop_rows for event in group):
                _fail("UNSUPPORTED_SEMANTICS", "DYNAMIC_HARDWARE_LOOP_MISMATCH")
            rows = [uop for event in group for uop in event["uops"]]
            for index, row in enumerate(rows):
                if any(row["args"][2] == prior["args"][2] for prior in rows[max(0, index-2):index]):
                    _fail("PROGRAM_ILLEGAL", "RUNTIME_VERIFYDEP_JOIN_SEAM")
            previous = deque(maxlen=interval)
            for values in expanded_uops({"uops": rows}):
                if values[2] in previous:
                    _fail("PROGRAM_ILLEGAL", "VERIFIED_HARDWARE_JOIN_SEAM")
                previous.append(values[2])

    def check_seams(candidate_function):
        events, _ = _trace(candidate_function, _limits(limits))
        # Runtime VerifyDep(2) is already checked in _trace. The hardware
        # contract additionally covers the expanded, effective destination
        # stream inside each joined GEMM, including hardware loop seams.
        from collections import deque
        from c3_valid_window import expanded_uops
        for event in events:
            if event["kind"] != "kernel" or event["mode_reset"] != [0, 0]:
                continue
            previous = deque(maxlen=interval)
            for values in expanded_uops(event):
                if values[2] in previous:
                    _fail("PROGRAM_ILLEGAL", "VERIFIED_HARDWARE_DESTINATION_INTERVAL_HAZARD")
                previous.append(values[2])
        candidate = audit_function(candidate_function, limits, hw_contract)
        for key in ("tensor_requests_sha256", "control_trace_sha256", "output_product_postprocess_sha256"):
            if candidate["semantic"][key] != baseline["semantic"][key]:
                _fail("PROGRAM_ILLEGAL", "AGGREGATION_EQUIVALENCE_FAILED:" + key)
        return candidate

    def transform(node, stage=None):
        # Recursion enters the existing loop body but never combines For
        # siblings, lifts work, or removes a host loop.
        if isinstance(node, tir.For):
            body = transform(node.body, stage)
            return node if body.same_as(node.body) else tir.For(node.loop_var, node.min, node.extent, node.kind,
                body, node.thread_binding, node.annotations, node.span)
        if isinstance(node, tir.AttrStmt):
            if node.attr_key == "coproc_uop_scope":
                return node
            nested_stage = integer(node.value) if node.attr_key == "coproc_scope" else stage
            body = transform(node.body, nested_stage)
            return node if body.same_as(node.body) else tir.AttrStmt(node.node, node.attr_key, node.value, body, node.span)
        if isinstance(node, tir.SeqStmt):
            children = [transform(child, stage) for child in node.seq]
            result, pending = [], []
            def flush(reason, category=None):
                if not pending:
                    return
                decision = {"segments": len(pending), "base_uops": sum(u["count"] for u in pending),
                            "action": "JOIN" if len(pending) > 1 else "PRESERVE", "stop": reason}
                if category is not None:
                    decision["category"] = category
                decisions.append(decision)
                if len(pending) > 1:
                    result.append(_merge(pending))
                else:
                    result.append(pending[0]["node"])
                pending.clear()
            for child in children:
                unit, reason = _candidate(child, stage)
                if unit is None:
                    category = "UNSUPPORTED_SEMANTICS" if reason and reason.startswith("SUPPORT_UNRESOLVED") else None
                    flush(reason, category)
                    if category is not None:
                        # Record unsupported candidate shapes even when there
                        # is no pending prefix to flush, otherwise a region of
                        # entirely opaque candidates would look like NO_JOIN.
                        decisions.append({"action": "PRESERVE", "segments": 1,
                                          "stop": reason, "category": category})
                    result.append(child)
                    continue
                if pending:
                    first, previous = pending[0], pending[-1]
                    reason, category = None, None
                    if (first["outer"] is None) != (unit["outer"] is None):
                        reason = "DIFFERENT_SCOPE_FORM"
                    elif previous["suffix"] or unit["prefix"]:
                        reason = "INTERNAL_NECESSARY_EVENT"
                    elif first["outer"] is not None and not tvm.ir.structural_equal(first["outer"].node, unit["outer"].node):
                        reason = "DIFFERENT_COPROCESSOR_AXIS"
                    elif len(first["begins"]) != len(unit["begins"]) or any(not tvm.ir.structural_equal(a, b)
                            for a, b in zip(first["begins"]+first["ends"], unit["begins"]+unit["ends"])):
                        reason = "INCOMPATIBLE_HARDWARE_LOOP"
                    elif sum(u["count"] for u in pending)+unit["count"] > CAPACITY["UOP"]:
                        reason = "UOP_CAPACITY"
                    elif interval is None:
                        reason = "SUPPORT_UNRESOLVED_HARDWARE_DEP_INTERVAL"
                        category = "SUPPORT_UNRESOLVED"
                    else:
                        try:
                            guard_prefix([*pending, unit])
                        except AuditFailure as error:
                            reason, category = error.reason, error.category
                    if reason is not None:
                        flush(reason, category)
                pending.append(unit)
            flush("REGION_END")
            rebuilt = statement_sequence(result)
            if all(a.same_as(b) for a, b in zip(children, node.seq)) and len(result) == len(children) and all(a.same_as(b) for a, b in zip(result, children)):
                return node
            return rebuilt
        if isinstance(node, tir.IfThenElse):
            then = transform(node.then_case, stage)
            other = transform(node.else_case, stage) if node.else_case is not None else None
            return node if then.same_as(node.then_case) and (other is None or other.same_as(node.else_case)) else tir.IfThenElse(node.condition, then, other, node.span)
        if isinstance(node, tir.LetStmt):
            body = transform(node.body, stage)
            return node if body.same_as(node.body) else tir.LetStmt(node.var, node.value, body, node.span)
        if isinstance(node, tir.Allocate):
            body = transform(node.body, stage)
            return node if body.same_as(node.body) else tir.Allocate(node.buffer_var, node.dtype, node.extents, node.condition, body, node.annotations, node.span)
        if isinstance(node, tir.AssertStmt):
            body = transform(node.body, stage)
            return node if body.same_as(node.body) else tir.AssertStmt(node.condition, node.message, body, node.span)
        return node

    candidate_function = function.with_body(transform(function.body))
    # Do not silently publish a semantically invalid bulk transform. Keep B
    # and report its natural opportunities; callers must not label this NO_WINDOW.
    join_count = sum(d["segments"]-1 for d in decisions if d["action"] == "JOIN")
    if join_count:
        try:
            candidate = check_seams(candidate_function)
        except AuditFailure as error:
            decisions.append({"action": "REJECT_FULL_CANDIDATE", "stop": error.reason, "category": error.category})
            status = ("SUPPORT_UNRESOLVED_BASELINE_PRESERVED" if error.category.startswith(("UNSUPPORTED", "SUPPORT_UNRESOLVED"))
                      else "CANDIDATE_REJECTED_BASELINE_PRESERVED")
            return function, {"status": status, "B": baseline, "A": baseline,
                              "decisions": decisions, "joins": 0, "attempted_joins": join_count,
                              "native_fifo_operand_certificate_required": True}
    else:
        candidate = baseline
        candidate_function = function
    before_loops, after_loops = _preserved_host_loops(function), _preserved_host_loops(candidate_function)
    if len(before_loops) != len(after_loops) or any(not a.loop_var.same_as(b.loop_var) or not tvm.ir.structural_equal(a.min, b.min)
            or not tvm.ir.structural_equal(a.extent, b.extent) or a.kind != b.kind for a, b in zip(before_loops, after_loops)):
        _fail("AUDIT_INTERNAL_ERROR", "HOST_LOOP_ITERATION_STRUCTURE_CHANGED")
    unresolved = any(d.get("category", "").startswith(("UNSUPPORTED", "SUPPORT_UNRESOLVED"))
                     or d["stop"].startswith("SUPPORT_UNRESOLVED") for d in decisions)
    status = ("SUPPORT_UNRESOLVED_PARTIAL_PROJECTED_NATIVE_PENDING" if join_count else "SUPPORT_UNRESOLVED_BASELINE_PRESERVED") if unresolved else "AGGREGATED_PROJECTED_NATIVE_PENDING" if join_count else "NO_JOIN_IN_SUPPORTED_STRUCTURE"
    return candidate_function, {"status": status, "B": baseline, "A": candidate, "decisions": decisions,
        "joins": join_count, "host_loop_iterations_preserved": True, "longest_prefix_policy": True,
        "native_fifo_operand_certificate_required": True,
        "hardware_contract_valid_for_join": interval is not None}


def AggregateOfficialWindows(hw_contract=None, decisions=None, limits=None):
    """Pass wrapper, appending one complete per-function audit report."""
    reports = decisions if decisions is not None else []
    @tir.transform.prim_func_pass(opt_level=0, name="tir.c3.AggregateOfficialWindows")
    def transform(function, _module, _context):
        result, report = aggregate_function(function, hw_contract, limits)
        reports.append(report)
        return result
    return transform
