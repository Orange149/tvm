"""Audit final TIR addresses and actual instruction dumps, never cycle occupancy."""
import hashlib
import json
import re
from collections import Counter

CAPACITY = {"INP": 2048, "WGT": 1024, "ACC": 2048, "OUT": 2048, "UOP": 8192}
ELEMENT_BYTES = {"INP": 16, "WGT": 256, "ACC": 64, "OUT": 16, "UOP": 4}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def bounded(start, size, capacity, label):
    if start < 0 or size < 0 or start + size > capacity:
        raise ValueError(f"{label} address exceeds capacity: {start}+{size}>{capacity}")


def final_tir_audit(module, semantics):
    """Evaluate host loops; account UOP loop factors without expanding all MACs."""
    import tvm
    from tvm import tir
    analyzer = tvm.arith.Analyzer()
    kernels, syncs, events = [], [], []
    loops, kernel = [], None
    steps = 0

    def integer(expr, bindings):
        value = analyzer.simplify(tir.stmt_functor.substitute(expr, bindings))
        if not isinstance(value, tir.IntImm):
            raise ValueError("nonstatic final address expression " + str(value))
        return int(value)

    def walk(node, bindings):
        nonlocal kernel, steps
        steps += 1
        if steps > 2000000:
            raise ValueError("bounded TIR audit expansion exceeded")
        if isinstance(node, tir.For):
            lo, extent = integer(node.min, bindings), integer(node.extent, bindings)
            if extent < 0:
                raise ValueError("negative loop")
            for i in range(lo, lo + extent):
                walk(node.body, {**bindings, node.loop_var: tir.IntImm(node.loop_var.dtype, i)})
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                walk(child, bindings)
        elif isinstance(node, tir.LetStmt):
            walk(node.body, {**bindings, node.var: analyzer.simplify(tir.stmt_functor.substitute(node.value, bindings))})
        elif isinstance(node, tir.IfThenElse):
            target = node.then_case if integer(node.condition, bindings) else node.else_case
            if target is not None:
                walk(target, bindings)
        elif isinstance(node, tir.AttrStmt):
            if node.attr_key == "coproc_uop_scope":
                if kernel is not None or loops:
                    raise ValueError("nested UOP kernel")
                kernel = []
                walk(node.body, bindings)
                bounded(0, len(kernel), CAPACITY["UOP"], "single UOP kernel")
                if loops or not kernel:
                    raise ValueError("incomplete UOP kernel")
                kernels.append(kernel)
                events.append(["kernel", kernel])
                kernel = None
            else:
                walk(node.body, bindings)
        elif isinstance(node, (tir.Allocate, tir.AssertStmt)):
            walk(node.body, bindings)
        elif isinstance(node, tir.Evaluate):
            call = node.value
            if not isinstance(call, tir.Call):
                return
            name, args = call.op.name, list(call.args)
            if name == "tir.call_extern":
                name, args = args[0].value, args[1:]
            if name == "VTAUopLoopBegin":
                values = [integer(x, bindings) for x in args]
                if values[0] <= 0 or len(loops) == 2:
                    raise ValueError("invalid UOP loop")
                loops.append(values)
            elif name == "VTAUopLoopEnd":
                if not loops:
                    raise ValueError("unbalanced UOP loops")
                loops.pop()
            elif name in ("tir.vta.uop_push", "VTAUopPush"):
                values = [integer(x, bindings) for x in args]
                if kernel is None or len(values) != 8:
                    raise ValueError("unscoped or unsupported UOP")
                mode, reset, acc, src, wgt, opcode, immediate, imm = values
                for index, base, domain in ((1, acc, "ACC"), (2, src, "INP" if mode == 0 else "ACC"), (3, wgt, "WGT")):
                    if index != 1 and (reset or (mode == 1 and (index == 3 or immediate))):
                        continue
                    offsets = [(loop[0] - 1) * loop[index] for loop in loops]
                    low = base + sum(min(0, x) for x in offsets)
                    high = base + sum(max(0, x) for x in offsets)
                    bounded(low, high - low + 1, CAPACITY[domain], domain + " UOP consumer")
                kernel.append(dict(args=values, loops=[list(x) for x in loops]))
            elif name in ("tir.vta.coproc_sync", "VTASynchronize"):
                syncs.append(len(events))
            elif name in ("VTALoadBuffer2D", "VTAStoreBuffer2D"):
                # Ordered numeric requests, normalized only for pointer identity.
                ignore = {0, 1} if name == "VTALoadBuffer2D" else {0, 3}
                events.append([name, [integer(x, bindings) for i, x in enumerate(args) if i not in ignore]])
            elif name.startswith("tir.vta.coproc_dep_") or name in ("VTADepPush", "VTADepPop"):
                pass
            else:
                raise ValueError("unhandled final call " + name)
        else:
            raise ValueError("unhandled final statement " + type(node).__name__)

    for request in semantics["request_groups"]:
        domain = request["memory_name"].upper()
        x, y = request["x_size"], request["y_size"]
        # padding order from VTALoadBuffer2D: left, top, right, bottom.
        left, top, right, bottom = request["padding"]
        size = (x + left + right) * (y + top + bottom)
        bounded(request["sram_offset_elements"], size, CAPACITY[domain], domain + " DMA")
        if not request["ddr_bounds_checked"]:
            raise ValueError("unknown DDR bounds")
    walk(module["main"].body, {})
    if not kernels:
        raise ValueError("no final computation audited")
    return dict(status="passed", kernels=kernels, ordered_events=events, sync_event_positions=syncs,
                max_kernel_uops=max(map(len, kernels)), event_sha256=digest(events),
                compute_sha256=digest(kernels), uop_effective_occupancy_peak=None)


def normalized_body(module):
    """Remove only dependency/sync effects; retain addresses, loops and UOP scope."""
    import tvm
    from tvm import tir

    def post(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "coproc_scope":
            return node.body
        if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            call = node.value
            name = call.op.name
            if name in ("tir.vta.coproc_sync", "tir.vta.coproc_dep_push", "tir.vta.coproc_dep_pop"):
                return tir.Evaluate(0)
    function = module["main"]
    body = tir.stmt_functor.ir_transform(function.body, None, post)
    normalized = tvm.IRModule({"main": function.with_body(body)})
    return tir.transform.Simplify()(normalized)["main"]


def control_match(left, right, left_module, right_module, kind):
    import tvm
    if not left or not right:
        return dict(matched=False, reason="candidate unqualified")
    if kind == "sync":
        checks = dict(
            normalized_structure=tvm.ir.structural_equal(normalized_body(left_module), normalized_body(right_module), map_free_vars=True),
            ordered_data_and_compute=left["event_sha256"] == right["event_sha256"],
            extra_sync=len(right["sync_event_positions"]) > len(left["sync_event_positions"]))
    else:
        # Compare host/UOP loop arrangement, mode and postprocessing; SRAM bases
        # are explicitly NOT equalized or claimed equal for residency controls.
        def layout(a):
            return [[[k["args"][:2], k["args"][5:], k["loops"]] for k in group] for group in a["kernels"]]
        def data(a, weight=False):
            return [event for event in a["ordered_events"] if event[0] != "kernel" and
                    not (event[0] == "VTALoadBuffer2D" and event[1][-1] == 1)]
        checks = dict(compute_context=layout(left) == layout(right),
                      nonweight_data=data(left) == data(right),
                      sync_count=len(left["sync_event_positions"]) == len(right["sync_event_positions"]))
    return dict(matched=all(checks.values()), checks=checks,
                left_event_sha256=left["event_sha256"], right_event_sha256=right["event_sha256"],
                left_sync_positions=left["sync_event_positions"], right_sync_positions=right["sync_event_positions"],
                interpretation="program effect, not isolated hardware cycles")


def instruction_dump(text):
    """Parse runtime DumpInsn. Raw opcodes and exclusive semantic classes differ."""
    blocks = []
    for match in re.finditer(r"There are (\d+) instructions\n(.*?)(?=There are \d+ instructions|\Z)", text, re.S):
        declared = int(match[1])
        records = list(re.finditer(r"INSTRUCTION (\d+): ([^\n]+)\n(.*?)(?=INSTRUCTION |\Z)", match[2], re.S))
        if len(records) != declared or [int(x[1]) for x in records] != list(range(declared)):
            raise ValueError("incomplete instruction dump")
        raw, semantic, loads, ranges = Counter(), Counter(), Counter(), []
        for record in records:
            desc, detail = record[2], record[3]
            if desc.startswith("NOP-"):
                raw["STORE" if "STORE" in desc else "LOAD"] += 1
                semantic[desc] += 1
            elif desc.startswith("LOAD ") or desc.startswith("STORE"):
                load = desc.startswith("LOAD ")
                domain = desc.split()[1] if load else "OUT"
                raw["LOAD" if load else "STORE"] += 1
                semantic[("LOAD_" if load else "STORE_") + domain] += 1
                address = re.search(r"SRAM:0x([0-9a-fA-F]+)", detail)
                x = re.search(r"x: size=(\d+), stride=(\d+), pad=\[(\d+), (\d+)\]", detail)
                y = re.search(r"y: size=(\d+), pad=\[(\d+), (\d+)\]", detail)
                if not address or not x or not y or domain not in CAPACITY:
                    raise ValueError("missing memory instruction fields")
                xx, yy = list(map(int, x.groups())), list(map(int, y.groups()))
                start = int(address[1], 16)
                size = (xx[0] + xx[2] + xx[3]) * sum(yy)
                bounded(start, size, CAPACITY[domain], domain + " instruction")
                if load:
                    loads[domain] += xx[0] * yy[0] * ELEMENT_BYTES[domain]
                if domain == "UOP":
                    ranges.append((start, start + size))
            elif desc == "GEMM" or desc.startswith("ALU -"):
                op = "GEMM" if desc == "GEMM" else "ALU"
                raw[op] += 1
                reset = re.search(r"reset_out: (\d+)", detail)
                extent = re.search(r"range \((\d+), (\d+)\)", detail)
                if not reset or not extent:
                    raise ValueError("missing compute instruction fields")
                begin, end = map(int, extent.groups())
                bounded(begin, end - begin, CAPACITY["UOP"], "UOP instruction reference")
                if end <= begin or not any(a <= begin and end <= b for a, b in ranges):
                    raise ValueError("UOP reference not covered by a preceding load")
                semantic["RESET_GEMM" if op == "GEMM" and int(reset[1]) else "MAC_GEMM" if op == "GEMM" else "ALU"] += 1
            elif desc == "FINISH":
                raw["FINISH"] += 1
                semantic["FINISH"] += 1
            else:
                raise ValueError("unknown opcode " + desc)
        if records[-1][2] != "FINISH" or sum(raw.values()) != declared or sum(semantic.values()) != declared:
            raise ValueError("instruction accounting mismatch")
        blocks.append(dict(instructions=declared, raw_opcode=dict(raw), semantic=dict(semantic),
                           load_bytes=dict(loads), uop_load_ranges=ranges))
    if not blocks:
        raise ValueError("no instruction dump")
    return dict(status="passed", submissions=blocks, submission_count=len(blocks),
                uop_load_bytes=sum(x["load_bytes"].get("UOP", 0) for x in blocks),
                effective_uop_occupancy_peak=None)
