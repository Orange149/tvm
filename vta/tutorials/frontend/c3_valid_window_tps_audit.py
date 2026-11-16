#!/usr/bin/env python3
"""Read-only audit of the sealed TPS program; never generate a TPS candidate."""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re

from c3_five_tradeoff_commands import completion_dag, decode_submissions
from c3_thread_sync_audit import CAPACITY, final_tir_audit
from audit_c3_banerjee_completion import native_certificate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract_tir(function):
    """Expand finite host loops, retaining every dependency and UOP scope."""
    import tvm
    from tvm import tir

    analyzer = tvm.arith.Analyzer()
    events, kernels, loops = [], [], []
    current = None
    parameters = [function.buffer_map[p].data for p in function.params]

    def integer(expr, bindings):
        value = analyzer.simplify(tir.stmt_functor.substitute(expr, bindings))
        require(isinstance(value, tir.IntImm), "unresolved finite TPS expression")
        return int(value)

    def emit(kind, **details):
        event = dict(event=len(events), kind=kind, **details)
        events.append(event)
        return event

    def walk(node, bindings):
        nonlocal current
        if isinstance(node, tir.For):
            lo, extent = integer(node.min, bindings), integer(node.extent, bindings)
            require(0 <= extent <= 8192, "unsupported host loop extent")
            for value in range(lo, lo + extent):
                walk(node.body, {**bindings, node.loop_var: tir.IntImm(node.loop_var.dtype, value)})
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                walk(child, bindings)
        elif isinstance(node, tir.LetStmt):
            walk(node.body, {**bindings, node.var: analyzer.simplify(
                tir.stmt_functor.substitute(node.value, bindings))})
        elif isinstance(node, tir.IfThenElse):
            chosen = node.then_case if integer(node.condition, bindings) else node.else_case
            if chosen is not None:
                walk(chosen, bindings)
        elif isinstance(node, tir.AttrStmt):
            if node.attr_key == "coproc_uop_scope":
                require(current is None and not loops, "nested TPS UOP scope")
                current = []
                walk(node.body, bindings)
                require(current and not loops, "empty or unclosed TPS kernel")
                kernels.append(current)
                first = current[0]["args"]
                require(all(k["args"][:2] == first[:2] for k in current), "mixed kernel mode")
                emit("kernel", kernel=len(kernels) - 1, mode=first[0], reset=first[1],
                     uops=len(current), loops=current[0]["loops"])
                current = None
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
                require(current is not None and len(loops) < 2, "unscoped hardware loop")
                loops.append([integer(a, bindings) for a in args])
            elif name == "VTAUopLoopEnd":
                require(loops, "unbalanced hardware loop")
                loops.pop()
            elif name in ("tir.vta.uop_push", "VTAUopPush"):
                require(current is not None, "unscoped UOP")
                current.append(dict(args=[integer(a, bindings) for a in args],
                                    loops=[list(loop) for loop in loops]))
            elif name in ("tir.vta.coproc_dep_push", "tir.vta.coproc_dep_pop"):
                emit(name.rsplit("_", 1)[-1], lane=[integer(a, bindings) for a in args])
            elif name == "VTALoadBuffer2D":
                object_id = next(i for i, p in enumerate(parameters) if args[1].same_as(p))
                emit("load", object=object_id, values=[integer(a, bindings) for a in args[2:]])
            elif name == "VTAStoreBuffer2D":
                object_id = next(i for i, p in enumerate(parameters) if args[3].same_as(p))
                emit("store", object=object_id,
                     values=[integer(a, bindings) for i, a in enumerate(args) if i not in (0, 3)])
            elif name == "tir.vta.coproc_sync":
                emit("sync")
            else:
                raise ValueError("unhandled TPS call " + name)
        else:
            raise ValueError("unhandled TPS statement " + type(node).__name__)

    walk(function.body, {})
    return events, kernels


def native_fields(record):
    detail = record["detail"]
    domain = record["description"]
    if domain.startswith("LOAD ") or domain.startswith("STORE"):
        address = re.search(r"DRAM: 0x([0-9a-fA-F]+), SRAM:0x([0-9a-fA-F]+)", detail)
        x = re.search(r"x: size=(\d+), stride=(\d+), pad=\[(\d+), (\d+)\]", detail)
        y = re.search(r"y: size=(\d+), pad=\[(\d+), (\d+)\]", detail)
        require(address and x and y, "missing native memory fields")
        xx, yy = list(map(int, x.groups())), list(map(int, y.groups()))
        return dict(dram=int(address[1], 16), sram=int(address[2], 16), x=xx[0], y=yy[0],
                    stride=xx[1], padding=[xx[2], yy[1], xx[3], yy[2]])
    extent = re.search(r"range \((\d+), (\d+)\)", detail)
    require(extent, "missing native kernel range")
    begin, end = map(int, extent.groups())
    return dict(begin=begin, end=end, uops=end - begin)


def bind_material(events, kernels, records):
    """Verify exact material order, relative DDR offsets, SRAM and descriptors."""
    material = [e for e in events if e["kind"] in ("kernel", "load", "store")]
    native = [r for r in records if r["description"] in ("LOAD INP", "LOAD WGT", "LOAD ACC", "STORE:", "GEMM")
              or r["description"].startswith("ALU -")]
    require(len(material) == len(native), "TIR/native material event count differs")
    pointers, mapping = {}, []
    memory = {1: "LOAD WGT", 2: "LOAD INP", 3: "LOAD ACC"}
    for event, record in zip(material, native):
        event["instruction"] = record["index"]
        desc, fields = record["description"], native_fields(record)
        if event["kind"] == "kernel":
            expected = "GEMM" if event["mode"] == 0 else "ALU -"
            require(desc.startswith(expected), "kernel opcode differs")
            require(event["uops"] == fields["uops"], "single kernel UOP count differs")
            require(f"reset_out: {event['reset']}" in record["detail"], "RESET mode differs")
            if event["mode"] == 0:
                raw = re.findall(r"(?:outer|inner) loop - iter: (\d+), wgt: (-?\d+), inp: (-?\d+), acc: (-?\d+)", record["detail"])
                factors = [[int(a), int(d), int(c), int(b)] for a, b, c, d in raw]
                require(factors == event["loops"], "GEMM hardware loop factors differ")
            else:
                uop = kernels[event["kernel"]][0]
                opcode, immediate = uop["args"][5:7]
                name = {0: "min", 1: "max", 2: "add", 3: "shr", 4: "mul"}[opcode]
                if immediate and opcode < 3:
                    name += " imm"
                require(desc == "ALU - " + name, "ALU opcode differs")
                loops = uop["loops"]
                expected_loops = ([[1, 0, 0, 0], [1, 0, 0, 0]] if not loops else
                                  [[1, 0, 0, 0], loops[0]] if len(loops) == 1 else loops)
                raw = re.findall(r"(?:outer|inner) loop - iter: (\d+), dst: (-?\d+), src: (-?\d+)", record["detail"])
                factors = [[int(a), int(b), int(c), 0] for a, b, c in raw]
                require(factors == expected_loops, "ALU hardware loop factors differ")
        else:
            values = event["values"]
            if event["kind"] == "load":
                offset, x, y, stride, left, top, right, bottom, sram, domain = values
                require(desc == memory[domain], "LOAD domain differs")
                padding = [left, top, right, bottom]
            else:
                sram, domain, offset, x, y, stride = values
                require(desc == "STORE:" and domain == 4, "STORE domain differs")
                padding = [0, 0, 0, 0]
            require((fields["sram"], fields["x"], fields["y"], fields["stride"], fields["padding"])
                    == (sram, x, y, stride, padding), "tensor descriptor differs")
            pointer = fields["dram"] - offset
            require(event["object"] not in pointers or pointers[event["object"]] == pointer,
                    "relative DDR request changes object version")
            pointers[event["object"]] = pointer
        mapping.append(dict(tir_event=event["event"], instruction=record["index"], kind=event["kind"],
                            description=desc, native=fields))
    return mapping


def bind_dependencies(events, records):
    """Bind per-lane TIR calls to instruction bits and pair actual FIFO tokens."""
    occurrences = defaultdict(list)
    for record in records:
        stage = record["stage"]
        for flag, kind, lane in zip(record["bits"], ("pop", "pop", "push", "push"),
                                   ((stage - 1, stage), (stage + 1, stage),
                                    (stage, stage - 1), (stage, stage + 1))):
            if flag:
                occurrences[(kind, lane)].append(record["index"])
    by_key = defaultdict(list)
    for event in events:
        if event["kind"] in ("push", "pop"):
            by_key[(event["kind"], tuple(event["lane"]))].append(event)
    mapping, drain = [], []
    for key, locations in occurrences.items():
        source = by_key[key]
        expected_drain = key in (("push", (3, 2)), ("push", (1, 2)),
                                 ("pop", (3, 2)), ("pop", (1, 2)))
        require(len(locations) == len(source) + int(expected_drain), "TIR/native token count differs")
        for ordinal, (event, instruction) in enumerate(zip(source, locations)):
            mapping.append(dict(tir_event=event["event"], kind=key[0], lane=list(key[1]),
                                ordinal=ordinal, instruction=instruction))
        for instruction in locations[len(source):]:
            drain.append(dict(kind=key[0], lane=list(key[1]), instruction=instruction,
                              origin="existing runtime Synchronize drain"))
    require(set(by_key) <= set(occurrences), "TIR token disappeared")
    pairs = []
    for lane in sorted({key[1] for key in occurrences}):
        producers = occurrences[("push", lane)]
        consumers = occurrences[("pop", lane)]
        require(len(producers) == len(consumers), "token lane does not close")
        pairs.extend(dict(lane=list(lane), ordinal=i, producer=a, consumer=b)
                     for i, (a, b) in enumerate(zip(producers, consumers)))
    return sorted(mapping, key=lambda x: x["tir_event"]), drain, pairs


def operand_versions(events, kernels, records):
    """Resolve every expanded operand address to a loaded version and DAG edge."""
    ancestors = completion_dag(records)
    before = lambda a, b: bool(ancestors[b] & (1 << a))
    loads = defaultdict(list)
    for event in events:
        if event["kind"] == "load" and event["values"][-1] in (1, 2):
            fields = native_fields(records[event["instruction"]])
            size = (fields["x"] + fields["padding"][0] + fields["padding"][2]) * (
                fields["y"] + fields["padding"][1] + fields["padding"][3])
            loads[event["values"][-1]].append(dict(instruction=event["instruction"],
                begin=fields["sram"], end=fields["sram"] + size,
                object=event["object"], ddr_offset=event["values"][0]))
    summary = []
    for event in events:
        if event["kind"] != "kernel" or event["mode"] or event["reset"]:
            continue
        read_addresses = {1: set(), 2: set()}
        output_counts = Counter()
        for uop in kernels[event["kernel"]]:
            loops = uop["loops"]
            for indices in itertools.product(*(range(loop[0]) for loop in loops)):
                acc, inp, wgt = uop["args"][2:5]
                acc += sum(i * loop[1] for i, loop in zip(indices, loops))
                inp += sum(i * loop[2] for i, loop in zip(indices, loops))
                wgt += sum(i * loop[3] for i, loop in zip(indices, loops))
                output_counts[acc] += 1
                read_addresses[1].add(wgt)
                read_addresses[2].add(inp)
        require(len(output_counts) == 784 and set(output_counts.values()) == {36},
                "TPS kernel reduction/output coverage differs")
        versions = {}
        for domain, addresses in read_addresses.items():
            used = {}
            for address in addresses:
                prior = [load for load in loads[domain] if load["instruction"] < event["instruction"]
                         and load["begin"] <= address < load["end"]]
                require(prior, "operand has no established version")
                version = prior[-1]
                require(before(version["instruction"], event["instruction"]), "LOAD/read not ordered")
                later = [load for load in loads[domain] if load["instruction"] > event["instruction"]
                         and load["begin"] <= address < load["end"]]
                if later:
                    require(before(event["instruction"], later[0]["instruction"]),
                            "read/overwrite not ordered")
                used[version["instruction"]] = version
            versions[str(domain)] = list(used.values())
        summary.append(dict(kernel=event["kernel"], instruction=event["instruction"],
            base_uops=event["uops"], expanded_vector_products=sum(output_counts.values()),
            output_acc_range=[min(output_counts), max(output_counts) + 1],
            products_per_output_vector=36, operand_versions=versions))
    return summary


def audit(root, out):
    import tvm
    import vta  # registers dependency and UOP operations

    program = "TPS-oc-reuse-ci4"
    local, arm = root / "local_final06" / program, root / "binaries" / program
    paths = [local / n for n in ("node24_final.tir.json", "address_audit.json", "request_semantics.json", "contract.json")]
    paths += [arm / n for n in ("node24_final.tir.json", "graphlib.so", "result.json")]
    paths += [root / n for n in ("selected_pool.json", "runtime_board_attempt02/build.json",
                               "runtime_board_attempt02/runtime.cc", "runtime_board_attempt02/libvta.so",
                               "qualification/calls.jsonl", "qualification/sessions.jsonl")]
    paths += [Path(__file__).resolve().parent / name for name in (
        "c3_five_tradeoff_commands.py", "c3_thread_sync_audit.py", "audit_c3_banerjee_completion.py")]
    paths += [root / ("qualification/qualification_%06d.txt" % i) for i in (75, 84, 85, 86)]
    sources = {str(p): digest(p) for p in paths}
    function = tvm.ir.load_json((local / "node24_final.tir.json").read_text())
    arm_function = tvm.ir.load_json((arm / "node24_final.tir.json").read_text())
    require(tvm.ir.structural_equal(function, arm_function, map_free_vars=True), "ARM/local TIR differs")
    require(json.loads((root / "selected_pool.json").read_text())["primary_TPS_baseline"] == program,
            "wrong sealed TPS baseline")
    arm_result = json.loads((arm / "result.json").read_text())
    require(arm_result["binary_sha256"] == digest(arm / "graphlib.so"), "ARM binary identity differs")
    runtime = json.loads((root / "runtime_board_attempt02/build.json").read_text())
    require(runtime["binary_sha256"] == digest(root / "runtime_board_attempt02/libvta.so"), "runtime identity differs")
    calls = [json.loads(line) for line in (root / "qualification/calls.jsonl").read_text().splitlines()]
    sessions = [json.loads(line) for line in (root / "qualification/sessions.jsonl").read_text().splitlines()]
    historical_bindings = []
    for i in (75, 84, 85, 86):
        call = next(c for c in calls if c["invocation_id"] == "qualification:%06d" % i)
        require(call["program"] == program and call["status"] == "PASS", "native dump invocation differs")
        log = root / "qualification" / call["dump_file"]
        require(log.name == "qualification_%06d.txt" % i and digest(log) == call["dump_sha256"],
                "historical native dump identity differs")
        session = next(s for s in sessions if s["session_id"] == call["session_id"])
        require(session["runtime_attestation"]["libvta.so"] == runtime["binary_sha256"],
                "actual loaded runtime identity differs")
        require(session["program"] == program and session["status"] == "PASS", "historical session differs")
        require(session["environment"]["boot_id"] == call["boot_id"], "historical boot differs")
        require(session["start_state"]["udmabuf_size"] == 192 * 1024 * 1024,
                "historical DMA contract differs")
        require(session["bitstream_load_record"]["sha256"] == "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6",
                "historical bitstream identity differs")
        require(session["inventory"]["replay_initial"]["mode"] == "disabled", "template replay differs")
        require(all(mask == "3" for mask in session["environment"]["masks"]), "historical affinity differs")
        historical_bindings.append(dict(invocation=call["invocation_id"], session=call["session_id"],
            boot=call["boot_id"], kind=call["call_kind"], dump_sha256=call["dump_sha256"],
            runtime_sha256=session["runtime_attestation"]["libvta.so"],
            bitstream_sha256=session["bitstream_load_record"]["sha256"]))
    contract = json.loads((local / "contract.json").read_text())
    address = final_tir_audit(tvm.IRModule({"main": function}), json.loads((local / "request_semantics.json").read_text()))
    prior_address = json.loads((local / "address_audit.json").read_text())
    require(address["kernels"] == prior_address["kernels"], "frozen UOP side table differs")
    events, kernels = extract_tir(function)
    require(kernels == address["kernels"], "independent TIR UOP extraction differs")
    diagnostics = []
    for i in (75, 84, 85, 86):
        text = (root / ("qualification/qualification_%06d.txt" % i)).read_text()
        blocks = decode_submissions(text)
        require(len(blocks) == 1, "sealed TPS submit count differs")
        records = blocks[0]
        material = bind_material(events, kernels, records)
        dependencies, drain, pairs = bind_dependencies(events, records)
        versions = operand_versions(events, kernels, records)
        certificate = native_certificate(text, contract)
        diagnostics.append(dict(invocation="qualification:%06d" % i, material_map=material,
            dependency_map=dependencies, runtime_drain_events=drain, token_pairs=pairs,
            mac_operand_versions=versions, certificate=certificate))
    macs = [e for e in events if e["kind"] == "kernel" and e["mode"] == 0 and e["reset"] == 0]
    boundaries = []
    for left, right in zip(macs, macs[1:]):
        internal = [e for e in events[left["event"] + 1:right["event"]] if e["kind"] != "kernel"]
        intervening_kernels = [e for e in events[left["event"] + 1:right["event"]] if e["kind"] == "kernel"]
        reasons = sorted(set(e["kind"] for e in internal))
        if intervening_kernels:
            reasons.append("RESET_or_postprocessing")
        require("push" in reasons and "pop" in reasons, "TPS boundary lacks expected events")
        boundaries.append(dict(left_kernel=left["kernel"], right_kernel=right["kernel"],
            left_instruction=left["instruction"], right_instruction=right["instruction"],
            intervening_events=[e["event"] for e in internal], reasons=reasons,
            decision="cut: no internal token relocation in current support domain"))
    require(all(digest(Path(path)) == value for path, value in sources.items()), "historical source changed")
    result = dict(schema="c3_valid_window_tps_static_audit_v1", status="SEALED_TPS_AUDIT_PASSED",
        program=program, tile=contract["selection"]["tile"], ABI=contract["ABI"],
        arm_binary_sha256=arm_result["binary_sha256"], board_runtime_sha256=runtime["binary_sha256"],
        new_board_contact=False, TPS_candidate_generated=False, historical_sources_unchanged=True,
        sources=sources, historical_invocation_bindings=historical_bindings,
        address_status=address["status"], max_kernel_uops=address["max_kernel_uops"],
        uop_capacity=CAPACITY["UOP"], capacity_only_max_equal_MAC_kernels=CAPACITY["UOP"] // 2352,
        supported_MAC_window_sizes=[1] * len(macs), boundary_checks=boundaries,
        expanded_TIR_events=events, diagnostics=diagnostics,
        conclusion="TPS already uses 2352-uop MAC kernels; all adjacent MAC boundaries carry retained events",
        limits=["No proof that a broader token-moving transformation could never be legal",
                "DumpInsn does not dump serialized UOP values or kernel signatures; no modified TPS cache claim",
                "No hardware timing, occupancy, or pipeline stall inference from static counts",
                "No TPS aggregation candidate or new performance claim"])
    out.mkdir(parents=True, exist_ok=True)
    target = out / "tps_static_audit.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "program", "max_kernel_uops",
                     "capacity_only_max_equal_MAC_kernels", "supported_MAC_window_sizes")}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.out.resolve())
