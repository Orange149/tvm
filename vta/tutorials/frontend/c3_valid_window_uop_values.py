"""Restricted, local certificate for actual compute FIFO UOP contents.

Inputs are an existing runtime debug dump and the independently expanded
final-TIR kernel trace. No VTA runtime or board is loaded. A cold first dump
must expose every unique kernel initializer. Warm dumps require the cache
definitions returned for that same module/session. Unsupported identities,
multiple submits or missing definitions are rejected conservatively.
"""

import hashlib
import json
from pathlib import Path
import re


HEADER = Path(__file__).resolve().parent / "c3_banerjee_uop_queue.h"
HEADER_SHA256 = "f0f3f7c881a2d58f078d3654aecb0881a07a17717c934a0fa418cb22d5d6459f"
DEPTH = 8192


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _normalize(kernel):
    rows = kernel["uops"] if isinstance(kernel, dict) else kernel
    _require(bool(rows), "empty kernel cannot be certified")
    args = [list(map(int, row["args"])) for row in rows]
    loops = [list(map(int, loop)) for loop in rows[0]["loops"]]
    _require(all(len(row) == 8 for row in args), "eight integer UOP arguments required")
    _require(all(row["loops"] == rows[0]["loops"] for row in rows), "per-UOP loop disagreement")
    mode, reset = args[0][:2]
    _require(mode in (0, 1) and reset in (0, 1), "unsupported mode/reset")
    _require(all(row[:2] == [mode, reset] for row in args), "mixed kernel mode/reset")
    _require(len(loops) <= 2 and all(len(loop) == 4 for loop in loops), "at most two four-field loops required")
    _require(all(0 < loop[0] < 16384 for loop in loops), "unrepresentable loop extent")
    if mode == 1:
        _require(all(row[5:] == args[0][5:] for row in args), "mixed ALU opcode/immediate")
    result = {"args": args, "loops": loops}
    identity = kernel.get("cache_identity") if isinstance(kernel, dict) else None
    # Default is conservative: if two static handles have identical full
    # content but the dump creates both, the initializer census will reject.
    result["identity"] = _hash(["explicit_cache_identity",identity]) if identity is not None else _hash([args, loops])
    result["values"] = [row[2:5] for row in args]
    return result


def _initializers(text):
    definitions = []
    for count, body in re.findall(r"There are (\d+) uops\n(.*?)(?=\nThere are|\Z)", text, re.S):
        values = [list(map(int, match)) for match in re.findall(r"\[\d+\]\s+acc=(\d+), inp=(\d+), wgt=(\d+)", body)]
        _require(len(values) == int(count), "incomplete initializer dump")
        definitions.append(values)
    return definitions


def _instructions(text):
    _require(len(re.findall(r"There are \d+ instructions", text)) == 1, "only one complete actual submit supported")
    _require("INSTRUCTION 0:" in text, "actual instruction stream absent")
    output = []
    for block in re.split(r"(?=INSTRUCTION \d+:)", text[text.index("INSTRUCTION 0:"):]):
        if re.match(r"INSTRUCTION \d+: LOAD UOP", block):
            source, begin = [int(value, 16) for value in re.search(r"DRAM: 0x([0-9a-f]+), SRAM:0x([0-9a-f]+)", block).groups()]
            size = int(re.search(r"x: size=(\d+)", block).group(1))
            output.append({"kind":"load", "source":source, "begin":begin, "size":size})
        elif re.match(r"INSTRUCTION \d+: (GEMM|ALU)", block):
            mode = 0 if re.match(r"INSTRUCTION \d+: GEMM", block) else 1
            begin, end = map(int, re.search(r"range \((\d+), (\d+)\)", block).groups())
            reset = int(re.search(r"reset_out: (\d+)", block).group(1))
            if mode == 0:
                loops = [[int(n), int(dst), int(src), int(wgt)] for n,wgt,src,dst in re.findall(r"loop - iter: (\d+), wgt: (\d+), inp: (\d+), acc: (\d+)", block)]
            else:
                loops = [[int(n), int(dst), int(src), 0] for n,dst,src in re.findall(r"loop - iter: (\d+), dst: (\d+), src: (\d+)", block)]
            output.append({"kind":"compute", "mode":mode, "reset":reset,
                           "begin":begin, "end":end, "loops":loops, "block":block})
    _require(bool(output), "empty relevant FIFO")
    return output


def _expected_loops(kernel):
    loops = kernel["loops"]
    defaults = [1,0,0,0]
    if kernel["args"][0][0] == 0:
        return [loops[0] if loops else defaults, loops[1] if len(loops) == 2 else defaults]
    return [loops[0] if len(loops) == 2 else defaults, loops[-1] if loops else defaults]


def _simulate(kernels, definitions):
    """The frozen queue's allocation, alias, eviction and greedy rules."""
    objects = {key:{"values":value, "begin":0,"end":0} for key,value in definitions.items()}
    cache, dram, instructions = [], [], []
    cache_idx, sram_end, wraps = 0, 0, 0
    for kernel in kernels:
        obj = objects[kernel["identity"]]
        if obj["begin"] == obj["end"]:
            size = len(obj["values"])
            _require(0 < size <= DEPTH, "single kernel exceeds UOP SRAM")
            prev = cache[cache_idx-1] if cache_idx else None
            if prev and prev["begin"] != prev["end"] and prev["values"] == obj["values"]:
                obj["begin"],obj["end"] = prev["begin"],prev["end"]
                cache.insert(cache_idx,obj)
                cache_idx += 1
            else:
                if not any(item is obj for item in dram):
                    dram.append(obj)
                begin = 0
                if sram_end + size > DEPTH:
                    cache_idx, sram_end = 0, size
                    wraps += 1
                else:
                    begin = sram_end
                    sram_end += size
                evict_begin = cache_idx
                while cache_idx < len(cache):
                    old = cache[cache_idx]
                    if old["begin"] >= sram_end:
                        break
                    old["begin"] = old["end"] = 0
                    cache_idx += 1
                obj["begin"],obj["end"] = begin,sram_end
                cache.insert(cache_idx,obj)
                del cache[evict_begin:cache_idx]
                cache_idx = evict_begin+1
                source = 0
                for item in dram:
                    if item is obj:
                        break
                    source += len(item["values"])
                prior = next((item for item in reversed(instructions) if item["kind"] == "load"), None)
                if prior and prior["begin"]+prior["size"] == begin and prior["source"]+prior["size"] == source and cache_idx > 1 and prior["begin"]+prior["size"]+size < DEPTH:
                    prior["size"] += size
                else:
                    instructions.append({"kind":"load","source":source,"begin":begin,"size":size})
        instructions.append({"kind":"compute","begin":obj["begin"],"end":obj["end"]})
    packed = [value for item in dram for value in item["values"]]
    _require(len(packed)*4 <= (1 << 25), "automatic host UOP submit outside support")
    return instructions,packed,wraps


def certify_uop_values(text, expected_kernels, cache_definitions=None):
    """Certify actual FIFO contents against a final-TIR trace.

    Returns JSON-serializable evidence, including `cache_definitions` to pass
    to warm calls of the SAME session/module. Callers must enforce that session
    binding externally. Identity ambiguity causes rejection; optional
    `cache_identity` on kernels can bind independently checked scope/signature.
    """
    _require(hashlib.sha256(HEADER.read_bytes()).hexdigest() == HEADER_SHA256, "frozen runtime header changed")
    kernels = [_normalize(kernel) for kernel in expected_kernels]
    _require(bool(kernels), "expected kernel trace empty")
    unique = {}
    unique_contract = {}
    for kernel in kernels:
        contract = {"args":kernel["args"],"loops":kernel["loops"]}
        _require(kernel["identity"] not in unique_contract or unique_contract[kernel["identity"]] == contract,
                 "same cache identity changes UOP/mode/loop content")
        unique_contract.setdefault(kernel["identity"],contract)
        unique.setdefault(kernel["identity"],kernel["values"])
    initial = _initializers(text)
    text_sha256 = hashlib.sha256(text.encode()).hexdigest()
    if cache_definitions is None:
        _require(len(initial) == len(unique), "cold initializer census/identity ambiguity; no content certificate")
        _require(initial == list(unique.values()), "actual initializer values/order differ from final TIR")
        frozen = {"schema":"c3_valid_window_uop_cache_definitions_v2", "header_sha256":HEADER_SHA256,
                  "origin_dump_sha256":text_sha256,"definitions":unique,"contracts":unique_contract}
        full_contract_binding = True
    else:
        _require(cache_definitions["schema"] in ("c3_valid_window_uop_cache_definitions_v1","c3_valid_window_uop_cache_definitions_v2") and cache_definitions["header_sha256"] == HEADER_SHA256, "invalid frozen initializer binding")
        _require(not initial, "warm dump unexpectedly creates new kernels")
        _require(cache_definitions["definitions"] == unique, "warm trace differs from frozen initializer identities")
        full_contract_binding = cache_definitions["schema"] == "c3_valid_window_uop_cache_definitions_v2"
        if full_contract_binding:
            _require(cache_definitions["contracts"] == unique_contract,
                     "warm full kernel contract changes args/mode/loop/immediate")
        frozen = cache_definitions
    simulated,packed,wraps = _simulate(kernels,frozen["definitions"])
    actual = _instructions(text)
    _require(len(actual) == len(simulated), "actual UOP/compute instruction census differs from frozen queue replay")
    sources = [row["source"] for row in actual if row["kind"] == "load"]
    _require(bool(sources), "actual cold SRAM load absent")
    dram_base = min(sources)
    live = [None] * DEPTH
    checks = []
    compute_idx = 0
    for index,(row,predicted) in enumerate(zip(actual,simulated)):
        _require(row["kind"] == predicted["kind"], "actual FIFO ordering differs from frozen queue replay")
        if row["kind"] == "load":
            source = row["source"] - dram_base
            _require({"kind":"load","source":source,"begin":row["begin"],"size":row["size"]} == predicted, "actual UOP load descriptor differs from serialized callback replay")
            start,size = row["begin"],row["size"]
            _require(source >= 0 and source+size <= len(packed) and start >= 0 and start+size <= DEPTH, "UOP load out of certified range")
            live[start:start+size] = packed[source:source+size]
        else:
            kernel = kernels[compute_idx]
            _require(row["begin"] == predicted["begin"] and row["end"] == predicted["end"], "actual UOP reference differs from frozen allocation replay")
            _require(row["mode"] == kernel["args"][0][0] and row["reset"] == kernel["args"][0][1], "actual mode/reset differs from final trace")
            _require(row["loops"] == _expected_loops(kernel), "actual hardware loop differs from final trace")
            if row["mode"] == 1:
                opcode = kernel["args"][0][5]
                names = {0:"min",1:"max",2:"add",3:"shr",4:"mul"}
                _require(opcode in names and ("ALU - "+names[opcode]) in row["block"], "actual ALU opcode differs")
            _require(live[row["begin"]:row["end"]] == kernel["values"], "actual FIFO reads overwritten or wrong UOP contents")
            checks.append({"compute_index":compute_idx,"fifo_index":index,"uop_begin":row["begin"],
                           "uop_end":row["end"],"kernel_identity":kernel["identity"],"values_match":True})
            compute_idx += 1
    _require(compute_idx == len(kernels), "actual compute reference count differs")
    return {"status":"ACTUAL_FIFO_UOP_VALUES_MATCH_FINAL_TRACE", "dump_sha256":text_sha256,
            "cache_definitions":frozen,"cache_definition_source":"this cold dump" if cache_definitions is None else "caller-bound same session/module cold dump",
            "compute_references":compute_idx,"uop_loads":len(sources),"serialized_uop_elements":len(packed),
            "serialized_uop_bytes":4*len(packed),"uop_sram_wraps":wraps,
            "checks":checks,"alu_imm_value_visible_in_dump":False,
            "full_cache_contract_bound_across_calls":full_contract_binding,
            "unprinted_alu_contract_unchanged":full_contract_binding,
            "legacy_definition_limit":None if full_contract_binding else "legacy v1 binds base UOP values only; does not certify unprinted ALU immediate unchanged across calls; re-enroll the same-session cold dump for v2",
            "scope":"UOP values, mode/reset, loop fields and opcode; v2 additionally freezes full args/loops across calls, including unprinted immediate contract; does not directly observe ALU immediate, tensor versions or token order"}
