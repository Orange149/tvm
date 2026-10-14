"""Pure, fail-closed protocol for the two fixed YOLO-224 additions.

No TVM import, board access, candidate search, or historical artifact mutation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median

SEEDS = (0, 20250901, 20260910)
GRAPH_SEEDS = (0, 20250901, 20260910, 0, 20250901,
               20260910, 0, 20250901, 20260910, 0)
ORDERS = {1: (("G0", "G1", "G2"), ("G2", "G1", "G0")),
          2: (("G1", "G2", "G0"), ("G0", "G2", "G1"))}
TARGETS = {
    19: {
        "task_id": "4e1a1134c7c42f5ac8673cf3c8b8a5a7d3fd63d75ab21e024093b6440c86944c",
        "input_program": "675b906865f59955be6c7680c095380d976f255dfb0672e36a6b2006bfe9aff6",
        "resident_program": "af47950cea9374b000366bf16b107c8f007ce12f9b9bf02f8d946dd87f0a2fc6",
        "tile": (1, 2, 14, 1, 1, 1, 1),
    },
    42: {
        "task_id": "6c5e86f7f880a657cb6ba45e6ea80fec090349ad4b68ef1c01dd49ca9a3ffb02",
        "input_program": "a8da4787f345bebeb4ec5cd53bb78c1f4160eebc2f108488a1cab07f4ac2d58f",
        "resident_program": "1d83a189e1dcdcc5d7272a6c12e9f72cd49e157c3938f906fa97f3fe154629ba",
        "tile": (1, 7, 1, 2, 1, 1, 1),
    },
}
TILE_KEYS = ("tile_b", "tile_h", "tile_w", "tile_ci", "tile_co",
             "oc_nthread", "h_nthread")
SCHEMA = "c3_yolo224_incremental_v1"


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def seal(root):
    root = Path(root)
    write_new(root / "artifact_hashes.json", {"artifacts": {
        str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "artifact_hashes.json"}})


def verify(root):
    root = Path(root).resolve()
    manifest = read(root / "artifact_hashes.json")
    if not manifest.get("artifacts"):
        raise ValueError("empty artifact ledger")
    def entries(items, prefix=""):
        for key, value in items.items():
            name = str(Path(prefix) / key)
            if isinstance(value, dict):
                yield from entries(value, name)
            else:
                yield name, value

    for name, expected in entries(manifest["artifacts"]):
        p = (root / name).resolve()
        if root not in p.parents or not p.is_file() or sha(p) != expected:
            raise ValueError("artifact drift: " + name)
    return sha(root / "artifact_hashes.json")


def bind_candidates(rows):
    by_id = {r["program_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate program identity")
    selected = {}
    for node, target in TARGETS.items():
        pair = []
        for key, mode in (("input_program", 1), ("resident_program", 11)):
            row = by_id[target[key]]
            tile = tuple(int(row["visible_features"][k]) for k in TILE_KEYS)
            if (row["execution_task_id"] != target["task_id"] or
                    int(row["implementation_mode"]) != mode or tile != target["tile"] or
                    row.get("local_qualification_passed") is not True):
                raise ValueError("fixed candidate identity/configuration drift")
            sites = row["graph_call_sites"]
            if len(sites) != 1 or int(sites[0]["node_id"]) != node:
                raise ValueError("target call-site drift")
            pair.append(row)
        if pair[0]["complete_config_entity"] != pair[1]["complete_config_entity"]:
            raise ValueError("Input/mode11 are not the same complete Tile entity")
        selected[str(node)] = pair
    return selected


def tensor_contract(graph, node, output=0):
    start, end = graph["node_row_ptr"][node:node + 2]
    if output < 0 or start + output >= end:
        raise ValueError("invalid tensor entry")
    entry = start + output
    attrs = graph["attrs"]
    shape = attrs["shape"][1][entry]
    strides, stride = [], 1
    for extent in reversed(shape):
        strides.insert(0, stride)
        stride *= extent
    return {"node": node, "output": output, "entry_id": entry, "shape": shape,
            "dtype": attrs["dltype"][1][entry], "device": attrs["device_index"][1][entry],
            "storage_id": attrs["storage_id"][1][entry], "byte_offset": 0,
            "strides_elements": strides}


def graph_inventory(graph):
    result = []
    for node, value in enumerate(graph["nodes"]):
        if value["op"] != "tvm_op":
            continue
        symbol = value["attrs"]["func_name"]
        result.append({"node": node, "symbol": symbol,
                       "wrapped": symbol not in ("__copy", "__nop"),
                       "inputs": [tensor_contract(graph, x[0], x[1]) for x in value["inputs"]],
                       "outputs": [tensor_contract(graph, node, i) for i in
                                   range(int(value["attrs"]["num_outputs"]))],
                       "completion": "host_call_only_until_compiled_completion_verified"})
    return result


def fusion_semantics(relay_texts, abi):
    """Recognize the exact observed int32 conv/add/shift/clip/cast chain.

    This deliberately rejects unfamiliar fusions instead of inferring semantics
    from the exported symbol. The full actual TIR remains the address evidence.
    """
    def tensor_text(item):
        return "Tensor[(" + ", ".join(map(str, item["shape"])) + "), " + item["dtype"] + "]"
    required = [tensor_text(x) for x in abi["inputs"] + abi["outputs"]]
    matches = []
    for text in relay_texts:
        blocks = re.findall(r"fn \([^}]+?Primitive=1\)[^{]+\{.*?\n  \} /\* ty=fn", text, re.S)
        matches += [b for b in blocks if all(t in b for t in required) and "nn.conv2d(" in b]
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise ValueError("FULL_FUSION_CONTRACT_UNRESOLVED: ambiguous Relay function")
    block = matches[0]
    header, body = block.split("{", 1)
    params = re.findall(r"(%[\w]+): Tensor", header)
    if len(params) != 3:
        raise ValueError("full fusion requires three inputs")
    conv = re.search(r"(%\w+) = nn\.conv2d\((%\w+), (%\w+), ([^\n]+)", body)
    if not conv or (conv[2], conv[3]) != tuple(params[:2]) or 'out_dtype="int32"' not in conv[4]:
        raise ValueError("unresolved int32 convolution")
    add = re.search(r"(%\w+) = add\(" + re.escape(conv[1]) + r", " + re.escape(params[2]) + r"\) /\* ty=Tensor\[[^\n]+, int32\]", body)
    if not add:
        raise ValueError("unresolved int32 addition before shift")
    shift = re.search(r"(%\w+) = right_shift\(" + re.escape(add[1]) + r", (\d+) /\* ty=int32 \*/\)", body)
    if not shift:
        raise ValueError("unresolved arithmetic shift")
    clip = re.search(r"(%\w+) = clip\(" + re.escape(shift[1]) + r", a_min=(-?[\d.]+)f, a_max=(-?[\d.]+)f\)", body)
    if not clip or not re.search(r"cast\(" + re.escape(clip[1]) + r', dtype="int8"\)', body):
        raise ValueError("unresolved clip/cast order")
    yshape, bshape = abi["outputs"][0]["shape"], abi["inputs"][2]["shape"]
    if len(bshape) > len(yshape):
        raise ValueError("invalid broadcast rank")
    offset = len(yshape) - len(bshape)
    indices = []
    for i, extent in enumerate(bshape):
        if extent not in (1, yshape[i + offset]):
            raise ValueError("invalid bias broadcast")
        indices.append("0" if extent == 1 else "output_axis_%d" % (i + offset))
    layouts = dict(re.findall(r'(data_layout|kernel_layout)="([^"]+)"', conv[4]))
    if set(layouts) != {"data_layout", "kernel_layout"}:
        raise ValueError("missing actual layouts")
    return {"status": "FULL_FUSION_CONTRACT_RESOLVED", "abi_order": ["X", "W", "ADD", "Y"],
            "accumulator_dtype": "int32", "add_dtype": "int32", "shift": int(shift[2]),
            "negative_shift_semantics": "signed arithmetic right shift",
            "clip": [float(clip[2]), float(clip[3])], "cast": "int8",
            "operation_order": ["conv", "add", "right_shift", "clip", "cast"],
            "add_broadcast_indices": indices, "layouts": layouts,
            "actual_relay_block": block, "relay_block_sha256": digest(block)}


def prefix_graph(graph, last_node, heads):
    """Stop a normal executor after a node; preserve allocation and prefix.

    Heads must be read from separate runs if their lifetimes overlap illegally.
    Used only for unmeasured snapshots, never for performance.
    """
    result = copy.deepcopy(graph)
    for node in result["nodes"][last_node + 1:]:
        if node["op"] == "tvm_op":
            node["attrs"]["func_name"] = "__nop"
    result["heads"] = heads
    if result["attrs"] != graph["attrs"] or result["node_row_ptr"] != graph["node_row_ptr"]:
        raise ValueError("prefix changed storage plan")
    return result


def protocol():
    return {"schema": SCHEMA, "input_shape": [1, 3, 224, 224],
            "targets": TARGETS, "seeds": SEEDS, "graph_seeds": GRAPH_SEEDS,
            "orders": ORDERS, "profile_order": [False, True, True, False],
            "profile_calls_per_block": 5, "profile_warmups_per_block": 1,
            "profile_perturbation_limit": .02, "graph_practical_gain_pct": 1.0,
            "graph_counts_per_boot": {"timing": 30, "warmup": 6, "qualification": 12},
            "board_paused": True, "completion_mode": "run_only",
            "clean_bitstream_per_rpc": True, "cpu": 3,
            "reselection": False, "core_certificate_is_fused_certificate": False,
            "fps_kind": "1000 / median graph milliseconds; NOT image pipeline throughput"}


def evaluation(rows, boot_index):
    expected = [(v, b, i, s) for b, order in zip(("A", "B"), ORDERS[boot_index])
                for v in order for i, s in enumerate(GRAPH_SEEDS[0:5] if b == "A" else GRAPH_SEEDS[5:])]
    timed = [r for r in rows if r["phase"] == "timing"]
    for key in ("boot_id", "contract_sha256", "build_ledger", "attempt_id"):
        if len({r.get(key) for r in timed}) != 1 or not timed or not timed[0].get(key):
            raise ValueError("missing or mixed sample identity: " + key)
    observed = [(r["variant"], r["block"], r["index"], r["seed"]) for r in timed]
    if observed != expected:
        raise ValueError("timing coverage/order/seed drift or spliced rows")
    for row in timed:
        if row.get("correct") is not True or row.get("diagnostic") is not False:
            raise ValueError("incorrect or diagnostic sample in performance")
        value = row.get("latency_ms")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("missing/non-positive/non-finite latency")
    stats = {}
    for variant in ("G0", "G1", "G2"):
        values = [r["latency_ms"] for r in timed if r["variant"] == variant]
        m = median(values)
        stats[variant] = {"median_ms": m, "equivalent_fps": 1000 / m,
                          "block_median_ms": {b: median([r["latency_ms"] for r in timed
                              if r["variant"] == variant and r["block"] == b]) for b in ("A", "B")}}
    gains = {v: 100 * (1 - stats["G2"]["median_ms"] / stats[v]["median_ms"]) for v in ("G0", "G1")}
    directions = {v: all(stats["G2"]["block_median_ms"][b] < stats[v]["block_median_ms"][b]
                         for b in ("A", "B")) for v in ("G0", "G1")}
    return {"statistics": stats, "gains_pct": gains, "block_directions": directions,
            "request_second_boot": gains["G0"] >= 1 and all(directions.values())}


def cross_boot(first, second):
    if (first["boot_id"] == second["boot_id"] or first["contract_sha256"] != second["contract_sha256"]
            or first["build_ledger"] != second["build_ledger"]):
        raise ValueError("cross-boot identity mismatch")
    if not first["evaluation"]["request_second_boot"]:
        raise ValueError("second boot was not eligible")
    for key in ("runtime_contract_sha256", "helper_sha256"):
        if not first.get(key) or first[key] != second.get(key):
            raise ValueError("cross-boot runtime/helper drift")
    for result in (first, second):
        if result.get("status") != "GRAPH_COMPLETE" or not result.get("all_correct"):
            raise ValueError("invalid graph session")
    both = [r["evaluation"] for r in (first, second)]
    overall = all(r["gains_pct"]["G0"] >= 1 and r["block_directions"]["G0"] for r in both)
    increment = all(r["gains_pct"]["G1"] > 0 and r["block_directions"]["G1"] for r in both)
    return {"practical_gain_over_current": overall, "positive_residency_increment": increment,
            "practical_residency_increment": increment and all(r["gains_pct"]["G1"] >= 1 for r in both),
            "statistical_significance_claim": False}
