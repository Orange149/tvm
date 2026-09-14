#!/usr/bin/env python3
"""Fail-closed C3-S input-pool audit. No model compilation or board mutation.

Static eligibility is conditional on the frozen GraphExecutor implementation;
it is not runtime physical-range qualification or permission to launch reuse.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align(n, size=256):
    return (n + size - 1) // size * size


def tensor_bytes(shape, dtype):
    match = re.fullmatch(r"(?:float|int|uint|bfloat)(\d+)(?:x(\d+))?", dtype)
    if not match or any(not isinstance(x, int) or x <= 0 for x in shape):
        raise ValueError("unsupported dtype/shape")
    bits = int(match[1]) * int(match[2] or 1)
    # GraphExecutor SetupStorage rounds each scalar, not total packed bits.
    return math.prod(shape) * ((bits + 7) // 8)


def graph_entries(graph, default_device):
    attrs = {k: v[1] for k, v in graph["attrs"].items()}
    row = graph["node_row_ptr"]
    count = row[-1]
    if any(len(attrs[k]) != count for k in ("shape", "dltype", "storage_id")):
        raise ValueError("incomplete entry attributes")
    readers = [[] for _ in range(count)]
    for nid, node in enumerate(graph["nodes"]):
        for port, (source, output, version) in enumerate(node["inputs"]):
            if version != 0 or not 0 <= output < row[source + 1] - row[source]:
                raise ValueError("unsupported input entry")
            readers[row[source] + output].append({"node": nid, "port": port,
                                                 "op": node["op"],
                                                 "func": node.get("attrs", {}).get("func_name")})
    heads = {row[n] + i for n, i, _ in graph["heads"]}
    result = []
    for nid, node in enumerate(graph["nodes"]):
        for eid in range(row[nid], row[nid + 1]):
            result.append({"entry": eid, "node": nid, "name": node["name"],
                           "op": node["op"], "storage_id": attrs["storage_id"][eid],
                           "shape": attrs["shape"][eid], "dtype": attrs["dltype"][eid],
                           "device": attrs.get("device_index", [default_device] * count)[eid],
                           "scope": attrs.get("storage_scope", [""] * count)[eid],
                           "bytes": tensor_bytes(attrs["shape"][eid], attrs["dltype"][eid]),
                           "readers": readers[eid],
                           "writers": [] if node["op"] == "null" else [nid],
                           "is_head": eid in heads, "is_arg": nid in graph["arg_nodes"]})
    for entry in result:
        aliases = [e for e in result if e["storage_id"] == entry["storage_id"]]
        entry["alias_entries"] = [e["entry"] for e in aliases]
        entry["pool_capacity_bytes"] = align(max(e["bytes"] for e in aliases), 4)
    return result


def classify(entries, name, parameter_names, linked_status, contract, direction):
    matches = [e for e in entries if e["name"] == name and e["is_arg"]]
    out = {"input_name": name, "status": "unknown", "reasons": []}
    if len(matches) != 1:
        out["reasons"] = ["input_entry_not_unique"]
        return out
    entry = matches[0]
    out.update(entry)
    out["is_parameter"] = None if parameter_names is None else name in parameter_names
    reject, unknown = [], []
    if direction != "cpu_to_vta":
        reject.append("direction_outside_v1")
    if len(entry["alias_entries"]) != 1:
        reject.append("internal_storage_alias_cross_frame_unsafe")
    if parameter_names is None:
        unknown.append("parameter_inventory_unknown")
    elif name in parameter_names:
        reject.append("parameter_input")
    if linked_status != "absent":
        unknown.append("linked_parameter_lookup_not_excluded")
    if entry["device"] != 12 or entry["scope"] not in ("", "global"):
        reject.append("not_plain_ext_dev_pool")
    if entry["storage_id"] < 0 or entry["is_head"] or entry["op"] != "null":
        reject.append("not_exclusive_plain_input")
    if entry["shape"] != contract.get("shape") or entry["dtype"] != contract.get("dtype"):
        reject.append("interface_mismatch")
    if not entry["readers"] or any(r["op"] != "tvm_op" or r["func"] in (None, "__nop")
                                   for r in entry["readers"]):
        unknown.append("zero_copy_use_coverage_unproven")
    out["status"] = "rejected" if reject else "unknown" if unknown else "eligible"
    out["reasons"] = reject + unknown or ["exclusive_nonparameter_input_all_tvm_op_uses_rebound"]
    return out


def finalize_bundles(edges):
    # Identity is (Executor, storage-id), never storage-id alone across Executors.
    claims = {}
    for edge in edges:
        for tensor in edge["tensors"]:
            if "storage_id" in tensor:
                key = (edge["consumer_stage"], tensor["storage_id"])
                claims.setdefault(key, []).append(tensor)
    for tensors in claims.values():
        if len(tensors) > 1:
            for tensor in tensors:
                tensor["status"] = "rejected"
                tensor["reasons"].append("cross_boundary_or_bundle_pool_conflict")
    for edge in edges:
        edge["source"] = "pool-anchor" if edge["tensors"] and all(
            t["status"] == "eligible" for t in edge["tensors"]) else "external"
        edge["external_k2_bytes"] = 2 * sum(align(t["contract_bytes"]) for t in edge["tensors"])
        edge["predicted_saved_bytes"] = (edge["external_k2_bytes"] // 2
                                         if edge["source"] == "pool-anchor" else 0)
    return edges


def verify_hashes(package, hashes):
    package = Path(package).resolve()
    for name, expected in hashes.items():
        path = (package / name).resolve()
        if not path.is_relative_to(package) or sha(path) != expected:
            raise ValueError("package hash mismatch: " + name)


def audit_package(package, expected_checks):
    import tvm  # Only needed to decode parameter names; no target module loaded.
    package = Path(package)
    checks = {line.split()[1]: line.split()[0]
              for line in Path(expected_checks).read_text().splitlines() if line.strip()}
    verify_hashes(package, checks)
    manifest = json.loads((package / "manifest.json").read_text())
    checks["manifest.json"] = sha(package / "manifest.json")
    stages, inventories, linked = manifest["stages"], {}, {}
    parameter_sets = {}
    for stage in stages:
        idx = stage["index"]
        graph = json.loads((package / stage["graph"]).read_text())
        inventories[idx] = graph_entries(graph, 12 if stage["device"] == "vta" else 1)
        parameter_sets[idx] = set(tvm.runtime.load_param_dict((package / stage["params"]).read_bytes()))
        # Full ELF symbol table (not just dynamic symbols). Stripped/unknown => fail closed.
        symbols = subprocess.run(["readelf", "-Ws", str(package / stage["lib"])],
                                 check=True, capture_output=True, text=True).stdout
        linked[idx] = ("present" if "__tvm_lookup_linked_param" in symbols else
                       "absent" if ".symtab" in symbols else "unknown")
        for entry in inventories[idx]:
            entry["is_parameter"] = entry["name"] in parameter_sets[idx]
    edges = []
    for producer, consumer in zip(stages, stages[1:]):
        direction = producer["device"] + "_to_" + consumer["device"]
        contracts = producer["output_schema"]["slots"]
        if len(contracts) != len(consumer["input_names"]):
            raise ValueError("boundary arity mismatch")
        tensors = []
        for i, (name, contract) in enumerate(zip(consumer["input_names"], contracts)):
            tensor = classify(inventories[consumer["index"]], name,
                              parameter_sets[consumer["index"]], linked[consumer["index"]],
                              contract, direction)
            tensor.update(tensor_index=i, contract_bytes=tensor_bytes(contract["shape"], contract["dtype"]))
            tensors.append(tensor)
        edges.append({"producer_stage": producer["index"], "consumer_stage": consumer["index"],
                      "direction": direction, "tensors": tensors})
    finalize_bundles(edges)
    pools = []
    for idx, entries in inventories.items():
        for sid in sorted({e["storage_id"] for e in entries}):
            aliases = [e for e in entries if e["storage_id"] == sid]
            if aliases[0]["device"] == 12:
                pools.append({"stage": idx, "storage_id": sid,
                              "request_bytes": aliases[0]["pool_capacity_bytes"],
                              "aligned_bytes": align(aliases[0]["pool_capacity_bytes"]),
                              "linked_status": linked[idx]})
    return {"schema": "c3s-static-audit-v1", "package": str(package.resolve()),
            "candidate_id": manifest["candidate_id"], "package_sha256": checks,
            "static_only": True, "runtime_qualified": False,
            "assumptions": ["frozen GraphExecutor SetInputZeroCopy/SetupOpExecs semantics",
                            "no hidden out-of-contract kernel writes; compiled package unchanged",
                            "runtime pool membership/alignment/owner retention still required"],
            "entries_by_stage": inventories, "ext_dev_pools": pools, "edges": edges,
            "graph_pool_aligned_bytes": sum(p["aligned_bytes"] for p in pools),
            "external_k2_bytes": sum(e["external_k2_bytes"] for e in edges),
            "predicted_saved_bytes": sum(e["predicted_saved_bytes"] for e in edges)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-checks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_package(args.package, args.expected_checks)
    result["audit_source_sha256"] = sha(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({k: result[k] for k in
                      ("candidate_id", "graph_pool_aligned_bytes", "external_k2_bytes", "predicted_saved_bytes")}))


if __name__ == "__main__":
    main()
