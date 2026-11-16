#!/usr/bin/env python3
"""Freeze an executable deployment manifest for the C1 shared-memory Top-K board test.

This command is deliberately board-free.  It resolves every frozen
candidate--policy configuration to an exact package, runtime and (where
needed) pool-anchor plan.  A topology-equivalent package may be reused only
when its compiled stages are identical and the difference is limited to the
runtime CPU thread map.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from shared_memory_topk import sha256_file, topology_id_from_candidate


HERE = Path(__file__).resolve().parent
REPORT = HERE / "report_out"
DEFAULT_FREEZE = (
    REPORT
    / "stage_tile_cotuning/compile_context_audit/c1_shared_memory_topk_run5"
)
DEFAULT_P7_PACKAGES = REPORT / "resource_aware_maxplus/v1_p7_top20_board_20260904/packages"
DEFAULT_P4R = REPORT / "resource_aware_maxplus/v1_p4r_candidate_manifest.json"
DEFAULT_P5A_AUDIT = REPORT / "resource_aware_maxplus/v1_p5a_compile_audit.json"
DEFAULT_C3S = REPORT / "stage_tile_cotuning/c3s_buffer_reuse/s1_board_run1"
DEFAULT_BUILD = REPORT / "stage_tile_cotuning/c3s_buffer_reuse/build_s1_run2"
REMOTE_LEGACY = "/media/sd-mmcblk1p2/v1_p7_top20/legacy_profile_rank{:02d}"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def candidate_threads(candidate_id):
    values = []
    for token in candidate_id.split("__"):
        match = re.fullmatch(r"(?:cpu|vta)-\d+-\d+_t(\d+)", token)
        if match is None:
            raise ValueError("invalid candidate token: {}".format(token))
        values.append(int(match.group(1)))
    return values


def package_hashes(package):
    package = Path(package)
    manifest = read_json(package / "manifest.json")
    files = {
        "manifest.json",
        "libtvm_runtime.so",
        "libvta.so",
    }
    for stage in manifest["stages"]:
        files.update((stage["graph"], stage["lib"], stage["params"]))
    return {name: sha256_file(package / name) for name in sorted(files)}


def scheme_signature(scheme_cfg):
    return tuple(
        (stage["device"], tuple(stage["unit_names"])) for stage in scheme_cfg
    )


def manifest_signature(manifest):
    return scheme_signature(manifest["stages"])


def interleaved_union(contract):
    """Round-robin the frozen strategies and retain each configuration once."""
    names = list(contract["strategies"])
    width = max(len(contract["strategies"][name]) for name in names)
    result = []
    for index in range(width):
        for name in names:
            order = contract["strategies"][name]
            if index < len(order) and order[index] not in result:
                result.append(order[index])
    if set(result) != set(contract["union_configuration_ids"]):
        raise ValueError("interleaving did not preserve the frozen configuration union")
    return result


def rotated_orders(base):
    """Freeze three balanced, distinct within-boot orders before any labels."""
    if len(base) < 3:
        raise ValueError("at least three configurations are required")
    shift = max(1, len(base) // 3)
    second = base[shift:] + base[:shift]
    reversed_base = list(reversed(base))
    third = reversed_base[shift:] + reversed_base[:shift]
    orders = [base, second, third]
    if len({tuple(order) for order in orders}) != 3:
        raise ValueError("boot orders are not distinct")
    return orders


def resolve_sources(contract, p7_packages, p4r, p5a_audit):
    exact = {}
    for manifest_path in Path(p7_packages).glob("*/package/manifest.json"):
        package = manifest_path.parent
        manifest = read_json(manifest_path)
        exact[manifest["candidate_id"]] = package

    p4r_by_topology = {}
    for row in p4r["rows"]:
        p4r_by_topology.setdefault(row["topology_id"], []).append(row)
    audit_by_candidate = {row["candidate_id"]: row for row in p5a_audit["rows"]}

    rows = {}
    for candidate_id in contract["union_candidate_ids"]:
        topology_id = topology_id_from_candidate(candidate_id)
        desired_threads = candidate_threads(candidate_id)
        package = exact.get(candidate_id)
        source_row = None
        source_kind = "exact_candidate_package"
        if package is None:
            alternatives = p4r_by_topology.get(topology_id, [])
            if not alternatives:
                raise ValueError("no frozen package topology for {}".format(candidate_id))
            audited = [
                row for row in alternatives
                if audit_by_candidate.get(row["candidate_id"], {}).get("audit_passed")
                and Path(audit_by_candidate[row["candidate_id"]]["package_dir"]).is_dir()
            ]
            if not audited:
                raise ValueError("no audited topology-equivalent package for {}".format(candidate_id))
            source_row = sorted(audited, key=lambda row: int(row["frozen_execution_order"]))[0]
            package = Path(audit_by_candidate[source_row["candidate_id"]]["package_dir"])
            source_kind = "topology_exact_runtime_thread_alias"
        manifest = read_json(package / "manifest.json")
        if source_row is None:
            matches = [
                row for row in p4r_by_topology.get(topology_id, [])
                if scheme_signature(row["scheme_cfg"]) == manifest_signature(manifest)
            ]
            source_row = matches[0] if matches else None
        if source_row is not None and scheme_signature(source_row["scheme_cfg"]) != manifest_signature(manifest):
            raise ValueError("package stage signature mismatch for {}".format(candidate_id))
        if len(desired_threads) != len(manifest["stages"]):
            raise ValueError("runtime thread count does not match stage count")
        if any(
            stage["device"] == "vta" and desired_threads[index] != 1
            for index, stage in enumerate(manifest["stages"])
        ):
            raise ValueError("VTA runtime thread identity changed")
        exact_rank = None
        match = re.match(r"(\d+)_", package.parent.name)
        if source_kind == "exact_candidate_package" and match:
            exact_rank = int(match.group(1))
        remote_package = (
            REMOTE_LEGACY.format(exact_rank)
            if exact_rank is not None
            else "/media/sd-mmcblk1p2/c1_shared_topk_{}".format(
                sha256_file(package / "manifest.json")[:12]
            )
        )
        rows[candidate_id] = {
            "candidate_id": candidate_id,
            "topology_id": topology_id,
            "desired_stage_runtime_threads": desired_threads,
            "source_kind": source_kind,
            "source_candidate_id": manifest["candidate_id"],
            "source_package": str(package.resolve()),
            "source_manifest_sha256": sha256_file(package / "manifest.json"),
            "compiled_stage_signature": [
                {"device": stage["device"], "unit_names": stage["unit_names"]}
                for stage in manifest["stages"]
            ],
            "package_sha256": package_hashes(package),
            "remote_package": remote_package,
            "requires_upload": source_kind != "exact_candidate_package",
            "reuse_proof": (
                "exact candidate identity"
                if source_kind == "exact_candidate_package"
                else "identical ordered stage DAG/artifacts; only CPU runtime thread map changes"
            ),
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--p7-packages", type=Path, default=DEFAULT_P7_PACKAGES)
    parser.add_argument("--p4r", type=Path, default=DEFAULT_P4R)
    parser.add_argument("--p5a-audit", type=Path, default=DEFAULT_P5A_AUDIT)
    parser.add_argument("--c3s", type=Path, default=DEFAULT_C3S)
    parser.add_argument("--diagnostic-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))

    contract_path = args.freeze_dir / "selection_contract.json"
    contract = read_json(contract_path)
    if contract.get("status") != "pre_board_frozen":
        raise ValueError("selection contract is not an active pre-board freeze")
    if not contract.get("label_free") or contract.get("board_contacted"):
        raise ValueError("selection contract is not label-free")
    packages = resolve_sources(
        contract, args.p7_packages, read_json(args.p4r), read_json(args.p5a_audit)
    )

    build_manifest_path = args.diagnostic_build / "build_manifest.json"
    build = read_json(build_manifest_path)
    for name, digest in build["artifacts"].items():
        if sha256_file(args.diagnostic_build / name) != digest:
            raise ValueError("diagnostic build hash mismatch: {}".format(name))

    rank_labels = {1: "A", 5: "B", 7: "C", 13: "D"}
    shared_plans = {}
    for rank, label in rank_labels.items():
        plan_path = args.c3s / (label + "_plan.json")
        plan = read_json(plan_path)
        candidate_id = plan["candidate_id"]
        if candidate_id not in packages:
            continue
        if plan["driver_sha256"] != build["artifacts"]["libvta.so"]:
            raise ValueError("plan/diagnostic driver mismatch")
        if plan["package_sha256"] != packages[candidate_id]["package_sha256"]:
            raise ValueError("plan/package hash mismatch for {}".format(candidate_id))
        shared_plans[candidate_id] = {
            "policy": "pool_anchor_k2_shared",
            "path": str(plan_path.resolve()),
            "sha256": sha256_file(plan_path),
            "predicted_saved_bytes": int(plan["predicted_saved_bytes"]),
            "additional_slot_bytes": int(
                sum(edge["external_k2_bytes"] for edge in plan["edges"])
                - plan["predicted_saved_bytes"]
            ),
        }

    configurations = []
    for configuration_id in contract["union_configuration_ids"]:
        candidate_id, policy = configuration_id.rsplit("::", 1)
        if policy == "pool_anchor_k2_shared" and candidate_id not in shared_plans:
            raise ValueError("missing exact pool-anchor plan for {}".format(candidate_id))
        configurations.append({
            "configuration_id": configuration_id,
            "candidate_id": candidate_id,
            "policy": policy,
            "additional_slot_bytes": (
                shared_plans[candidate_id]["additional_slot_bytes"]
                if policy == "pool_anchor_k2_shared" else 0
            ),
            "shared_plan_sha256": (
                shared_plans[candidate_id]["sha256"]
                if policy == "pool_anchor_k2_shared" else None
            ),
        })

    input_names = [
        line.strip() for line in (args.c3s / "inputs.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:3]
    if len(input_names) != 3:
        raise ValueError("the correctness protocol requires three frozen inputs")
    correctness_inputs = []
    for name in input_names:
        path = args.c3s / name
        if not path.is_file():
            raise ValueError("missing correctness input: {}".format(path))
        correctness_inputs.append({
            "path": str(path.resolve()), "name": name, "sha256": sha256_file(path)
        })

    base_order = interleaved_union(contract)
    payload = {
        "schema": "vta_shared_memory_topk_deployment_manifest_v1",
        "status": "board_ready_frozen_before_new_performance_labels",
        "selection_contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256_file(contract_path),
        },
        "correctness_protocol": {
            "input_count": 3,
            "input_source": str((args.c3s / "inputs.txt").resolve()),
            "inputs": correctness_inputs,
            "gate": "ordinary serial full bytes equal each configuration pipeline full bytes before timing",
        },
        "performance_protocol": {
            "runs": 302,
            "warmup_runs": 2,
            "discard_first_scored_rows": 2,
            "estimator": "completion window II over remaining 300 rows",
            "experimental_unit": "independent board boot",
        },
        "boot_configuration_orders": rotated_orders(base_order),
        "packages": packages,
        "shared_plans": shared_plans,
        "configurations": configurations,
        "diagnostic_build": {
            "path": str(args.diagnostic_build.resolve()),
            "manifest_sha256": sha256_file(build_manifest_path),
            "artifacts": build["artifacts"],
        },
        "board_runner": {
            "path": str((HERE / "run_vta_shared_memory_topk_board_boot.py").resolve()),
            "sha256": sha256_file(HERE / "run_vta_shared_memory_topk_board_boot.py"),
        },
        "deployment_preparer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(__file__),
        },
        "board_requirements": {
            "fpga_state": "operating",
            "fpga_firmware": "vta_hpc.bit",
            "udmabuf_bytes": 201326592,
            "cpu_frequency_khz": 1066666,
            "bitstream_sha256": "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6",
            "no_rpc_or_native_runner_process": True,
        },
    }
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "deployment_manifest.json"
    write_json(manifest_path, payload)
    write_json(args.output_dir / "artifact_hashes.json", {
        "deployment_manifest.json": sha256_file(manifest_path),
        "selection_contract.json": sha256_file(contract_path),
    })
    print(json.dumps({
        "configurations": len(configurations),
        "packages": len(packages),
        "thread_alias_packages": sum(row["requires_upload"] for row in packages.values()),
        "shared_plans": len(shared_plans),
        "deployment_manifest_sha256": sha256_file(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
