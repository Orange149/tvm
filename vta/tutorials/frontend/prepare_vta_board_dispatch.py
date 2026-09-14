"""Prepare an immutable, offline VTA board-dispatch contract.

This tool validates the frozen P5b shortlist against its P4b static inputs and
writes a dispatch manifest.  It deliberately contains no RPC or board action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from plan_vta_predispatch_shortlist import (
    FEATURE_VERSION,
    PLANNER_VERSION,
    STRATEGIES,
    build_shortlists,
    extract_prefeature,
    file_sha256,
)


SCHEMA = "c3_offline_board_dispatch_contract_v1"
CONTRACT_VERSION = "c3_p6a_dispatch_contract_v1"
DEFAULT_CORRECTNESS_SEEDS = (0, 20250901, 20260910)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _expect_hash(path, expected, label):
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError("{} SHA-256 mismatch: expected={} actual={}".format(label, expected, actual))
    return actual


def _validate_complete_config(entity, config_index, candidate_id):
    if not isinstance(entity, dict) or set(entity) != {"index", "code_hash", "entity"}:
        raise ValueError("incomplete ConfigEntity for {}".format(candidate_id))
    if entity["index"] != config_index or not isinstance(entity["entity"], list) or not entity["entity"]:
        raise ValueError("invalid ConfigEntity index/body for {}".format(candidate_id))
    names = []
    for knob in entity["entity"]:
        if not isinstance(knob, list) or len(knob) != 3 or not isinstance(knob[0], str):
            raise ValueError("invalid ConfigEntity knob for {}".format(candidate_id))
        names.append(knob[0])
    if len(names) != len(set(names)):
        raise ValueError("duplicate ConfigEntity knob for {}".format(candidate_id))


def _pool_identity_index(pool):
    identities = {}
    workloads = {}

    def add(candidate_id, workload_id, role, mode, config_index, config_entity):
        if candidate_id in identities:
            raise ValueError("duplicate candidate_id in candidate pool: {}".format(candidate_id))
        _validate_complete_config(config_entity, config_index, candidate_id)
        identities[candidate_id] = {
            "workload_id": workload_id,
            "candidate_role": role,
            "residence_mode": mode,
            "config_index": config_index,
            "complete_config_entity": config_entity,
        }

    for workload in pool["workloads"]:
        workload_id = workload["workload_id"]
        if workload_id in workloads:
            raise ValueError("duplicate workload_id in candidate pool: {}".format(workload_id))
        workloads[workload_id] = workload
        incumbent = workload["protected_original_incumbent"]
        add(
            incumbent["candidate_id"],
            workload_id,
            "protected_original_incumbent",
            "original",
            incumbent["config_index"],
            incumbent["complete_config_entity"],
        )
        for mapping in workload["mapping"]["records"]:
            for mode, candidate_id in mapping["candidate_ids"].items():
                role = "same_tile_original_control" if mode == "original" else "residency_experiment"
                add(
                    candidate_id,
                    workload_id,
                    role,
                    mode,
                    mapping["mapped_config_index"],
                    mapping["complete_config_entity"],
                )
    return identities, workloads


def validate_frozen_inputs(
    shortlist_path,
    candidate_pool_path,
    results_path,
    qualification_results_path,
    qualification_summary_path,
):
    """Validate source hashes and reproduce the complete feature/shortlist payload."""
    shortlist_path = Path(shortlist_path)
    candidate_pool_path = Path(candidate_pool_path)
    results_path = Path(results_path)
    qualification_results_path = Path(qualification_results_path)
    qualification_summary_path = Path(qualification_summary_path)
    p5b_dir = shortlist_path.parent
    input_manifest_path = p5b_dir / "input_manifest.json"
    artifact_hashes_path = p5b_dir / "artifact_hashes.json"
    if not input_manifest_path.is_file() or not artifact_hashes_path.is_file():
        raise ValueError("shortlist must be accompanied by frozen input_manifest.json and artifact_hashes.json")

    frozen_inputs = _load_json(input_manifest_path)
    artifact_hashes = _load_json(artifact_hashes_path)
    files = artifact_hashes.get("files", {})
    _expect_hash(shortlist_path, files.get("shortlist.json"), "shortlist")
    _expect_hash(input_manifest_path, files.get("input_manifest.json"), "P5b input manifest")
    pool_hash = _expect_hash(
        candidate_pool_path, frozen_inputs["candidate_pool"]["sha256"], "candidate pool"
    )
    results_hash = _expect_hash(results_path, frozen_inputs["results"]["sha256"], "P4b results")
    if files.get("p4b/candidate_pool.json") != pool_hash or files.get("p4b/results.jsonl") != results_hash:
        raise ValueError("P5b artifact hash ledger disagrees with the P4b input hashes")

    planner_path = Path(__file__).with_name("plan_vta_predispatch_shortlist.py")
    _expect_hash(planner_path, frozen_inputs["planner_source_sha256"], "P5b planner source")
    shortlist = _load_json(shortlist_path)
    pool = _load_json(candidate_pool_path)
    records = _load_jsonl(results_path)
    rebuilt, features, failures, screening = build_shortlists(pool, records, shortlist["budgets"])
    if rebuilt != shortlist:
        raise ValueError("shortlist semantic reproduction mismatch")
    if shortlist["feature_version"] != FEATURE_VERSION:
        raise ValueError("unsupported feature version")
    if shortlist["feature_set_sha256"] != frozen_inputs["feature_set_sha256"]:
        raise ValueError("feature-set SHA-256 mismatch")
    if shortlist["planner_version"] != PLANNER_VERSION:
        raise ValueError("unsupported shortlist planner version")

    qualification_hashes_path = qualification_results_path.parent / "artifact_hashes.json"
    if qualification_summary_path.parent != qualification_results_path.parent or not qualification_hashes_path.is_file():
        raise ValueError("P4e results and summary must share their frozen artifact_hashes.json directory")
    qualification_hashes = _load_json(qualification_hashes_path)
    qualification_results_hash = _expect_hash(
        qualification_results_path,
        qualification_hashes.get("results.jsonl"),
        "P4e qualification results",
    )
    qualification_summary_hash = _expect_hash(
        qualification_summary_path,
        qualification_hashes.get("summary.json"),
        "P4e qualification summary",
    )
    qualification_records = _load_jsonl(qualification_results_path)
    qualification_summary = _load_json(qualification_summary_path)
    qualification_by_id = {record["candidate_id"]: record for record in qualification_records}
    result_by_id = {record["candidate_id"]: record for record in records}
    if len(qualification_by_id) != len(qualification_records) or set(qualification_by_id) != set(result_by_id):
        raise ValueError("P4e/P4b candidate identity sets do not match exactly")
    fsim_passed = 0
    lower_failed = 0
    for candidate_id, record in result_by_id.items():
        qualification = qualification_by_id[candidate_id]
        for field in ("workload_id", "candidate_role", "residence_mode"):
            if qualification[field] != record[field]:
                raise ValueError("P4e candidate identity mismatch for {} {}".format(candidate_id, field))
        if record.get("status") == "ok" and record.get("failure") is None:
            if (
                qualification.get("local_status") != "fsim_passed"
                or qualification.get("lower_status") != "ok"
                or tuple(qualification.get("correctness_seeds", ())) != DEFAULT_CORRECTNESS_SEEDS
                or qualification.get("tir_sha256") != record.get("tir_sha256")
            ):
                raise ValueError("P4e FSim qualification mismatch for {}".format(candidate_id))
            fsim_passed += 1
        else:
            if qualification.get("local_status") != "lower_failed" or qualification.get("lower_status") != "failed":
                raise ValueError("P4e lower-failure qualification mismatch for {}".format(candidate_id))
            lower_failed += 1
    if (
        qualification_summary.get("candidate_records") != len(records)
        or qualification_summary.get("unique_candidate_ids") != len(records)
        or qualification_summary.get("fsim_passed_all_three_seeds") != fsim_passed
        or qualification_summary.get("lower_failed") != lower_failed
        or qualification_summary.get("unqualified_lower_success") != 0
        or qualification_summary.get("board_access") is not False
        or qualification_summary.get("performance_measurement") != "not_collected"
    ):
        raise ValueError("P4e qualification summary disagrees with joined records")
    return {
        "shortlist": shortlist,
        "candidate_pool": pool,
        "records": records,
        "features": features,
        "failures": failures,
        "screening": screening,
        "qualification_by_id": qualification_by_id,
        "qualification_summary": qualification_summary,
        "provenance": {
            "shortlist": {"path": str(shortlist_path), "sha256": file_sha256(shortlist_path)},
            "candidate_pool": {"path": str(candidate_pool_path), "sha256": pool_hash},
            "results": {"path": str(results_path), "sha256": results_hash},
            "p5b_input_manifest": {
                "path": str(input_manifest_path),
                "sha256": file_sha256(input_manifest_path),
            },
            "p5b_artifact_hashes": {
                "path": str(artifact_hashes_path),
                "sha256": file_sha256(artifact_hashes_path),
            },
            "p4e_qualification_results": {
                "path": str(qualification_results_path),
                "sha256": qualification_results_hash,
            },
            "p4e_qualification_summary": {
                "path": str(qualification_summary_path),
                "sha256": qualification_summary_hash,
            },
            "p4e_artifact_hashes": {
                "path": str(qualification_hashes_path),
                "sha256": file_sha256(qualification_hashes_path),
            },
        },
    }


def build_dispatch_manifest(validated, workload_id, strategy, budget, correctness_seeds):
    shortlist = validated["shortlist"]
    pool = validated["candidate_pool"]
    records = validated["records"]
    qualification_by_id = validated["qualification_by_id"]
    if strategy not in STRATEGIES:
        raise ValueError("unsupported strategy: {}".format(strategy))
    if workload_id not in shortlist["workloads"]:
        raise ValueError("unknown workload: {}".format(workload_id))
    budget_key = str(int(budget))
    try:
        selection = shortlist["workloads"][workload_id]["strategies"][strategy]["budgets"][budget_key]
    except KeyError as error:
        raise ValueError("budget {} is not frozen for {} {}".format(budget, workload_id, strategy)) from error

    seeds = tuple(int(seed) for seed in correctness_seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three unique correctness seeds are required")
    candidate_ids = selection["candidate_ids"]
    if len(candidate_ids) != len(set(candidate_ids)) or len(candidate_ids) != selection["effective_budget"]:
        raise ValueError("dispatch prefix candidate IDs are not unique/complete")
    incumbent = shortlist["workloads"][workload_id]["protected_original_incumbent"]
    if not candidate_ids or candidate_ids[0] != incumbent:
        raise ValueError("protected original incumbent must be first")

    pool_identities, workloads = _pool_identity_index(pool)
    result_by_id = {record["candidate_id"]: record for record in records}
    if len(result_by_id) != len(records):
        raise ValueError("duplicate candidate_id in P4b results")
    shortlist_details = selection["candidates"]
    if [item["candidate_id"] for item in shortlist_details] != candidate_ids:
        raise ValueError("shortlist candidate detail order disagrees with candidate_ids")

    ordered = []
    for dispatch_order, detail in enumerate(shortlist_details):
        candidate_id = detail["candidate_id"]
        if candidate_id not in pool_identities or candidate_id not in result_by_id:
            raise ValueError("candidate identity missing: {}".format(candidate_id))
        identity = pool_identities[candidate_id]
        record = result_by_id[candidate_id]
        qualification = qualification_by_id[candidate_id]
        if identity["workload_id"] != workload_id or record["workload_id"] != workload_id:
            raise ValueError("candidate/workload identity mismatch: {}".format(candidate_id))
        if record.get("status") != "ok" or record.get("failure") is not None:
            raise ValueError("non-lower-successful candidate in dispatch prefix: {}".format(candidate_id))
        if (
            qualification.get("local_status") != "fsim_passed"
            or tuple(qualification.get("correctness_seeds", ())) != DEFAULT_CORRECTNESS_SEEDS
        ):
            raise ValueError("candidate lacks required three-seed FSim qualification: {}".format(candidate_id))
        for field in ("candidate_role", "residence_mode"):
            if detail[field] != identity[field] or detail[field] != record[field]:
                raise ValueError("candidate {} mismatch for {}".format(candidate_id, field))
        if detail["config_index"] != identity["config_index"] or record["debug"]["config_index"] != identity["config_index"]:
            raise ValueError("candidate config index mismatch: {}".format(candidate_id))
        record_config = record["identity"]["complete_config_entity"]
        pool_config_without_index = {
            "code_hash": identity["complete_config_entity"]["code_hash"],
            "entity": identity["complete_config_entity"]["entity"],
        }
        if record_config != pool_config_without_index:
            raise ValueError("candidate ConfigEntity identity mismatch: {}".format(candidate_id))
        feature = extract_prefeature(record)
        if detail["feature_version"] != FEATURE_VERSION or detail["feature_sha256"] != feature["feature_sha256"]:
            raise ValueError("candidate feature hash mismatch: {}".format(candidate_id))
        if record["identity"]["workload"] != workloads[workload_id]["workload"]:
            raise ValueError("candidate workload payload mismatch: {}".format(candidate_id))
        ordered.append(
            {
                "dispatch_order": dispatch_order,
                "candidate_id": candidate_id,
                "candidate_role": identity["candidate_role"],
                "workload_id": workload_id,
                "residence_mode": identity["residence_mode"],
                "template_name": record["identity"]["template_name"],
                "schedule_version": record["identity"]["schedule_version"],
                "config_index": identity["config_index"],
                "complete_config_entity": identity["complete_config_entity"],
                "feature_version": detail["feature_version"],
                "feature_sha256": detail["feature_sha256"],
                "static_tir_sha256": record["tir_sha256"],
                "local_qualification": {
                    "status": qualification["local_status"],
                    "correctness_seeds": qualification["correctness_seeds"],
                    "evidence_source": qualification["evidence_source"],
                },
            }
        )

    expected_board = pool["hardware_fingerprint"]
    return {
        "schema": SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "board_executed": False,
        "selection": {
            "workload_id": workload_id,
            "workload": workloads[workload_id]["workload"],
            "strategy": strategy,
            "requested_budget": int(budget),
            "effective_budget": len(ordered),
            "protected_original_incumbent": incumbent,
            "ordered_candidate_ids": candidate_ids,
        },
        "candidates": ordered,
        "correctness_protocol": {
            "status": "pending_board_execution",
            "seeds": list(seeds),
            "required_passes_per_candidate": len(seeds),
            "reference": "NumPy reference output",
        },
        "timing_protocol": {
            "status": "must_be_filled_and_frozen_before_board_execution",
            "warmup_runs": None,
            "timed_runs": None,
            "timer_number": None,
            "summary_statistic": None,
        },
        "board_fingerprint": {
            "status": "must_be_filled_and_matched_before_board_execution",
            "expected": expected_board,
            "actual": {
                "target": None,
                "vta_config_sha256": None,
                "bitstream_sha256": None,
                "boot_id": None,
            },
        },
        "feature_contract": {
            "feature_version": shortlist["feature_version"],
            "feature_set_sha256": shortlist["feature_set_sha256"],
        },
        "source_artifacts": validated["provenance"],
        "claim_boundary": {
            "offline_contract_only": True,
            "latency_measured": False,
            "correctness_executed": False,
            "G6_passed": False,
        },
    }


def prepare_dispatch(
    shortlist_path,
    candidate_pool_path,
    results_path,
    qualification_results_path,
    qualification_summary_path,
    workload_id,
    strategy,
    budget,
    correctness_seeds=DEFAULT_CORRECTNESS_SEEDS,
):
    validated = validate_frozen_inputs(
        shortlist_path,
        candidate_pool_path,
        results_path,
        qualification_results_path,
        qualification_summary_path,
    )
    return build_dispatch_manifest(validated, workload_id, strategy, budget, correctness_seeds)


def write_dispatch_manifest(output_dir, manifest):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dispatch_manifest.json"
    if output_path.exists():
        raise FileExistsError("refusing to overwrite frozen dispatch manifest: {}".format(output_path))
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--qualification-results", required=True)
    parser.add_argument("--qualification-summary", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--correctness-seeds", default="0,20250901,20260910")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.correctness_seeds.split(",") if value)
    manifest = prepare_dispatch(
        args.shortlist,
        args.candidate_pool,
        args.results,
        args.qualification_results,
        args.qualification_summary,
        args.workload,
        args.strategy,
        args.budget,
        seeds,
    )
    output_path = write_dispatch_manifest(args.output_dir, manifest)
    print(
        json.dumps(
            {
                "status": "offline_dispatch_contract_frozen",
                "output": str(output_path),
                "workload": args.workload,
                "strategy": args.strategy,
                "effective_budget": manifest["selection"]["effective_budget"],
                "board_executed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
