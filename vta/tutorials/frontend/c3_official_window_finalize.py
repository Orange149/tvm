#!/usr/bin/env python3
"""Close local evidence and obey the approved no-difference stop condition.

No target function, RPC or board operation is executed. Historical failed
tool attempts remain separate from fresh complete local qualifications.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from c3_official_window_protocol import canonical_submit_sequence, digest as object_digest, freeze_pool
from c3_official_window_workflow import RUN, REPO, BASELINES, digest, read, write


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def ancestor_values(certificate):
    if certificate.get("ancestor_encoding") == "hexadecimal_integer_bitset":
        return [int(value, 16) for value in certificate["ancestors"]]
    return certificate["ancestors"]


def finalize(output, qualifications):
    import tvm
    import vta  # Register the captured VTA TIR operations before JSON loading.
    from c3_official_window_llvm import llvm_alu_contract
    from c3_official_window_runtime_native import HEADER
    from c3_valid_window_uop_values import _instructions

    output.mkdir(parents=True, exist_ok=False)
    runtime = RUN / "runtime_local_final"
    build = read(runtime / "build.json")
    require(digest(runtime / "libvta_fsim.so") == build["binary_sha256"], "RUNTIME_CHANGED")
    require(build["source_bindings"][str(HEADER.resolve())] == digest(HEADER), "QUEUE_MODEL_BUILD_UNBOUND")
    for name in BASELINES:
        compiler = read(RUN / "baseline" / name / "compiler_identity.json")
        require(digest(compiler["official_log"]) == compiler["official_log_sha256"], "OFFICIAL_LOG_CHANGED")
        for path, sha in compiler["source_sha256"].items():
            actual = RUN / "source_snapshots/baseline_at_capture.py" if Path(path).name == "c3_official_window_baseline.py" else Path(path)
            require(digest(actual) == sha, "UNEXPLAINED_COMPILER_OR_TEMPLATE_SOURCE_CHANGE:" + path)
    codegen_binding = {"codegen_source_path": str(REPO / "src/target/llvm/codegen_cpu.cc"),
        "codegen_source_sha256_at_closure": digest(REPO / "src/target/llvm/codegen_cpu.cc"),
        "actual_libtvm_sha256": digest(REPO / "build/libtvm.so"),
        "compiler_module_identity": "library frozen at baseline capture; source is the explicit declaration-order contract, not a fabricated compiler build provenance"}
    inventory = read(qualifications[0] / "semantic_instance_inventory.json")
    require(len(inventory) == 26, "COMPLETE_FUSED_INSTANCE_COVERAGE_CHANGED")
    mandatory = {}
    for root in qualifications[1:]:
        for entry in read(root / "semantic_instance_inventory.json"):
            mandatory[entry["object_id"]] = root / entry["object_id"]
    audits, capacities, calls_index, strengthened = [], [], [], []
    for row in inventory:
        oid = row["object_id"]
        candidates = [root / oid for root in qualifications
                      if all((root / oid / mode / "summary.json").exists() for mode in ("detailed", "minimal"))]
        require(bool(candidates), "MISSING_COMPLETE_LOCAL_QUALIFICATION:" + oid)
        base = candidates[-1]
        require(oid not in mandatory or base == mandatory[oid], "FRESH_MANDATORY_QUALIFICATION_FAILED:" + oid)
        require(row["official_baseline_bound"] and row["roundtrip_passed"], "BASELINE_IDENTITY_NOT_PASSED")
        require(digest(row["parent_module"]) == row["parent_module_sha256"], "PARENT_MODULE_CHANGED")
        summaries, calls = {}, {}
        for mode in ("detailed", "minimal"):
            directory = base / mode
            summary = read(directory / "summary.json")
            require(summary["status"] == "ALL_LOCAL_OFFICIAL_WINDOW_GATES_PASSED", "LOCAL_GATES_FAILED")
            require(summary["runtime_sha256"] == build["binary_sha256"], "QUALIFICATION_RUNTIME_CHANGED")
            require(summary["abi"] == row["abi"], "QUALIFICATION_ABI_CHANGED")
            require(digest(summary["helper_path"]) == summary["helper_sha256"], "QUALIFICATION_HELPER_CHANGED")
            require(digest(row["final_module"]) == row["final_module_sha256"], "FINAL_MODULE_CHANGED")
            module = tvm.ir.load_json(Path(row["final_module"]).read_text())
            llvm_certificate = llvm_alu_contract(module[row["global_var"]], (directory / "module.ll").read_text())
            llvm_certificate["frozen_codegen_binding"] = codegen_binding
            strong_path = output / (oid + "_" + mode + "_initializer_binding.json")
            write(strong_path, llvm_certificate)
            strengthened.append({"object_id": oid, "mode": mode, "certificate": str(strong_path), "status": llvm_certificate["status"]})
            local_calls = read(directory / "calls.json")
            require(len(local_calls) == 7 and summary["local_target_calls"] == 7, "LOCAL_CALL_BUDGET_CHANGED")
            require([c["call_kind"] for c in local_calls] == ["three_seed"]*3 +
                    ["weight_ABA_W0", "weight_ABA_W1", "weight_ABA_W0_return", "directed"], "LOCAL_CASE_ORDER_CHANGED")
            require([c["seed"] for c in local_calls[:3]] == [0, 20250901, 20260910], "QUALIFICATION_SEEDS_CHANGED")
            require(local_calls[3]["input_sha256"] == local_calls[5]["input_sha256"] and
                    local_calls[3]["input_sha256"] != local_calls[4]["input_sha256"] and
                    local_calls[3]["expected_sha256"] == local_calls[5]["expected_sha256"] and
                    local_calls[3]["expected_sha256"] != local_calls[4]["expected_sha256"], "WEIGHT_RETURN_NOT_OBSERVABLE")
            require(local_calls[3]["input_sha256"][1] != local_calls[4]["input_sha256"][1] and
                    all(a == b for i, (a, b) in enumerate(zip(local_calls[3]["input_sha256"], local_calls[4]["input_sha256"])) if i != 1),
                    "WEIGHT_CHECK_CHANGED_ANOTHER_INPUT")
            for index, call in enumerate(local_calls, 1):
                require(call["correct"] and call["arguments_unchanged"] and call["target_calls"] == 1 and
                        call["helper_count"] == index and not call["performance_sample"] and not call["board_contacted"], "LOCAL_CALL_CONTRACT")
                require(call["call_index"] == index, "LOCAL_CALL_INDEX_CHANGED")
                helper_meta = call["helper_metadata"]
                require(helper_meta["started_calls"] == helper_meta["completed_calls"] == index and
                        helper_meta["packed_calls_per_invocation"] == 1 and not helper_meta["adds_synchronize"] and
                        helper_meta["parameter_count"] == len(row["abi"]) == len(helper_meta["parameters"]), "HELPER_COMPLETION_OR_ABI_CHANGED")
                require(len(call["input_sha256"]) == len(row["abi"])-1, "INPUT_IDENTITY_LIST_INCOMPLETE")
                for expected_param, actual_param in zip(row["abi"], helper_meta["parameters"]):
                    require(expected_param["shape"] == actual_param["shape"] and
                            int(expected_param["dtype"].removeprefix("int")) == actual_param["dtype_bits"] and
                            actual_param["dtype_code"] == 0 and actual_param["dtype_lanes"] == 1 and
                            actual_param["byte_offset"] == 0, "HELPER_BUFFER_LAYOUT_CHANGED")
                require(call["output_sha256"] == call["expected_sha256"], "INDEPENDENT_OUTPUT_MISMATCH")
                require(call["actual_storage"] == local_calls[0]["actual_storage"] and
                        call["buffer_handles"] == local_calls[0]["buffer_handles"], "WEIGHT_STORAGE_CHANGED")
                canonical_submit_sequence(call["runtime_metadata"], mode)
                require(call["native_output_terms_exactly_match_independent_workload"], "ACTUAL_PRODUCTS_UNBOUND")
                native_path = directory / ("native_certificate_%02d.json" % index)
                native_record = read(native_path)
                require(native_record["status"] == call["native_certificate_status"] == "ACTUAL_BOUNDED_NATIVE_FIFO_TOKEN_UOP_CERTIFIED", "NATIVE_STATUS_FAILED")
                require(native_record["dump_sha256"] == call["native_dump_sha256"] ==
                        digest(directory / ("native_call_%02d.txt" % index)), "NATIVE_DUMP_CHANGED")
                require(native_record["cache_definitions"]["header_sha256"] == digest(HEADER), "NATIVE_MODEL_SOURCE_CHANGED")
                del native_record
                if "collection_status_after_call" in call:
                    observed = call["collection_status_after_call"]
                    require(observed["mode"] == mode and observed["debug_flag"] == 0, "COLLECTION_STATE_CHANGED")
                    require(observed["replay"]["mode"] == "disabled" and observed["queue"]["replay_policy"] == "disabled", "REPLAY_STATE_CHANGED")
                    require(mode != "minimal" or observed["detailed_clock_reads"] == 0, "MINIMAL_CLOCKS_REMAIN")
                calls_index.append({"object_id": oid, "mode": mode, "call_index": index,
                    "call_record": str(directory / ("dispatch_%02d_end.json" % index)),
                    "native_certificate": str(directory / ("native_certificate_%02d.json" % index)),
                    "performance_sample": False, "collection_status_observed": "collection_status_after_call" in call,
                    "saved_fsim_module_sha256": digest(directory / "fsim_module.so"),
                    "module_binding_evidence": "same worker exported and loaded this DSO; saved SHA measured at closure, not per historical call"})
            require(local_calls[-1]["output_distribution"]["nonzero_unsaturated_outputs"] > 0, "DIRECTED_OUTPUT_INVISIBLE")
            summaries[mode], calls[mode] = summary, local_calls
        require(summaries["detailed"]["execution_sha256"] == summaries["minimal"]["execution_sha256"], "MODE_EXECUTION_CHANGED")
        for index, (detailed, minimal) in enumerate(zip(calls["detailed"], calls["minimal"]), 1):
            require(all(detailed[k] == minimal[k] for k in ("call_kind", "seed", "input_sha256", "expected_sha256", "output_sha256")), "MODE_ARGUMENT_OR_RESULT_CHANGED")
            require(canonical_submit_sequence(detailed["runtime_metadata"]) == canonical_submit_sequence(minimal["runtime_metadata"]), "MODE_SUBMISSIONS_CHANGED")
            left = read(base / "detailed" / ("native_certificate_%02d.json" % index))
            right = read(base / "minimal" / ("native_certificate_%02d.json" % index))
            require(ancestor_values(left) == ancestor_values(right), "MODE_ACTUAL_DEPENDENCIES_CHANGED")
            require(left["material_map"] == right["material_map"] and left["dependency_map"] == right["dependency_map"] and
                    left["uop_values"] == right["uop_values"], "MODE_OPERANDS_OR_UOPS_CHANGED")
            del left, right
        certificate = read(base / "detailed/window_certificate.json")
        events = certificate["B"]["events"]
        # Classification uses the method's explicit support status, independently
        # of numerical/native correctness of the preserved baseline.
        supported_no_join = certificate["status"] == "NO_JOIN_IN_SUPPORTED_STRUCTURE"
        require(certificate["B"]["events"] == certificate["A"]["events"] and
                summaries["detailed"]["B_A_representation_equal"], "DIFFERENT_PROGRAM_REQUIRES_SEPARATE_A_QUALIFICATION")
        kernels = [event for event in events if event["kind"] == "kernel"]
        count = Counter((event["mode_reset"][0], event["mode_reset"][1]) for event in kernels)
        native = read(base / "detailed/native_certificate_01.json")
        definitions = native["cache_definitions"]["definitions"]
        distinct_lists = {tuple(tuple(uop) for uop in values) for values in definitions.values()}
        native_instructions = _instructions((base / "detailed/native_call_01.txt").read_text())
        device_count = Counter((instruction["mode"], instruction["reset"]) for instruction in native_instructions if instruction["kind"] == "compute")
        require(count == device_count, "DYNAMIC_GEMM_CENSUS_MISMATCH")
        values = native["uop_values"]
        loads = [instruction for instruction in native_instructions if instruction["kind"] == "load"]
        submits = calls["detailed"][0]["runtime_metadata"]["submits"]
        # This run has only one explicit submission. Do not pretend that a
        # fragment-relative parser supplies an unimplemented general report.
        require(len(submits) == 1, "REPORT_REQUIRES_EXPLICIT_MULTISUBMIT_CAPACITY_BINDING")
        capacity = {"object_id": oid, "single_kernel_max_uops": max(len(event["uops"]) for event in kernels),
            "host_cache_identity_count": len(definitions), "host_identity_uops_sum": sum(map(len, definitions.values())),
            "distinct_value_list_count": len(distinct_lists), "distinct_value_list_uops_sum": sum(map(len, distinct_lists)),
            "per_submit_serialized_uop_bytes": [submit["serialized_uop_bytes"] for submit in submits],
            "cumulative_uop_load_bytes_per_call": 4*sum(instruction["size"] for instruction in loads),
            "uop_load_instructions_per_call": len(loads), "wraps_per_call": values["uop_sram_wraps"],
            "greedy_conflicts_guarded_per_call": values["greedy_read_conflicts_guarded"], "submissions_per_call": len(submits),
            "dynamic_mac_gemm_insns": count[(0, 0)], "dynamic_reset_gemm_insns": count[(0, 1)],
            "dynamic_alu_insns": sum(n for (mode, reset), n in count.items() if mode == 1),
            "expanded_uop_steps": certificate["B"]["counts"]["expanded_uop_steps"],
            "reload_reason": "first submission: first load; later completed calls: submission Reset, not capacity eviction",
            "B_A_equal": True, "native_dump": str(base / "detailed/native_call_01.txt")}
        require(capacity["wraps_per_call"] == 0 and capacity["distinct_value_list_uops_sum"] <= 8192, "NATURAL_CAPACITY_PRESSURE_REQUIRES_DETAILED_EVICTION_REPORT")
        capacity["capacity_evictions_per_call"] = 0
        capacity["capacity_reloads_per_call"] = 0
        capacities.append(capacity)
        abi_sha = object_digest(row["abi"])
        audits.append({"object_id": oid, "network": row["network"], "earliest_graph_node": row["earliest_graph_node"],
            "semantic_hash": row["semantic_hash"], "classes": [4] if supported_no_join else [],
            "eligible": supported_no_join, "aggregation_status": certificate["status"],
            "rejection_reason": None if supported_no_join else "SUPPORT_UNRESOLVED",
            "delta_mac_insns": 0, "official_baseline_bound": True, "local_gates_passed": supported_no_join,
            "baseline_native_passed": True, "hardware_join_contract_verified": certificate["hardware_contract_valid_for_join"],
            "natural_adjacent_mac_seams": certificate["B"]["statistics"]["counts"]["natural_adjacent_seams"],
            "qualification": str(base), "graph_callsite_aliases": row["graph_callsite_aliases"],
            "programs": {variant: {"program_id": oid+"_"+variant, "execution_sha256": summaries["detailed"]["execution_sha256"],
                "abi_sha256": abi_sha} for variant in ("B", "A")}})
        print(json.dumps({"object": oid, "status": "LOCAL_EVIDENCE_CLOSED", "calls": 14}), flush=True)
    pool = freeze_pool(audits)
    require(not pool["board_allowed"], "BOARD_POOL_ADMITTED_REQUIRES_NEXT_AUTHORIZED_STAGE")
    write(output / "audit_rows.json", audits)
    write(output / "capacity_report.json", capacities)
    write(output / "strong_initializer_bindings.json", strengthened)
    write(output / "local_call_index.json", calls_index)
    write(output / "pool.json", pool)
    write(output / "board_call_manifest.json", {"status": "NOT_GENERATED_LOCAL_STOP", "pool_sha256": pool["pool_sha256"],
        "stop_reason": pool["status"], "target_calls": 0, "calls": [], "sessions": [], "performance_samples": 0,
        "board_contacted": False, "G": None, "T": None, "comparison_status": "STRUCTURE_IDENTICAL_NOT_MEASURED"})
    funnel = {"graph_vta_callsites": 32, "complete_semantic_instances": 26,
        "baseline_identity_passed": 26, "roundtrip_passed": 26, "analysis_completed": len(audits),
        "baseline_legal_and_native_passed": len(audits), "aggregation_support_certified": sum(a["eligible"] for a in audits),
        "support_unresolved": sum(not a["eligible"] for a in audits), "joinable_objects": 0,
        "selected_structural_controls": len(pool["objects"]), "different_selected_programs": 0,
        "qualified_local_target_calls": len(calls_index), "FPGA_calls": 0}
    write(output / "coverage_funnel.json", funnel)
    write(output / "completion.json", {"status": "COMPLETED_LOCAL_STOP_NO_DIFFERENT_OFFICIAL_PROGRAM",
        "funnel": funnel, "pool_status": pool["status"], "qualified_calls_are_performance_samples": False,
        "hardware_feedback_interval": "UNRESOLVED_NO_JOIN_SEAMS", "board_contacted": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qualification", action="append", required=True, help="Original full audit first, later fresh object qualifications next")
    args = parser.parse_args()
    finalize(Path(args.output_dir).resolve(), [Path(value).resolve() for value in args.qualification])
