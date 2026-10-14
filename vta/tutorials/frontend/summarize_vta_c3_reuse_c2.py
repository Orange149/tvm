#!/usr/bin/env python3
"""Close C2, preserving incomplete preflight costs and deferring node-D claims."""
import argparse
from pathlib import Path
import time

from run_vta_c3_reuse_supplement import verify, read, write, seal, file_hash, check_bindings


def run(args):
    root = Path(args.root).resolve()
    names = {"c1": "20260915_c3_reuse_fixed_driver_c1_board_run01",
             "c2": "20260915_c3_reuse_fixed_driver_c2_board_run02",
             "restore": "20260915_c3_reuse_c2_boot_restore_run01",
             "analysis": "20260915_c3_reuse_fixed_driver_c2_analysis_run01"}
    contracts = {key: verify(root / name) for key, name in names.items()}
    for c in contracts.values():
        check_bindings(c.get("source_bindings", {}))
    c1, c2 = [read(root / names[key] / "summary.json") for key in ("c1", "c2")]
    if c1["boot_id"] == c2["boot_id"] or not c2["status"].startswith("C2_COMPLETE"):
        raise ValueError("requires complete actual-second-boot session")
    for key in ("parent_manifest_sha256", "dispatch_context_sha256", "runtime_sha256"):
        if contracts["c1"][key] != contracts["c2"][key]:
            raise ValueError("frozen runtime/candidate context changed: " + key)
    failed = root / "20260915_c3_reuse_fixed_driver_c2_board_run01"
    ledger = read(failed / "artifact_hashes.json")["artifacts"]
    actual = {str(p.relative_to(failed)) for p in failed.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"}
    if set(ledger) != actual or any(file_hash(failed / name) != sha for name, sha in ledger.items()):
        raise ValueError("early-failure ledger changed")
    if not read(failed / "invalid_session.json")["all_timing_invalid"]:
        raise ValueError("early failure must remain invalid")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(output / "contract.json", {"phase": "c2_closeout", "node_D_complete": False,
          "source_sha256": file_hash(__file__), "input_manifests": {
              str(root / name): file_hash(root / name / "artifact_hashes.json") for name in [*names.values(), failed.name]},
          "early_failed_contract_missing": True, "sessions_not_spliced": True})
    records = [{"session": key, **value} for key, value in (("c1", c1), ("c2", c2))]
    records += [{"session": "early_preflight_failure", "invalid": True,
                 "completed_operator_calls": 0, "wall_seconds": None,
                 "reason": "missing volatile module/firmware; no terminal timer before original runner failed"},
                {"session": "volatile_boot_restore", **read(root / names["restore"] / "summary.json")}]
    from run_vta_c3_reuse_supplement import append
    for r in records:
        append(output / "results.jsonl", r)
    # This timeline describes analysis only, not an artificial continuous board session.
    append(output / "timeline.jsonl", {"event": "analysis_start", "wall_unix": time.time()})
    analysis = read(root / names["analysis"] / "summary.json")
    report = {"status": "C2_COMPLETE_PAUSED_BEFORE_D", "boot_ids": [c1["boot_id"], c2["boot_id"]],
              "correct_identities_per_boot": [c1["correct_identities"], c2["correct_identities"]],
              "timed_calls_per_boot": [c1["timed_complete_operator_calls"], c2["timed_complete_operator_calls"]],
              "all_completed_calls_per_boot": [c1["all_completed_operator_calls"], c2["all_completed_operator_calls"]],
              "W0_seconds_per_boot": [c1["W0_to_T1_seconds"], c2["W0_to_T1_seconds"]],
              "T0_seconds_per_boot": [c1["T0_to_T1_seconds"], c2["T0_to_T1_seconds"]],
              "C2_four_measured": analysis["four_measured"],
              "C2_fastest_mode_counts": analysis["fastest_mode_counts"],
              "C2_static_FPGA_verified_profiles": analysis["static_FSim_FPGA_logical_DMA_verified_profiles"],
              "C2_best_original_vs_extended": analysis["best_original_vs_extended"],
              "stable_cross_boot_claim": False, "full_total_including_failed_wall_seconds": None,
              "node_D_complete": False, "failure_cost_unknown_not_zero": True,
              "legacy_analysis_status_is_generic_single_boot_label": True}
    write(output / "summary.json", report)
    append(output / "timeline.jsonl", {"event": "analysis_end", "elapsed_seconds": time.monotonic() - started})
    seal(output)
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output-dir", required=True)
    run(p.parse_args())
