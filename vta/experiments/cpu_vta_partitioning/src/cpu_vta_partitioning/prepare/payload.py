"""Host archive / board execution closure and conservative per-job disk budgets."""
import math
from pathlib import Path

from cpu_vta_partitioning.common import read, write, inventory, verify_inventory


EXECUTION_POLICY = {
    "revision": 1,
    "udmabuf_bytes": 201326592,
    "storage_margin_bytes": 256 * 1024**2,
    "diagnostic_order": "paired_reverse_blocks_across_six_boots",
    "diagnostic_anchor_drift": .05,
    "diagnostic_replacement_blocks": 2,
    "recovery_io": "session_lock_excludes_measurement_and_bulk_io",
    "input_validation": "per_frame_scheduled_sequence_and_qualification_coverage",
    "analysis_evidence": "hashed_receipts_raw_files_and_matched_six_boots",
    "analysis_identity": "run_and_bundle_controller_match_before_outputs",
    "preparation_lock": "held_before_workspace_initialization_through_seal",
    "sdk_cache": "effective_toolchain_identity_and_verified_sdk_runtime_closure",
    "direction_gate": "same_budget_all_primary_and_mechanism_host_ablation_separate",
    "random_controls": "seeded_shuffled_reservoir_excluding_top5_union",
    "cost_accounting": "accepted_and_observed_frames_with_unknown_consumption",
    "signal_recovery": "retain_boot_lock_on_wrapper_hup_int_quit_term",
}


def execution_files(bundle):
    """References and controller sources are consumed only by the host."""
    bundle = Path(bundle)
    files = {}
    for folder in ("runtime", "runner", "firmware", "inputs"):
        files.update({folder + "/" + k: v for k, v in inventory(bundle / folder).items()})
    for name, program in read(bundle / "plans/programs.json").items():
        for filename in ("graph.json", "module.so", "params.bin"):
            files[f"programs/{name}/{filename}"] = program["files"][filename]
    files["plans/board_execution.json"] = inventory(bundle / "plans")["board_execution.json"]
    return files


def create_payload_plan(workspace):
    write(workspace / "plans/board_execution.json", EXECUTION_POLICY, replace=True)
    files = execution_files(workspace)
    plan = {"revision": 1, "files": files, "unpacked_bytes": sum(v["bytes"] for v in files.values()),
            "transfer": "gzip_tar_stream_no_remote_archive", "host_only": ["references", "sources"]}
    write(workspace / "plans/board_payload.json", plan, replace=True)
    return plan


def validate_payload(bundle):
    bundle = Path(bundle)
    plan = read(bundle / "plans/board_payload.json")
    if read(bundle / "plans/board_execution.json") != EXECUTION_POLICY:
        raise ValueError("board execution policy differs from controller")
    if plan["files"] != execution_files(bundle) or plan["unpacked_bytes"] != sum(v["bytes"] for v in plan["files"].values()):
        raise ValueError("board execution closure mismatch")
    verify_inventory(bundle, plan["files"])
    return plan


def job_storage_bytes(document, programs):
    """Raw tensor dumps plus a conservative JSON/log budget, excluding free margin."""
    frames = document["warmup"] + document["scored"] + document["tail"]
    total = frames * max(1, len(document["stages"])) * 65536 + 8 * 1024**2
    if document["kind"].startswith("qualify"):
        sizes = []
        for stage in document["stages"]:
            sizes.append(sum(math.prod(t["shape"]) * int("".join(filter(str.isdigit, t["dtype"]))) // 8
                             for t in programs[stage["program"]]["output_types"]))
        # The native report dumps the last frontier both as final and stage output.
        total += frames * (sum(sizes) + sizes[-1])
    return total
