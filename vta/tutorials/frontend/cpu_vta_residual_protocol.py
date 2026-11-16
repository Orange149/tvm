#!/usr/bin/env python3
"""Freeze same-program assets for an independently measured residual campaign.

This module prepares files only. It does not build, deploy, or execute programs.
All public paths in a protocol are relative to its RunStore directory.
"""
import copy
import json
from pathlib import Path
import shutil
import uuid

from thesis_supplement_run import RunStore, sha256, utc, write_new_json


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / (
    "vta/tutorials/frontend/report_out/stage_tile_cotuning/"
    "thesis_supplement_experiments/results/06/"
    "20260916T124859Z_single_image_audit_bfec496d"
)
PROTOCOL_VERSION = "cpu_vta_residual_same_program_v1"
PROGRAM_NAMES = ("cpu_00_02", "vta_03_17", "cpu_18_20", "vta_03_15", "cpu_16_20")
TOPOLOGIES = {
    "03:17": ("cpu_00_02", "vta_03_17", "cpu_18_20"),
    "03:15": ("cpu_00_02", "vta_03_15", "cpu_16_20"),
}
BLOCK_ORDERS = [
    ["P0", "P1", "P2", "P3", "P4", "P5"],
    ["P5", "P4", "P3", "P2", "P1", "P0"],
    ["P2", "P3", "P4", "P5", "P0", "P1"],
]


def matrix():
    """Return fresh copies of the six preregistered conditions."""
    rows = [
        ("P0", [4, 1, 4], [[0, 1, 2, 3]] * 3, 0),
        ("P1", [4, 1, 1], [[0, 1, 2, 3], [0, 1, 2, 3], [0]], 0),
        ("P2", [4, 1, 2], [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1]], 0),
        ("P3", [3, 1, 1], [[0, 1, 2], [0, 1, 2, 3], [0]], 0),
        ("P4", [3, 1, 1], [[0, 1, 2], [0, 1, 2, 3], [3]], 0),
        ("P5", [4, 1, 4], [[0, 1, 2, 3]] * 3, 300000),
    ]
    return copy.deepcopy([
        {"condition": name, "threads": threads, "affinity": masks,
         "spin_count": spin, "queue_depth": 2, "actual_policy": "ordinary_copy"}
        for name, threads, masks, spin in rows
    ])


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(path):
    value = Path(path)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError("Unsafe protocol asset path: " + str(path))
    return value


def _check(path, identity):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Required asset missing: " + str(path))
    if sha256(path) != identity["sha256"]:
        raise ValueError("Asset SHA256 mismatch: " + str(path))
    if "bytes" in identity and path.stat().st_size != identity["bytes"]:
        raise ValueError("Asset byte count mismatch: " + str(path))


def verify_source(source_dir):
    """Verify selected sealed source evidence, excluding reconciled script collisions."""
    source = RunStore(source_dir)
    r = source.directory
    hashes = _json(r / "artifact_hashes.json")

    def sealed(relative):
        relative = str(_relative(relative))
        if relative not in hashes:
            raise ValueError("Source asset has no sealed hash: " + relative)
        _check(r / relative, hashes[relative])
        return r / relative

    summary = _json(sealed("summary.json"))
    if summary.get("status") != "SINGLE_IMAGE_SMOKE_PASS":
        raise ValueError("Source must have SINGLE_IMAGE_SMOKE_PASS")
    manifest_path = sealed("raw/c1_components_int8/component_manifest.json")
    manifest = _json(manifest_path)
    if manifest.get("canonical_qir_sha256") != source.contract["canonical_qir_sha256"]:
        raise ValueError("Component manifest uses a different canonical qIR")
    if manifest.get("extra_quantize_calls") != 0:
        raise ValueError("Components must retain the canonical quantization")
    if source.contract.get("canonical_source"):
        _check(source.contract["canonical_source"], {"sha256": source.contract["canonical_qir_sha256"]})
    programs = {row["name"]: row for row in manifest["programs"]}
    if len(programs) != len(manifest["programs"]):
        raise ValueError("Duplicate component program name")
    for name in PROGRAM_NAMES:
        if name not in programs:
            raise ValueError("Required component missing: " + name)
        row = programs[name]
        if len(row["input_names"]) != len(row["input_node_ids"]):
            raise ValueError("Incomplete input frontier: " + name)
        if len(row["output_sources"]) != len(row["output_node_ids"]):
            raise ValueError("Incomplete output frontier: " + name)
        for output_index, output in enumerate(row["output_sources"]):
            if output["kind"] not in ("live_input", "executor_output"):
                raise ValueError("Unknown output source kind")
            if output["kind"] == "live_input" and (
                type(output.get("live_input_index")) is not int
                or not 0 <= output["live_input_index"] < len(row["input_names"])
            ):
                raise ValueError("Invalid live_input frontier index")
            if output["kind"] == "live_input" and (
                row["output_node_ids"][output_index] != row["input_node_ids"][output["live_input_index"]]
            ):
                raise ValueError("Live input output names a different canonical node")
        for filename, identity in row["files"].items():
            _check(r / "raw/c1_components_int8" / name / _relative(filename), identity)
    for names in TOPOLOGIES.values():
        for left, right in zip(names, names[1:]):
            if programs[left]["output_node_ids"] != programs[right]["input_node_ids"]:
                raise ValueError("Adjacent stages have different live frontiers")

    input_contract = _json(sealed("analysis/input_contract.json"))
    _check(sealed("raw/inputs/input.bin"), {"sha256": input_contract["input_bin_sha256"]})
    _check(sealed("raw/inputs/input.npy"), {"sha256": input_contract["input_sha256"]})
    for name in PROGRAM_NAMES:
        for index in range(len(programs[name]["input_names"])):
            sealed(f"raw/c1_component_inputs/{name}/input{index}.bin")
    for name in ("reference_logits.npy", "reference_boundaries.npz"):
        sealed("raw/host_reference/" + name)

    bindings = _json(sealed("analysis/board_artifact_bindings.json"))
    for name in ("libtvm_runtime.so", "libvta.so", "tvm_rpc", "vta_hpc.bit", "u-dma-buf.ko"):
        _check(bindings[name]["host_path"], bindings[name])
    if bindings["vta_hpc.bit"]["sha256"] != source.contract["bitstream_sha256"]:
        raise ValueError("Runtime firmware differs from source contract")
    runner = _json(sealed("analysis/native_runner_selection.json"))
    _check(sealed("raw/" + str(_relative(runner["name"]))), runner)
    proxy_path = sealed("raw/c1_frozen/fair_support_contract.json")
    proxy = _json(proxy_path)
    if not proxy.get("label_free") or proxy.get("actual_policy") != "ordinary_copy":
        raise ValueError("Old proxy must retain its label-free ordinary-copy identity")
    proxy_runner = runner.get("source_C1_runner_selection")
    if not proxy_runner:
        raise ValueError("Source does not identify the original proxy executable")
    if proxy_runner:
        _check(sealed("raw/" + str(_relative(proxy_runner["name"]))), proxy_runner)
        for row in proxy["support"]:
            identity = row["full_identity"]
            if identity["runner_sha256"] != proxy_runner["sha256"]:
                raise ValueError("Old proxy runner identity differs")
            if identity["canonical_qir_sha256"] != source.contract["canonical_qir_sha256"]:
                raise ValueError("Old proxy canonical qIR differs")
            if identity["input_sha256"] != input_contract["input_sha256"]:
                raise ValueError("Old proxy numeric input differs")
            for name, binding in identity.get("runtime_bindings", {}).items():
                if name in bindings and binding["sha256"] != bindings[name]["sha256"]:
                    raise ValueError("Old proxy runtime identity differs: " + name)
    return {"store": source, "sealed": sealed, "manifest": manifest, "programs": programs,
            "input_contract": input_contract, "bindings": bindings, "runner": runner,
            "proxy_runner": proxy_runner, "proxy": proxy}


def freeze_protocol(source_dir, output_dir):
    """Create a new immutable RunStore and copy the exact assets needed by both cuts."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Existing run cannot be overwritten: " + str(output))
    verified = verify_source(source_dir)
    source = verified["store"]
    assets = {}

    def add(source_path, target):
        target = str(_relative(target))
        source_path = Path(source_path).resolve()
        if target in assets:
            raise ValueError("Duplicate destination asset: " + target)
        assets[target] = {"source_path": str(source_path), "sha256": sha256(source_path),
                          "bytes": source_path.stat().st_size}

    sealed = verified["sealed"]
    for name in PROGRAM_NAMES:
        for filename in verified["programs"][name]["files"]:
            add(source.directory / "raw/c1_components_int8" / name / filename,
                f"raw/programs/{name}/{filename}")
        for index in range(len(verified["programs"][name]["input_names"])):
            relative = f"raw/c1_component_inputs/{name}/input{index}.bin"
            add(sealed(relative), f"raw/component_inputs/{name}/input{index}.bin")
    for name in ("input.bin", "input.npy"):
        add(sealed("raw/inputs/" + name), "raw/inputs/" + name)
    for name in ("reference_logits.npy", "reference_boundaries.npz"):
        add(sealed("raw/host_reference/" + name), "raw/reference/" + name)
    for name, binding in verified["bindings"].items():
        if name in ("libtvm_runtime.so", "libvta.so", "tvm_rpc", "vta_hpc.bit", "u-dma-buf.ko"):
            add(binding["host_path"], "raw/runtime/" + name)
    add(sealed("raw/" + verified["runner"]["name"]), "raw/baseline_runner")
    if verified["proxy_runner"]:
        add(sealed("raw/" + verified["proxy_runner"]["name"]), "raw/proxy_runner")
    evidence = {
        "component_manifest": "raw/c1_components_int8/component_manifest.json",
        "input_contract": "analysis/input_contract.json",
        "proxy": "raw/c1_frozen/fair_support_contract.json",
        "source_summary": "summary.json",
        "source_runner_selection": "analysis/native_runner_selection.json",
        "source_runtime_bindings": "analysis/board_artifact_bindings.json",
    }
    paths = {"input": "raw/inputs/input.bin", "numeric_input": "raw/inputs/input.npy",
             "reference_dir": "raw/reference", "reference_logits": "raw/reference/reference_logits.npy",
             "frontiers": "raw/reference/reference_boundaries.npz", "runtime_dir": "raw/runtime",
             "baseline_runner": "raw/baseline_runner", "proxy_runner": "raw/proxy_runner",
             "bitstream": "raw/runtime/vta_hpc.bit", "tvm_rpc": "raw/runtime/tvm_rpc"}
    for key, relative in evidence.items():
        paths[key] = "raw/source_evidence/" + key + ".json"
        add(sealed(relative), paths[key])
    programs = {}
    for name in PROGRAM_NAMES:
        row = copy.deepcopy(verified["programs"][name])
        row.update(directory="raw/programs/" + name,
                   graph=f"raw/programs/{name}/graph.json", lib=f"raw/programs/{name}/module.so",
                   params=f"raw/programs/{name}/params.bin",
                   input_files=[f"raw/component_inputs/{name}/input{i}.bin"
                                for i in range(len(row["input_names"]))])
        programs[name] = row
    run_id = output.name
    contract = {
        "run_id": run_id, "stage": "residual", "campaign_id": "cpu_vta_residual_" + uuid.uuid4().hex[:12],
        "protocol_version": PROTOCOL_VERSION, "scope": "single_input_one_boot_mechanism_campaign",
        "created_at": utc(), "completion": {"execution_context_reset": {
            "required_before_each_new_command_queue": True}},
        "source_identity": {"directory": str(source.directory), "run_id": source.contract["run_id"],
            "contract_sha256": sha256(source.directory / "contract.json"),
            "artifact_registry_sha256": sha256(source.directory / "artifact_hashes.json"),
            "summary_sha256": sha256(source.directory / "summary.json"), "status": "SINGLE_IMAGE_SMOKE_PASS",
            "baseline_runner": verified["runner"], "proxy_runner": verified["proxy_runner"],
            "baseline_runner_role": "source executable audit anchor, not required primary performance baseline"},
        "canonical_qir_sha256": source.contract["canonical_qir_sha256"],
        "numeric": copy.deepcopy(source.contract["numeric"]),
        "numeric_input": verified["input_contract"], "bitstream_sha256": source.contract["bitstream_sha256"],
        "board_host": source.contract["board_host"], "udmabuf_bytes": source.contract["udmabuf_bytes"],
        "assets": assets, "paths": paths, "programs": programs,
        "topologies": {key: list(names) for key, names in TOPOLOGIES.items()},
        "main_topology": "03:17", "holdout_topology": "03:15", "conditions": matrix(),
        "block_orders": copy.deepcopy(BLOCK_ORDERS), "holdout_conditions": ["P0", "P1", "P2", "P3", "P4", "P5"],
        "performance_window": {
            "pipeline_rows": 307, "warmup_rows": 5, "settling_rows": 2,
            "scored_completion_rows": 300, "scored_intervals": 299,
            "ii_formula": "(completion[306] - completion[7]) / 299",
            "serial_rows": 20, "serial_warmup_rows": 5, "serial_scored_rows": 15,
            "stage_service_statistic": "mean of matched scored rows within each block",
            "pilot_statistic": "median of the three block means",
            "replicate_statistic": "paired within-boot mean of two balanced block means",
            "fill_drain": "report separately; never add to steady-state II",
        },
        "correctness": {
            "repeats": 10, "modes": ["serial", "pipeline"],
            "complete_live_frontiers_required": True, "all_outputs_finite_required": True,
            "reference": "copied same-input canonical LLVM reference",
            "integer_comparison": "byte_exact", "float_comparison": "frozen numeric atol/rtol",
            "failure_rule": "stop before performance and preserve raw outputs",
        },
        "timeouts_seconds": {
            "build": 1200, "preflight": 60, "correctness": 60,
            "serial": 60, "window": 600, "reset": 60,
        },
        "thresholds": {
            "cpu_run_absolute_effect_fraction_min": 0.05,
            "critical_cpu_busy_fraction_min": 0.90,
            "stage_vs_global_ii_relative_difference_max": 0.05,
            "replicate_median_absolute_effect_fraction_min": 0.05,
            "replicate_effect_direction_consistent_across_boots": True,
            "paired_boot_mean_ci_confidence": 0.95,
            "paired_boot_mean_ci_degrees_of_freedom": 2,
            "paired_boot_mean_ci_critical_t": 4.3026527299,
            "paired_boot_mean_ci_must_exclude_zero": True,
            "timing_telemetry_ii_relative_difference_max": 0.02,
            "timing_telemetry_exceedance_policy": "use telemetry-disabled primary timing; keep on/off evidence",
        },
        "selection": {
            "contrasts": [["P0", "P1"], ["P0", "P2"], ["P0", "P5"], ["P3", "P4"]],
            "eligible_stages": [0, 2], "common_pilot_blocks_required": 3,
            "identity_matched_serial_and_pipeline_required": True,
            "critical_trajectory_required_in_either_arm": True,
            "metric": "absolute fractional change of median of three block mean CPU run services",
            "order": ["descending absolute effect_fraction", "ascending intervention condition id",
                      "ascending baseline condition id", "ascending stage index"],
            "cpu_stage_preference": None,
            "fps_used_for_selection": False, "freeze_before_new_boots": True,
            "no_eligible_contrast": "report no contrast selected; do not infer a missing mechanism",
        },
        "replication": {
            "independent_boots_required": 3, "current_boot_counts_as_first": True,
            "user_physical_reboots_required": 2, "rpc_restart_is_new_boot": False,
            "new_boot_identity_must_differ_from_all_previous": True,
            "blocks_per_boot": 2, "orders": ["control_then_contrast", "contrast_then_control"],
            "pilot_rows_used_for_replication": False,
            "conditions": "pilot-selected frozen control and contrast only",
            "statistical_unit": "paired independent boot mean; frames and blocks are not independent boots",
            "gate": "all three boot effects same direction, median absolute change >=5%; paired-boot t95 CI excludes zero",
            "holdout": "same frozen contrast and current executables on topology 03:15; report separately",
        },
        "missing_baselines": {
            "same_executable_same_program_isolated_service": "must be measured before concurrency attribution",
            "uninstrumented_executable": "same new executable with telemetry flags disabled; copied source runner is audit only",
            "instrumentation_overhead": "same executable P0 telemetry on/off in two rotating blocks, 2% primary-timing gate",
            "source_runner_equivalence": "optional short same-boot diagnostic; no mandatory source/new executable equivalence gate",
            "isolated_missing_components": [
                {"name": "cpu_00_02", "threads": 3, "affinity": [0, 1, 2]},
                {"name": "cpu_18_20", "threads": 2, "affinity": [0, 1]},
                {"name": "cpu_18_20", "threads": 1, "affinity": [3]},
            ],
            "ddr_attribution": "unknown without independently identified DDR-specific evidence",
        },
        "conditional_diagnostics": {
            "allowed_only_after_pilot_matrix": True,
            "cpu_pair_and_cpu_vta": "same-program pressure controls, separate from exact pilot scores",
            "queue_depth": "if no CPU contrast is eligible, P0 depth 1/2/4 then 4/2/1; separate from P0..P5",
            "vta_shape_service": "held-out topology and isolated inclusive run; no additive profiler cost",
            "vta_profile_trigger": "any matched pilot pipeline VTA mean run exceeds serial by 5%",
            "vta_profile_trigger_topologies": ["03:15", "03:17"],
            "predeclared_triggered_branches_only": True,
        },
        "board_state": {
            "cpu_frequency_khz": "actual_unknown_until_each_session_preflight",
            "cpu_governor": "actual_unknown_until_each_session_preflight",
            "requested_cpu_frequency_khz": 1066666, "requested_cpu_governor": "userspace",
            "report_actual_frequency_and_governor": True,
            "claim_domain": "actual attested boot/runtime/frequency; desired values are not measurements",
        },
        "immutability": {
            "existing_run_directory": "reject", "existing_session_output": "reject",
            "partial_failed_session": "preserve and stop; new run for retry",
            "labels_may_change_protocol": False, "source_evidence_overwrite": False,
        },
        "actual_policy": "ordinary_copy", "queue_depth": 2,
        "stop_rule": "numeric failure, timeout, asset drift, incomplete trace: preserve evidence and stop; retry in new run",
        "proxy_rule": "old component service is an identity-bound diagnostic, not refitted with campaign labels",
        "instrumented_runner_binding": "separate immutable analysis artifact; primary on/off baseline uses the same new executable; source executable is audit only",
    }
    store = RunStore.create(output, contract)
    for relative, identity in assets.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _check(identity["source_path"], identity)
        with target.open("xb") as destination, Path(identity["source_path"]).open("rb") as original:
            shutil.copyfileobj(original, destination)
        shutil.copymode(identity["source_path"], target)
        _check(target, identity)
    bindings = {name: {"host_path": str(output / ("raw/runtime/" + name)),
                      "sha256": assets["raw/runtime/" + name]["sha256"],
                      "bytes": assets["raw/runtime/" + name]["bytes"]}
                for name in ("libtvm_runtime.so", "libvta.so", "tvm_rpc", "vta_hpc.bit", "u-dma-buf.ko")}
    write_new_json(output / "analysis/board_artifact_bindings.json", bindings)
    store.event("protocol_assets_frozen", asset_count=len(assets),
                board_bindings_sha256=sha256(output / "analysis/board_artifact_bindings.json"))
    verify_assets(store)
    return store


def verify_assets(store):
    """Fail closed if any frozen execution/diagnostic asset changed."""
    store.verify_contract()
    for relative, identity in store.contract["assets"].items():
        _check(store.directory / _relative(relative), identity)
    bindings = _json(store.directory / "analysis/board_artifact_bindings.json")
    required = {"libtvm_runtime.so", "libvta.so", "tvm_rpc", "vta_hpc.bit", "u-dma-buf.ko"}
    if set(bindings) != required:
        raise ValueError("Incomplete or unexpected deployment runtime bindings")
    for name, binding in bindings.items():
        relative = "raw/runtime/" + str(_relative(name))
        identity = store.contract["assets"][relative]
        if (binding["host_path"] != str(store.directory / relative)
                or binding["sha256"] != identity["sha256"] or binding.get("bytes") != identity["bytes"]):
            raise ValueError("Deployment binding differs from frozen runtime: " + name)
        _check(binding["host_path"], identity)
    return True


def load_protocol(run_dir):
    store = RunStore(run_dir)
    if store.contract.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Not a CPU/VTA residual protocol")
    verify_assets(store)
    return store


def protocol_paths(store):
    """Resolve all protocol path aliases without changing the frozen contract."""
    return {name: store.directory / _relative(relative)
            for name, relative in store.contract["paths"].items()}


def stages_for(store, topology="03:17"):
    """Return copied complete-frontier stage records for either frozen topology."""
    if topology not in store.contract["topologies"]:
        raise ValueError("Unknown frozen topology: " + topology)
    stages = []
    for index, name in enumerate(store.contract["topologies"][topology]):
        stage = copy.deepcopy(store.contract["programs"][name])
        stage["index"] = index
        stages.append(stage)
    return stages
