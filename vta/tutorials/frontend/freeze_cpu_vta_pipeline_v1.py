#!/usr/bin/env python3
"""Freeze the CPU-VTA Pipeline V1 protocol without running performance tests."""

from __future__ import absolute_import, print_function

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import tvm
import vta

from split_resnet18_stages import (  # pylint: disable=import-error
    UNIT_ORDER,
    _unit_kind,
    build_resnet18_unit_compute_metadata,
    build_resnet18_unit_io_schemas,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "cpu_vta_pipeline_v1"
DEFAULT_BOARD_HOST = "192.168.1.234"
DEFAULT_RPC_PORT = 9090
DEFAULT_SSH_USER = "root"
CPU_THREAD_CHOICES = [1, 2, 3, 4]
MAX_VTA_ISLANDS = 3
QUEUE_DEPTH = 2
POLL_SLEEP_NS = 1000


def repo_root():
    return Path(__file__).resolve().parents[3]


def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_sha256(payload):
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def seal_artifact(payload):
    sealed = copy.deepcopy(payload)
    sealed.pop("artifact_sha256", None)
    sealed["artifact_sha256"] = payload_sha256(sealed)
    return sealed


def validate_artifact_sha256(payload):
    expected = payload.get("artifact_sha256")
    unsealed = copy.deepcopy(payload)
    unsealed.pop("artifact_sha256", None)
    if expected != payload_sha256(unsealed):
        raise ValueError("artifact_sha256 does not match canonical payload")


def load_sealed_artifact(path, expected_kind):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_artifact_sha256(payload)
    if payload.get("kind") != expected_kind:
        raise ValueError("expected {}, got {}".format(expected_kind, payload.get("kind")))
    return payload


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(args), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def command_value(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cmake_cache_value(path, key):
    prefix = key + ":"
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix) and "=" in line:
            return line.split("=", 1)[1]
    return "unknown"


def _normalized_slots(schema, preserve_roles):
    slots = []
    for slot in schema["slots"]:
        item = {
            "shape": [int(value) for value in slot["shape"]],
            "dtype": str(slot["dtype"]),
        }
        if preserve_roles:
            item["role"] = str(slot["role"])
        slots.append(item)
    return slots


def contracts_compatible(producer, consumer):
    if int(producer["arity"]) != int(consumer["arity"]):
        return False
    preserve_roles = int(producer["arity"]) > 1
    return _normalized_slots(producer, preserve_roles) == _normalized_slots(
        consumer, preserve_roles
    )


def _candidate_device_support(unit_name):
    if unit_name in {"stem", "head"}:
        return ["cpu"]
    return ["cpu", "vta"]


def build_unit_and_boundary_schema(batch=1, image_size=224):
    io_schemas = build_resnet18_unit_io_schemas(batch, image_size)
    compute_metadata = build_resnet18_unit_compute_metadata(batch, image_size)
    units = []
    for index, name in enumerate(UNIT_ORDER):
        units.append(
            {
                "index": index,
                "name": name,
                "kind": _unit_kind(name),
                "candidate_device_support": _candidate_device_support(name),
                "static_compute": compute_metadata[name],
                "input_contract": io_schemas[name]["input_schema"],
                "output_contract": io_schemas[name]["output_schema"],
            }
        )

    boundaries = []
    for cut_position in range(1, len(UNIT_ORDER)):
        producer = UNIT_ORDER[cut_position - 1]
        consumer = UNIT_ORDER[cut_position]
        producer_contract = io_schemas[producer]["output_schema"]
        consumer_contract = io_schemas[consumer]["input_schema"]
        boundaries.append(
            {
                "cut_position": cut_position,
                "producer_unit": producer,
                "consumer_unit": consumer,
                "producer_contract": producer_contract,
                "consumer_contract": consumer_contract,
                "contract_compatible": contracts_compatible(
                    producer_contract, consumer_contract
                ),
                "native_lowering_status": "not_checked_until_V1-P1",
                "heterogeneous_transfer": {
                    "transfer_resource": "stage_dma",
                    "ps_pl_owner": "lowered_vta_load_store",
                    "boundary_owner": "host_adapter_set_get_copy",
                    "accounting_id_templates": {
                        "host_adapter": "boundary:{cut_position}:host_adapter:{ordinal}",
                        "ps_pl_load": "vta_stage:{stage_index}:load_instruction:{ordinal}",
                        "ps_pl_store": "vta_stage:{stage_index}:store_instruction:{ordinal}",
                    },
                },
            }
        )

    if not all(item["contract_compatible"] for item in boundaries):
        raise ValueError("ResNet18 unit contracts are not compatible at every cut")

    return seal_artifact(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "cpu_vta_pipeline_v1_unit_and_boundary_schema",
            "protocol_id": PROTOCOL_ID,
            "model": {
                "name": "resnet18_v1",
                "batch": int(batch),
                "image_size": int(image_size),
                "atomic_unit_count": len(units),
                "cut_position_count": len(boundaries),
                "unit_order": list(UNIT_ORDER),
            },
            "candidate_grammar": {
                "coverage": "all units exactly once in UNIT_ORDER",
                "stage_shape": "nonempty contiguous unit interval",
                "canonicalization": "merge adjacent stages on the same device",
                "cut_legality_level": "static tensor contract; native lowering required in V1-P1",
                "required_device_pattern": "cpu,(vta,cpu){1,3}",
                "first_stage_device": "cpu",
                "last_stage_device": "cpu",
                "minimum_vta_islands": 1,
                "maximum_vta_islands": MAX_VTA_ISLANDS,
                "maximum_stage_count": 2 * MAX_VTA_ISLANDS + 1,
                "cpu_thread_choices": CPU_THREAD_CHOICES,
                "cpu_thread_parameter_scope": "independent TVM runtime parameter per CPU stage",
                "cpu_thread_sum_is_a_constraint": False,
                "cpu_topology": "four_homogeneous_cores",
                "cpu_affinity_policy": {
                    "pipeline": "each CPU stage independently uses [0,threads); masks may overlap",
                    "serial": "each CPU stage independently uses [0,threads)",
                    "tvm_affinity_mode": -3,
                    "big_little_assumption": False,
                },
                "vta_window_rules": {
                    "must_contain_kind": "main_preadd",
                    "allowed_start_kinds": ["main_preadd"],
                    "allowed_end_kinds": [
                        "main_preadd",
                        "skip_proj",
                        "add_relu_tail",
                    ],
                },
                "all_vta_baseline_separate_from_search": True,
                "all_cpu_baseline_separate_from_search": True,
            },
            "resource_ownership": {
                "single_vta_mutex": "one token shared by all VTA stages",
                "vta_mutex_scope": "native RunStage set_input/run/get_output",
                "cpu_stage": "CPU compute and stage-local memory service",
                "shared_ddr": "unique per-frame physical traffic by accounting id",
                "boundary": "host adapter and set/get copy only",
                "vta_dma": "lowered VTA LOAD/STORE including fixed-schedule spill",
                "submit_sync": "VTA device-run service",
                "additive_spill_penalty": False,
            },
            "units": units,
            "legal_cut_positions": boundaries,
        }
    )


def validate_candidate(stages, schema):
    errors = []
    if not stages:
        return ["candidate has no stages"]

    flattened = []
    previous_device = None
    vta_islands = 0
    for index, stage in enumerate(stages):
        device = stage.get("device")
        unit_names = list(stage.get("unit_names") or [])
        if device not in {"cpu", "vta"}:
            errors.append("stage {} has invalid device".format(index))
        if not unit_names:
            errors.append("stage {} is empty".format(index))
        if previous_device == device:
            errors.append("adjacent stages {} and {} use the same device".format(index - 1, index))
        previous_device = device
        flattened.extend(unit_names)

        if device == "cpu":
            threads = stage.get("threads")
            if threads not in CPU_THREAD_CHOICES:
                errors.append("CPU stage {} has invalid threads".format(index))
        elif device == "vta":
            vta_islands += 1
            kinds = [_unit_kind(name) for name in unit_names if name in UNIT_ORDER]
            if "main_preadd" not in kinds:
                errors.append("VTA stage {} has no main_preadd unit".format(index))
            if kinds and kinds[0] != "main_preadd":
                errors.append("VTA stage {} starts at an illegal unit kind".format(index))
            if kinds and kinds[-1] not in {"main_preadd", "skip_proj", "add_relu_tail"}:
                errors.append("VTA stage {} ends at an illegal unit kind".format(index))

    expected_order = list(schema["model"]["unit_order"])
    if flattened != expected_order:
        errors.append("stages do not cover UNIT_ORDER exactly once")
    if stages[0].get("device") != "cpu" or stages[-1].get("device") != "cpu":
        errors.append("searched pipeline must start and end on CPU")
    if not 1 <= vta_islands <= MAX_VTA_ISLANDS:
        errors.append("VTA island count is outside [1, {}]".format(MAX_VTA_ISLANDS))
    return errors


def build_cpu_affinity_manifest(stages, serial=False):
    """Pin each TVM thread pool independently; masks may overlap across stages."""
    manifest = []
    for index, stage in enumerate(stages):
        if stage.get("device") != "cpu":
            manifest.append({"stage_index": index, "device": stage.get("device"), "cpu_mask": []})
            continue
        threads = int(stage.get("threads", 0))
        if threads not in CPU_THREAD_CHOICES:
            raise ValueError("CPU stage {} has invalid threads".format(index))
        manifest.append(
            {
                "stage_index": index,
                "device": "cpu",
                "threads": threads,
                "cpu_mask": list(range(threads)),
                "tvm_affinity_mode": -3,
                "affinity_policy": "independent_overlapping_prefix_masks_v1",
            }
        )
    return manifest


def build_hardware_fingerprint(args):
    root = repo_root()
    env = vta.get_env()
    vta_config_path = root / "3rdparty" / "vta-hw" / "config" / "vta_config.json"
    with vta_config_path.open("r", encoding="utf-8") as inp:
        vta_config = json.load(inp)

    source_paths = [
        "vta/tutorials/frontend/split_resnet18_stages.py",
        "vta/tutorials/frontend/profile_split_resnet18_stages.py",
        "vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py",
        "vta/apps/native_deploy/vta_stage_pipeline_runner.cc",
        "vta/python/vta/build_module.py",
        "vta/python/vta/top/graphpack.py",
        "vta/python/vta/top/op.py",
        "vta/python/vta/top/vta_conv2d.py",
        "python/tvm/relay/quantize/quantize.py",
        "src/runtime/thread_pool.cc",
        "src/runtime/threading_backend.cc",
        "apps/vta_rpc/start_axu5evb_cpp_rpc.sh",
        "apps/vta_rpc/start_axu5evb_hpc_rpc.sh",
    ]
    source_sha256 = {name: file_sha256(root / name) for name in source_paths}
    cmake_cache_path = root / "build_axu_aarch64/CMakeCache.txt"
    cross_compiler = cmake_cache_value(cmake_cache_path, "CMAKE_CXX_COMPILER")
    preflight = load_sealed_artifact(
        args.preflight_evidence, "cpu_vta_pipeline_v1_preflight_evidence"
    )
    endpoint = preflight["endpoint"]
    remote = preflight["remote"]
    remote_hashes = preflight["component_hashes"]["remote"]
    local_component_paths = {
        "bitstream": root / "apps/vta_rpc/firmware/vta_hpc.bit",
        "tvm_rpc": root / "build_axu_aarch64/tvm_rpc",
        "libtvm_runtime.so": root / "build_axu_aarch64/libtvm_runtime.so",
        "libvta.so": root / "build_axu_aarch64/libvta.so",
    }
    local_component_hashes = {
        name: file_sha256(path) for name, path in local_component_paths.items()
    }
    remote_identity = {
        "board_instance_proxy": preflight["board_instance_proxy"],
        "bitstream_name": preflight["expected_runtime"]["bitstream_name"],
        "bitstream_sha256": remote_hashes.get("bitstream"),
        "runtime_components_sha256": {
            "tvm_rpc": remote_hashes.get("tvm_rpc"),
            "libtvm_runtime.so": remote_hashes.get("libtvm_runtime.so"),
            "libvta.so": remote_hashes.get("libvta.so"),
        },
        "cpu_topology": "four_homogeneous_cores",
        "cpu_frequencies_khz": remote.get("cpu_frequencies_khz"),
        "cpu_governors": remote.get("cpu_governors"),
        "fpga_state": remote.get("fpga_state"),
    }
    required_remote_fields = []
    if not preflight.get("passed"):
        required_remote_fields.append("preflight_passed")
    if endpoint.get("ssh_target") != "{}@{}".format(args.ssh_user, args.board_host):
        required_remote_fields.append("ssh_endpoint_match")
    if endpoint.get("rpc_host") != args.board_host or int(endpoint.get("rpc_port", -1)) != int(
        args.rpc_port
    ):
        required_remote_fields.append("rpc_endpoint_match")
    if not all(preflight.get("gate_checks", {}).values()):
        required_remote_fields.append("all_preflight_gate_checks")
    if any(remote_hashes.get(name) != digest for name, digest in local_component_hashes.items()):
        required_remote_fields.append("remote_components_match_current_local_build")

    stable_identity = {
        "expected_target": str(env.TARGET),
        "vta_config": vta_config,
        "vta_config_sha256": file_sha256(vta_config_path),
        "compiler": {
            "tvm_python_version": str(tvm.__version__),
            "tvm_git_head": git_value(root, "rev-parse", "HEAD"),
            "vta_hw_git_head": git_value(root / "3rdparty" / "vta-hw", "rev-parse", "HEAD"),
            "source_sha256": source_sha256,
            "cross_compiler_version": command_value([cross_compiler, "--version"]),
            "cross_build_cmake_cache_sha256": file_sha256(cmake_cache_path),
            "cross_sysroot": cmake_cache_value(cmake_cache_path, "CMAKE_SYSROOT"),
            "target_host": str(env.target_host),
            "target_vta_cpu": str(env.target_vta_cpu),
        },
        "frozen_lowering": {
            "status": "configuration_frozen; actual_segment_tir_and_module_hashes_required_in_V1-P1",
            "relay_pass_opt_level": 3,
            "quantize_global_scale": 8.0,
            "quantize_skip_conv_layers": [],
            "graph_pack_mode": "boundary_bridge",
            "graph_pack_batch": int(env.BATCH),
            "graph_pack_block_out": int(env.BLOCK_OUT),
            "graph_pack_weight_width": int(env.WGT_WIDTH),
            "vta_build_opt_level": 3,
            "disabled_passes": ["AlterOpLayout", "tir.CommonSubexprElimTIR"],
        },
        "runtime_policy": {
            "queue_depth": QUEUE_DEPTH,
            "poll_sleep_ns": POLL_SLEEP_NS,
            "single_vta_mutex": True,
            "stage_workers": "one persistent host thread per stage",
            "cpu_topology": "four homogeneous cores; no big.LITTLE assumption",
            "threadpool_affinity_mode": -3,
            "cpu_affinity_policy": "independent [0,threads) masks; overlap allowed",
            "cpu_thread_parameter_scope": "independent per CPU stage; sums are not constrained",
        },
        "remote_identity": remote_identity,
    }
    profile_allowed = not required_remote_fields
    return seal_artifact(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "cpu_vta_pipeline_v1_hardware_fingerprint",
            "protocol_id": PROTOCOL_ID,
            "status": "complete" if profile_allowed else "draft_remote_preflight_incomplete",
            "profile_allowed": profile_allowed,
            "stable_hardware_fingerprint_sha256": (
                payload_sha256(stable_identity) if profile_allowed else None
            ),
            "local_build_fingerprint_sha256": payload_sha256(
                {key: value for key, value in stable_identity.items() if key != "remote_identity"}
            ),
            "missing_required_evidence": sorted(set(required_remote_fields)),
            "connection": {
                "ssh_target": "{}@{}".format(args.ssh_user, args.board_host),
                "ssh_options": [
                    "HostKeyAlgorithms=+ssh-rsa",
                    "PubkeyAcceptedAlgorithms=+ssh-rsa",
                ],
                "rpc_host": args.board_host,
                "rpc_port": int(args.rpc_port),
                "included_in_stable_hardware_identity": False,
            },
            "preflight_evidence_artifact_sha256": preflight["artifact_sha256"],
            "stable_identity": stable_identity,
        }
    )


def build_protocol(unit_schema, hardware_fingerprint):
    profile_allowed = bool(hardware_fingerprint["profile_allowed"])
    return seal_artifact(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "cpu_vta_pipeline_v1_protocol",
            "protocol_id": PROTOCOL_ID,
            "status": "frozen" if profile_allowed else "draft_blocked_by_remote_preflight",
            "profile_allowed": profile_allowed,
            "model": {
                "name": "resnet18_v1",
                "unit_schema_artifact_sha256": unit_schema["artifact_sha256"],
                "atomic_unit_count": unit_schema["model"]["atomic_unit_count"],
            },
            "hardware": {
                "stable_hardware_fingerprint_sha256": hardware_fingerprint[
                    "stable_hardware_fingerprint_sha256"
                ],
            },
            "decision_variables": {
                "unit_device": ["cpu", "vta"],
                "stage_cuts": "legal positions from unit schema",
                "cpu_stage_threads": CPU_THREAD_CHOICES,
                "cpu_stage_affinity": "each stage independently uses [0,threads); overlap allowed",
            },
            "fixed_variables": {
                "vta_schedule_and_tile": "frozen_lowering in hardware fingerprint",
                "quantization_and_graph_pack": "frozen_lowering in hardware fingerprint",
                "queue_depth": QUEUE_DEPTH,
                "poll_sleep_ns": POLL_SLEEP_NS,
                "runtime_policy": "one worker per stage; one global VTA mutex",
                "cpu_affinity_policy": "independent_overlapping_prefix_masks_v1",
                "maximum_vta_islands": MAX_VTA_ISLANDS,
            },
            "constraints": {
                "cpu_stage_threads_independent": True,
                "cpu_thread_sum_is_a_constraint": False,
                "single_physical_vta": True,
                "shared_ddr_accounting": "unique transaction ids; aggregate demand lower bound",
                "adjacent_same_device_stages": "canonical merge",
                "native_pipeline_pattern": "cpu,(vta,cpu){1,3}",
                "compiler_validity_required": True,
            },
            "objective": {
                "name": "predicted_pipeline_initiation_interval_ms",
                "formula": "max(max_cpu_stage_ms,single_vta_service_plus_mutex_boundary_ms,"
                "total_host_core_ms/4,shared_ddr_demand_ms)",
                "maximize": "predicted_fps=1000/predicted_pipeline_initiation_interval_ms",
            },
            "metrics": {
                "primary": [
                    "throughput_regret_at_1_3_5_10_20",
                    "board_evaluations_to_95_percent_measured_pool_oracle",
                    "top_k_recall",
                    "profile_build_search_wall_clock",
                ],
                "throughput_regret_formula": "1-best_fps_in_top_k/measured_pool_oracle_fps",
                "global_oracle_claim_allowed": False,
            },
            "budget_governance": {
                "prospective_top_k_max": 20,
                "profile_budget": "freeze after P1 signature deduplication and user review",
                "performance_experiment_requires_user_confirmation": True,
                "full_candidate_board_enumeration_forbidden": True,
            },
            "correctness": {
                "independent_reference_required": True,
                "serial_pipeline_equivalence_required": True,
                "fallback_allowed": False,
            },
            "required_gates": {
                "P0": [
                    "auditable preflight evidence",
                    "TVM thread parameter semantics without static core partition",
                    "stable hardware fingerprint",
                ],
                "P1": "segment native lowering validity and actual TIR/module hashes",
            },
        }
    )


def build_execution_state(protocol, hardware_fingerprint):
    p0_passed = bool(hardware_fingerprint["profile_allowed"])
    return seal_artifact(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "cpu_vta_pipeline_v1_execution_state",
            "protocol_id": PROTOCOL_ID,
            "protocol_artifact_sha256": protocol["artifact_sha256"],
            "hardware_fingerprint_artifact_sha256": hardware_fingerprint["artifact_sha256"],
            "preflight_evidence_artifact_sha256": hardware_fingerprint[
                "preflight_evidence_artifact_sha256"
            ],
            "current_stage": "V1-P0",
            "current_stage_status": "completed_awaiting_user_confirmation"
            if p0_passed
            else "blocked",
            "next_stage": "V1-P1",
            "next_stage_may_start": False,
            "advance_requires_user_confirmation": True,
            "blocking_evidence": hardware_fingerprint["missing_required_evidence"],
        }
    )


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze_artifacts(args):
    unit_schema = build_unit_and_boundary_schema(args.batch, args.image_size)
    hardware_fingerprint = build_hardware_fingerprint(args)
    protocol = build_protocol(unit_schema, hardware_fingerprint)
    execution_state = build_execution_state(protocol, hardware_fingerprint)
    output_dir = Path(args.output_dir)
    outputs = {
        "v1_unit_and_boundary_schema.json": unit_schema,
        "v1_hardware_fingerprint.json": hardware_fingerprint,
        "v1_protocol.json": protocol,
        "v1_execution_state.json": execution_state,
    }
    for name, payload in outputs.items():
        validate_artifact_sha256(payload)
        write_json(output_dir / name, payload)
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(
            repo_root()
            / "vta"
            / "tutorials"
            / "frontend"
            / "report_out"
            / "resource_aware_maxplus"
        ),
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--board-host", default=DEFAULT_BOARD_HOST)
    parser.add_argument("--rpc-port", type=int, default=DEFAULT_RPC_PORT)
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER)
    parser.add_argument(
        "--preflight-evidence",
        default=str(
            repo_root()
            / "vta/tutorials/frontend/report_out/resource_aware_maxplus/v1_preflight_evidence.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = freeze_artifacts(args)
    for name, payload in outputs.items():
        print("{} {}".format(name, payload["artifact_sha256"]))
    if not outputs["v1_protocol.json"]["profile_allowed"]:
        print("P0_GATE=blocked_remote_preflight")


if __name__ == "__main__":
    main()
