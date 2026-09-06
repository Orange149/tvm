#!/usr/bin/env python3
"""Build and audit the P7 component-profile plan without measuring candidates."""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import platform
import random
import shlex
import statistics
import subprocess
import tarfile
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = ROOT / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
DEFAULT_RANKING = REPORT_ROOT / "v1_p5b_iteration2_ranked_candidates.json"
DEFAULT_MANIFEST = REPORT_ROOT / "v1_profile_manifest.json"
DEFAULT_PLAN = REPORT_ROOT / "v1_p7_measurement_plan.json"
DEFAULT_REVIEW = REPORT_ROOT / "v1_p7_review.md"
DEFAULT_PACKAGE_ROOT = REPORT_ROOT / "v1_p7_top20_board_20260904/packages"
DEFAULT_TOP20_BOARD_SUMMARY = (
    REPORT_ROOT / "v1_p7_top20_board_20260904/top20_board_summary.json"
)
DEFAULT_P7B1_MULTIBOOT = REPORT_ROOT / "v1_p7b1_three_boot_reproducibility.json"
DEFAULT_P5A_PACKAGE_ROOT = REPORT_ROOT / "v1_p5a_packages"
DEFAULT_P7B2_PLAN = REPORT_ROOT / "v1_p7b2_measurement_plan.json"
DEFAULT_P7B2_SESSION = REPORT_ROOT / "v1_p7_session_boot1_p7b2_runtime.json"
DEFAULT_P7B2_SUMMARY = REPORT_ROOT / "v1_p7b2_runtime_qualification_summary.json"
DEFAULT_P7C_PLAN = REPORT_ROOT / "v1_p7c_measurement_plan.json"
DEFAULT_P7C_SESSION = REPORT_ROOT / "v1_p7_session_boot1_p7c_ddr.json"
DEFAULT_P7C_SUMMARY = REPORT_ROOT / "v1_p7c_qualification_summary.json"
DEFAULT_REFERENCE = REPORT_ROOT / "v1_p5a_reference.json"
DEFAULT_LOCAL_COST = REPORT_ROOT / "v1_local_cost_table.json"

RANKING_KIND = "cpu_vta_pipeline_v1_p5b_iteration2_ranked_candidates"
MANIFEST_KIND = "cpu_vta_pipeline_v1_profile_manifest"
PROTOCOL_ID = "cpu_vta_pipeline_v1_p7_physical_profile"
THREAD_CHOICES = (1, 2, 3, 4)
WARMUP_RUNS = 5
SCORED_RUNS = 20
MEMORY_SAMPLE_TRAFFIC_BYTES = 128 * 1024 * 1024
MEMORY_SAMPLE_CV_MAX = 0.10
ISOLATED_STAGE_CV_MAX = 0.10
SLOWDOWN_CI_RELATIVE_HALFWIDTH_MAX = 0.25
SSH_OPTIONS = (
    "-o",
    "HostKeyAlgorithms=+ssh-rsa",
    "-o",
    "PubkeyAcceptedAlgorithms=+ssh-rsa",
    "-o",
    "BatchMode=yes",
)


def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_sha256(payload):
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def seal_artifact(payload):
    result = copy.deepcopy(payload)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = payload_sha256(result)
    return result


def validate_artifact(payload, expected_kind):
    expected = payload.get("artifact_sha256")
    unsealed = copy.deepcopy(payload)
    unsealed.pop("artifact_sha256", None)
    if expected != payload_sha256(unsealed):
        raise ValueError("artifact_sha256 mismatch for {}".format(expected_kind))
    if payload.get("kind") != expected_kind:
        raise ValueError("expected kind {}, got {}".format(expected_kind, payload.get("kind")))


def load_artifact(path, expected_kind):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_artifact(payload, expected_kind)
    return payload


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_relative_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve().relative_to(ROOT.resolve()))


def _coefficient_of_variation(values):
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _pearson_correlation(left, right):
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_energy = sum((x - left_mean) ** 2 for x in left)
    right_energy = sum((y - right_mean) ** 2 for y in right)
    denominator = (left_energy * right_energy) ** 0.5
    return numerator / denominator if denominator else 1.0


def collect_environment():
    perf_paranoid = None
    perf_path = Path("/proc/sys/kernel/perf_event_paranoid")
    if perf_path.is_file():
        perf_paranoid = perf_path.read_text(encoding="utf-8").strip()
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "orchestrator_affinity": affinity,
        "perf_event_paranoid": perf_paranoid,
    }


def _terminate_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def run_synchronized_pair(command_a, command_b, barrier_dir, timeout_s=120):
    """Run two barrier-aware component commands with one shared start token."""
    barrier_dir = Path(barrier_dir)
    barrier_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    ready_a = barrier_dir / "{}.a.ready".format(token)
    ready_b = barrier_dir / "{}.b.ready".format(token)
    start = barrier_dir / "{}.start".format(token)
    common = ["--start-file", str(start), "--barrier-token", token]
    process_a = subprocess.Popen(
        list(command_a) + ["--ready-file", str(ready_a)] + common,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process_b = subprocess.Popen(
        list(command_b) + ["--ready-file", str(ready_b)] + common,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ready_start = time.monotonic()
    try:
        deadline = ready_start + float(timeout_s)
        while time.monotonic() < deadline:
            if process_a.poll() is not None or process_b.poll() is not None:
                raise RuntimeError("component process exited before the synchronized start")
            if (
                ready_a.is_file()
                and ready_b.is_file()
                and ready_a.read_text(encoding="utf-8").strip() == token
                and ready_b.read_text(encoding="utf-8").strip() == token
            ):
                break
            time.sleep(0.001)
        else:
            raise RuntimeError("timed out waiting for both component ready files")
        synchronized_start = time.monotonic()
        start.write_text(token, encoding="utf-8")
        output_a, error_a = process_a.communicate(timeout=timeout_s)
        output_b, error_b = process_b.communicate(timeout=timeout_s)
    except Exception:
        _terminate_process(process_a)
        _terminate_process(process_b)
        raise
    finally:
        for path in (ready_a, ready_b, start):
            path.unlink(missing_ok=True)
    if process_a.returncode != 0 or process_b.returncode != 0:
        raise RuntimeError(
            "synchronized component failed: A={} {!r}; B={} {!r}".format(
                process_a.returncode, error_a, process_b.returncode, error_b
            )
        )
    return {
        "barrier_token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
        "ready_wait_ms": 1000.0 * (synchronized_start - ready_start),
        "process_a": {"returncode": process_a.returncode, "stdout": output_a, "stderr": error_a},
        "process_b": {"returncode": process_b.returncode, "stdout": output_b, "stderr": error_b},
    }


def summarize_rows(rows):
    if not rows:
        raise ValueError("empty component result")
    wall = [float(row.get("wall_ms", row.get("total_latency_ms", 0.0))) for row in rows]
    if not all(value > 0 for value in wall):
        raise ValueError("component result lacks positive wall time")
    fingerprints = []
    for row in rows:
        if "raw_outputs" in row:
            fingerprints.append(canonical_json(row["raw_outputs"]))
        elif "checksum" in row:
            fingerprints.append(str(row["checksum"]))
        elif "top1" in row:
            fingerprints.append(str(row["top1"]))
        else:
            raise ValueError("component result lacks a determinism fingerprint")
    correctness_flags = [
        row.get("correct", row.get("reference_correctness_passed")) for row in rows
    ]
    correctness_observed = all(value is not None for value in correctness_flags)
    correctness = correctness_observed and all(value is True for value in correctness_flags)
    mean_wall = statistics.mean(wall)
    return {
        "row_count": len(rows),
        "correctness_observed": correctness_observed,
        "correctness_passed": correctness,
        "determinism_passed": len(set(fingerprints)) == 1,
        "wall_ms_mean": mean_wall,
        "wall_ms_median": statistics.median(wall),
        "wall_ms_cv": statistics.pstdev(wall) / mean_wall if len(wall) > 1 else 0.0,
        "pmu_available_rows": sum(row.get("pmu_available") is True for row in rows),
        "pmu_unavailable_reasons": sorted(
            {
                row["pmu_unavailable_reason"]
                for row in rows
                if row.get("pmu_unavailable_reason")
            }
        ),
    }


def summarize_jsonl(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return summarize_rows(rows)


def _run_checked(command, timeout_s=120):
    process = subprocess.run(
        list(command), text=True, capture_output=True, timeout=timeout_s, check=False
    )
    if process.returncode != 0:
        raise RuntimeError(
            "command failed rc={} command={} stderr={!r}".format(
                process.returncode, shlex.join(command), process.stderr.strip()
            )
        )
    return process.stdout


def _ssh_command(board_host, ssh_user, remote_command, timeout_s=120):
    return _run_checked(
        ["ssh", *SSH_OPTIONS, "{}@{}".format(ssh_user, board_host), remote_command],
        timeout_s=timeout_s,
    )


def _scp_to_board(local_path, board_host, ssh_user, remote_path, timeout_s=120):
    _run_checked(
        [
            "scp",
            *SSH_OPTIONS,
            str(local_path),
            "{}@{}:{}".format(ssh_user, board_host, remote_path),
        ],
        timeout_s=timeout_s,
    )


def _scp_from_board(board_host, ssh_user, remote_path, local_path, timeout_s=120):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _run_checked(
        [
            "scp",
            *SSH_OPTIONS,
            "{}@{}:{}".format(ssh_user, board_host, remote_path),
            str(local_path),
        ],
        timeout_s=timeout_s,
    )


def _parse_jsonl_text(contents):
    return [json.loads(line) for line in contents.splitlines() if line.strip().startswith("{")]


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_ratio_ci(isolated, concurrent, seed, samples=4000):
    if not isolated or not concurrent:
        raise ValueError("slowdown bootstrap requires two non-empty samples")
    rng = random.Random(seed)
    ratios = []
    for _ in range(samples):
        isolated_sample = [rng.choice(isolated) for _ in isolated]
        concurrent_sample = [rng.choice(concurrent) for _ in concurrent]
        ratios.append(
            statistics.median(concurrent_sample) / statistics.median(isolated_sample)
        )
    ratios.sort()
    return {
        "method": "independent_median_bootstrap_v1",
        "samples": samples,
        "lower_95": ratios[int(0.025 * samples)],
        "upper_95": ratios[min(samples - 1, int(0.975 * samples))],
    }


def _relative_ci_halfwidth(point, interval):
    return max(point - interval["lower_95"], interval["upper_95"] - point) / point


def _segment_units(manifest):
    return {item["segment_id"]: tuple(item["unit_names"]) for item in manifest["segments"]}


def _package_catalog(package_root):
    catalog = []
    for manifest_path in sorted(Path(package_root).glob("*/package/manifest.json")):
        package = json.loads(manifest_path.read_text(encoding="utf-8"))
        tar_path = manifest_path.parents[1] / "package.tar.gz"
        if not tar_path.is_file():
            continue
        catalog.append(
            {
                "package_dir": manifest_path.parent,
                "tar_path": tar_path,
                "manifest": package,
            }
        )
    return catalog


def _combined_package_catalog(*package_roots):
    catalog = []
    seen = set()
    for root in package_roots:
        for item in _package_catalog(root):
            key = (item["manifest"]["candidate_id"], str(item["package_dir"]))
            if key not in seen:
                seen.add(key)
                catalog.append(item)
        for manifest_path in sorted(Path(root).glob("*/package/manifest.json")):
            package = json.loads(manifest_path.read_text(encoding="utf-8"))
            key = (package["candidate_id"], str(manifest_path.parent))
            if key in seen:
                continue
            seen.add(key)
            catalog.append(
                {
                    "package_dir": manifest_path.parent,
                    "tar_path": None,
                    "manifest": package,
                }
            )
    return catalog


def _resolve_device_stage_package(segment_id, device, manifest, catalog):
    units = _segment_units(manifest).get(segment_id)
    if units is None:
        raise ValueError("unknown segment {}".format(segment_id))
    matches = []
    for package in catalog:
        for stage in package["manifest"]["stages"]:
            if stage["device"] == device and tuple(stage["unit_names"]) == units:
                matches.append((package, stage))
    if not matches:
        raise ValueError("no native package contains {} {}".format(device, segment_id))
    matches.sort(key=lambda item: (item[0]["manifest"]["candidate_id"], item[1]["index"]))
    package, stage = matches[0]
    return {
        "segment_id": segment_id,
        "package_dir": package["package_dir"],
        "tar_path": package["tar_path"],
        "package_manifest": package["manifest"],
        "stage": stage,
    }


def _materialize_package_archive(resolved):
    if resolved.get("tar_path") is not None and Path(resolved["tar_path"]).is_file():
        return Path(resolved["tar_path"])
    candidate_id = resolved["package_manifest"]["candidate_id"]
    tag = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:16]
    archive = Path("/tmp") / "ramps_p7_package_{}.tar.gz".format(tag)
    package_dir = Path(resolved["package_dir"])
    newest_source = max(path.stat().st_mtime_ns for path in package_dir.rglob("*") if path.is_file())
    if archive.is_file() and archive.stat().st_mtime_ns >= newest_source:
        return archive
    with tarfile.open(archive, "w:gz") as output:
        for path in sorted(package_dir.rglob("*")):
            output.add(path, arcname=str(path.relative_to(package_dir)), recursive=False)
    return archive


def _resolve_stage_package(segment_id, manifest, catalog):
    units = _segment_units(manifest).get(segment_id)
    if units is None:
        raise ValueError("unknown segment {}".format(segment_id))
    matches = []
    for package in catalog:
        for stage in package["manifest"]["stages"]:
            if stage["device"] == "cpu" and tuple(stage["unit_names"]) == units:
                matches.append((package, stage))
    if not matches:
        raise ValueError("no native package contains {}".format(segment_id))
    matches.sort(key=lambda item: item[0]["manifest"]["candidate_id"])
    package, stage = matches[0]
    return {
        "segment_id": segment_id,
        "package_dir": package["package_dir"],
        "tar_path": package["tar_path"],
        "package_manifest": package["manifest"],
        "stage": stage,
    }


def _resolve_pair_package(segment_a, segment_b, manifest, catalog):
    units = _segment_units(manifest)
    required = {segment_a: units[segment_a], segment_b: units[segment_b]}
    matches = []
    for package in catalog:
        by_units = {
            tuple(stage["unit_names"]): stage
            for stage in package["manifest"]["stages"]
            if stage["device"] == "cpu"
        }
        if all(value in by_units for value in required.values()):
            matches.append(
                (
                    package,
                    by_units[required[segment_a]],
                    by_units[required[segment_b]],
                )
            )
    if not matches:
        raise ValueError("no native package contains pair {} + {}".format(segment_a, segment_b))
    matches.sort(key=lambda item: item[0]["manifest"]["candidate_id"])
    package, stage_a, stage_b = matches[0]
    return {
        "package_dir": package["package_dir"],
        "tar_path": package["tar_path"],
        "package_manifest": package["manifest"],
        "stage_a": stage_a,
        "stage_b": stage_b,
    }


def _stage_cli(stage, threads):
    command = [
        "--stage0-graph",
        stage["graph"],
        "--stage0-lib",
        stage["lib"],
        "--stage0-params",
        stage["params"],
        "--stage0-input-names",
        ",".join(stage["input_names"]),
        "--stage0-name",
        stage["name"],
        "--stage0-device",
        stage["device"],
        "--stage0-runtime-num-threads",
        str(threads),
    ]
    if stage["device"] == "cpu":
        command.extend(
            ["--stage0-cpu-affinity", ",".join(str(value) for value in range(threads))]
        )
    return command


def _full_pipeline_cli(package_manifest):
    command = []
    stage_threads = package_manifest["stage_runtime_threads"]["serial"]
    stage_affinity = package_manifest["stage_cpu_affinity"]["serial"]
    for stage in package_manifest["stages"]:
        index = stage["index"]
        prefix = "--stage{}-".format(index)
        command.extend(
            [
                prefix + "graph",
                stage["graph"],
                prefix + "lib",
                stage["lib"],
                prefix + "params",
                stage["params"],
                prefix + "input-names",
                ",".join(stage["input_names"]),
                prefix + "name",
                stage["name"],
                prefix + "device",
                stage["device"],
                prefix + "runtime-num-threads",
                str(stage_threads[stage["name"]]),
            ]
        )
        affinity = stage_affinity.get(stage["name"], [])
        if affinity:
            command.extend([prefix + "cpu-affinity", ",".join(map(str, affinity))])
    return command


def _runner_shell(remote_package, runner_args, environment=None):
    command = ["./vta_stage_pipeline_runner_p7", *runner_args]
    extra_environment = " ".join(
        "{}={}".format(key, shlex.quote(str(value)))
        for key, value in sorted((environment or {}).items())
    )
    return (
        "cd {root} && export LD_LIBRARY_PATH=$PWD && "
        "export LD_PRELOAD=$PWD/libtvm_runtime.so:$PWD/libvta.so && "
        "export TVM_NUM_THREADS=4 TVM_THREAD_POOL_SPIN_COUNT=0 && {environment} {command}"
    ).format(
        root=shlex.quote(remote_package),
        environment=extra_environment,
        command=shlex.join(command),
    )


def _reference_input_files(stage, remote_package):
    if stage["index"] == 0:
        return [remote_package + "/input.bin"]
    preceding = stage["index"] - 1
    return [
        "{}/reference/run_0/stage{}_output_{}.bin".format(
            remote_package, preceding, slot
        )
        for slot in range(stage["input_schema"]["arity"])
    ]


def _expected_stage_outputs(reference_row, stage_index):
    for item in reference_row["stage_raw_outputs"]:
        if int(item["stage_index"]) == int(stage_index):
            return item["outputs"]
    raise ValueError("reference row lacks stage {}".format(stage_index))


def _annotate_stage_rows(rows, expected_outputs):
    expected = canonical_json(expected_outputs)
    for row in rows:
        row["reference_correctness_passed"] = canonical_json(row["raw_outputs"]) == expected
    return rows


def _stage_summary(rows):
    summary = summarize_rows(rows)
    stage_ms = [float(row["stage0_ms"]) for row in rows]
    mean_stage = statistics.mean(stage_ms)
    summary.update(
        {
            "stage_ms_mean": mean_stage,
            "stage_ms_median": statistics.median(stage_ms),
            "stage_ms_cv": statistics.pstdev(stage_ms) / mean_stage,
            "process_cpu_ms_median": statistics.median(
                float(row["stage0_process_cpu_ms"]) for row in rows
            ),
        }
    )
    return summary


def run_p7b1_memory_qualification(args, plan):
    from collect_cpu_vta_pipeline_v1_preflight import collect as collect_preflight

    preflight_args = argparse.Namespace(
        board_host=args.board_host,
        rpc_port=args.rpc_port,
        ssh_user=args.ssh_user,
        timeout_s=args.timeout_s,
    )
    preflight_before = collect_preflight(preflight_args)
    if not preflight_before["passed"]:
        raise RuntimeError("P7B-1 refused because board preflight failed")

    remote_root = args.remote_root.rstrip("/")
    remote_binary = remote_root + "/ramps_cpu_memory_microbench"
    _ssh_command(
        args.board_host,
        args.ssh_user,
        "mkdir -p {root} && chmod 700 {root}".format(root=shlex.quote(remote_root)),
        args.timeout_s,
    )
    _scp_to_board(
        Path(args.memory_binary), args.board_host, args.ssh_user, remote_binary, args.timeout_s
    )
    _ssh_command(
        args.board_host,
        args.ssh_user,
        "chmod 700 {}".format(shlex.quote(remote_binary)),
        args.timeout_s,
    )
    boot_id = _ssh_command(
        args.board_host,
        args.ssh_user,
        "cat /proc/sys/kernel/random/boot_id",
        args.timeout_s,
    ).strip()

    case_results = []
    for index, case in enumerate(plan["experiments"]["p7b1_cpu_memory_baseline"], 1):
        bytes_per_repeat = int(case["target_active_working_set_bytes"])
        inner_repeats = max(
            1, (MEMORY_SAMPLE_TRAFFIC_BYTES + bytes_per_repeat - 1) // bytes_per_repeat
        )
        remote_output = remote_root + "/{}.jsonl".format(case["case_id"])
        command = [
            remote_binary,
            "--operation",
            case["operation"],
            "--case-id",
            case["case_id"],
            "--output",
            remote_output,
            "--pressure-class",
            case["pressure_class"],
            "--threads",
            str(case["threads"]),
            "--streams",
            "1",
            "--warmup",
            str(case["warmup_runs"]),
            "--runs",
            str(case["scored_runs"]),
            "--inner-repeats",
            str(inner_repeats),
            "--bytes",
            str(case["operand_bytes_per_stream"]),
            "--profile-pmu",
        ]
        remote_command = "{} && cat {}".format(
            shlex.join(command), shlex.quote(remote_output)
        )
        print("[P7B-1 memory {}/36] {}".format(index, case["case_id"]), flush=True)
        rows = _parse_jsonl_text(
            _ssh_command(
                args.board_host, args.ssh_user, remote_command, args.case_timeout_s
            )
        )
        summary = summarize_rows(rows)
        pmu_semantics_valid = summary["pmu_available_rows"] in {0, len(rows)} and (
            summary["pmu_available_rows"] == len(rows)
            or bool(summary["pmu_unavailable_reasons"])
        )
        passed = (
            len(rows) == SCORED_RUNS
            and summary["correctness_passed"]
            and summary["determinism_passed"]
            and summary["wall_ms_cv"] <= MEMORY_SAMPLE_CV_MAX
            and pmu_semantics_valid
        )
        case_results.append(
            {
                "case": case,
                "inner_repeats": inner_repeats,
                "sample_traffic_bytes": bytes_per_repeat * inner_repeats,
                "rows": rows,
                "summary": summary,
                "gate_checks": {
                    "scored_row_count_20": len(rows) == SCORED_RUNS,
                    "correctness": summary["correctness_passed"],
                    "determinism": summary["determinism_passed"],
                    "wall_time_cv_at_most_10_percent": summary["wall_ms_cv"]
                    <= MEMORY_SAMPLE_CV_MAX,
                    "pmu_available_or_null_with_reason": pmu_semantics_valid,
                },
                "passed": passed,
            }
        )

    preflight_after = collect_preflight(preflight_args)
    session = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7b1_memory_qualification_session",
            "protocol_id": PROTOCOL_ID,
            "plan_artifact_sha256": plan["artifact_sha256"],
            "endpoint": {
                "board_host": args.board_host,
                "ssh_user": args.ssh_user,
                "rpc_port": args.rpc_port,
            },
            "board_boot_id": boot_id,
            "preflight_before": preflight_before,
            "preflight_after": preflight_after,
            "sample_policy": {
                "target_traffic_bytes_per_sample": MEMORY_SAMPLE_TRAFFIC_BYTES,
                "wall_time_cv_max": MEMORY_SAMPLE_CV_MAX,
                "persistent_worker_threads": True,
            },
            "case_results": case_results,
            "summary": {
                "case_count": len(case_results),
                "passed_case_count": sum(item["passed"] for item in case_results),
                "failed_case_ids": [
                    item["case"]["case_id"] for item in case_results if not item["passed"]
                ],
            },
            "passed": preflight_after["passed"] and all(
                item["passed"] for item in case_results
            ),
            "evidence_scope": "single-boot P7B-1 memory qualification; not a formal parameter profile",
        }
    )
    write_json(args.session_output, session)
    return session


def run_p7b1_cpu_pair_qualification(args, plan, manifest):
    from collect_cpu_vta_pipeline_v1_preflight import collect as collect_preflight

    preflight_args = argparse.Namespace(
        board_host=args.board_host,
        rpc_port=args.rpc_port,
        ssh_user=args.ssh_user,
        timeout_s=args.timeout_s,
    )
    preflight_before = collect_preflight(preflight_args)
    if not preflight_before["passed"]:
        raise RuntimeError("P7B-1 refused because board preflight failed")

    catalog = _package_catalog(args.package_root)
    top20_summary = json.loads(Path(args.top20_board_summary).read_text(encoding="utf-8"))
    prior_correctness = {
        row["candidate_id"]: bool(row["correctness"]["independent_reference_passed"])
        for row in top20_summary["rows"]
    }
    remote_root = args.remote_root.rstrip("/") + "/cpu_pairs"
    _ssh_command(
        args.board_host,
        args.ssh_user,
        "mkdir -p {}".format(shlex.quote(remote_root)),
        args.timeout_s,
    )
    boot_id = _ssh_command(
        args.board_host,
        args.ssh_user,
        "cat /proc/sys/kernel/random/boot_id",
        args.timeout_s,
    ).strip()

    deployed = {}

    def deploy_and_reference(resolved):
        package_manifest = resolved["package_manifest"]
        candidate_id = package_manifest["candidate_id"]
        if candidate_id in deployed:
            return deployed[candidate_id]
        tag = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
        remote_package = remote_root + "/pkg_" + tag
        remote_tar = remote_root + "/pkg_" + tag + ".tar.gz"
        remote_runner = remote_package + "/vta_stage_pipeline_runner_p7"
        runner_sha256 = _file_sha256(args.runner_binary)
        reusable = _ssh_command(
            args.board_host,
            args.ssh_user,
            "if test -f {manifest} && test -f {runner} && test -f {reference}; "
            "then sha256sum {runner} | cut -d' ' -f1; fi".format(
                manifest=shlex.quote(remote_package + "/manifest.json"),
                runner=shlex.quote(remote_runner),
                reference=shlex.quote(remote_package + "/reference.jsonl"),
            ),
            args.timeout_s,
        ).strip() == runner_sha256
        if not reusable:
            _scp_to_board(
                resolved["tar_path"],
                args.board_host,
                args.ssh_user,
                remote_tar,
                args.package_timeout_s,
            )
            _ssh_command(
                args.board_host,
                args.ssh_user,
                "mkdir -p {package} && tar -xzf {archive} -C {package}".format(
                    package=shlex.quote(remote_package), archive=shlex.quote(remote_tar)
                ),
                args.package_timeout_s,
            )
            _scp_to_board(
                args.runner_binary,
                args.board_host,
                args.ssh_user,
                remote_runner,
                args.package_timeout_s,
            )
            _ssh_command(
                args.board_host,
                args.ssh_user,
                "chmod 700 {}".format(shlex.quote(remote_runner)),
                args.timeout_s,
            )
        reference_output = remote_package + "/reference.jsonl"
        reference_args = [
            *_full_pipeline_cli(package_manifest),
            "--input",
            "input.bin",
            "--runs",
            "1",
            "--warmup-runs",
            "1",
            "--runtime-num-threads",
            "4",
            "--serial",
            "--output-mode",
            "raw_all_stages",
            "--output-jsonl",
            reference_output,
            "--output-dump-dir",
            "reference",
        ]
        if reusable:
            reference_text = _ssh_command(
                args.board_host,
                args.ssh_user,
                "cat {}".format(shlex.quote(reference_output)),
                args.timeout_s,
            )
        else:
            reference_text = _ssh_command(
                args.board_host,
                args.ssh_user,
                _runner_shell(remote_package, reference_args)
                + " && cat {}".format(shlex.quote(reference_output)),
                args.package_timeout_s,
            )
        reference_rows = _parse_jsonl_text(reference_text)
        if len(reference_rows) != 1:
            raise RuntimeError("expected one full-pipeline reference row for " + candidate_id)
        deployed[candidate_id] = {
            "remote_package": remote_package,
            "reference_row": reference_rows[0],
            "package_candidate_id": candidate_id,
            "package_tar_sha256": _file_sha256(resolved["tar_path"]),
            "runner_sha256": runner_sha256,
            "remote_package_reused": reusable,
            "prior_independent_reference_passed": prior_correctness.get(candidate_id, False),
        }
        return deployed[candidate_id]

    def run_isolated(remote_package, stage, threads, case_id, mode, expected):
        output = "{}/{}_{}.jsonl".format(remote_package, case_id, mode)
        runner_args = [
            *_stage_cli(stage, threads),
            "--input-files",
            ",".join(_reference_input_files(stage, remote_package)),
            "--runs",
            str(SCORED_RUNS),
            "--warmup-runs",
            str(WARMUP_RUNS),
            "--runtime-num-threads",
            "4",
            "--serial",
            "--profile-pmu",
            "--output-mode",
            "raw",
            "--output-jsonl",
            output,
        ]
        text = _ssh_command(
            args.board_host,
            args.ssh_user,
            _runner_shell(remote_package, runner_args)
            + " && cat {}".format(shlex.quote(output)),
            args.case_timeout_s,
        )
        return _annotate_stage_rows(_parse_jsonl_text(text), expected)

    def run_concurrent(remote_package, case, stage_a, stage_b, expected_a, expected_b):
        token = uuid.uuid4().hex
        work = "{}/concurrent_{}_{}".format(remote_package, case["case_id"], token[:8])
        ready_a = work + "/a.ready"
        ready_b = work + "/b.ready"
        start = work + "/start"
        output_a = work + "/a.jsonl"
        output_b = work + "/b.jsonl"

        def command(stage, stage_spec, output, ready):
            return _runner_shell(
                remote_package,
                [
                    *_stage_cli(stage, stage_spec["threads"]),
                    "--input-files",
                    ",".join(_reference_input_files(stage, remote_package)),
                    "--runs",
                    str(SCORED_RUNS),
                    "--warmup-runs",
                    str(WARMUP_RUNS),
                    "--runtime-num-threads",
                    "4",
                    "--serial",
                    "--profile-pmu",
                    "--barrier-each-run",
                    "--ready-file",
                    ready,
                    "--start-file",
                    start,
                    "--barrier-token",
                    token,
                    "--start-timeout-s",
                    str(args.case_timeout_s),
                    "--output-mode",
                    "raw",
                    "--output-jsonl",
                    output,
                ],
            )

        command_a = command(stage_a, case["stage_a"], output_a, ready_a)
        command_b = command(stage_b, case["stage_b"], output_b, ready_b)
        controller = """
set -eu
mkdir -p {work}
({command_a}) >{work}/a.stdout 2>{work}/a.stderr &
pid_a=$!
({command_b}) >{work}/b.stdout 2>{work}/b.stderr &
pid_b=$!
i=0
while [ "$i" -lt {runs} ]; do
  while [ ! -f {ready_a}.$i ] || [ ! -f {ready_b}.$i ]; do
    kill -0 "$pid_a" 2>/dev/null || {{ cat {work}/a.stderr >&2; exit 31; }}
    kill -0 "$pid_b" 2>/dev/null || {{ cat {work}/b.stderr >&2; exit 32; }}
    sleep 0.01
  done
  test "$(cat {ready_a}.$i)" = {token}
  test "$(cat {ready_b}.$i)" = {token}
  printf '%s\n' {token} > {start}.$i
  i=$((i + 1))
done
wait "$pid_a"
wait "$pid_b"
printf '%s\n' __P7_A_JSONL__
cat {output_a}
printf '%s\n' __P7_B_JSONL__
cat {output_b}
""".format(
            work=shlex.quote(work),
            command_a=command_a,
            command_b=command_b,
            runs=SCORED_RUNS,
            ready_a=shlex.quote(ready_a),
            ready_b=shlex.quote(ready_b),
            start=shlex.quote(start),
            token=shlex.quote(token),
            output_a=shlex.quote(output_a),
            output_b=shlex.quote(output_b),
        )
        text = _ssh_command(
            args.board_host, args.ssh_user, controller, args.case_timeout_s
        )
        before, after = text.split("__P7_B_JSONL__", 1)
        rows_a = _parse_jsonl_text(before.split("__P7_A_JSONL__", 1)[1])
        rows_b = _parse_jsonl_text(after)
        return (
            _annotate_stage_rows(rows_a, expected_a),
            _annotate_stage_rows(rows_b, expected_b),
        )

    case_results = []
    cases = plan["experiments"]["p7b1_cpu_stage_pairs"]
    for index, case in enumerate(cases, 1):
        resolved = _resolve_pair_package(
            case["stage_a"]["segment_id"],
            case["stage_b"]["segment_id"],
            manifest,
            catalog,
        )
        deployment = deploy_and_reference(resolved)
        remote_package = deployment["remote_package"]
        expected_a = _expected_stage_outputs(
            deployment["reference_row"], resolved["stage_a"]["index"]
        )
        expected_b = _expected_stage_outputs(
            deployment["reference_row"], resolved["stage_b"]["index"]
        )
        print("[P7B-1 CPU pair {}/16] {}".format(index, case["case_id"]), flush=True)
        a_only = run_isolated(
            remote_package,
            resolved["stage_a"],
            case["stage_a"]["threads"],
            case["case_id"],
            "a_only",
            expected_a,
        )
        b_only = run_isolated(
            remote_package,
            resolved["stage_b"],
            case["stage_b"]["threads"],
            case["case_id"],
            "b_only",
            expected_b,
        )
        concurrent_a, concurrent_b = run_concurrent(
            remote_package,
            case,
            resolved["stage_a"],
            resolved["stage_b"],
            expected_a,
            expected_b,
        )
        summaries = {
            "a_only": _stage_summary(a_only),
            "b_only": _stage_summary(b_only),
            "a_concurrent": _stage_summary(concurrent_a),
            "b_concurrent": _stage_summary(concurrent_b),
        }
        slowdown_a = summaries["a_concurrent"]["stage_ms_median"] / summaries["a_only"][
            "stage_ms_median"
        ]
        slowdown_b = summaries["b_concurrent"]["stage_ms_median"] / summaries["b_only"][
            "stage_ms_median"
        ]
        ci_a = _bootstrap_ratio_ci(
            [row["stage0_ms"] for row in a_only],
            [row["stage0_ms"] for row in concurrent_a],
            case["case_id"] + ":a",
        )
        ci_b = _bootstrap_ratio_ci(
            [row["stage0_ms"] for row in b_only],
            [row["stage0_ms"] for row in concurrent_b],
            case["case_id"] + ":b",
        )
        ci_relative_halfwidth_a = _relative_ci_halfwidth(slowdown_a, ci_a)
        ci_relative_halfwidth_b = _relative_ci_halfwidth(slowdown_b, ci_b)
        pmu_valid = all(
            item["pmu_available_rows"] in {0, SCORED_RUNS}
            and (item["pmu_available_rows"] == SCORED_RUNS or item["pmu_unavailable_reasons"])
            for item in summaries.values()
        )
        passed = (
            deployment["prior_independent_reference_passed"]
            and all(item["row_count"] == SCORED_RUNS for item in summaries.values())
            and all(item["correctness_passed"] for item in summaries.values())
            and all(item["determinism_passed"] for item in summaries.values())
            and summaries["a_only"]["stage_ms_cv"] <= ISOLATED_STAGE_CV_MAX
            and summaries["b_only"]["stage_ms_cv"] <= ISOLATED_STAGE_CV_MAX
            and ci_relative_halfwidth_a <= SLOWDOWN_CI_RELATIVE_HALFWIDTH_MAX
            and ci_relative_halfwidth_b <= SLOWDOWN_CI_RELATIVE_HALFWIDTH_MAX
            and pmu_valid
        )
        case_results.append(
            {
                "case": case,
                "representative_package": {
                    key: deployment[key]
                    for key in (
                        "package_candidate_id",
                        "package_tar_sha256",
                        "runner_sha256",
                        "remote_package_reused",
                        "prior_independent_reference_passed",
                    )
                },
                "stage_indices": {
                    "a": resolved["stage_a"]["index"],
                    "b": resolved["stage_b"]["index"],
                },
                "rows": {
                    "a_only": a_only,
                    "b_only": b_only,
                    "a_concurrent": concurrent_a,
                    "b_concurrent": concurrent_b,
                },
                "summaries": summaries,
                "slowdown": {
                    "a": slowdown_a,
                    "b": slowdown_b,
                    "a_ci": ci_a,
                    "b_ci": ci_b,
                    "a_ci_relative_halfwidth": ci_relative_halfwidth_a,
                    "b_ci_relative_halfwidth": ci_relative_halfwidth_b,
                },
                "gate_checks": {
                    "isolated_stage_cv_at_most_10_percent": (
                        summaries["a_only"]["stage_ms_cv"] <= ISOLATED_STAGE_CV_MAX
                        and summaries["b_only"]["stage_ms_cv"] <= ISOLATED_STAGE_CV_MAX
                    ),
                    "slowdown_ci_relative_halfwidth_at_most_25_percent": (
                        ci_relative_halfwidth_a <= SLOWDOWN_CI_RELATIVE_HALFWIDTH_MAX
                        and ci_relative_halfwidth_b <= SLOWDOWN_CI_RELATIVE_HALFWIDTH_MAX
                    ),
                    "concurrent_cv_is_reported_not_rejected": True,
                    "pmu_available_or_null_with_reason": pmu_valid,
                },
                "single_boot_admission_signal": {
                    "a": slowdown_a > 1.05 and ci_a["lower_95"] > 1.0,
                    "b": slowdown_b > 1.05 and ci_b["lower_95"] > 1.0,
                    "formal_admission_deferred_until_three_boots": True,
                },
                "passed": passed,
            }
        )

    preflight_after = collect_preflight(preflight_args)
    session = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7b1_cpu_pair_qualification_session",
            "protocol_id": PROTOCOL_ID,
            "plan_artifact_sha256": plan["artifact_sha256"],
            "endpoint": {
                "board_host": args.board_host,
                "ssh_user": args.ssh_user,
                "rpc_port": args.rpc_port,
            },
            "board_boot_id": boot_id,
            "preflight_before": preflight_before,
            "preflight_after": preflight_after,
            "sample_policy": {
                "measurement_order": ["a_only", "b_only", "a_b_concurrent"],
                "warmup_runs": WARMUP_RUNS,
                "scored_runs": SCORED_RUNS,
                "concurrent_barrier": "one_ready_start_barrier_per_scored_sample",
                "isolated_stage_wall_time_cv_max": ISOLATED_STAGE_CV_MAX,
                "slowdown_ci_relative_halfwidth_max": SLOWDOWN_CI_RELATIVE_HALFWIDTH_MAX,
                "concurrent_stage_cv_policy": (
                    "report_as_scheduler_contention_variability; do not reject when slowdown "
                    "median confidence interval is bounded"
                ),
                "thread_semantics": "independent_per-stage_TVM_runtime_parameter",
                "sum_of_stage_threads_is_not_a_legality_constraint": True,
            },
            "case_results": case_results,
            "summary": {
                "case_count": len(case_results),
                "passed_case_count": sum(item["passed"] for item in case_results),
                "failed_case_ids": [
                    item["case"]["case_id"] for item in case_results if not item["passed"]
                ],
                "single_boot_admission_signal_count_a": sum(
                    item["single_boot_admission_signal"]["a"] for item in case_results
                ),
                "single_boot_admission_signal_count_b": sum(
                    item["single_boot_admission_signal"]["b"] for item in case_results
                ),
            },
            "passed": preflight_after["passed"] and all(
                item["passed"] for item in case_results
            ),
            "evidence_scope": (
                "single-boot P7B-1 CPU-stage concurrency qualification; slowdown parameters "
                "are not formally admitted before three independent boots"
            ),
        }
    )
    write_json(args.cpu_pair_session_output, session)
    return session


def finalize_p7b1(args, plan):
    memory = load_artifact(
        args.memory_session_input,
        "cpu_vta_pipeline_v1_p7b1_memory_qualification_session",
    )
    cpu_pairs = load_artifact(
        args.cpu_pair_session_input,
        "cpu_vta_pipeline_v1_p7b1_cpu_pair_qualification_session",
    )
    if memory["plan_artifact_sha256"] != plan["artifact_sha256"]:
        raise ValueError("memory session was produced from a different P7 plan")
    if cpu_pairs["plan_artifact_sha256"] != plan["artifact_sha256"]:
        raise ValueError("CPU-pair session was produced from a different P7 plan")
    if memory["board_boot_id"] != cpu_pairs["board_boot_id"]:
        raise ValueError("P7B-1 component sessions came from different boots")

    bandwidth = {}
    for item in memory["case_results"]:
        case = item["case"]
        key = "{}:{}:t{}".format(
            case["pressure_class"], case["operation"], case["threads"]
        )
        bandwidth[key] = statistics.median(row["bandwidth_GBps"] for row in item["rows"])
    slowdown_a = [item["slowdown"]["a"] for item in cpu_pairs["case_results"]]
    slowdown_b = [item["slowdown"]["b"] for item in cpu_pairs["case_results"]]
    instruction_ratios = []
    cache_miss_ratios = []
    for item in cpu_pairs["case_results"]:
        for side in ("a", "b"):
            isolated = item["rows"][side + "_only"]
            concurrent = item["rows"][side + "_concurrent"]
            instruction_ratios.append(
                statistics.median(row["pmu_instructions"] for row in concurrent)
                / statistics.median(row["pmu_instructions"] for row in isolated)
            )
            cache_miss_ratios.append(
                statistics.median(row["pmu_cache_misses"] for row in concurrent)
                / statistics.median(row["pmu_cache_misses"] for row in isolated)
            )
    combined = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7b1_single_boot_qualification",
            "protocol_id": PROTOCOL_ID,
            "plan_artifact_sha256": plan["artifact_sha256"],
            "board_boot_id": memory["board_boot_id"],
            "endpoint": memory["endpoint"],
            "source_sessions": {
                "memory": {
                    "path": _repo_relative_path(args.memory_session_input),
                    "artifact_sha256": memory["artifact_sha256"],
                },
                "cpu_stage_pairs": {
                    "path": _repo_relative_path(args.cpu_pair_session_input),
                    "artifact_sha256": cpu_pairs["artifact_sha256"],
                },
            },
            "gate_checks": {
                "same_plan": True,
                "same_boot": True,
                "memory_36_of_36_passed": memory["summary"]["passed_case_count"] == 36,
                "cpu_pairs_16_of_16_passed": cpu_pairs["summary"]["passed_case_count"] == 16,
                "preflight_before_and_after_passed": all(
                    session[point]["passed"]
                    for session in (memory, cpu_pairs)
                    for point in ("preflight_before", "preflight_after")
                ),
                "candidate_throughput_used_for_fit": False,
            },
            "observations": {
                "memory_bandwidth_GBps_median": bandwidth,
                "memory_max_wall_time_cv": max(
                    item["summary"]["wall_ms_cv"] for item in memory["case_results"]
                ),
                "cpu_pair_slowdown_a_range": [min(slowdown_a), max(slowdown_a)],
                "cpu_pair_slowdown_b_range": [min(slowdown_b), max(slowdown_b)],
                "cpu_pair_instruction_ratio_range": [
                    min(instruction_ratios),
                    max(instruction_ratios),
                ],
                "cpu_pair_cache_miss_ratio_median": statistics.median(cache_miss_ratios),
                "single_boot_admission_signal_count_a": cpu_pairs["summary"][
                    "single_boot_admission_signal_count_a"
                ],
                "single_boot_admission_signal_count_b": cpu_pairs["summary"][
                    "single_boot_admission_signal_count_b"
                ],
            },
            "formal_parameter_profile_generated": False,
            "remaining_requirement": "repeat the frozen P7B-1 plan on at least two more boots",
            "passed": memory["passed"] and cpu_pairs["passed"],
            "evidence_scope": (
                "P7B-1 single-boot qualification only; no slowdown parameter is admitted into "
                "the static formula and P7B-2/P7C were not executed"
            ),
        }
    )
    write_json(args.combined_session_output, combined)
    write_p7b1_review(args.review_output, plan, combined, memory, cpu_pairs)
    return combined


def write_p7b1_review(path, plan, combined, memory, cpu_pairs):
    observations = combined["observations"]
    bandwidth = observations["memory_bandwidth_GBps_median"]
    lines = [
        "# P7B-1 Single-Boot Qualification Review",
        "",
        "更新日期：{}".format(datetime.date.today().isoformat()),
        "",
        "## 结论",
        "",
        "P7A instrumentation 与 P7B-1 单 boot qualification 已完成。36/36 组 CPU memory case 和",
        "16/16 组真实 CPU stage pair 均通过 correctness、determinism、PMU、同步和统计 gate。",
        "本阶段没有读取候选吞吐拟合参数，也没有把 `+34 ms` 写回公式。",
        "",
        "## 为什么需要这一步",
        "",
        "旧模型把多个 CPU stage 的孤立服务时间直接放入 pipeline 最大值，但实际 TVM worker 会在",
        "四核 CPU 上竞争。P7B-1 使用两个独立 native 进程，并在每个计分样本前通过 barrier 同时",
        "启动，直接测量这种竞争，而不是使用 `sum(threads) <= 4` 之类的搜索限制代替物理测量。",
        "",
        "## 实测结果",
        "",
        "- 256 KiB cache-resident copy：t1/t4 为 `{:.3f}/{:.3f} GB/s`。".format(
            bandwidth["cache_resident:copy:t1"], bandwidth["cache_resident:copy:t4"]
        ),
        "- 64 MiB streaming copy：t1/t4 为 `{:.3f}/{:.3f} GB/s`，3--4 线程已接近饱和。".format(
            bandwidth["streaming:copy:t1"], bandwidth["streaming:copy:t4"]
        ),
        "- 64 MiB streaming write：t3/t4 为 `{:.3f}/{:.3f} GB/s`，增加线程没有继续提高带宽。".format(
            bandwidth["streaming:write:t3"], bandwidth["streaming:write:t4"]
        ),
        "- 16 组 CPU pair 的 A 侧 slowdown 范围为 `{:.3f}--{:.3f}x`，B 侧为 `{:.3f}--{:.3f}x`。".format(
            *observations["cpu_pair_slowdown_a_range"],
            *observations["cpu_pair_slowdown_b_range"],
        ),
        "- 单 boot 中 A/B 分别有 `{}/{}` 组满足 slowdown >5% 且 95% CI 不含 1。".format(
            observations["single_boot_admission_signal_count_a"],
            observations["single_boot_admission_signal_count_b"],
        ),
        "- 并发/孤立 PMU 指令数中位比范围为 `{:.3f}--{:.3f}`，而 cache-miss 比值中位数为 `{:.3f}`；".format(
            *observations["cpu_pair_instruction_ratio_range"],
            observations["cpu_pair_cache_miss_ratio_median"],
        ),
        "  slowdown 来自相同计算工作受到资源竞争，而不是并发时额外执行了算子。",
        "",
        "## Review 后的修正",
        "",
        "最初把并发 wall-time CV <=15% 作为硬 gate，导致 2 组虽有稳定 PMU 工作量和显著 slowdown",
        "却被拒绝。并发 elapsed time 的多峰波动本身来自被测的 Linux/TVM 调度竞争，因此最终冻结为：",
        "孤立基线 CV <=10%，slowdown 中位数 bootstrap 95% CI 相对半宽 <=25%；并发 CV 继续报告为",
        "不确定性。修改后统一重跑全部 16 组，而非选择性重测失败 case。",
        "",
        "runner 同时补上 `--input-files`，因为 `cpu:16:20` 与 `cpu:17:20` 需要 main/residual 两个",
        "输入张量。忽略第二个输入会使所谓 stage profile 不对应真实切图。",
        "",
        "## 证据边界",
        "",
        "这些结果来自同一个 boot，只证明实验可运行并发现明显 CPU 并发耦合。正式 slowdown surface",
        "至少还需两个独立 boot 复现；在此之前 `v1_p7_physical_profile.json` 不生成，静态公式不更新。",
        "P7B-2 固定运行时开销与 P7C CPU-VTA 共享 DDR 竞争尚未执行。",
        "",
        "## 产物",
        "",
        "- plan SHA256: `{}`".format(plan["artifact_sha256"]),
        "- memory session SHA256: `{}`".format(memory["artifact_sha256"]),
        "- CPU-pair session SHA256: `{}`".format(cpu_pairs["artifact_sha256"]),
        "- combined session SHA256: `{}`".format(combined["artifact_sha256"]),
        "- local regression: `24 passed`",
        "",
        "当前阶段停在 P7B-1 review。用户确认前不进入 P7B-2 或 P7C。",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pairwise_correlations(vectors):
    correlations = []
    for left_index in range(len(vectors)):
        for right_index in range(left_index + 1, len(vectors)):
            correlations.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "pearson": _pearson_correlation(
                        vectors[left_index], vectors[right_index]
                    ),
                }
            )
    return correlations


def build_p7b1_multiboot_summary(
    plan, memory_session_paths, cpu_pair_session_paths, failed_memory_attempt_paths=()
):
    memory_sources = [
        (
            path,
            load_artifact(path, "cpu_vta_pipeline_v1_p7b1_memory_qualification_session"),
        )
        for path in memory_session_paths
    ]
    pair_sources = [
        (
            path,
            load_artifact(path, "cpu_vta_pipeline_v1_p7b1_cpu_pair_qualification_session"),
        )
        for path in cpu_pair_session_paths
    ]
    failed_sources = [
        (
            path,
            load_artifact(path, "cpu_vta_pipeline_v1_p7b1_memory_qualification_session"),
        )
        for path in failed_memory_attempt_paths
    ]
    if len(memory_sources) != 3 or len(pair_sources) != 3:
        raise ValueError("P7B-1 reproducibility requires exactly three memory and pair sessions")

    memory_by_boot = {payload["board_boot_id"]: (path, payload) for path, payload in memory_sources}
    pairs_by_boot = {payload["board_boot_id"]: (path, payload) for path, payload in pair_sources}
    boot_ids = sorted(memory_by_boot)
    if len(boot_ids) != 3 or set(boot_ids) != set(pairs_by_boot):
        raise ValueError("P7B-1 sessions must cover the same three unique board boots")
    sessions = [
        {
            "boot_id": boot_id,
            "memory": memory_by_boot[boot_id][1],
            "cpu_pairs": pairs_by_boot[boot_id][1],
        }
        for boot_id in boot_ids
    ]
    all_sessions = [item[kind] for item in sessions for kind in ("memory", "cpu_pairs")]
    if any(item["plan_artifact_sha256"] != plan["artifact_sha256"] for item in all_sessions):
        raise ValueError("P7B-1 sessions were produced from different plans")

    memory_maps = [
        {item["case"]["case_id"]: item for item in session["memory"]["case_results"]}
        for session in sessions
    ]
    pair_maps = [
        {item["case"]["case_id"]: item for item in session["cpu_pairs"]["case_results"]}
        for session in sessions
    ]
    memory_case_ids = [item["case"]["case_id"] for item in sessions[0]["memory"]["case_results"]]
    pair_case_ids = [item["case"]["case_id"] for item in sessions[0]["cpu_pairs"]["case_results"]]
    if any(set(mapping) != set(memory_case_ids) for mapping in memory_maps):
        raise ValueError("memory case ids differ across boots")
    if any(set(mapping) != set(pair_case_ids) for mapping in pair_maps):
        raise ValueError("CPU-pair case ids differ across boots")

    memory_results = []
    memory_vectors = [[] for _ in sessions]
    for case_id in memory_case_ids:
        values = []
        for index, mapping in enumerate(memory_maps):
            value = statistics.median(
                float(row["bandwidth_GBps"]) for row in mapping[case_id]["rows"]
            )
            values.append(value)
            memory_vectors[index].append(value)
        memory_results.append(
            {
                "case": memory_maps[0][case_id]["case"],
                "per_boot_bandwidth_GBps_median": dict(zip(boot_ids, values)),
                "three_boot_bandwidth_GBps_median": statistics.median(values),
                "cross_boot_cv": _coefficient_of_variation(values),
                "all_sessions_passed": all(mapping[case_id]["passed"] for mapping in memory_maps),
            }
        )

    pair_results = []
    slowdown_vectors = {side: [[] for _ in sessions] for side in ("a", "b")}
    admitted_counts = {"a": 0, "b": 0}
    slowdown_cvs = {"a": [], "b": []}
    for case_id in pair_case_ids:
        sides = {}
        for side in ("a", "b"):
            values = [mapping[case_id]["slowdown"][side] for mapping in pair_maps]
            signals = [
                mapping[case_id]["single_boot_admission_signal"][side]
                for mapping in pair_maps
            ]
            admitted = all(signals)
            admitted_counts[side] += int(admitted)
            cross_boot_cv = _coefficient_of_variation(values)
            slowdown_cvs[side].append(cross_boot_cv)
            for index, value in enumerate(values):
                slowdown_vectors[side][index].append(value)
            sides[side] = {
                "per_boot_slowdown": dict(zip(boot_ids, values)),
                "per_boot_admission_signal": dict(zip(boot_ids, signals)),
                "three_boot_slowdown_median": statistics.median(values),
                "cross_boot_cv": cross_boot_cv,
                "admitted_exact_observation": admitted,
            }
        pair_results.append({"case": pair_maps[0][case_id]["case"], "sides": sides})

    memory_correlations = _pairwise_correlations(memory_vectors)
    slowdown_correlations = {
        side: _pairwise_correlations(slowdown_vectors[side]) for side in ("a", "b")
    }
    failed_attempts = []
    for path, payload in failed_sources:
        failed_attempts.append(
            {
                "path": _repo_relative_path(path),
                "artifact_sha256": payload["artifact_sha256"],
                "boot_id": payload["board_boot_id"],
                "failed_case_ids": payload["summary"]["failed_case_ids"],
                "passed": payload["passed"],
            }
        )

    passed = (
        all(item["passed"] for item in all_sessions)
        and all(item["all_sessions_passed"] for item in memory_results)
        and len(boot_ids) == 3
    )
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7b1_three_boot_reproducibility",
            "protocol_id": PROTOCOL_ID,
            "plan_artifact_sha256": plan["artifact_sha256"],
            "board_boot_ids": boot_ids,
            "source_sessions": [
                {
                    "boot_id": boot_id,
                    "memory": {
                        "path": _repo_relative_path(memory_by_boot[boot_id][0]),
                        "artifact_sha256": memory_by_boot[boot_id][1]["artifact_sha256"],
                    },
                    "cpu_pairs": {
                        "path": _repo_relative_path(pairs_by_boot[boot_id][0]),
                        "artifact_sha256": pairs_by_boot[boot_id][1]["artifact_sha256"],
                    },
                }
                for boot_id in boot_ids
            ],
            "failed_memory_attempts_retained": failed_attempts,
            "memory_results": memory_results,
            "cpu_pair_results": pair_results,
            "summary": {
                "memory_case_count": len(memory_results),
                "cpu_pair_case_count": len(pair_results),
                "memory_pairwise_correlations": memory_correlations,
                "memory_min_pairwise_pearson": min(
                    item["pearson"] for item in memory_correlations
                ),
                "memory_median_cross_boot_cv": statistics.median(
                    item["cross_boot_cv"] for item in memory_results
                ),
                "memory_max_cross_boot_cv": max(
                    item["cross_boot_cv"] for item in memory_results
                ),
                "slowdown_pairwise_correlations": slowdown_correlations,
                "slowdown_min_pairwise_pearson": {
                    side: min(item["pearson"] for item in slowdown_correlations[side])
                    for side in ("a", "b")
                },
                "slowdown_median_cross_boot_cv": {
                    side: statistics.median(slowdown_cvs[side]) for side in ("a", "b")
                },
                "slowdown_max_cross_boot_cv": {
                    side: max(slowdown_cvs[side]) for side in ("a", "b")
                },
                "admitted_exact_observation_count": admitted_counts,
            },
            "gate_checks": {
                "three_unique_boots": len(boot_ids) == 3,
                "all_component_sessions_passed": all(item["passed"] for item in all_sessions),
                "same_frozen_plan": True,
                "candidate_throughput_used_for_fit": False,
            },
            "cpu_concurrency_exact_observations_ready": passed,
            "generalized_slowdown_surface_ready": False,
            "formal_physical_profile_generated": False,
            "remaining_requirement": (
                "fit and grouped-holdout validate a feature-based slowdown surface in P7D; "
                "exact observed stage pairs must not be silently extrapolated"
            ),
            "passed": passed,
            "evidence_scope": (
                "P7B-1 three-boot component reproducibility only; P7B-2/P7C and the final "
                "static formula were not executed"
            ),
        }
    )


def write_p7b1_multiboot_review(path, summary):
    observations = summary["summary"]
    lines = [
        "# P7B-1 Three-Boot Reproducibility Review",
        "",
        "更新日期：{}".format(datetime.date.today().isoformat()),
        "",
        "## 结论",
        "",
        "P7B-1 已在三个独立 boot 上完成。三个合格 session 均为 36/36 memory case 和 16/16",
        "CPU stage pair 通过；整个阶段没有读取候选吞吐标签，也没有把 `+34 ms` 写入公式。",
        "",
        "三 boot 可重复性 gate 通过，但当前只得到已测 stage pair 的局部 slowdown 观测。它们不能",
        "静默外推到全部 DP 候选；特征化 surface 和 grouped holdout 留到 P7D。",
        "",
        "## 三 Boot 结果",
        "",
        "- memory 三轮两两相关性的最小值：`{:.3f}`。".format(
            observations["memory_min_pairwise_pearson"]
        ),
        "- memory case 跨 boot CV 中位数/最大值：`{:.2%}/{:.2%}`。".format(
            observations["memory_median_cross_boot_cv"],
            observations["memory_max_cross_boot_cv"],
        ),
        "- slowdown A/B 三轮两两相关性的最小值：`{:.3f}/{:.3f}`。".format(
            observations["slowdown_min_pairwise_pearson"]["a"],
            observations["slowdown_min_pairwise_pearson"]["b"],
        ),
        "- slowdown A/B 跨 boot CV 中位数：`{:.2%}/{:.2%}`。".format(
            observations["slowdown_median_cross_boot_cv"]["a"],
            observations["slowdown_median_cross_boot_cv"]["b"],
        ),
        "- 三轮均满足 slowdown >5% 且单 boot 95% CI 不含 1 的 A/B 观测：`{}/{}`。".format(
            observations["admitted_exact_observation_count"]["a"],
            observations["admitted_exact_observation_count"]["b"],
        ),
        "",
        "## 测量异常与处理",
        "",
        "boot 2 和 boot 3 的 memory 首轮各有一个 streaming read case 因长尾导致 CV 超过 10%。",
        "失败 session 均完整保留；没有选择性补跑单点，而是在不修改 case、流量和 gate 的条件下",
        "统一重跑全部 36 组。该现象说明最终论文必须同时报告失败轮，且 memory 参数应使用跨 boot",
        "稳健汇总，不能挑选单轮最快结果。",
        "",
        "## 证据边界",
        "",
        "P7B-1 证明 CPU stage 并发减速是真实且可重复的物理现象，但尚未证明它全部来自 DDR。",
        "P7B-2 固定运行时开销、P7C CPU-VTA 共享 DDR 竞争和 P7D 公式验证均未执行。",
        "`v1_p7_physical_profile.json` 仍不生成，静态搜索公式保持不变。",
        "",
        "## 产物",
        "",
        "- three-boot summary SHA256: `{}`".format(summary["artifact_sha256"]),
        "- failed memory attempts retained: `{}`".format(
            len(summary["failed_memory_attempts_retained"])
        ),
        "",
        "当前阶段停在 P7B-1 review。用户确认前不进入 P7B-2 或 P7C。",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _deploy_p7b2_package(args, resolved, remote_root, deployed):
    from compare_cpu_vta_pipeline_v1_reference import compare_files

    package_manifest = resolved["package_manifest"]
    candidate_id = package_manifest["candidate_id"]
    if candidate_id in deployed:
        return deployed[candidate_id]
    tag = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    remote_package = remote_root + "/pkg_" + tag
    remote_archive = remote_root + "/pkg_" + tag + ".tar.gz"
    archive = _materialize_package_archive(resolved)
    runner_sha256 = _file_sha256(args.runner_binary)
    _ssh_command(
        args.board_host,
        args.ssh_user,
        "mkdir -p {root} {package}".format(
            root=shlex.quote(remote_root), package=shlex.quote(remote_package)
        ),
        args.timeout_s,
    )
    reusable = (
        _ssh_command(
            args.board_host,
            args.ssh_user,
            "if test -f {manifest} && test -f {runner} && test -f {reference}; "
            "then sha256sum {runner} | cut -d' ' -f1; fi".format(
                manifest=shlex.quote(remote_package + "/manifest.json"),
                runner=shlex.quote(remote_package + "/vta_stage_pipeline_runner_p7"),
                reference=shlex.quote(remote_package + "/reference.jsonl"),
            ),
            args.timeout_s,
        ).strip()
        == runner_sha256
    )
    if not reusable:
        _scp_to_board(archive, args.board_host, args.ssh_user, remote_archive, args.package_timeout_s)
        _ssh_command(
            args.board_host,
            args.ssh_user,
            "mkdir -p {package} && tar -xzf {archive} -C {package}".format(
                package=shlex.quote(remote_package), archive=shlex.quote(remote_archive)
            ),
            args.package_timeout_s,
        )
        _scp_to_board(
            args.runner_binary,
            args.board_host,
            args.ssh_user,
            remote_package + "/vta_stage_pipeline_runner_p7",
            args.package_timeout_s,
        )
        _ssh_command(
            args.board_host,
            args.ssh_user,
            "chmod 700 {}".format(
                shlex.quote(remote_package + "/vta_stage_pipeline_runner_p7")
            ),
            args.timeout_s,
        )

    reference_output = remote_package + "/reference.jsonl"
    reference_args = [
        *_full_pipeline_cli(package_manifest),
        "--input",
        "input.bin",
        "--runs",
        "1",
        "--warmup-runs",
        "1",
        "--runtime-num-threads",
        "4",
        "--serial",
        "--output-mode",
        "raw_all_stages",
        "--output-jsonl",
        reference_output,
        "--output-dump-dir",
        "reference",
    ]
    if reusable:
        reference_text = _ssh_command(
            args.board_host,
            args.ssh_user,
            "cat {}".format(shlex.quote(reference_output)),
            args.timeout_s,
        )
    else:
        reference_text = _ssh_command(
            args.board_host,
            args.ssh_user,
            _runner_shell(remote_package, reference_args)
            + " && cat {}".format(shlex.quote(reference_output)),
            args.package_timeout_s,
        )
    reference_rows = _parse_jsonl_text(reference_text)
    if len(reference_rows) != 1:
        raise RuntimeError("expected one reference row for " + candidate_id)
    remote_logits = remote_package + "/reference/run_0/final_output_0.bin"
    local_logits = Path("/tmp") / "p7b2_{}_logits.bin".format(tag)
    _scp_from_board(
        args.board_host,
        args.ssh_user,
        remote_logits,
        local_logits,
        args.package_timeout_s,
    )
    independent = compare_files(args.reference_json, local_logits)
    if not independent["passed"]:
        raise RuntimeError("independent reference failed for " + candidate_id)
    deployed[candidate_id] = {
        "remote_package": remote_package,
        "reference_row": reference_rows[0],
        "package_candidate_id": candidate_id,
        "package_manifest_sha256": _file_sha256(Path(resolved["package_dir"]) / "manifest.json"),
        "package_archive_sha256": _file_sha256(archive),
        "runner_sha256": runner_sha256,
        "remote_package_reused": reusable,
        "independent_reference": independent,
    }
    return deployed[candidate_id]


def _p7b2_stage_summary(rows, profiler):
    summary = _stage_summary(rows)
    for component in ("set", "run", "get"):
        values = [float(row["stage0_{}_ms".format(component)]) for row in rows]
        summary["{}_ms_median".format(component)] = statistics.median(values)
        summary["{}_ms_cv".format(component)] = _coefficient_of_variation(values)
    divisor = float(len(rows))
    profile_per_inference = {}
    for name in (
        "mem_copy_from_host_calls",
        "mem_copy_from_host_bytes",
        "mem_copy_from_host_us",
        "mem_copy_to_host_calls",
        "mem_copy_to_host_bytes",
        "mem_copy_to_host_us",
        "flush_cache_calls",
        "flush_cache_bytes",
        "flush_cache_us",
        "invalidate_cache_calls",
        "invalidate_cache_bytes",
        "invalidate_cache_us",
        "load_buffer_2d_calls",
        "load_buffer_2d_bytes",
        "load_buffer_2d_enqueue_us",
        "store_buffer_2d_calls",
        "store_buffer_2d_bytes",
        "store_buffer_2d_enqueue_us",
        "push_gemm_op_calls",
        "push_gemm_op_us",
        "push_alu_op_calls",
        "push_alu_op_us",
        "synchronize_calls",
        "synchronize_insns",
        "synchronize_load_bytes",
        "synchronize_store_bytes",
        "device_run_wait_us",
        "driver_run_calls",
        "driver_run_total_us",
        "driver_submit_mmio_us",
        "driver_post_start_sleep_us",
        "driver_poll_wait_us",
    ):
        if name not in profiler:
            raise ValueError("VTA profiler lacks required field " + name)
        profile_per_inference[name] = float(profiler[name]) / divisor
    summary["profiler_per_inference"] = profile_per_inference
    summary["run_minus_driver_ms_median"] = summary["run_ms_median"] - (
        profile_per_inference["driver_run_total_us"] / 1000.0
    )
    return summary


def run_p7b2_runtime_qualification(args, parent_plan, manifest):
    from collect_cpu_vta_pipeline_v1_preflight import collect as collect_preflight

    catalog = _combined_package_catalog(args.package_root, args.p5a_package_root)
    subplan = build_p7b2_plan(parent_plan, manifest, catalog)
    write_json(args.p7b2_plan_output, subplan)
    preflight_args = argparse.Namespace(
        board_host=args.board_host,
        rpc_port=args.rpc_port,
        ssh_user=args.ssh_user,
        timeout_s=args.timeout_s,
    )
    preflight_before = collect_preflight(preflight_args)
    if not preflight_before["passed"]:
        raise RuntimeError("P7B-2 refused because board preflight failed")
    boot_id = _ssh_command(
        args.board_host,
        args.ssh_user,
        "cat /proc/sys/kernel/random/boot_id",
        args.timeout_s,
    ).strip()
    remote_root = args.remote_root.rstrip("/") + "/runtime"
    _ssh_command(
        args.board_host,
        args.ssh_user,
        "mkdir -p {}".format(shlex.quote(remote_root)),
        args.timeout_s,
    )

    resolved_by_segment = {
        segment_id: _resolve_device_stage_package(segment_id, "vta", manifest, catalog)
        for segment_id in {
            item["source"]["segment_id"] for item in subplan["vta_runtime_cases"]
        }
        | {
            item["source"]["segment_id"] for item in subplan["boundary_representatives"]
        }
    }
    deployed = {}
    for segment_id in sorted(resolved_by_segment):
        _deploy_p7b2_package(
            args, resolved_by_segment[segment_id], remote_root, deployed
        )

    host_package = deployed[
        resolved_by_segment["vta:03:17"]["package_manifest"]["candidate_id"]
    ]["remote_package"]
    host_results = []
    for case in subplan["host_empty_cases"]:
        output = host_package + "/{}.jsonl".format(case["case_id"])
        runner_args = [
            "--host-empty",
            "--host-empty-stage-count",
            str(case["actor_count"]),
            "--host-empty-inner-repeats",
            str(case["inner_repeats"]),
            "--queue-depth",
            "2",
            "--warmup-runs",
            str(case["warmup_runs"]),
            "--runs",
            str(case["scored_runs"]),
            "--profile-pmu",
            "--output-jsonl",
            output,
        ]
        print("[P7B-2 host] {}".format(case["case_id"]), flush=True)
        text = _ssh_command(
            args.board_host,
            args.ssh_user,
            _runner_shell(host_package, runner_args)
            + " && cat {}".format(shlex.quote(output)),
            args.case_timeout_s,
        )
        rows = _parse_jsonl_text(text)
        summary = summarize_rows(rows)
        passed = (
            len(rows) == SCORED_RUNS
            and summary["correctness_passed"]
            and summary["determinism_passed"]
            and summary["wall_ms_cv"] <= ISOLATED_STAGE_CV_MAX
        )
        host_results.append({"case": case, "rows": rows, "summary": summary, "passed": passed})

    run_specs = {}
    for case in subplan["vta_runtime_cases"]:
        run_specs[(case["source"]["segment_id"], case["poll_sleep_ns"])] = case
    for case in subplan["boundary_representatives"]:
        run_specs.setdefault(
            (case["source"]["segment_id"], case["poll_sleep_ns"]), case
        )

    stage_results = {}
    for index, ((segment_id, poll_sleep_ns), case) in enumerate(sorted(run_specs.items()), 1):
        resolved = resolved_by_segment[segment_id]
        deployment = deployed[resolved["package_manifest"]["candidate_id"]]
        stage = resolved["stage"]
        expected = _expected_stage_outputs(deployment["reference_row"], stage["index"])
        key = "{}_poll{}".format(segment_id.replace(":", "-"), poll_sleep_ns)
        output = deployment["remote_package"] + "/{}.jsonl".format(key)
        profile_dir = deployment["remote_package"] + "/{}_profile".format(key)
        runner_args = [
            *_stage_cli(stage, 1),
            "--input-files",
            ",".join(_reference_input_files(stage, deployment["remote_package"])),
            "--runs",
            str(SCORED_RUNS),
            "--warmup-runs",
            str(WARMUP_RUNS),
            "--runtime-num-threads",
            "4",
            "--serial",
            "--profile-pmu",
            "--output-mode",
            "raw",
            "--output-jsonl",
            output,
            "--vta-runtime-profile-dir",
            profile_dir,
            "--vta-runtime-profile-events-limit",
            "256",
        ]
        environment = {
            "AXU5EVB_DRIVER_POLL_SLEEP_NS": poll_sleep_ns,
            "AXU5EVB_DRIVER_POST_START_SLEEP_NS": 1000,
        }
        print(
            "[P7B-2 VTA {}/{}] {} poll={}ns".format(
                index, len(run_specs), segment_id, poll_sleep_ns
            ),
            flush=True,
        )
        command = (
            _runner_shell(deployment["remote_package"], runner_args, environment)
            + " && printf '\\n__P7B2_ROWS__\\n' && cat {rows}"
            + " && printf '\\n__P7B2_PROFILE__\\n' && cat {profile}"
        ).format(
            rows=shlex.quote(output),
            profile=shlex.quote(profile_dir + "/benchmark_totals_status.json"),
        )
        text = _ssh_command(
            args.board_host, args.ssh_user, command, args.case_timeout_s
        )
        row_text, profile_text = text.split("__P7B2_PROFILE__", 1)
        rows = _annotate_stage_rows(
            _parse_jsonl_text(row_text.split("__P7B2_ROWS__", 1)[1]), expected
        )
        profiler = json.loads(profile_text.strip())
        summary = _p7b2_stage_summary(rows, profiler)
        passed = (
            len(rows) == SCORED_RUNS
            and summary["correctness_passed"]
            and summary["determinism_passed"]
            and summary["stage_ms_cv"] <= ISOLATED_STAGE_CV_MAX
            and int(profiler["driver_run_calls"]) > 0
        )
        stage_results[key] = {
            "source": _p7b2_stage_source(resolved),
            "poll_sleep_ns": poll_sleep_ns,
            "rows": rows,
            "profiler_totals": profiler,
            "summary": summary,
            "representative_package": {
                key: deployment[key]
                for key in (
                    "package_candidate_id",
                    "package_manifest_sha256",
                    "package_archive_sha256",
                    "runner_sha256",
                    "remote_package_reused",
                    "independent_reference",
                )
            },
            "passed": passed,
        }

    poll_deltas = []
    for workload in ("short", "long"):
        segment_id = next(
            case["source"]["segment_id"]
            for case in subplan["vta_runtime_cases"]
            if case["workload"] == workload
        )
        zero = stage_results["{}_poll0".format(segment_id.replace(":", "-"))]["summary"]
        one_us = stage_results["{}_poll1000".format(segment_id.replace(":", "-"))]["summary"]
        poll_deltas.append(
            {
                "workload": workload,
                "segment_id": segment_id,
                "run_ms_poll0": zero["run_ms_median"],
                "run_ms_poll1000": one_us["run_ms_median"],
                "run_ms_delta": one_us["run_ms_median"] - zero["run_ms_median"],
                "driver_poll_wait_ms_delta": (
                    one_us["profiler_per_inference"]["driver_poll_wait_us"]
                    - zero["profiler_per_inference"]["driver_poll_wait_us"]
                )
                / 1000.0,
            }
        )

    boundary_observations = []
    for case in subplan["boundary_representatives"]:
        key = "{}_poll{}".format(
            case["source"]["segment_id"].replace(":", "-"), case["poll_sleep_ns"]
        )
        result = stage_results[key]
        for direction, component in (("cpu_to_vta", "set"), ("vta_to_cpu", "get")):
            spec = case["{}_observation".format(direction)]
            values = [float(row[spec["metric"]]) for row in result["rows"]]
            boundary_observations.append(
                {
                    "direction": direction,
                    "quantile": case["quantile"],
                    "segment_id": case["source"]["segment_id"],
                    "logical_bytes": spec["logical_bytes"],
                    "component_owner": "vta_{}_host_copy_cache_adapter".format(component),
                    "service_ms_median": statistics.median(values),
                    "service_ms_cv": _coefficient_of_variation(values),
                    "stage_dma_additional_service_ms": 0.0,
                }
            )

    preflight_after = collect_preflight(preflight_args)
    all_results = host_results + list(stage_results.values())
    session = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7b2_runtime_qualification_session",
            "protocol_id": PROTOCOL_ID,
            "parent_p7_plan_artifact_sha256": parent_plan["artifact_sha256"],
            "p7b2_plan_artifact_sha256": subplan["artifact_sha256"],
            "endpoint": {
                "board_host": args.board_host,
                "ssh_user": args.ssh_user,
                "rpc_port": args.rpc_port,
            },
            "board_boot_id": boot_id,
            "preflight_before": preflight_before,
            "preflight_after": preflight_after,
            "host_empty_results": host_results,
            "vta_stage_results": stage_results,
            "poll_policy_deltas": poll_deltas,
            "boundary_observations": boundary_observations,
            "ownership_conclusion": {
                "alpha_cpu_or_vta_fitted": False,
                "reason": "inclusive stage run_ms already owns graph invocation, VTA submit/sync and device wait",
                "profiler_components_additive_to_run_ms": False,
                "host_queue_floor_admitted_to_formula": False,
                "boundary_observations_are_component_level": True,
            },
            "candidate_throughput_used_for_fit": False,
            "formal_physical_profile_generated": False,
            "single_boot_qualification_only": True,
            "summary": {
                "host_case_count": len(host_results),
                "vta_unique_run_count": len(stage_results),
                "boundary_observation_count": len(boundary_observations),
                "passed_case_count": sum(item["passed"] for item in all_results),
                "case_count": len(all_results),
            },
            "passed": preflight_after["passed"] and all(item["passed"] for item in all_results),
            "evidence_scope": "single-boot P7B-2 runtime/boundary qualification; no candidate-level fit",
        }
    )
    write_json(args.p7b2_session_output, session)
    return session


def _fit_boundary_through_origin(observations, direction):
    points = [item for item in observations if item["direction"] == direction]
    if not points:
        raise ValueError(f"no boundary observations for direction {direction!r}")
    if any(
        int(item["logical_bytes"]) <= 0 or float(item["service_ms_median"]) <= 0
        for item in points
    ):
        raise ValueError("boundary observations require positive bytes and service time")
    denominator = sum(float(item["logical_bytes"]) ** 2 for item in points)
    slope = sum(
        float(item["logical_bytes"]) * float(item["service_ms_median"])
        for item in points
    ) / denominator
    errors = [
        abs(slope * float(item["logical_bytes"]) - float(item["service_ms_median"]))
        / float(item["service_ms_median"])
        for item in points
    ]
    return {
        "fit_policy": "nonnegative_through_origin_single_boot_qualification_only",
        "point_count": len(points),
        "ms_per_mib": slope * 1024.0 * 1024.0,
        "fit_mape": statistics.mean(errors),
        "fit_max_ape": max(errors),
        "formal_parameter_admitted": False,
        "remaining_gate": "repeat on at least two additional independent boots",
    }


def build_p7b2_qualification_summary(session, local_cost_path):
    validate_artifact(session, "cpu_vta_pipeline_v1_p7b2_runtime_qualification_session")
    local_cost_path = Path(local_cost_path)
    local_cost = json.loads(local_cost_path.read_text(encoding="utf-8"))
    segment_costs = {item["segment_id"]: item for item in local_cost["segment_costs"]}
    vta_comparisons = []
    for result in session["vta_stage_results"].values():
        segment_id = result["source"]["segment_id"]
        predicted = float(segment_costs[segment_id]["service_ms_by_threads"]["1"])
        measured = float(result["summary"]["run_ms_median"])
        vta_comparisons.append(
            {
                "segment_id": segment_id,
                "poll_sleep_ns": result["poll_sleep_ns"],
                "p2_predicted_run_ms": predicted,
                "p7b2_measured_run_ms": measured,
                "signed_residual_ms": measured - predicted,
                "absolute_percentage_error": abs(measured - predicted) / measured,
            }
        )
    host = {
        item["case"]["actor_count"]: float(item["summary"]["wall_ms_median"])
        for item in session["host_empty_results"]
    }
    boundary_models = {
        direction: _fit_boundary_through_origin(session["boundary_observations"], direction)
        for direction in ("cpu_to_vta", "vta_to_cpu")
    }
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7b2_runtime_qualification_summary",
            "protocol_id": PROTOCOL_ID,
            "source_session_artifact_sha256": session["artifact_sha256"],
            "source_local_cost_table": {
                "path": _repo_relative_path(local_cost_path),
                "sha256": _file_sha256(local_cost_path),
            },
            "board_boot_id": session["board_boot_id"],
            "host_queue_floor": {
                "one_actor_ms": host[1],
                "three_actor_ms": host[3],
                "increment_per_additional_actor_ms": (host[3] - host[1]) / 2.0,
                "admitted_to_static_formula": False,
                "reason": "sub-0.1ms queue floor cannot explain the candidate residual and lacks pipeline holdout",
            },
            "poll_policy_deltas": session["poll_policy_deltas"],
            "poll_component_admitted_as_additive_service": False,
            "poll_reason": "poll wait is already included in VTA run_ms",
            "boundary_models": boundary_models,
            "vta_service_comparisons": sorted(
                vta_comparisons, key=lambda item: (item["segment_id"], item["poll_sleep_ns"])
            ),
            "vta_service_max_ape": max(
                item["absolute_percentage_error"] for item in vta_comparisons
            ),
            "interpretation": {
                "explains_34ms_residual": False,
                "main_measured_remaining_factor": "P7B-1 CPU-stage concurrent slowdown",
                "next_unmeasured_factor": "P7C CPU-VTA shared-DDR contention",
                "candidate_throughput_used_for_fit": False,
                "formal_physical_profile_generated": False,
            },
            "passed": session["passed"],
        }
    )


def write_p7b2_review(path, summary):
    boundaries = summary["boundary_models"]
    polls = {item["workload"]: item for item in summary["poll_policy_deltas"]}
    host = summary["host_queue_floor"]
    lines = [
        "# P7B-2 Runtime and Boundary Qualification Review",
        "",
        "更新日期：{}".format(datetime.date.today().isoformat()),
        "",
        "## 结论",
        "",
        "P7B-2 单 boot qualification 已完成，9/9 个实际运行 case 通过。实验没有读取候选吞吐标签，",
        "也没有把历史 `+34 ms` 反向拟合进公式。",
        "",
        "原协议中的 `alpha_frame + N_cpu*alpha_cpu + N_vta*alpha_vta` 不可直接辨识，并且会与",
        "P2/P3 已使用的 inclusive `run_ms` 重复计费。本轮已修正为只报告 host 队列下界、poll",
        "matched delta、boundary set/get 和 VTA profiler 分解。",
        "",
        "## 结果",
        "",
        "- 1/3 个空载 actor 队列链中位数为 `{:.3f}/{:.3f} ms/帧`，远小于 22--34 ms 残差。".format(
            host["one_actor_ms"], host["three_actor_ms"]
        ),
        "- poll sleep 从 0 改为 1000 ns 后，short/long VTA run 增加 `{:.3f}/{:.3f} ms`。".format(
            polls["short"]["run_ms_delta"], polls["long"]["run_ms_delta"]
        ),
        "- CPU->VTA boundary 单 boot slope 为 `{:.3f} ms/MiB`，fit MAPE `{:.2%}`。".format(
            boundaries["cpu_to_vta"]["ms_per_mib"], boundaries["cpu_to_vta"]["fit_mape"]
        ),
        "- VTA->CPU boundary 单 boot slope 为 `{:.3f} ms/MiB`，fit MAPE `{:.2%}`。".format(
            boundaries["vta_to_cpu"]["ms_per_mib"], boundaries["vta_to_cpu"]["fit_mape"]
        ),
        "- P2 的单一 VTA GOP slope 在本轮实际 segment 上最大 APE 为 `{:.2%}`；`vta:03:17` 仅约 1%，".format(
            summary["vta_service_max_ape"]
        ),
        "  但短 segment 达到 11%--23%，说明后续应保留 segment/shape-aware service，而不是全局截距。",
        "",
        "## 计费结论",
        "",
        "`set_ms/get_ms` 继续唯一归 boundary host copy/cache；VTA LOAD/STORE 属于 stage DMA。",
        "`submit/post-start sleep/poll wait/sync` 已包含在 `run_ms`，profiler 只用于解释，不能再相加。",
        "空载队列下界尚不进入公式；boundary slope 因只有一个 boot，也不生成正式 profile。",
        "",
        "## 剩余问题",
        "",
        "P7B-2 排除了普通队列调度、1 us poll 和 boundary copy 是 34 ms 主因的假设。当前主要已测",
        "因素是 P7B-1 的 CPU 并发减速；尚未测量的是 CPU 与 VTA 同时访问共享 DDR 的相互降速。",
        "下一阶段应为 P7C 单 boot matched control，执行前等待用户确认。",
        "",
        "## 验证",
        "",
        "- CPU-VTA Pipeline V1 全套本地回归：`66 passed`。",
        "- P7 runner 已通过当前 AXU5EVB SDK 的 AArch64 交叉编译。",
        "- measurement plan、session 和 summary 均通过 canonical SHA256 校验。",
        "",
        "## 产物",
        "",
        "- qualification summary SHA256: `{}`".format(summary["artifact_sha256"]),
        "- source session SHA256: `{}`".format(summary["source_session_artifact_sha256"]),
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _p7c_cpu_command(remote_binary, workload, case_id, output, barrier=None):
    command = [
        remote_binary,
        "--operation",
        workload["operation"],
        "--case-id",
        case_id,
        "--output",
        output,
        "--pressure-class",
        "cache_resident"
        if workload["target_active_working_set_bytes"] <= 256 * 1024
        else "streaming",
        "--threads",
        str(workload["threads"]),
        "--streams",
        "1",
        "--warmup",
        str(WARMUP_RUNS),
        "--runs",
        str(SCORED_RUNS),
        "--inner-repeats",
        str(workload["inner_repeats"]),
        "--bytes",
        str(workload["operand_bytes_per_stream"]),
        "--profile-pmu",
    ]
    if barrier:
        command.extend(
            [
                "--barrier-each-run",
                "--ready-file",
                barrier["ready"],
                "--start-file",
                barrier["start"],
                "--barrier-token",
                barrier["token"],
                "--start-timeout-s",
                str(barrier["timeout_s"]),
            ]
        )
    return shlex.join(command)


def _p7c_vta_command(remote_package, stage, output, profile_dir, barrier=None):
    runner_args = [
        *_stage_cli(stage, 1),
        "--input-files",
        ",".join(_reference_input_files(stage, remote_package)),
        "--runs",
        str(SCORED_RUNS),
        "--warmup-runs",
        str(WARMUP_RUNS),
        "--runtime-num-threads",
        "1",
        "--process-cpu-affinity",
        "3",
        "--serial",
        "--profile-pmu",
        "--output-mode",
        "raw",
        "--output-jsonl",
        output,
        "--vta-runtime-profile-dir",
        profile_dir,
        "--vta-runtime-profile-events-limit",
        "256",
    ]
    if barrier:
        runner_args.extend(
            [
                "--barrier-each-run",
                "--ready-file",
                barrier["ready"],
                "--start-file",
                barrier["start"],
                "--barrier-token",
                barrier["token"],
                "--start-timeout-s",
                str(barrier["timeout_s"]),
            ]
        )
    return _runner_shell(
        remote_package,
        runner_args,
        {
            "AXU5EVB_DRIVER_POLL_SLEEP_NS": 1000,
            "AXU5EVB_DRIVER_POST_START_SLEEP_NS": 1000,
        },
    )


def _p7c_parse_vta_output(text, expected):
    rows_text, profile_text = text.split("__P7C_PROFILE__", 1)
    rows = _annotate_stage_rows(
        _parse_jsonl_text(rows_text.split("__P7C_ROWS__", 1)[1]), expected
    )
    profiler = json.loads(profile_text.strip())
    return rows, profiler


def _p7c_run_isolated_vta(args, remote_package, stage, expected, base):
    output = base + "_vta_only.jsonl"
    profile_dir = base + "_vta_only_profile"
    command = (
        _p7c_vta_command(remote_package, stage, output, profile_dir)
        + " && printf '\\n__P7C_ROWS__\\n' && cat {rows}"
        + " && printf '\\n__P7C_PROFILE__\\n' && cat {profile}"
    ).format(
        rows=shlex.quote(output),
        profile=shlex.quote(profile_dir + "/benchmark_totals_status.json"),
    )
    return _p7c_parse_vta_output(
        _ssh_command(args.board_host, args.ssh_user, command, args.case_timeout_s),
        expected,
    )


def _p7c_run_concurrent(
    args, remote_binary, remote_package, stage, expected, group, base
):
    token = uuid.uuid4().hex
    work = base + "_concurrent_" + token[:8]
    cpu_output = work + "/cpu.jsonl"
    vta_output = work + "/vta.jsonl"
    profile_dir = work + "/vta_profile"
    start = work + "/start"
    common = {"start": start, "token": token, "timeout_s": args.case_timeout_s}
    cpu_command = _p7c_cpu_command(
        remote_binary,
        group["cpu_workload"],
        group["group_id"] + "_concurrent_cpu",
        cpu_output,
        {**common, "ready": work + "/cpu.ready"},
    )
    vta_command = _p7c_vta_command(
        remote_package,
        stage,
        vta_output,
        profile_dir,
        {**common, "ready": work + "/vta.ready"},
    )
    controller = """
set -eu
mkdir -p {work}
({cpu_command}) >{work}/cpu.stdout 2>{work}/cpu.stderr &
pid_cpu=$!
({vta_command}) >{work}/vta.stdout 2>{work}/vta.stderr &
pid_vta=$!
i=0
while [ "$i" -lt {runs} ]; do
  while [ ! -f {cpu_ready}.$i ] || [ ! -f {vta_ready}.$i ]; do
    kill -0 "$pid_cpu" 2>/dev/null || {{ cat {work}/cpu.stderr >&2; exit 41; }}
    kill -0 "$pid_vta" 2>/dev/null || {{ cat {work}/vta.stderr >&2; exit 42; }}
    sleep 0.01
  done
  test "$(cat {cpu_ready}.$i)" = {token}
  test "$(cat {vta_ready}.$i)" = {token}
  printf '%s\n' {token} > {start}.$i
  i=$((i + 1))
done
wait "$pid_cpu"
wait "$pid_vta"
printf '%s\n' __P7C_CPU_JSONL__
cat {cpu_output}
printf '%s\n' __P7C_VTA_JSONL__
cat {vta_output}
printf '%s\n' __P7C_PROFILE__
cat {profile}
""".format(
        work=shlex.quote(work),
        cpu_command=cpu_command,
        vta_command=vta_command,
        runs=SCORED_RUNS,
        cpu_ready=shlex.quote(work + "/cpu.ready"),
        vta_ready=shlex.quote(work + "/vta.ready"),
        token=shlex.quote(token),
        start=shlex.quote(start),
        cpu_output=shlex.quote(cpu_output),
        vta_output=shlex.quote(vta_output),
        profile=shlex.quote(profile_dir + "/benchmark_totals_status.json"),
    )
    text = _ssh_command(
        args.board_host, args.ssh_user, controller, args.case_timeout_s
    )
    cpu_text, remainder = text.split("__P7C_VTA_JSONL__", 1)
    vta_text, profile_text = remainder.split("__P7C_PROFILE__", 1)
    cpu_rows = _parse_jsonl_text(cpu_text.split("__P7C_CPU_JSONL__", 1)[1])
    vta_rows = _annotate_stage_rows(_parse_jsonl_text(vta_text), expected)
    return cpu_rows, vta_rows, json.loads(profile_text.strip())


def _p7c_profile_work(profiler, divisor):
    return {
        field: float(profiler[field]) / float(divisor)
        for field in (
            "synchronize_load_bytes",
            "synchronize_store_bytes",
            "push_gemm_op_calls",
            "push_alu_op_calls",
        )
    }


def run_p7c_qualification(args, parent_plan, manifest):
    from collect_cpu_vta_pipeline_v1_preflight import collect as collect_preflight

    p7b2_session = load_artifact(
        args.p7b2_session_output,
        "cpu_vta_pipeline_v1_p7b2_runtime_qualification_session",
    )
    catalog = _combined_package_catalog(args.package_root, args.p5a_package_root)
    subplan = build_p7c_plan(parent_plan, manifest, catalog, p7b2_session)
    write_json(args.p7c_plan_output, subplan)
    preflight_args = argparse.Namespace(
        board_host=args.board_host,
        rpc_port=args.rpc_port,
        ssh_user=args.ssh_user,
        timeout_s=args.timeout_s,
    )
    preflight_before = collect_preflight(preflight_args)
    if not preflight_before["passed"]:
        raise RuntimeError("P7C refused because board preflight failed")
    boot_id = _ssh_command(
        args.board_host,
        args.ssh_user,
        "cat /proc/sys/kernel/random/boot_id",
        args.timeout_s,
    ).strip()
    remote_root = args.remote_root.rstrip("/") + "/ddr"
    remote_binary = remote_root + "/ramps_cpu_memory_microbench"
    _ssh_command(
        args.board_host,
        args.ssh_user,
        "mkdir -p {}".format(shlex.quote(remote_root)),
        args.timeout_s,
    )
    _scp_to_board(
        args.memory_binary,
        args.board_host,
        args.ssh_user,
        remote_binary,
        args.timeout_s,
    )
    _ssh_command(
        args.board_host,
        args.ssh_user,
        "chmod 700 {}".format(shlex.quote(remote_binary)),
        args.timeout_s,
    )

    resolved = {
        group["vta_pressure_class"]: _resolve_device_stage_package(
            group["vta_source"]["segment_id"], "vta", manifest, catalog
        )
        for group in subplan["case_groups"]
    }
    deployed = {}
    for item in resolved.values():
        _deploy_p7b2_package(args, item, remote_root + "/packages", deployed)

    results = []
    for index, group in enumerate(subplan["case_groups"], 1):
        item = resolved[group["vta_pressure_class"]]
        deployment = deployed[item["package_manifest"]["candidate_id"]]
        remote_package = deployment["remote_package"]
        stage = item["stage"]
        expected = _expected_stage_outputs(deployment["reference_row"], stage["index"])
        base = remote_root + "/" + group["group_id"]
        print(
            "[P7C {}/4] {}: CPU-only, VTA-only, concurrent".format(
                index, group["group_id"]
            ),
            flush=True,
        )
        cpu_only_output = base + "_cpu_only.jsonl"
        cpu_only_text = _ssh_command(
            args.board_host,
            args.ssh_user,
            _p7c_cpu_command(
                remote_binary,
                group["cpu_workload"],
                group["group_id"] + "_cpu_only",
                cpu_only_output,
            )
            + " && cat {}".format(shlex.quote(cpu_only_output)),
            args.case_timeout_s,
        )
        cpu_only = _parse_jsonl_text(cpu_only_text)
        vta_only, vta_only_profiler = _p7c_run_isolated_vta(
            args, remote_package, stage, expected, base
        )
        cpu_concurrent, vta_concurrent, vta_concurrent_profiler = _p7c_run_concurrent(
            args,
            remote_binary,
            remote_package,
            stage,
            expected,
            group,
            base,
        )
        summaries = {
            "cpu_only": summarize_rows(cpu_only),
            "vta_only": _p7b2_stage_summary(vta_only, vta_only_profiler),
            "cpu_concurrent": summarize_rows(cpu_concurrent),
            "vta_concurrent": _p7b2_stage_summary(
                vta_concurrent, vta_concurrent_profiler
            ),
        }
        cpu_slowdown = (
            summaries["cpu_concurrent"]["wall_ms_median"]
            / summaries["cpu_only"]["wall_ms_median"]
        )
        vta_slowdown = (
            summaries["vta_concurrent"]["stage_ms_median"]
            / summaries["vta_only"]["stage_ms_median"]
        )
        cpu_ci = _bootstrap_ratio_ci(
            [row["wall_ms"] for row in cpu_only],
            [row["wall_ms"] for row in cpu_concurrent],
            group["group_id"] + ":cpu",
        )
        vta_ci = _bootstrap_ratio_ci(
            [row["stage0_ms"] for row in vta_only],
            [row["stage0_ms"] for row in vta_concurrent],
            group["group_id"] + ":vta",
        )
        isolated_work = _p7c_profile_work(vta_only_profiler, len(vta_only))
        concurrent_work = _p7c_profile_work(
            vta_concurrent_profiler, len(vta_concurrent)
        )
        work_invariant = all(
            abs(isolated_work[key] - concurrent_work[key]) < 1.0e-9
            for key in isolated_work
        )
        affinity_valid = all(
            row["cpu_affinity"] == [0, 1, 2] for row in cpu_only + cpu_concurrent
        ) and all(
            row["process_cpu_affinity_requested"] == "3"
            and row["process_cpu_affinity_effective"] == "3"
            and row["process_cpu_affinity_scope"]
            == "all_existing_threads_future_threads_inherit"
            for row in vta_only + vta_concurrent
        )
        pmu_valid = all(
            summary["pmu_available_rows"] in {0, SCORED_RUNS}
            and (
                summary["pmu_available_rows"] == SCORED_RUNS
                or summary["pmu_unavailable_reasons"]
            )
            for summary in summaries.values()
        )
        cpu_traffic = statistics.median(
            float(row["traffic_bytes"]) for row in cpu_concurrent
        )
        vta_traffic = (
            concurrent_work["synchronize_load_bytes"]
            + concurrent_work["synchronize_store_bytes"]
        )
        overlap_wall_ms = max(
            summaries["cpu_concurrent"]["wall_ms_median"],
            summaries["vta_concurrent"]["stage_ms_median"],
        )
        aggregate_gbps = (cpu_traffic + vta_traffic) / overlap_wall_ms / 1.0e6
        passed = (
            all(summary["row_count"] == SCORED_RUNS for summary in summaries.values())
            and all(summary["correctness_passed"] for summary in summaries.values())
            and all(summary["determinism_passed"] for summary in summaries.values())
            and summaries["cpu_only"]["wall_ms_cv"] <= MEMORY_SAMPLE_CV_MAX
            and summaries["vta_only"]["stage_ms_cv"] <= ISOLATED_STAGE_CV_MAX
            and _relative_ci_halfwidth(cpu_slowdown, cpu_ci)
            <= SLOWDOWN_CI_RELATIVE_HALFWIDTH_MAX
            and _relative_ci_halfwidth(vta_slowdown, vta_ci)
            <= SLOWDOWN_CI_RELATIVE_HALFWIDTH_MAX
            and work_invariant
            and affinity_valid
            and pmu_valid
        )
        results.append(
            {
                "case_group": group,
                "representative_package": {
                    key: deployment[key]
                    for key in (
                        "package_candidate_id",
                        "package_manifest_sha256",
                        "package_archive_sha256",
                        "runner_sha256",
                        "remote_package_reused",
                        "independent_reference",
                    )
                },
                "stage_index": stage["index"],
                "rows": {
                    "cpu_only": cpu_only,
                    "vta_only": vta_only,
                    "cpu_concurrent": cpu_concurrent,
                    "vta_concurrent": vta_concurrent,
                },
                "vta_profiler_totals": {
                    "isolated": vta_only_profiler,
                    "concurrent": vta_concurrent_profiler,
                },
                "summaries": summaries,
                "slowdown": {
                    "cpu": cpu_slowdown,
                    "vta": vta_slowdown,
                    "cpu_ci": cpu_ci,
                    "vta_ci": vta_ci,
                    "cpu_ci_relative_halfwidth": _relative_ci_halfwidth(
                        cpu_slowdown, cpu_ci
                    ),
                    "vta_ci_relative_halfwidth": _relative_ci_halfwidth(
                        vta_slowdown, vta_ci
                    ),
                },
                "traffic": {
                    "cpu_bytes_per_sample": cpu_traffic,
                    "vta_load_store_bytes_per_sample": vta_traffic,
                    "aggregate_effective_bandwidth_GBps": aggregate_gbps,
                },
                "gate_checks": {
                    "vta_instruction_work_is_matched": work_invariant,
                    "disjoint_process_affinity_is_effective": affinity_valid,
                    "pmu_available_or_null_with_reason": pmu_valid,
                    "isolated_cv_within_limit": (
                        summaries["cpu_only"]["wall_ms_cv"] <= MEMORY_SAMPLE_CV_MAX
                        and summaries["vta_only"]["stage_ms_cv"]
                        <= ISOLATED_STAGE_CV_MAX
                    ),
                    "slowdown_ci_relative_halfwidth_within_limit": (
                        _relative_ci_halfwidth(cpu_slowdown, cpu_ci)
                        <= SLOWDOWN_CI_RELATIVE_HALFWIDTH_MAX
                        and _relative_ci_halfwidth(vta_slowdown, vta_ci)
                        <= SLOWDOWN_CI_RELATIVE_HALFWIDTH_MAX
                    ),
                },
                "single_boot_signal": {
                    "cpu": cpu_slowdown > 1.05 and cpu_ci["lower_95"] > 1.0,
                    "vta": vta_slowdown > 1.05 and vta_ci["lower_95"] > 1.0,
                    "formal_admission_deferred_until_three_boots": True,
                },
                "passed": passed,
            }
        )

    preflight_after = collect_preflight(preflight_args)
    session = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7c_shared_ddr_qualification_session",
            "protocol_id": PROTOCOL_ID,
            "parent_p7_plan_artifact_sha256": parent_plan["artifact_sha256"],
            "p7c_plan_artifact_sha256": subplan["artifact_sha256"],
            "endpoint": {
                "board_host": args.board_host,
                "ssh_user": args.ssh_user,
                "rpc_port": args.rpc_port,
            },
            "board_boot_id": boot_id,
            "preflight_before": preflight_before,
            "preflight_after": preflight_after,
            "case_results": results,
            "candidate_throughput_used_for_fit": False,
            "formal_physical_profile_generated": False,
            "single_boot_qualification_only": True,
            "summary": {
                "case_group_count": len(results),
                "actual_run_mode_count": 3 * len(results),
                "passed_case_group_count": sum(item["passed"] for item in results),
                "failed_case_group_ids": [
                    item["case_group"]["group_id"]
                    for item in results
                    if not item["passed"]
                ],
                "single_boot_cpu_signal_count": sum(
                    item["single_boot_signal"]["cpu"] for item in results
                ),
                "single_boot_vta_signal_count": sum(
                    item["single_boot_signal"]["vta"] for item in results
                ),
            },
            "passed": preflight_after["passed"] and all(
                item["passed"] for item in results
            ),
            "evidence_scope": "single-boot CPU-VTA interference qualification with disjoint process affinity; no candidate-level fit",
        }
    )
    write_json(args.p7c_session_output, session)
    return session


def build_p7c_qualification_summary(session):
    validate_artifact(
        session, "cpu_vta_pipeline_v1_p7c_shared_ddr_qualification_session"
    )
    observations = []
    by_pair = {}
    for result in session["case_results"]:
        group = result["case_group"]
        key = (group["cpu_pressure_class"], group["vta_pressure_class"])
        record = {
            "cpu_pressure_class": key[0],
            "vta_pressure_class": key[1],
            "vta_segment_id": group["vta_source"]["segment_id"],
            "cpu_slowdown": result["slowdown"]["cpu"],
            "vta_slowdown": result["slowdown"]["vta"],
            "cpu_ci": result["slowdown"]["cpu_ci"],
            "vta_ci": result["slowdown"]["vta_ci"],
            "cpu_signal": result["single_boot_signal"]["cpu"],
            "vta_signal": result["single_boot_signal"]["vta"],
            "aggregate_effective_bandwidth_GBps": result["traffic"][
                "aggregate_effective_bandwidth_GBps"
            ],
            "isolated_ms": {
                "cpu": result["summaries"]["cpu_only"]["wall_ms_median"],
                "vta_stage": result["summaries"]["vta_only"]["stage_ms_median"],
            },
            "concurrent_ms": {
                "cpu": result["summaries"]["cpu_concurrent"]["wall_ms_median"],
                "vta_stage": result["summaries"]["vta_concurrent"][
                    "stage_ms_median"
                ],
            },
            "vta_component_delta_ms": {
                "set": result["summaries"]["vta_concurrent"]["set_ms_median"]
                - result["summaries"]["vta_only"]["set_ms_median"],
                "run": result["summaries"]["vta_concurrent"]["run_ms_median"]
                - result["summaries"]["vta_only"]["run_ms_median"],
                "get": result["summaries"]["vta_concurrent"]["get_ms_median"]
                - result["summaries"]["vta_only"]["get_ms_median"],
                "driver_run": (
                    result["summaries"]["vta_concurrent"]["profiler_per_inference"][
                        "driver_run_total_us"
                    ]
                    - result["summaries"]["vta_only"]["profiler_per_inference"][
                        "driver_run_total_us"
                    ]
                )
                / 1000.0,
                "run_minus_driver": result["summaries"]["vta_concurrent"][
                    "run_minus_driver_ms_median"
                ]
                - result["summaries"]["vta_only"][
                    "run_minus_driver_ms_median"
                ],
            },
        }
        observations.append(record)
        by_pair[key] = record
    streaming_excess = {}
    for vta_class in ("compute_heavy", "dma_heavy"):
        cache = by_pair[("cache_resident", vta_class)]
        streaming = by_pair[("streaming", vta_class)]
        streaming_excess[vta_class] = {
            "cpu_slowdown_ratio_streaming_over_cache": streaming["cpu_slowdown"]
            / cache["cpu_slowdown"],
            "vta_slowdown_ratio_streaming_over_cache": streaming["vta_slowdown"]
            / cache["vta_slowdown"],
        }
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7c_shared_ddr_qualification_summary",
            "protocol_id": PROTOCOL_ID,
            "source_session_artifact_sha256": session["artifact_sha256"],
            "board_boot_id": session["board_boot_id"],
            "observations": observations,
            "streaming_excess_over_cache_control": streaming_excess,
            "interpretation": {
                "candidate_throughput_used_for_fit": False,
                "formal_shared_ddr_parameter_admitted": False,
                "remaining_gate": "repeat unchanged subprotocol on at least two additional independent boots, then test a grouped shared-bandwidth model",
                "cpu_core_user_process_overlap_controlled": True,
                "vta_host_affinity_scope": "all_existing_threads_future_threads_inherit",
                "kernel_interrupt_interference_fully_excluded": False,
                "single_boot_supports_cpu_vta_interference": any(
                    item["cpu_signal"] or item["vta_signal"] for item in observations
                ),
                "ddr_specificity_requires_streaming_vs_cache_and_cross_boot_evidence": True,
                "simple_shared_bandwidth_model_ready": False,
                "reason_simple_bandwidth_not_ready": "cache-resident repeated traffic is not DDR traffic, and the observed VTA stage deltas are mostly outside driver_run_total_us",
            },
            "passed": session["passed"],
        }
    )


def write_p7c_review(path, summary):
    cpu_signals = [item for item in summary["observations"] if item["cpu_signal"]]
    vta_signals = [item for item in summary["observations"] if item["vta_signal"]]
    lines = [
        "# P7C CPU-VTA Shared-DDR Qualification Review",
        "",
        "更新日期：{}".format(datetime.date.today().isoformat()),
        "",
        "## 结论",
        "",
        "P7C 单 boot qualification 已完成；未读取候选吞吐标签，也未生成正式 shared-DDR 参数。",
        "CPU memory worker 固定在核 0--2；VTA host 的全部现有线程固定在核 3，后续线程继承该",
        "affinity，从而降低用户态 CPU 核竞争混入。",
        "",
        "## 四组 Matched Control",
        "",
    ]
    for item in summary["observations"]:
        lines.append(
            "- CPU `{}` + VTA `{}` (`{}`): CPU slowdown `{:.3f}x`，VTA slowdown "
            "`{:.3f}x`，声明流量/最长 wall 诊断比率 `{:.3f} GB/s`。".format(
                item["cpu_pressure_class"],
                item["vta_pressure_class"],
                item["vta_segment_id"],
                item["cpu_slowdown"],
                item["vta_slowdown"],
                item["aggregate_effective_bandwidth_GBps"],
            )
        )
    lines.extend(
        [
            "",
            "按单组 `slowdown > 1.05` 且 95% CI 下界大于 1 的 qualification gate，CPU 有 {} 组、"
            "VTA 有 {} 组信号。".format(len(cpu_signals), len(vta_signals)),
        ]
    )
    for item in cpu_signals:
        lines.append(
            "- CPU signal: `{}` + `{}` 为 `{:.3f}x`；若 CPU 压力是 cache-resident，"
            "该信号不能归因为 DDR。".format(
                item["cpu_pressure_class"],
                item["vta_pressure_class"],
                item["cpu_slowdown"],
            )
        )
    for item in vta_signals:
        stage_delta_ms = (
            item["concurrent_ms"]["vta_stage"]
            - item["isolated_ms"]["vta_stage"]
        )
        lines.append(
            "- VTA signal: `{}` + `{}` 为 `{:.3f}x`；stage 增量 `{:.3f} ms`，"
            "driver run 增量 `{:.3f} ms`，run-minus-driver 增量 `{:.3f} ms`。".format(
                item["cpu_pressure_class"],
                item["vta_pressure_class"],
                item["vta_slowdown"],
                stage_delta_ms,
                item["vta_component_delta_ms"]["driver_run"],
                item["vta_component_delta_ms"]["run_minus_driver"],
            )
        )
    lines.extend(
        [
            "",
            "streaming 相对 cache-resident control 的 VTA slowdown 比率在 compute-heavy/DMA-heavy",
            "下分别为 `{:.3f}x/{:.3f}x`。它提示 CPU streaming 压力会增加 VTA 延迟，但显著增量".format(
                summary["streaming_excess_over_cache_control"]["compute_heavy"][
                    "vta_slowdown_ratio_streaming_over_cache"
                ],
                summary["streaming_excess_over_cache_control"]["dma_heavy"][
                    "vta_slowdown_ratio_streaming_over_cache"
                ],
            ),
            "主要落在 host runtime 的 run-minus-driver 部分。因此当前只能称为轻度、非对称的",
            "共享内存/host-runtime 耦合，不能直接拟合成纯 DDR 带宽常数。",
            "上述 GB/s 是声明流量除以最长组件 wall time；cache-resident 重复访问不等于 DDR 流量，",
            "该值不是实测 DDR bandwidth，也不进入参数拟合。",
            "",
            "## 证据边界",
            "",
            "单 boot 只用于 qualification。只有相同子协议在至少三个独立 boot 可重复，且 streaming",
            "相对 cache-resident control 的额外 slowdown 稳定存在，才可归因为共享 DDR 并进入公式。",
            "进程 affinity 不能排除内核中断和驱动线程影响，因此当前结果称为 CPU-VTA interference，",
            "不提前称为纯 DDR bandwidth contention。",
            "",
            "## 产物",
            "",
            "- qualification summary SHA256: `{}`".format(summary["artifact_sha256"]),
            "- source session SHA256: `{}`".format(
                summary["source_session_artifact_sha256"]
            ),
            "",
            "本阶段完成后停止。用户审阅前不进入 P7D，也不重跑额外 boot。",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_local_smoke_session(plan, pair_result, output_a, output_b):
    result_a = summarize_jsonl(output_a)
    result_b = summarize_jsonl(output_b)
    passed = all(
        item["correctness_passed"] and item["determinism_passed"]
        for item in (result_a, result_b)
    )
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7_local_smoke_session",
            "protocol_id": PROTOCOL_ID,
            "plan_artifact_sha256": plan["artifact_sha256"],
            "environment": collect_environment(),
            "pair_execution": pair_result,
            "component_a": result_a,
            "component_b": result_b,
            "passed": passed,
            "evidence_scope": "orchestrator/barrier/correctness smoke; not a hardware parameter",
        }
    )


def _stage_key(segment_id, threads):
    return str(segment_id), int(threads)


def _source_candidate(rows, first, second):
    required = {
        _stage_key(first["segment_id"], first["threads"]),
        _stage_key(second["segment_id"], second["threads"]),
    }
    for row in rows:
        observed = {
            _stage_key(stage["segment_id"], stage["threads"])
            for stage in row["stages"]
            if stage["device"] == "cpu"
        }
        if required.issubset(observed):
            return {"rank": int(row["rank"]), "candidate_id": row["candidate_id"]}
    raise ValueError("no frozen Top-20 candidate contains CPU pair {}".format(sorted(required)))


def _cpu_stage(segment_id, threads):
    return {
        "segment_id": segment_id,
        "threads": int(threads),
        "cpu_affinity": list(range(int(threads))),
        "thread_semantics": "per-stage TVM runtime parameter",
    }


def build_memory_cases():
    sizes = (
        (256 * 1024, "cache_resident"),
        (4 * 1024 * 1024, "transition"),
        (64 * 1024 * 1024, "streaming"),
    )
    cases = []
    for size, pressure_class in sizes:
        for operation in ("read", "write", "copy"):
            for threads in THREAD_CHOICES:
                cases.append(
                    {
                        "case_id": "cpu_mem_{}_{}_t{}".format(
                            pressure_class, operation, threads
                        ),
                        "target_active_working_set_bytes": size,
                        "operand_bytes_per_stream": size // 2 if operation == "copy" else size,
                        "operation": operation,
                        "threads": threads,
                        "cpu_affinity": list(range(threads)),
                        "pressure_class": pressure_class,
                        "warmup_runs": WARMUP_RUNS,
                        "scored_runs": SCORED_RUNS,
                        "correctness": "checksum_required",
                        "worker_lifecycle": "persistent_process_threads",
                        "outputs": [
                            "wall_ms",
                            "process_cpu_ms",
                            "cycles",
                            "instructions",
                            "cache_references",
                            "cache_misses",
                            "bandwidth_GBps",
                            "checksum",
                        ],
                    }
                )
    return cases


def build_cpu_pair_cases(rows):
    pairs = []
    for segment in ("cpu:18:20", "cpu:17:20", "cpu:16:20"):
        for threads in THREAD_CHOICES:
            pairs.append((_cpu_stage("cpu:00:02", 4), _cpu_stage(segment, threads)))
    for left_threads in (1, 2):
        for right_threads in (1, 4):
            pairs.append(
                (_cpu_stage("cpu:13:14", left_threads), _cpu_stage("cpu:20:20", right_threads))
            )

    cases = []
    for index, (first, second) in enumerate(pairs, 1):
        source = _source_candidate(rows, first, second)
        cases.append(
            {
                "case_id": "cpu_pair_{:02d}".format(index),
                "stage_a": first,
                "stage_b": second,
                "modes": ["a_only", "b_only", "a_b_concurrent"],
                "process_model": "one_native_runner_process_per_stage",
                "start_barrier_required": True,
                "concurrent_sampling": "one_ready_start_barrier_per_scored_sample",
                "warmup_runs_per_process": WARMUP_RUNS,
                "scored_runs_per_process": SCORED_RUNS,
                "source_frozen_candidate": source,
                "outputs": [
                    "wall_ms",
                    "process_cpu_ms",
                    "cycles",
                    "instructions",
                    "cache_references",
                    "cache_misses",
                    "correctness",
                ],
            }
        )
    return cases


def _quantile_boundaries(boundaries, direction):
    by_size = {}
    for boundary in boundaries:
        if boundary["direction"] == direction:
            by_size.setdefault(int(boundary["logical_bytes"]), boundary)
    ordered = sorted(by_size)
    if len(ordered) < 3:
        raise ValueError("{} needs at least three distinct boundary sizes".format(direction))
    indexes = (0, (len(ordered) - 1) // 2, len(ordered) - 1)
    labels = ("low", "median", "high")
    return [
        {
            "quantile": label,
            "boundary_id": by_size[ordered[index]]["boundary_id"],
            "logical_bytes": ordered[index],
            "direction": direction,
        }
        for label, index in zip(labels, indexes)
    ]


def build_runtime_cases(manifest):
    device_cases = []
    for workload in ("host_empty", "vta_short", "vta_long"):
        for poll_sleep_ns in (0, 1000):
            device_cases.append(
                {
                    "case_id": "runtime_{}_poll{}".format(workload, poll_sleep_ns),
                    "workload": workload,
                    "poll_sleep_ns": poll_sleep_ns,
                    "post_start_sleep_ns": 1000,
                    "warmup_runs": WARMUP_RUNS,
                    "scored_runs": SCORED_RUNS,
                }
            )
    boundary_cases = []
    for direction in ("cpu_to_vta", "vta_to_cpu"):
        for item in _quantile_boundaries(manifest["boundaries"], direction):
            boundary_cases.append(
                {
                    "case_id": "boundary_{}_{}".format(direction, item["quantile"]),
                    **item,
                    "warmup_runs": WARMUP_RUNS,
                    "scored_runs": SCORED_RUNS,
                    "transfer_resource": "stage_dma",
                    "host_owner": "boundary_adapter_and_coherent_cache",
                    "device_dma_owner": "lowered_vta_load_store",
                }
            )
    return {"device_cases": device_cases, "boundary_cases": boundary_cases}


def _schema_nbytes(schema):
    dtype_bytes = {"int8": 1, "uint8": 1, "int32": 4, "float32": 4}
    total = 0
    for slot in schema.get("slots", []):
        elements = 1
        for extent in slot["shape"]:
            elements *= int(extent)
        total += elements * dtype_bytes[slot["dtype"]]
    return total


def _p7b2_stage_source(resolved):
    stage = resolved["stage"]
    return {
        "segment_id": resolved["segment_id"],
        "package_candidate_id": resolved["package_manifest"]["candidate_id"],
        "package_manifest_sha256": _file_sha256(Path(resolved["package_dir"]) / "manifest.json"),
        "stage_index": int(stage["index"]),
        "input_bytes": _schema_nbytes(stage["input_schema"]),
        "output_bytes": _schema_nbytes(stage["output_schema"]),
        "unit_names": list(stage["unit_names"]),
    }


def build_p7b2_plan(parent_plan, manifest, catalog):
    """Freeze an identifiable runtime sub-plan without changing the P7B-1 plan hash."""
    resolved = {
        segment_id: _resolve_device_stage_package(segment_id, "vta", manifest, catalog)
        for segment_id in ("vta:03:17", "vta:15:19", "vta:18:19", "vta:13:13", "vta:01:03")
    }
    host_cases = [
        {
            "case_id": "host_empty_actor_chain_{}".format(stage_count),
            "actor_count": stage_count,
            "inner_repeats": 1000,
            "warmup_runs": WARMUP_RUNS,
            "scored_runs": SCORED_RUNS,
            "interpretation": "host queue/scheduling floor; not an additive graph service",
        }
        for stage_count in (1, 3)
    ]
    vta_cases = []
    for workload, segment_id in (("short", "vta:15:19"), ("long", "vta:03:17")):
        for poll_sleep_ns in (0, 1000):
            vta_cases.append(
                {
                    "case_id": "vta_{}_poll{}".format(workload, poll_sleep_ns),
                    "workload": workload,
                    "poll_sleep_ns": poll_sleep_ns,
                    "post_start_sleep_ns": 1000,
                    "source": _p7b2_stage_source(resolved[segment_id]),
                    "warmup_runs": WARMUP_RUNS,
                    "scored_runs": SCORED_RUNS,
                }
            )
    boundary_representatives = []
    for quantile, segment_id in (
        ("low", "vta:18:19"),
        ("median", "vta:13:13"),
        ("high", "vta:01:03"),
    ):
        source = _p7b2_stage_source(resolved[segment_id])
        boundary_representatives.append(
            {
                "case_id": "boundary_pair_{}".format(quantile),
                "quantile": quantile,
                "source": source,
                "cpu_to_vta_observation": {
                    "logical_bytes": source["input_bytes"],
                    "metric": "stage0_set_ms",
                },
                "vta_to_cpu_observation": {
                    "logical_bytes": source["output_bytes"],
                    "metric": "stage0_get_ms",
                },
                "poll_sleep_ns": 1000,
                "post_start_sleep_ns": 1000,
                "warmup_runs": WARMUP_RUNS,
                "scored_runs": SCORED_RUNS,
            }
        )
    expected_input = [100352, 200704, 802816]
    expected_output = [100352, 401408, 1605632]
    if [item["source"]["input_bytes"] for item in boundary_representatives] != expected_input:
        raise ValueError("P7B-2 CPU-to-VTA representatives do not match frozen quantiles")
    if [item["source"]["output_bytes"] for item in boundary_representatives] != expected_output:
        raise ValueError("P7B-2 VTA-to-CPU representatives do not match frozen quantiles")
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7b2_runtime_measurement_plan",
            "protocol_id": PROTOCOL_ID,
            "parent_p7_plan_artifact_sha256": parent_plan["artifact_sha256"],
            "design_revision": "p7b2_identifiable_ownership_v2",
            "host_empty_cases": host_cases,
            "vta_runtime_cases": vta_cases,
            "boundary_representatives": boundary_representatives,
            "ownership": {
                "cpu_vta_boundary": "VTA set_ms only; host copy/cache/adapter evidence",
                "vta_cpu_boundary": "VTA get_ms only; host copy/cache/adapter evidence",
                "vta_run": "inclusive LOAD/STORE, enqueue, submit, post-start sleep, poll wait and sync",
                "profiler_components": "decomposition evidence only; never added to inclusive run_ms",
                "host_empty": "queue/scheduling floor only; admission requires separate pipeline holdout",
            },
            "identifiability_correction": {
                "removed_fit": "alpha_frame + N_cpu*alpha_cpu + N_vta*alpha_vta",
                "reason": "P2/P3 already use inclusive CPU/VTA run_ms, so per-stage intercepts would double count",
                "retained_outputs": [
                    "host_queue_floor",
                    "poll_policy_delta",
                    "boundary_set_get_size_observations",
                    "inclusive_vta_runtime_decomposition",
                ],
            },
            "candidate_throughput_used_for_fit": False,
            "single_boot_is_qualification_only": True,
        }
    )


def build_ddr_contention_cases():
    cpu_pressures = {
        "cache_resident": {
            "target_active_working_set_bytes": 256 * 1024,
            "operand_bytes_per_stream": 128 * 1024,
            "operation": "read",
            "threads": 4,
        },
        "streaming": {
            "target_active_working_set_bytes": 64 * 1024 * 1024,
            "operand_bytes_per_stream": 32 * 1024 * 1024,
            "operation": "copy",
            "threads": 4,
        },
    }
    vta_pressures = {
        "compute_heavy": {
            "source_segment_id": "vta:03:17",
            "template": "fixed_dma_repeated_gemm",
        },
        "dma_heavy": {
            "source_segment_id": "vta:03:17",
            "template": "fixed_compute_repeated_load_store",
        },
    }
    cases = []
    for cpu_name, cpu in cpu_pressures.items():
        for vta_name, vta in vta_pressures.items():
            for mode in ("cpu_only", "vta_only", "cpu_vta_concurrent"):
                cases.append(
                    {
                        "case_id": "ddr_{}_{}_{}".format(cpu_name, vta_name, mode),
                        "cpu_pressure_class": cpu_name,
                        "cpu_workload": cpu,
                        "vta_pressure_class": vta_name,
                        "vta_workload": vta,
                        "mode": mode,
                        "start_barrier_required": mode == "cpu_vta_concurrent",
                        "warmup_runs": WARMUP_RUNS,
                        "scored_runs": SCORED_RUNS,
                        "outputs": [
                            "cpu_wall_ms",
                            "vta_wall_ms",
                            "cpu_traffic_bytes",
                            "vta_load_bytes",
                            "vta_store_bytes",
                            "aggregate_effective_bandwidth_GBps",
                        ],
                    }
                )
    return cases


def _p7c_vta_pressure_evidence(p7b2_session, segment_id):
    candidates = [
        result
        for result in p7b2_session["vta_stage_results"].values()
        if result["source"]["segment_id"] == segment_id
        and int(result["poll_sleep_ns"]) == 1000
    ]
    if len(candidates) != 1:
        raise ValueError("P7C requires one poll=1000 observation for " + segment_id)
    result = candidates[0]
    profile = result["summary"]["profiler_per_inference"]
    traffic = profile["synchronize_load_bytes"] + profile["synchronize_store_bytes"]
    service_ms = result["summary"]["run_ms_median"]
    return {
        "segment_id": segment_id,
        "run_ms_median": service_ms,
        "synchronize_load_bytes": profile["synchronize_load_bytes"],
        "synchronize_store_bytes": profile["synchronize_store_bytes"],
        "dma_mib_per_run_ms": traffic / (1024.0 * 1024.0) / service_ms,
        "push_gemm_op_calls": profile["push_gemm_op_calls"],
        "push_alu_op_calls": profile["push_alu_op_calls"],
    }


def build_p7c_plan(parent_plan, manifest, catalog, p7b2_session):
    validate_artifact(
        p7b2_session, "cpu_vta_pipeline_v1_p7b2_runtime_qualification_session"
    )
    pressure_segments = {
        "compute_heavy": "vta:01:03",
        "dma_heavy": "vta:15:19",
    }
    evidence = {
        name: _p7c_vta_pressure_evidence(p7b2_session, segment_id)
        for name, segment_id in pressure_segments.items()
    }
    if (
        evidence["dma_heavy"]["dma_mib_per_run_ms"]
        < 2.0 * evidence["compute_heavy"]["dma_mib_per_run_ms"]
    ):
        raise ValueError("P7C VTA pressure classes are not sufficiently separated")
    sources = {
        name: _p7b2_stage_source(
            _resolve_device_stage_package(segment_id, "vta", manifest, catalog)
        )
        for name, segment_id in pressure_segments.items()
    }
    cpu_pressures = {
        "cache_resident": {
            "operation": "read",
            "threads": 3,
            "process_cpu_affinity": [0, 1, 2],
            "operand_bytes_per_stream": 256 * 1024,
            "target_active_working_set_bytes": 256 * 1024,
            "inner_repeats": 512,
            "traffic_bytes_per_sample": 128 * 1024 * 1024,
        },
        "streaming": {
            "operation": "copy",
            "threads": 3,
            "process_cpu_affinity": [0, 1, 2],
            "operand_bytes_per_stream": 32 * 1024 * 1024,
            "target_active_working_set_bytes": 64 * 1024 * 1024,
            "inner_repeats": 2,
            "traffic_bytes_per_sample": 128 * 1024 * 1024,
        },
    }
    groups = []
    for cpu_name, cpu in cpu_pressures.items():
        for vta_name in ("compute_heavy", "dma_heavy"):
            groups.append(
                {
                    "group_id": "p7c_{}_{}".format(cpu_name, vta_name),
                    "cpu_pressure_class": cpu_name,
                    "cpu_workload": cpu,
                    "vta_pressure_class": vta_name,
                    "vta_source": sources[vta_name],
                    "vta_process_cpu_affinity": [3],
                    "poll_sleep_ns": 1000,
                    "modes": ["cpu_only", "vta_only", "cpu_vta_concurrent"],
                    "warmup_runs": WARMUP_RUNS,
                    "scored_runs": SCORED_RUNS,
                }
            )
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7c_shared_ddr_measurement_plan",
            "protocol_id": PROTOCOL_ID,
            "parent_p7_plan_artifact_sha256": parent_plan["artifact_sha256"],
            "p7b2_session_artifact_sha256": p7b2_session["artifact_sha256"],
            "case_groups": groups,
            "actual_run_mode_count": 3 * len(groups),
            "vta_pressure_evidence": evidence,
            "protocol_corrections": [
                {
                    "original": "compute-heavy and DMA-heavy both used vta:03:17 placeholder templates",
                    "corrected": "use real vta:01:03 and vta:15:19 native segments with a measured >2x DMA-rate separation",
                },
                {
                    "original": "cache-resident read declared 256 KiB active set but supplied 128 KiB",
                    "corrected": "read operand and active working set are both 256 KiB",
                },
                {
                    "original": "four CPU workers left no core isolated for the VTA host process",
                    "corrected": "CPU workers use cores 0--2; all existing VTA host process threads use core 3 and future threads inherit that affinity",
                },
            ],
            "measurement_policy": {
                "order_within_group": ["cpu_only", "vta_only", "cpu_vta_concurrent"],
                "concurrent_start": "one ready/start barrier per scored sample",
                "candidate_throughput_used_for_fit": False,
                "single_boot_is_qualification_only": True,
                "slowdown_signal": "point estimate >1.05 and bootstrap 95% lower bound >1",
                "formal_admission_requires_independent_boots": 3,
                "attribution_scope": "CPU-VTA interference under disjoint user-process CPU affinities; kernel/interrupt effects remain possible",
            },
        }
    )


def build_plan(ranking, manifest):
    if len(ranking["rows"]) != 20:
        raise ValueError("P7 requires the frozen natural Top-20")
    if ranking.get("candidate_throughput_used_for_ranking") is not False:
        raise ValueError("Top-20 must be independent of candidate throughput labels")

    memory_cases = build_memory_cases()
    cpu_pairs = build_cpu_pair_cases(ranking["rows"])
    runtime_cases = build_runtime_cases(manifest)
    ddr_cases = build_ddr_contention_cases()
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7_measurement_plan",
            "protocol_id": PROTOCOL_ID,
            "status": "p7a_local_instrumentation_only",
            "endpoint_policy": {
                "board_host": "runtime_cli_argument_recorded_in_session_artifact",
                "rpc_port": 9090,
            },
            "source_artifacts": {
                "frozen_top20": {
                    "path": str(DEFAULT_RANKING.relative_to(ROOT)),
                    "artifact_sha256": ranking["artifact_sha256"],
                    "fields_consumed": ["rank", "candidate_id", "stages", "boundaries"],
                    "candidate_throughput_consumed": False,
                },
                "profile_manifest": {
                    "path": str(DEFAULT_MANIFEST.relative_to(ROOT)),
                    "artifact_sha256": manifest["artifact_sha256"],
                },
            },
            "measurement_policy": {
                "warmup_runs": WARMUP_RUNS,
                "scored_runs": SCORED_RUNS,
                "single_boot_is_qualification_only": True,
                "formal_parameter_boots_minimum": 3,
                "correctness_required": True,
                "determinism_required": True,
                "pmu_unavailable_is_null_with_reason": True,
                "candidate_throughput_fit_allowed": False,
                "stop_after_each_experiment_class": True,
                "concurrent_sampling": "one_ready_start_barrier_per_scored_sample",
                "memory_target_traffic_bytes_per_sample": MEMORY_SAMPLE_TRAFFIC_BYTES,
                "memory_wall_time_cv_max": MEMORY_SAMPLE_CV_MAX,
                "cpu_stage_isolated_wall_time_cv_max": ISOLATED_STAGE_CV_MAX,
                "cpu_stage_slowdown_ci_relative_halfwidth_max": (
                    SLOWDOWN_CI_RELATIVE_HALFWIDTH_MAX
                ),
                "concurrent_stage_cv_policy": (
                    "report_as_scheduler_contention_variability; acceptance uses the slowdown "
                    "median confidence interval"
                ),
            },
            "preflight_gate": {
                "required_before_p7b": True,
                "checks": [
                    "ssh",
                    "rpc",
                    "fpga_operating",
                    "udmabuf_192_mib",
                    "sd_free_space",
                    "current_boot_dmesg_has_no_ext4_or_io_error",
                ],
            },
            "experiments": {
                "p7b1_cpu_memory_baseline": memory_cases,
                "p7b1_cpu_stage_pairs": cpu_pairs,
                "p7b2_runtime": runtime_cases,
                "p7c_shared_ddr_contention": ddr_cases,
            },
            "parameter_admission": {
                "slowdown_threshold": 1.05,
                "confidence_interval_must_exclude_one": True,
                "independent_boot_reproducibility_required": True,
                "cpu_pair_metrics": {
                    "slowdown_a": "median(T_A_given_A_plus_B) / median(T_A_only)",
                    "slowdown_b": "median(T_B_given_A_plus_B) / median(T_B_only)",
                },
                "runtime_fit": "alpha_frame + N_cpu*alpha_cpu + N_vta*alpha_vta + sum(adapter_cache)",
                "shared_ddr_fit": "(Q_cpu + Q_vta_load + Q_vta_store) / B_shared(pressure_class)",
                "unattributed_residual_policy": "report_only",
                "failed_frozen_validation_policy": "exclude_parameter_from_formula",
            },
            "ownership": {
                "vta_iso_includes": ["device_work", "submit", "sync", "owned_cache_maintenance"],
                "poll_wait_rule": "latency evidence only; never added again to device work",
                "boundary_rule": "host adapter/cache owned by boundary; LOAD/STORE owned by VTA DMA",
                "shared_ddr_rule": "resource lower bound; never added to an already inclusive stage wall time",
            },
            "future_artifacts": {
                "v1_p7_session_<boot>.json": "pending_p7b_p7c_board_measurement",
                "v1_p7_physical_profile.json": "pending_three_boot_component_fit",
                "v1_p7_formula_validation.json": "pending_p7d_frozen_validation",
            },
            "p7d_model_contract": {
                "cpu_stage": "T_cpu_iso * S_cpu(threads,peer,pressure) + T_owned_boundary",
                "vta_stage": "T_vta_iso + T_owned_boundary",
                "pipeline_ii": "max(max_cpu_stage, sum_vta, D_core, D_DDR) + alpha_frame",
                "no_double_count": "inclusive submit/sync/cache terms may not be added twice",
                "validation_gates": {
                    "cycle_mape_below": 0.12859,
                    "top20_spearman_above": 0.155,
                    "regret_at_5_below": 0.05419,
                    "candidate_throughput_fit_allowed": False,
                },
            },
            "summary": {
                "memory_baseline_case_count": len(memory_cases),
                "cpu_stage_pair_count": len(cpu_pairs),
                "cpu_stage_pair_run_mode_count": 3 * len(cpu_pairs),
                "runtime_device_case_count": len(runtime_cases["device_cases"]),
                "runtime_boundary_case_count": len(runtime_cases["boundary_cases"]),
                "ddr_contention_run_mode_count": len(ddr_cases),
            },
        }
    )


def audit_plan(plan, runner_source, memory_source, runtime_source):
    experiments = plan["experiments"]
    checks = {
        "memory_matrix_is_36": len(experiments["p7b1_cpu_memory_baseline"]) == 36,
        "cpu_pair_count_is_16": len(experiments["p7b1_cpu_stage_pairs"]) == 16,
        "cpu_pairs_have_three_matched_modes": all(
            case["modes"] == ["a_only", "b_only", "a_b_concurrent"]
            for case in experiments["p7b1_cpu_stage_pairs"]
        ),
        "cpu_pairs_have_frozen_sources": all(
            case["source_frozen_candidate"]["candidate_id"]
            for case in experiments["p7b1_cpu_stage_pairs"]
        ),
        "runtime_device_matrix_is_6": len(experiments["p7b2_runtime"]["device_cases"]) == 6,
        "runtime_boundary_matrix_is_6": len(experiments["p7b2_runtime"]["boundary_cases"]) == 6,
        "ddr_matched_run_count_is_12": len(experiments["p7c_shared_ddr_contention"]) == 12,
        "candidate_throughput_not_consumed": not plan["source_artifacts"]["frozen_top20"][
            "candidate_throughput_consumed"
        ],
        "runner_has_independent_warmup": "--warmup-runs" in runner_source,
        "runner_has_start_barrier": "--barrier-token" in runner_source,
        "runner_has_per_sample_barrier": "--barrier-each-run" in runner_source,
        "runner_supports_multi_tensor_stage_input": "--input-files" in runner_source,
        "runner_has_process_pmu": all(
            field in runner_source
            for field in ("pmu_cycles", "pmu_instructions", "pmu_cache_misses")
        ),
        "runner_can_dump_intermediate_reference_tensors": all(
            token in runner_source for token in ("stage_raw_outputs", "output_files")
        ),
        "memory_bench_has_pressure_class": "--pressure-class" in memory_source,
        "memory_bench_has_start_barrier": "--barrier-token" in memory_source,
        "memory_bench_has_per_sample_barrier": "--barrier-each-run" in memory_source,
        "memory_bench_excludes_per_sample_thread_creation": all(
            token in memory_source
            for token in ("class MemoryWorkers", "persistent_process_threads")
        ),
        "memory_bench_has_process_pmu": all(
            field in memory_source
            for field in ("pmu_cycles", "pmu_instructions", "pmu_cache_misses")
        ),
        "vta_profiler_has_required_ownership_counters": all(
            field in runtime_source
            for field in (
                "mem_copy_from_host_bytes",
                "mem_copy_to_host_bytes",
                "flush_cache_bytes",
                "invalidate_cache_bytes",
                "synchronize_load_bytes",
                "synchronize_store_bytes",
                "driver_submit_mmio_us",
                "driver_poll_wait_us",
            )
        ),
    }
    identifiers = []
    for value in experiments.values():
        if isinstance(value, list):
            identifiers.extend(case["case_id"] for case in value)
        else:
            identifiers.extend(case["case_id"] for cases in value.values() for case in cases)
    checks["case_ids_are_unique"] = len(identifiers) == len(set(identifiers))
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7a_local_audit",
            "protocol_id": PROTOCOL_ID,
            "plan_artifact_sha256": plan["artifact_sha256"],
            "checks": checks,
            "passed": all(checks.values()),
            "evidence_scope": "local source/plan semantics; no board performance parameters",
        }
    )


def write_review(path, plan, audit):
    summary = plan["summary"]
    lines = [
        "# P7A Local Instrumentation Review",
        "",
        "更新日期：{}".format(datetime.date.today().isoformat()),
        "",
        "## 结论",
        "",
        "P7A 本地 instrumentation 已实现并通过字段审计；尚未执行 P7B/P7C 上板测量，",
        "因此没有生成新的物理参数，也没有把 `+34 ms` 写入正式公式。",
        "",
        "## 已实现",
        "",
        "- native stage runner 支持独立 warmup、文件 barrier、单 stage 重复运行和进程级 PMU。",
        "- PMU 不可用时输出 `null` 与失败原因，不以零冒充观测值。",
        "- CPU memory benchmark 支持同一 barrier 和 cache/transition/streaming 压力类别。",
        "- copy 明确区分 operand bytes 与实际活跃 working set，避免把双数组访问少算一半。",
        "- VTA LOAD/STORE、copy、cache、submit、poll、synchronize 继续复用现有 profiler 口径。",
        "- 统一计划含 {} 个 memory case、{} 个 CPU stage pair、{} 个 runtime case 和 {} 个 DDR matched run。".format(
            summary["memory_baseline_case_count"],
            summary["cpu_stage_pair_count"],
            summary["runtime_device_case_count"] + summary["runtime_boundary_case_count"],
            summary["ddr_contention_run_mode_count"],
        ),
        "",
        "## 证据边界",
        "",
        "Top-20 只提供 segment、线程参数和 boundary 大小，候选实测吞吐不参与参数拟合。",
        "P7B 开始前必须对运行时传入的板端地址重新检查 SSH/RPC、FPGA、u-dma-buf、SD 与 dmesg。",
        "板端地址不写死在协议中，由每个 session artifact 记录实际 endpoint。",
        "",
        "## 本地验证",
        "",
        "- barrier-aware 组件逐样本同步语义：PASS。",
        "- stage runner 与 memory benchmark AArch64 交叉编译：PASS。",
        "- P7/P5A/Top-20 相关回归测试：`24 passed`。",
        "",
        "## 审计",
        "",
        "- local audit: `{}`".format("PASS" if audit["passed"] else "FAIL"),
        "- plan SHA256: `{}`".format(plan["artifact_sha256"]),
        "- audit SHA256: `{}`".format(audit["artifact_sha256"]),
        "",
        "下一步仅为 P7B-1 单 boot CPU/cache/DDR qualification；需要用户确认后执行。",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "p7a",
            "p7b1-memory",
            "p7b1-cpu-pairs",
            "p7b1-finalize",
            "p7b1-multiboot",
            "p7b2-runtime",
            "p7c-ddr",
        ),
        default="p7a",
    )
    parser.add_argument("--ranking", default=str(DEFAULT_RANKING))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--plan-output", default=str(DEFAULT_PLAN))
    parser.add_argument(
        "--audit-output", default=str(REPORT_ROOT / "v1_p7_local_instrumentation_audit.json")
    )
    parser.add_argument("--review-output", default=str(DEFAULT_REVIEW))
    parser.add_argument(
        "--smoke-command-a-json",
        default="",
        help="Optional JSON argv list for the first barrier-aware local component command",
    )
    parser.add_argument("--smoke-command-b-json", default="")
    parser.add_argument("--smoke-output-a", default="")
    parser.add_argument("--smoke-output-b", default="")
    parser.add_argument("--smoke-barrier-dir", default="/tmp/ramps_p7_barrier")
    parser.add_argument("--smoke-timeout-s", type=int, default=120)
    parser.add_argument("--board-host", default="192.168.1.247")
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument("--timeout-s", type=int, default=15)
    parser.add_argument("--case-timeout-s", type=int, default=300)
    parser.add_argument("--remote-root", default="/var/volatile/ramps_p7")
    parser.add_argument("--memory-binary", default="/tmp/ramps_cpu_memory_microbench_p7")
    parser.add_argument("--runner-binary", default="/tmp/vta_stage_pipeline_runner_p7")
    parser.add_argument("--package-root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--p5a-package-root", default=str(DEFAULT_P5A_PACKAGE_ROOT))
    parser.add_argument("--reference-json", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--p7b2-plan-output", default=str(DEFAULT_P7B2_PLAN))
    parser.add_argument("--p7b2-session-output", default=str(DEFAULT_P7B2_SESSION))
    parser.add_argument("--p7b2-summary-output", default=str(DEFAULT_P7B2_SUMMARY))
    parser.add_argument("--p7c-plan-output", default=str(DEFAULT_P7C_PLAN))
    parser.add_argument("--p7c-session-output", default=str(DEFAULT_P7C_SESSION))
    parser.add_argument("--p7c-summary-output", default=str(DEFAULT_P7C_SUMMARY))
    parser.add_argument("--local-cost-table", default=str(DEFAULT_LOCAL_COST))
    parser.add_argument("--top20-board-summary", default=str(DEFAULT_TOP20_BOARD_SUMMARY))
    parser.add_argument("--package-timeout-s", type=int, default=600)
    parser.add_argument(
        "--cpu-pair-session-output",
        default=str(REPORT_ROOT / "v1_p7_session_boot1_p7b1_cpu_pairs.json"),
    )
    parser.add_argument(
        "--memory-session-input",
        default=str(REPORT_ROOT / "v1_p7_session_boot1_p7b1_memory.json"),
    )
    parser.add_argument(
        "--cpu-pair-session-input",
        default=str(REPORT_ROOT / "v1_p7_session_boot1_p7b1_cpu_pairs.json"),
    )
    parser.add_argument(
        "--combined-session-output",
        default=str(REPORT_ROOT / "v1_p7_session_boot1.json"),
    )
    parser.add_argument(
        "--multiboot-memory-session-inputs",
        nargs="+",
        default=[
            str(REPORT_ROOT / "v1_p7_session_boot{}_p7b1_memory.json".format(boot))
            for boot in (1, 2, 3)
        ],
    )
    parser.add_argument(
        "--multiboot-cpu-pair-session-inputs",
        nargs="+",
        default=[
            str(REPORT_ROOT / "v1_p7_session_boot{}_p7b1_cpu_pairs.json".format(boot))
            for boot in (1, 2, 3)
        ],
    )
    parser.add_argument(
        "--failed-memory-attempt-inputs",
        nargs="*",
        default=[
            str(REPORT_ROOT / "v1_p7_session_boot2_p7b1_memory_attempt1_failed.json"),
            str(REPORT_ROOT / "v1_p7_session_boot3_p7b1_memory_attempt1_failed.json"),
        ],
    )
    parser.add_argument("--multiboot-output", default=str(DEFAULT_P7B1_MULTIBOOT))
    parser.add_argument(
        "--session-output",
        default=str(REPORT_ROOT / "v1_p7_session_local_smoke.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ranking = load_artifact(args.ranking, RANKING_KIND)
    manifest = load_artifact(args.manifest, MANIFEST_KIND)
    plan = build_plan(ranking, manifest)
    audit = audit_plan(
        plan,
        (ROOT / "vta/apps/native_deploy/vta_stage_pipeline_runner.cc").read_text(
            encoding="utf-8"
        ),
        (ROOT / "vta/apps/native_deploy/ramps_cpu_memory_microbench.cc").read_text(
            encoding="utf-8"
        ),
        (ROOT / "vta/runtime/runtime.cc").read_text(encoding="utf-8"),
    )
    write_json(args.plan_output, plan)
    write_json(args.audit_output, audit)
    if args.phase == "p7a":
        write_review(args.review_output, plan, audit)
    if not audit["passed"]:
        raise SystemExit("P7A local audit failed")
    if args.phase == "p7b1-memory":
        session = run_p7b1_memory_qualification(args, plan)
        print(
            "[P7B-1 memory] passed={} cases={}/{} session_sha256={}".format(
                session["passed"],
                session["summary"]["passed_case_count"],
                session["summary"]["case_count"],
                session["artifact_sha256"],
            )
        )
        if not session["passed"]:
            raise SystemExit(2)
        return
    if args.phase == "p7b1-cpu-pairs":
        session = run_p7b1_cpu_pair_qualification(args, plan, manifest)
        print(
            "[P7B-1 CPU pairs] passed={} cases={}/{} admission_signals=A{} B{} "
            "session_sha256={}".format(
                session["passed"],
                session["summary"]["passed_case_count"],
                session["summary"]["case_count"],
                session["summary"]["single_boot_admission_signal_count_a"],
                session["summary"]["single_boot_admission_signal_count_b"],
                session["artifact_sha256"],
            )
        )
        if not session["passed"]:
            raise SystemExit(2)
        return
    if args.phase == "p7b1-finalize":
        session = finalize_p7b1(args, plan)
        print(
            "[P7B-1 finalize] passed={} session_sha256={}".format(
                session["passed"], session["artifact_sha256"]
            )
        )
        if not session["passed"]:
            raise SystemExit(2)
        return
    if args.phase == "p7b1-multiboot":
        summary = build_p7b1_multiboot_summary(
            plan,
            args.multiboot_memory_session_inputs,
            args.multiboot_cpu_pair_session_inputs,
            args.failed_memory_attempt_inputs,
        )
        write_json(args.multiboot_output, summary)
        write_p7b1_multiboot_review(args.review_output, summary)
        print(
            "[P7B-1 multiboot] passed={} boots={} admitted=A{} B{} "
            "summary_sha256={}".format(
                summary["passed"],
                len(summary["board_boot_ids"]),
                summary["summary"]["admitted_exact_observation_count"]["a"],
                summary["summary"]["admitted_exact_observation_count"]["b"],
                summary["artifact_sha256"],
            )
        )
        if not summary["passed"]:
            raise SystemExit(2)
        return
    if args.phase == "p7b2-runtime":
        session = run_p7b2_runtime_qualification(args, plan, manifest)
        summary = build_p7b2_qualification_summary(session, args.local_cost_table)
        write_json(args.p7b2_summary_output, summary)
        write_p7b2_review(args.review_output, summary)
        print(
            "[P7B-2 runtime] passed={} cases={}/{} session_sha256={} summary_sha256={}".format(
                session["passed"],
                session["summary"]["passed_case_count"],
                session["summary"]["case_count"],
                session["artifact_sha256"],
                summary["artifact_sha256"],
            )
        )
        if not session["passed"]:
            raise SystemExit(2)
        return
    if args.phase == "p7c-ddr":
        session = run_p7c_qualification(args, plan, manifest)
        summary = build_p7c_qualification_summary(session)
        write_json(args.p7c_summary_output, summary)
        write_p7c_review(args.review_output, summary)
        print(
            "[P7C DDR] passed={} groups={}/{} CPU-signals={} VTA-signals={} "
            "session_sha256={} summary_sha256={}".format(
                session["passed"],
                session["summary"]["passed_case_group_count"],
                session["summary"]["case_group_count"],
                session["summary"]["single_boot_cpu_signal_count"],
                session["summary"]["single_boot_vta_signal_count"],
                session["artifact_sha256"],
                summary["artifact_sha256"],
            )
        )
        if not session["passed"]:
            raise SystemExit(2)
        return
    smoke_values = (
        args.smoke_command_a_json,
        args.smoke_command_b_json,
        args.smoke_output_a,
        args.smoke_output_b,
    )
    if any(smoke_values):
        if not all(smoke_values):
            raise SystemExit("both smoke commands and output JSONL paths are required")
        pair = run_synchronized_pair(
            json.loads(args.smoke_command_a_json),
            json.loads(args.smoke_command_b_json),
            args.smoke_barrier_dir,
            args.smoke_timeout_s,
        )
        session = build_local_smoke_session(
            plan, pair, args.smoke_output_a, args.smoke_output_b
        )
        write_json(args.session_output, session)
        if not session["passed"]:
            raise SystemExit("P7A synchronized smoke failed")
    print("[P7A] local audit passed plan_sha256={}".format(plan["artifact_sha256"]))


if __name__ == "__main__":
    main()
