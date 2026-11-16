#!/usr/bin/env python3
"""Execute one preregistered boot of the C1 shared-memory-aware Top-K union.

The script is fail-closed: it verifies the deployment freeze, board state,
package/runtime hashes and three full output tensors before each configuration
is timed.  A failed run is retained and is never retried in the same output
directory.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shlex
import statistics
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from shared_memory_topk import sha256_file


SSH_OPTIONS = [
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
    "-o", "HostKeyAlgorithms=+ssh-rsa",
    "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def remote(host, command, check=True, timeout=300):
    return subprocess.run(
        ["ssh", *SSH_OPTIONS, host, command], check=check,
        capture_output=True, timeout=timeout,
    )


def upload(host, source, destination, timeout=600):
    subprocess.run(
        ["scp", *SSH_OPTIONS, str(source), host + ":" + destination],
        check=True, capture_output=True, timeout=timeout,
    )


def parse_preflight(raw):
    fields = {}
    for line in raw.decode().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


def board_preflight(host, requirements, initialize_udmabuf=False, initialize_fpga=False):
    probe = """
echo BOOT=$(cat /proc/sys/kernel/random/boot_id)
echo FPGA=$(cat /sys/class/fpga_manager/fpga0/state 2>/dev/null || echo unavailable)
echo FIRMWARE=$(cat /sys/class/fpga_manager/fpga0/firmware 2>/dev/null || echo unavailable)
echo UDMABUF=$(cat /sys/class/u-dma-buf/udmabuf0/size 2>/dev/null || echo unavailable)
echo FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo unavailable)
echo GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unavailable)
echo BITSTREAM=$(sha256sum /lib/firmware/vta_hpc.bit 2>/dev/null | awk '{print $1}')
echo BUSY=$(ps -eo args | grep -E '[t]vm_rpc|[v]ta_stage_pipeline_runner|[v]ta_stage_pair_runner' | wc -l)
echo FREE_KB=$(df -Pk /media/sd-mmcblk1p2 | awk 'NR==2 {print $4}')
""".strip()
    raw = remote(host, probe).stdout
    fields = parse_preflight(raw)
    fpga_programmed = False
    if initialize_fpga and (
        fields.get("BITSTREAM") != requirements["bitstream_sha256"]
        or fields.get("FIRMWARE") != requirements["fpga_firmware"]
    ):
        sources = (
            "/media/sd-mmcblk1p2/tvm_deploy/firmware/vta_hpc.bit",
            "/media/sd-mmcblk1p2/lib/firmware/vta_wrapper.bit",
        )
        command = (
            "for source in {sources}; do "
            "if test -f \"$source\" && test \"$(sha256sum \"$source\" | awk '{{print $1}}')\" = {digest}; then "
            "cp \"$source\" /lib/firmware/{firmware} && "
            "echo {firmware} > /sys/class/fpga_manager/fpga0/firmware; exit $?; fi; "
            "done; exit 7"
        ).format(
            sources=" ".join(map(shlex.quote, sources)),
            digest=shlex.quote(requirements["bitstream_sha256"]),
            firmware=shlex.quote(requirements["fpga_firmware"]),
        )
        remote(host, command, timeout=120)
        fpga_programmed = True
        raw = remote(host, probe).stdout
        fields = parse_preflight(raw)
    fields["FPGA_PROGRAMMED_BY_PREFLIGHT"] = str(fpga_programmed).lower()
    if fields.get("UDMABUF") == "unavailable" and initialize_udmabuf:
        modules = (
            "/mnt/sd/tvm_deploy/hp/u-dma-buf.ko",
            "/media/sd-mmcblk1p2/tvm_deploy/hp/u-dma-buf.ko",
        )
        command = (
            "for module in {modules}; do "
            "if test -f \"$module\"; then exec /sbin/insmod \"$module\" udmabuf0={n}; fi; "
            "done; exit 6"
        ).format(
            modules=" ".join(map(shlex.quote, modules)),
            n=int(requirements["udmabuf_bytes"]),
        )
        remote(host, command)
        raw = remote(host, probe).stdout
        fields = parse_preflight(raw)
    expected = {
        "FPGA": requirements["fpga_state"],
        "UDMABUF": str(requirements["udmabuf_bytes"]),
        "FREQ": str(requirements["cpu_frequency_khz"]),
        "BITSTREAM": requirements["bitstream_sha256"],
        "BUSY": "0",
    }
    mismatches = {
        key: {"expected": value, "observed": fields.get(key)}
        for key, value in expected.items() if fields.get(key) != value
    }
    if mismatches:
        raise RuntimeError("board preflight failed: {}".format(mismatches))
    return fields, raw


def manifest_boundary_bytes(manifest):
    return sum(int(stage["output_bytes"]) for stage in manifest["stages"][:-1])


def runner_args(manifest, threads, input_list, runs, output_jsonl, output_dump,
                profile_dir, allocation_snapshot, serial=False, shared_plan=""):
    args = []
    for index, (stage, thread_count) in enumerate(zip(manifest["stages"], threads)):
        prefix = "--stage{}-".format(index)
        args.extend([
            prefix + "graph", stage["graph"],
            prefix + "lib", stage["lib"],
            prefix + "params", stage["params"],
            prefix + "input-names", ",".join(stage["input_names"]),
            prefix + "name", stage["name"],
            prefix + "device", stage["device"],
            prefix + "runtime-num-threads", str(thread_count),
        ])
        if stage["device"] == "cpu":
            args.extend([prefix + "cpu-affinity", ",".join(map(str, range(thread_count)))])
    args.extend([
        "--input-list", input_list,
        "--runs", str(runs),
        "--queue-depth", "2",
        "--runtime-num-threads", "4",
        "--output-mode", "raw",
        "--output-jsonl", output_jsonl,
        "--vta-runtime-profile-dir", profile_dir,
        "--vta-runtime-profile-events-limit", "200",
        "--vta-runtime-profile-checkpoint-every", "0",
    ])
    if serial:
        args.append("--serial")
    else:
        args.extend(["--warmup-runs", "2"])
    if output_dump:
        args.extend(["--output-dump-dir", output_dump])
    if allocation_snapshot:
        args.extend(["--allocation-snapshot", allocation_snapshot])
    if shared_plan:
        args.extend(["--p8-managed-slots", "2", "--shared-buffer-plan", shared_plan])
    return args


def run_native(host, package, runner, driver, args, stdout_path, stderr_path,
               timeout=900):
    env = {
        "LD_LIBRARY_PATH": package,
        "LD_PRELOAD": package + "/libtvm_runtime.so:" + driver,
        "TVM_NUM_THREADS": "4",
        "TVM_THREAD_POOL_SPIN_COUNT": "0",
        "AXU5EVB_DRIVER_POST_START_SLEEP_NS": "1000",
        "AXU5EVB_DRIVER_POLL_SLEEP_NS": "1000",
        "AXU5EVB_DRIVER_SAFE_COPY": "0",
    }
    shell = "cd {package} && env {env} {runner} {args}".format(
        package=shlex.quote(package),
        env=" ".join(shlex.quote(key + "=" + value) for key, value in env.items()),
        runner=shlex.quote(runner),
        args=shlex.join(args),
    )
    started = time.monotonic()
    proc = remote(host, shell, check=False, timeout=timeout)
    elapsed = time.monotonic() - started
    Path(stdout_path).write_bytes(proc.stdout)
    Path(stderr_path).write_bytes(proc.stderr)
    if proc.returncode:
        raise RuntimeError("native runner failed with return code {}".format(proc.returncode))
    return shell, elapsed


def remote_jsonl(host, path, local_path):
    raw = remote(host, "cat " + shlex.quote(path)).stdout
    Path(local_path).write_bytes(raw)
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("empty JSONL: {}".format(path))
    return rows


def remote_archive(host, directory, local_path):
    raw = remote(
        host, "tar -C {} -cf - .".format(shlex.quote(directory)), timeout=300
    ).stdout
    Path(local_path).write_bytes(raw)
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            if member.isfile():
                result[member.name.lstrip("./")] = archive.extractfile(member).read()
    return result


def completion_window_ii(rows, discard):
    ordered = sorted(rows, key=lambda row: int(row["completion_index"]))[discard:]
    if len(ordered) < 2:
        raise ValueError("not enough scored rows for II")
    first, last = float(ordered[0]["completion_ms"]), float(ordered[-1]["completion_ms"])
    if not math.isfinite(first + last) or last <= first:
        raise ValueError("invalid completion window")
    return (last - first) / (len(ordered) - 1)


def boundary_api_ms(rows, stage_count):
    values = []
    for row in rows:
        values.append(sum(
            float(row["stage{}_get_ms".format(edge)])
            + float(row["stage{}_set_ms".format(edge + 1)])
            for edge in range(stage_count - 1)
        ))
    return float(statistics.median(values))


def snapshot_high_water(snapshots):
    phases = {row["phase"]: row for row in snapshots}
    if "after_steady_state" not in phases:
        raise ValueError("allocation snapshot lacks after_steady_state")
    return int(phases["after_steady_state"]["high_water_bytes"])


def verify_remote_hashes(host, package_rows, diagnostic, work_dir):
    expected = {}
    for row in package_rows.values():
        # Deployed manifests can contain board-only path changes and the old
        # package-local libvta is intentionally absent: every mode preloads the
        # one hash-frozen diagnostic driver.  Compiled graph/lib/params and the
        # TVM runtime must stay exact.
        for name, digest in row["package_sha256"].items():
            if name not in ("manifest.json", "libvta.so"):
                expected[row["remote_package"] + "/" + name] = digest
    for name, digest in diagnostic["artifacts"].items():
        expected[work_dir + "/" + name] = digest
    proc = remote(
        host, "sha256sum " + " ".join(map(shlex.quote, expected)),
        check=False, timeout=300,
    )
    output = proc.stdout.decode()
    if proc.returncode:
        raise RuntimeError("remote sha256sum failed: {}".format(proc.stderr.decode().strip()))
    observed = {line.split()[1]: line.split()[0] for line in output.splitlines()}
    if observed != expected:
        missing = {path: digest for path, digest in expected.items() if observed.get(path) != digest}
        raise RuntimeError("remote artifact hash mismatch: {}".format(missing))
    return output


def deploy_assets(host, deployment, work_dir, output_dir):
    diagnostic = deployment["diagnostic_build"]
    for name in diagnostic["artifacts"]:
        upload(host, Path(diagnostic["path"]) / name, work_dir + "/")
    for candidate_id, plan in deployment["shared_plans"].items():
        upload(host, plan["path"], work_dir + "/plan_" + sha256_file(plan["path"])[:12] + ".json")
    input_lines = []
    for item in deployment["correctness_protocol"]["inputs"]:
        if sha256_file(item["path"]) != item["sha256"]:
            raise RuntimeError("local frozen input hash mismatch")
        upload(host, item["path"], work_dir + "/" + item["name"])
        input_lines.append(item["name"])
    local_input_list = Path(output_dir) / "inputs.txt"
    local_input_list.write_text("\n".join(input_lines) + "\n", encoding="ascii")
    upload(host, local_input_list, work_dir + "/inputs.txt")

    for row in deployment["packages"].values():
        if not row["requires_upload"]:
            continue
        package = Path(row["source_package"])
        remote_package = row["remote_package"]
        exists = remote(host, "test -d " + shlex.quote(remote_package), check=False).returncode == 0
        if not exists:
            with tempfile.TemporaryDirectory(prefix="c1_shared_topk_") as temp:
                archive = Path(temp) / "package.tar.gz"
                with tarfile.open(archive, "w:gz") as output:
                    output.add(package / "manifest.json", arcname="manifest.json")
                    output.add(package / "stages", arcname="stages")
                remote_archive_path = work_dir + "/" + archive.name
                upload(host, archive, remote_archive_path, timeout=1200)
                runtime_source = next(
                    item["remote_package"] for item in deployment["packages"].values()
                    if not item["requires_upload"]
                )
                command = (
                    "mkdir {target} && tar -C {target} -xzf {archive} && "
                    "ln -s {runtime}/libtvm_runtime.so {target}/libtvm_runtime.so && "
                    "ln -s {runtime}/libvta.so {target}/libvta.so"
                ).format(
                    target=shlex.quote(remote_package), archive=shlex.quote(remote_archive_path),
                    runtime=shlex.quote(runtime_source),
                )
                remote(host, command, timeout=1200)

    # The four historical board packages intentionally deduplicated large
    # files through v1_p7_top20/_file_cache.  A board cleanup may leave their
    # package symlinks while removing the cache.  Restore only the exact
    # missing digests, reusing regular files from the newly uploaded alias
    # topology whenever possible.
    cache_dir = "/media/sd-mmcblk1p2/v1_p7_top20/_file_cache"
    remote(host, "mkdir -p " + shlex.quote(cache_dir))
    uploaded_by_digest = {}
    for row in deployment["packages"].values():
        if not row["requires_upload"]:
            continue
        for name, digest in row["package_sha256"].items():
            if name not in ("manifest.json", "libvta.so"):
                uploaded_by_digest[digest] = row["remote_package"] + "/" + name
    restored = set()
    for row in deployment["packages"].values():
        if row["requires_upload"]:
            continue
        for name, digest in row["package_sha256"].items():
            if name in ("manifest.json", "libvta.so"):
                continue
            remote_file = row["remote_package"] + "/" + name
            if remote(host, "test -f " + shlex.quote(remote_file), check=False).returncode == 0:
                continue
            cache_file = cache_dir + "/" + digest
            if digest not in restored and remote(
                host, "test -f " + shlex.quote(cache_file), check=False
            ).returncode != 0:
                reusable = uploaded_by_digest.get(digest)
                if reusable and remote(
                    host, "test -f " + shlex.quote(reusable), check=False
                ).returncode == 0:
                    remote(host, "ln -s " + shlex.quote(reusable) + " " + shlex.quote(cache_file))
                else:
                    upload(host, Path(row["source_package"]) / name, cache_file, timeout=1200)
                observed = remote(host, "sha256sum " + shlex.quote(cache_file)).stdout.decode().split()[0]
                if observed != digest:
                    raise RuntimeError("restored file-cache digest mismatch: " + digest)
                restored.add(digest)
            if remote(host, "test -f " + shlex.quote(remote_file), check=False).returncode != 0:
                raise RuntimeError("legacy package remains broken after cache restore: " + remote_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--boot-index", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="root@192.168.1.247")
    parser.add_argument("--initialize-udmabuf", action="store_true")
    parser.add_argument("--initialize-fpga", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    deployment = read_json(args.deployment_manifest)
    if deployment.get("status") != "board_ready_frozen_before_new_performance_labels":
        raise ValueError("deployment manifest is not board-ready")
    if sha256_file(__file__) != deployment["board_runner"]["sha256"]:
        raise ValueError("board runner changed after deployment freeze")
    contract_info = deployment["selection_contract"]
    if sha256_file(contract_info["path"]) != contract_info["sha256"]:
        raise ValueError("selection contract changed after deployment freeze")
    contract = read_json(contract_info["path"])
    order = deployment["boot_configuration_orders"][args.boot_index]
    if set(order) != set(contract["union_configuration_ids"]):
        raise ValueError("boot order differs from frozen union")

    args.output_dir.mkdir(parents=True)
    report = {
        "schema": "vta_shared_memory_topk_board_boot_v1",
        "status": "preflight",
        "boot_index": args.boot_index,
        "selection_contract_sha256": contract_info["sha256"],
        "deployment_manifest_sha256": sha256_file(args.deployment_manifest),
        "configuration_order": order,
        "rows": [],
    }
    write_json(args.output_dir / "preregistered.json", report)
    try:
        board, raw_preflight = board_preflight(
            args.host, deployment["board_requirements"], args.initialize_udmabuf,
            args.initialize_fpga,
        )
        report["boot_id"] = board["BOOT"]
        report["board_preflight"] = board
        work_dir = "/tmp/c1_shared_topk_{}_{}_b{}".format(
            board["BOOT"][:8], report["deployment_manifest_sha256"][:8], args.boot_index
        )
        remote(args.host, "mkdir " + shlex.quote(work_dir))
        report["remote_work_dir"] = work_dir
        (args.output_dir / "board_preflight.txt").write_bytes(raw_preflight)
        deploy_assets(args.host, deployment, work_dir, args.output_dir)
        hashes = verify_remote_hashes(
            args.host, deployment["packages"], deployment["diagnostic_build"], work_dir
        )
        (args.output_dir / "remote_artifact_sha256.txt").write_text(hashes, encoding="utf-8")
        runner = work_dir + "/vta_stage_pipeline_runner"
        driver = work_dir + "/libvta.so"
        input_list = work_dir + "/inputs.txt"

        # Freeze a byte-exact serial reference for each topology before any
        # performance process is launched.
        references = {}
        for candidate_id in contract["union_candidate_ids"]:
            package_row = deployment["packages"][candidate_id]
            package = package_row["remote_package"]
            manifest = read_json(Path(package_row["source_package"]) / "manifest.json")
            stem = "reference_" + sha256_file(Path(package_row["source_package"]) / "manifest.json")[:12]
            output_jsonl = work_dir + "/" + stem + ".jsonl"
            output_dump = work_dir + "/" + stem + "_outputs"
            command = runner_args(
                manifest, package_row["desired_stage_runtime_threads"], input_list, 3,
                output_jsonl, output_dump, "", "", serial=True,
            )
            shell, elapsed = run_native(
                args.host, package, runner, driver, command,
                args.output_dir / (stem + ".stdout"), args.output_dir / (stem + ".stderr"),
            )
            (args.output_dir / (stem + "_command.sh")).write_text(shell + "\n", encoding="utf-8")
            rows = remote_jsonl(args.host, output_jsonl, args.output_dir / (stem + ".jsonl"))
            if len(rows) != 3 or [int(row["input_index"]) for row in rows] != [0, 1, 2]:
                raise RuntimeError("serial correctness reference did not cover three inputs")
            archive_path = args.output_dir / (stem + "_outputs.tar")
            values = remote_archive(args.host, output_dump, archive_path)
            if len(values) != 3:
                raise RuntimeError("serial correctness reference lacks full outputs")
            references[candidate_id] = values
            report.setdefault("correctness_references", {})[candidate_id] = {
                "output_archive_sha256": sha256_file(archive_path),
                "full_output_files": len(values),
                "wall_clock_s": elapsed,
            }
            write_json(args.output_dir / "partial.json", report)

        config_by_id = {row["configuration_id"]: row for row in deployment["configurations"]}
        for position, configuration_id in enumerate(order):
            config = config_by_id[configuration_id]
            candidate_id = config["candidate_id"]
            package_row = deployment["packages"][candidate_id]
            package = package_row["remote_package"]
            manifest = read_json(Path(package_row["source_package"]) / "manifest.json")
            token = "cfg{:02d}_{}".format(position, sha256_text(configuration_id)[:12])
            plan_remote = ""
            if config["policy"] == "pool_anchor_k2_shared":
                plan = deployment["shared_plans"][candidate_id]
                plan_remote = work_dir + "/plan_" + plan["sha256"][:12] + ".json"

            correctness_jsonl = work_dir + "/" + token + "_correctness.jsonl"
            correctness_dump = work_dir + "/" + token + "_correctness_outputs"
            correctness_args = runner_args(
                manifest, package_row["desired_stage_runtime_threads"], input_list, 3,
                correctness_jsonl, correctness_dump, "", "", shared_plan=plan_remote,
            )
            shell, correctness_wall = run_native(
                args.host, package, runner, driver, correctness_args,
                args.output_dir / (token + "_correctness.stdout"),
                args.output_dir / (token + "_correctness.stderr"),
            )
            (args.output_dir / (token + "_correctness_command.sh")).write_text(
                shell + "\n", encoding="utf-8"
            )
            correctness_rows = remote_jsonl(
                args.host, correctness_jsonl, args.output_dir / (token + "_correctness.jsonl")
            )
            correctness_archive = args.output_dir / (token + "_correctness_outputs.tar")
            observed = remote_archive(args.host, correctness_dump, correctness_archive)
            correct = observed == references[candidate_id]
            if len(correctness_rows) != 3 or not correct:
                raise RuntimeError("full-output correctness gate failed for " + configuration_id)

            performance_jsonl = work_dir + "/" + token + "_performance.jsonl"
            profile_dir = work_dir + "/" + token + "_profile"
            allocation = work_dir + "/" + token + "_allocation.jsonl"
            performance_args = runner_args(
                manifest, package_row["desired_stage_runtime_threads"], input_list,
                int(deployment["performance_protocol"]["runs"]), performance_jsonl,
                "", profile_dir, allocation, shared_plan=plan_remote,
            )
            shell, performance_wall = run_native(
                args.host, package, runner, driver, performance_args,
                args.output_dir / (token + "_performance.stdout"),
                args.output_dir / (token + "_performance.stderr"), timeout=1200,
            )
            (args.output_dir / (token + "_performance_command.sh")).write_text(
                shell + "\n", encoding="utf-8"
            )
            performance_rows = remote_jsonl(
                args.host, performance_jsonl, args.output_dir / (token + "_performance.jsonl")
            )
            expected_runs = int(deployment["performance_protocol"]["runs"])
            if len(performance_rows) != expected_runs:
                raise RuntimeError("performance row count mismatch")
            if any(
                tuple(item["fnv1a64"] for item in row["raw_outputs"])
                != tuple(item["fnv1a64"] for item in correctness_rows[int(row["input_index"])] ["raw_outputs"])
                for row in performance_rows
            ):
                raise RuntimeError("performance output drift after correctness gate")
            profile_path = profile_dir + "/benchmark_totals_status.json"
            profile = json.loads(remote(args.host, "cat " + shlex.quote(profile_path)).stdout)
            local_profile = args.output_dir / (token + "_profile.json")
            write_json(local_profile, profile)
            snapshots = remote_jsonl(
                args.host, allocation, args.output_dir / (token + "_allocation.jsonl")
            )
            discard = int(deployment["performance_protocol"]["discard_first_scored_rows"])
            logical_boundary = manifest_boundary_bytes(manifest)
            profiled_frames = len(performance_rows)
            dma_bytes = int(profile.get("load_buffer_2d_bytes", 0)) + int(
                profile.get("store_buffer_2d_bytes", 0)
            )
            dma_calls = int(profile.get("load_buffer_2d_calls", 0)) + int(
                profile.get("store_buffer_2d_calls", 0)
            )
            if dma_bytes % profiled_frames or dma_calls % profiled_frames:
                raise RuntimeError("VTA profile counters are not integral per frame")
            report["rows"].append({
                "configuration_id": configuration_id,
                "candidate_id": candidate_id,
                "policy": config["policy"],
                "correct": True,
                "correctness_full_output_archive_sha256": sha256_file(correctness_archive),
                "ii_ms": completion_window_ii(performance_rows, discard),
                "framework_materialization_bytes": (
                    2 * logical_boundary if config["policy"] == "ordinary_copy"
                    else max(int(row["p8_framework_materialization_bytes"]) for row in performance_rows)
                ),
                "additional_slot_bytes": int(config["additional_slot_bytes"]),
                "boundary_api_service_ms": boundary_api_ms(performance_rows, len(manifest["stages"])),
                "slot_wait_ms_median": float(statistics.median(
                    float(row.get("p8_total_slot_wait_ms", 0.0)) for row in performance_rows
                )),
                "pool_high_water_bytes": snapshot_high_water(snapshots),
                "vta_internal_dma_bytes": dma_bytes // profiled_frames,
                "vta_dma_calls": dma_calls // profiled_frames,
                "correctness_wall_clock_s": correctness_wall,
                "performance_wall_clock_s": performance_wall,
                "performance_result_sha256": sha256_file(args.output_dir / (token + "_performance.jsonl")),
                "profile_sha256": sha256_file(local_profile),
            })
            report["status"] = "running"
            write_json(args.output_dir / "partial.json", report)
            print("[RESULT] {}/{} {} II={:.6f} ms".format(
                position + 1, len(order), configuration_id, report["rows"][-1]["ii_ms"]
            ), flush=True)

        observed, final_raw = board_preflight(args.host, deployment["board_requirements"])
        if observed["BOOT"] != report["boot_id"]:
            raise RuntimeError("board boot changed during experiment")
        (args.output_dir / "board_after.txt").write_bytes(final_raw)
        report["status"] = "complete_one_independent_boot"
        write_json(args.output_dir / "boot.json", report)
        print("[DONE]", report["boot_id"], len(report["rows"]), flush=True)
    except Exception as error:
        report["status"] = "failed_no_retry"
        report["error"] = repr(error)
        write_json(args.output_dir / "failed.json", report)
        raise


def sha256_text(value):
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
