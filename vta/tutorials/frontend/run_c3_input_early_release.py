#!/usr/bin/env python3
"""Frozen, fail-closed ResNet18 input-lease experiment; never edits production code."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPORTS = HERE / "report_out/stage_tile_cotuning"
PACKAGE = HERE / "report_out/resource_aware_maxplus/v1_p7_top20_board_20260904/packages/01_cpu-00-02_t4__vta-03-17_t1__cpu-18-20_t1/package"
A_AUDIT = REPORTS / "c3s_buffer_reuse/s0_static_run1/A.json"
DRIVER = REPORTS / "c3s_buffer_reuse/build_s1_run2/libvta.so"
TVM_RUNTIME = PACKAGE / "libtvm_runtime.so"
BITSTREAM = ROOT / "apps/vta_rpc/firmware/vta_hpc.bit"
BIT_SHA = "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6"
SDK = Path("/home/orange/alinx_plsdk/install/sysroots")
CXX = SDK / "x86_64-petalinux-linux/usr/bin/aarch64-xilinx-linux/aarch64-xilinx-linux-g++"
TARGET = "tvmgen_default_fused_reshape_transpose_multiply_transpose_reshape_round_clip"
ORDER = (("L1", "L2", "E1", "C1"), ("L2", "C1", "L1", "E1"),
         ("C1", "E1", "L2", "L1"), ("E1", "L1", "C1", "L2"))
ENV = {"TVM_NUM_THREADS": "2", "TVM_THREAD_POOL_SPIN_COUNT": "0",
       "AXU5EVB_DRIVER_SAFE_COPY": "0", "AXU5EVB_DRIVER_POST_START_SLEEP_NS": "1000",
       "AXU5EVB_DRIVER_POLL_SLEEP_NS": "1000", "AXU5EVB_DRIVER_DEBUG": "0",
       "AXU5EVB_DRIVER_INFO": "0", "VTA_RUNTIME_TRACE": "0",
       "VTA_QUEUE_DIAGNOSTICS": "0", "VTA_REPLAY_POLICY": "disabled"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".pending")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def run_logged(argv, out, name, *, timeout=120, env=None):
    start = time.monotonic()
    try:
        result = subprocess.run([str(x) for x in argv], capture_output=True, text=True,
                                timeout=timeout, env=env)
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value
        write(Path(out) / (name + ".json"), {
            "argv": [str(x) for x in argv], "returncode": None, "timeout": True,
            "stdout": decode(error.stdout), "stderr": decode(error.stderr),
            "wall_seconds": time.monotonic() - start,
            "target_completion_unknown": True, "automatic_retry": False})
        raise
    write(Path(out) / (name + ".json"), {
        "argv": [str(x) for x in argv], "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
        "wall_seconds": time.monotonic() - start})
    if result.returncode:
        raise RuntimeError(f"{name} failed: {result.stderr[-3000:]}")
    return result.stdout


def source_paths():
    return [Path(__file__), HERE / "c3_input_early_release_runner.cc",
            HERE / "c3_input_early_release_audit.py", HERE / "c3_input_early_release_analysis.py",
            HERE / "test_c3_input_early_release.py", HERE / "test_c3_input_early_release_analysis.py",
            HERE / "c3_input_early_release_validate.py", HERE / "test_c3_input_early_release_validate.py",
            HERE / "test_c3_input_early_release_control.py",
            REPORTS / "c3_dma_residency_autotune/00_governance/C3_INPUT_EARLY_RELEASE_PILOT_V1_1.md",
            ROOT / "vta/apps/native_deploy/vta_stage_pipeline_runner.cc",
            ROOT / "vta/apps/native_deploy/vta_shared_buffer_plan.h",
            ROOT / "vta/apps/native_deploy/vta_managed_lifecycle_control.h",
            ROOT / "vta/runtime/runtime.cc", ROOT / "src/runtime/graph_executor/graph_executor.cc",
            ROOT / "src/runtime/thread_pool.cc"]


def verify_bindings(prepared):
    for filename, expected in prepared["bindings"].items():
        if sha(filename) != expected:
            raise ValueError("frozen input/source changed: " + filename)


def prepare(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        if args.parent_local_attempt is not None:
            parent = args.parent_local_attempt.resolve()
            failed = read(parent / "invalid_prepare.json")
            if failed.get("board_contacted") is not False or (parent / "prepared.json").exists():
                raise ValueError("only a pre-board local preparation failure may have a successor here")
            write(out / "local_attempt_lineage.json", {"parent": str(parent),
                  "parent_failure_sha256": sha(parent / "invalid_prepare.json"),
                  "prior_board_calls": 0, "prior_performance_samples": 0,
                  "reason": "local preparation infrastructure correction before sealing"})
        from c3_input_early_release_audit import audit
        local = audit(PACKAGE, A_AUDIT, TVM_RUNTIME, DRIVER)
        write(out / "local_certificate.json", local)
        if local.get("local_static_passed") is not True or local.get("board_qualified") is not False:
            raise ValueError("local last-reader proof incomplete; see local_certificate.json")
        for file in source_paths():
            if not file.is_file():
                raise ValueError("implementation/test file missing: " + str(file))
        if sha(BITSTREAM) != BIT_SHA:
            raise ValueError("bitstream artifact changed")
        run_logged([sys.executable, "-m", "pytest", "-q", str(HERE / "test_c3_input_early_release.py"),
                    str(HERE / "test_c3_input_early_release_analysis.py"),
                    str(HERE / "test_c3_input_early_release_validate.py"),
                    str(HERE / "test_c3_input_early_release_control.py")], out, "local_tests", timeout=180,
                   env={**os.environ, "PYTHONPATH": os.pathsep.join(
                       [str(HERE), str(ROOT / "python"), os.environ.get("PYTHONPATH", "")])})
        import numpy as np
        from PIL import Image
        images = sorted(p for p in (ROOT / "n02123597").iterdir()
                        if p.is_file() and p.suffix.lower() in (".jpeg", ".jpg"))[:100]
        if len(images) != 100:
            raise ValueError("need first 100 lexical JPEG inputs; no replacement allowed")
        input_dir = out / "inputs"
        input_dir.mkdir()
        inputs = []
        for index, image in enumerate(images):
            with Image.open(image) as opened:
                array = np.array(opened.resize((224, 224)).convert("RGB"), dtype="float32")
            array = (array - np.array([123., 117., 104.], dtype="float32")) / np.array(
                [58.395, 57.12, 57.375], dtype="float32")
            array = np.ascontiguousarray(array.transpose(2, 0, 1)[None], dtype="float32")
            target = input_dir / f"{index:03d}.bin"
            target.write_bytes(array.tobytes())
            inputs.append({"path": str(target), "sha256": sha(target),
                           "image": str(image), "image_sha256": sha(image)})
        libraries = {}
        lib_dir = out / "lib"
        lib_dir.mkdir()
        for label, source in (("tvm_runtime", TVM_RUNTIME), ("vta_runtime", DRIVER)):
            target = lib_dir / ("libtvm_runtime.so" if label == "tvm_runtime" else "libvta.so")
            shutil.copy2(source, target)
            libraries[label] = {"path": str(target), "sha256": sha(source)}
        stages = []
        for index, (device, threads, cpus) in enumerate((("cpu", 2, [0, 1]),
                                                       ("vta", 1, [3]), ("cpu", 1, [2]))):
            stage = {"name": f"stage{index}", "device": device,
                     "threads": threads, "cpus": cpus, "input_names": ["data0"]}
            for label, name in (("graph", "graph.json"), ("lib", "graphlib.so"),
                                ("params", "params.params")):
                path = PACKAGE / f"stages/stage{index}" / name
                stage[label] = str(path)
                stage[label + "_sha256"] = sha(path)
            stages.append(stage)
        plan = {"schema": "c3-input-last-read-v1", "stages": stages,
                "inputs": inputs, "libraries": libraries,
                "target": {"node": 1, "function": TARGET, "entry": 0,
                           "storage_id": 0, "bytes": 802816},
                "warmup": 5, "slot_timeout_seconds": 60, "queue_depth": 2,
                "release_boundary": "after_node1_success_and_node2_entry_range_validation_before_node2_execute",
                "environment": ENV, "formal_order": ORDER,
                "identity": {"bitstream_sha256": BIT_SHA, "udmabuf_bytes": 201326592},
                "protocol": {"reference_frames": 100, "qualification_frames": 32,
                             "event_frames": 3, "proxy_frames": 16, "formal_frames": 100,
                             "planned_full_frames": 1956, "planned_stage_calls": 5868}}
        write(out / "plan.json", plan)
        bin_dir = out / "bin"
        bin_dir.mkdir()
        command = [str(CXX), "--sysroot=" + str(SDK / "aarch64-xilinx-linux"),
                   "-std=c++17", "-O2", "-pthread", "-ffunction-sections", "-fdata-sections",
                   "-DTVM_KALLOC_ALIGNMENT=256", "-DTVM_INDEX_DEFAULT_I64=1",
                   "-DDMLC_USE_FOPEN64=0", "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>"]
        for folder in ("include", "3rdparty/dlpack/include", "3rdparty/dmlc-core/include",
                       "3rdparty/vta-hw/include", "3rdparty/picojson"):
            command.append("-I" + str(ROOT / folder))
        command += [str(HERE / "c3_input_early_release_runner.cc"), "-L" + str(lib_dir),
                    "-Wl,--gc-sections", "-Wl,-rpath,$ORIGIN/../lib", "-ltvm_runtime", "-lvta",
                    "-ldl", "-lrt", "-o", str(bin_dir / "c3_input_early_release_runner")]
        run_logged(command, out, "cross_compile", timeout=180)
        host_command = ["g++", "-std=c++17", "-O2", "-pthread", "-ffunction-sections", "-fdata-sections",
                        "-DC3_LAST_READ_SELFTEST_ONLY", "-DTVM_KALLOC_ALIGNMENT=256", "-DTVM_INDEX_DEFAULT_I64=1",
                        "-DDMLC_USE_FOPEN64=0", "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>"]
        host_command += ["-I" + str(ROOT / folder) for folder in
                         ("include", "3rdparty/dlpack/include", "3rdparty/dmlc-core/include",
                          "3rdparty/vta-hw/include", "3rdparty/picojson")]
        host_command += [str(HERE / "c3_input_early_release_runner.cc"), "-Wl,--gc-sections",
                         "-o", str(bin_dir / "lease_selftest_host")]
        run_logged(host_command, out, "host_selftest_compile", timeout=180)
        run_logged([bin_dir / "lease_selftest_host"], out, "host_selftest", timeout=15)
        bindings = {str(p): sha(p) for p in source_paths()}
        for stage in stages:
            for key in ("graph", "lib", "params"):
                bindings[stage[key]] = stage[key + "_sha256"]
        for item in inputs:
            bindings[item["path"]] = item["sha256"]
            bindings[item["image"]] = item["image_sha256"]
        for item in libraries.values():
            bindings[item["path"]] = item["sha256"]
        bindings[str(BITSTREAM)] = BIT_SHA
        bindings[str(out / "plan.json")] = sha(out / "plan.json")
        bindings[str(bin_dir / "c3_input_early_release_runner")] = sha(bin_dir / "c3_input_early_release_runner")
        prepared = {"status": "PREPARED", "bindings": bindings,
                    "board_contacted": False, "timing_valid": False,
                    "local_certificate_sha256": sha(out / "local_certificate.json")}
        write(out / "prepared.json", prepared)
        write(out / "state.json", {"phase": "PREPARED", "global_stop": False})
        report(out, "本地准备完成，尚无板端性能结果。")
    except BaseException as error:
        write(out / "invalid_prepare.json", {"status": "LOCAL_GATE_FAILED", "error": str(error),
                                            "board_contacted": False, "timing_valid": False})
        report(out, "本地准备停止：" + str(error))
        raise


class Board:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o",
                        "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + args.known_hosts,
                        "-o", "HostKeyAlgorithms=+ssh-rsa", "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa"]
        self.remote = "/var/volatile/" + self.out.name

    def ssh(self, command, name, timeout=30):
        return run_logged(["ssh", *self.options, "root@" + self.args.host, command],
                          self.out / "infrastructure", name, timeout=timeout)

    def scp(self, source, destination, name, *, receive=False):
        remote = "root@" + self.args.host + ":" + destination
        argv = ["scp", "-O", *self.options]
        argv += [remote, str(source)] if receive else [str(source), remote]
        return run_logged(argv, self.out / "infrastructure", name, timeout=120)

    def preflight(self, name):
        raw = self.ssh("cat /proc/sys/kernel/random/boot_id; cat /proc/version; mount; ps; "
                       "cat /sys/class/fpga_manager/fpga0/state; "
                       "cat /sys/class/u-dma-buf/udmabuf0/size; "
                       "sha256sum /lib/firmware/vta_hpc.bit; "
                       "command -v taskset; sha256sum /usr/bin/taskset; cat /sys/devices/system/cpu/online; "
                       "dmesg | tail -n 30", name)
        if "none on / type rootfs (rw)" not in raw or "tmpfs on /var/volatile type tmpfs" not in raw:
            raise ValueError("RAM rootfs/tmpfs not attested; no board writes allowed")
        if BIT_SHA not in raw or "\n201326592\n" not in raw or "\n0-3\n" not in raw:
            raise ValueError("frozen firmware/DMA/CPU prerequisites missing; stop without reconfiguration")
        if "\noperating\n" not in raw or "\n/usr/bin/taskset\n" not in raw:
            raise ValueError("FPGA state or taskset prerequisite missing")
        # The probe command itself contains neither experiment executable nor RPC server name.
        for line in raw.splitlines():
            if line.strip().split()[:1] and line.lstrip()[0:1].isdigit():
                if re.search(r"(?:tvm[._]rpc|vta_stage_pipeline_runner|c3\S*runner|vta\.exec\.rpc_server|run_vta_)", line):
                    raise ValueError("another board experiment is active; do not stop it")
        taskset_hash = next((line.split()[0] for line in raw.splitlines()
                             if line.endswith("  /usr/bin/taskset")), None)
        if taskset_hash is None:
            raise ValueError("cannot attest actual board taskset")
        if hasattr(self, "taskset_hash") and taskset_hash != self.taskset_hash:
            raise ValueError("board affinity tool changed inside experiment")
        self.taskset_hash = taskset_hash
        return {"boot_id": raw.splitlines()[0], "taskset_sha256": taskset_hash, "raw": raw}

    def deploy(self, plan):
        import tarfile
        package = self.out / "deployment.tar"
        remote_plan = json.loads(json.dumps(plan))
        with tarfile.open(package, "x") as archive:
            for stage in remote_plan["stages"]:
                for label, filename in (("graph", "graph.json"), ("lib", "graphlib.so"),
                                        ("params", "params.params")):
                    rel = "stages/" + stage["name"] + "/" + filename
                    archive.add(stage[label], arcname=rel, recursive=False)
                    stage[label] = self.remote + "/" + rel
            for index, item in enumerate(remote_plan["inputs"]):
                rel = f"inputs/{index:03d}.bin"
                archive.add(item["path"], arcname=rel, recursive=False)
                item["path"] = self.remote + "/" + rel
            for label, item in remote_plan["libraries"].items():
                rel = "lib/" + Path(item["path"]).name
                archive.add(item["path"], arcname=rel, recursive=False)
                item["path"] = self.remote + "/" + rel
            archive.add(self.out / "bin/c3_input_early_release_runner", arcname="bin/c3_input_early_release_runner")
        remote_plan["identity"]["boot_id"] = self.boot_id
        write(self.out / "remote_plan.json", remote_plan)
        q = shlex.quote
        self.ssh("test ! -e " + q(self.remote) + " && mkdir " + q(self.remote), "deploy_mkdir")
        self.scp(package, self.remote + "/deployment.tar", "deploy_archive")
        self.scp(self.out / "remote_plan.json", self.remote + "/plan.json", "deploy_plan")
        self.ssh("cd " + q(self.remote) + " && test \"$(sha256sum deployment.tar | cut -d' ' -f1)\" = "
                 + sha(package) + " && tar -xf deployment.tar && chmod 755 bin/c3_input_early_release_runner",
                 "deploy_unpack")

    def invoke(self, name, variant, mode, frames, *, proxy_off=False):
        q = shlex.quote
        local = self.out / "board" / name
        if local.exists():
            raise ValueError("never overwrite/retry a board attempt: " + name)
        local.mkdir(parents=True)
        before = self.preflight(name + "_before")
        if before["boot_id"] != self.boot_id:
            raise ValueError("Linux boot changed")
        self.ssh("test \"$(sha256sum /lib/firmware/vta_hpc.bit | cut -d' ' -f1)\" = " + BIT_SHA
                 + " && echo vta_hpc.bit > /sys/class/fpga_manager/fpga0/firmware"
                 + " && test \"$(cat /sys/class/fpga_manager/fpga0/state)\" = operating", name + "_reload")
        binary = self.remote + "/bin/c3_input_early_release_runner"
        destination = self.remote + "/results/" + name
        env = {**ENV, "LD_LIBRARY_PATH": self.remote + "/lib",
               "LD_PRELOAD": self.remote + "/lib/libtvm_runtime.so:" + self.remote + "/lib/libvta.so"}
        argv = ["taskset", "-c", "0-3", "env", *[k + "=" + v for k, v in env.items()], binary,
                "--plan", self.remote + "/plan.json", "--variant", variant, "--mode", mode,
                "--frames", str(frames), "--output-dir", destination]
        if proxy_off:
            argv.append("--proxy-off")
        begin = time.monotonic()
        append(self.out / "dispatch.jsonl", {"event": "dispatch", "name": name, "variant": variant,
               "mode": mode, "frames": frames, "boot_id": self.boot_id, "argv": argv})
        try:
            self.ssh(shlex.join(argv), name + "_run", timeout=180)
        except BaseException:
            # Do not retry, kill a foreign process, or launch another target. Try read-only retrieval.
            try:
                self.scp(local / "result.json", destination + "/result.json", name + "_failed_result", receive=True)
            except Exception:
                pass
            raise
        self.scp(local / "result.json", destination + "/result.json", name + "_result", receive=True)
        self.scp(local / "outputs.bin", destination + "/outputs.bin", name + "_outputs", receive=True)
        result = read(local / "result.json")
        result.update(variant=variant, mode=mode, attempt=name, boot_id=self.boot_id,
                      local_directory=str(local), wall_seconds=time.monotonic() - begin)
        if result.get("status") not in ("passed", "PASSED", "complete", "COMPLETE"):
            raise ValueError("native attempt rejected: " + name)
        result["status"] = "passed"
        if self.preflight(name + "_after")["boot_id"] != self.boot_id:
            raise ValueError("boot changed inside attempt")
        write(local / "collected.json", result)
        append(self.out / "dispatch.jsonl", {"event": "complete", "name": name,
               "frames": len(result["frames"]), "wall_seconds": result["wall_seconds"]})
        return result


def compare_outputs(result, reference):
    expected = Path(reference["local_directory"], "outputs.bin").read_bytes()
    actual = Path(result["local_directory"], "outputs.bin").read_bytes()
    refs = {}
    reference_offset = 0
    for row in reference["frames"]:
        offset, size = row["output_offset"], row["output_bytes"]
        if (row["input_index"] in refs or offset != reference_offset or size <= 0
                or offset + size > len(expected) or row.get("complete") is not True):
            raise ValueError("invalid/duplicate full reference snapshot")
        reference_offset += size
        refs[row["input_index"]] = expected[offset:offset + size]
    if set(refs) != set(range(100)) or sum(map(len, refs.values())) != len(expected):
        raise ValueError("reference must contain exactly 100 complete, distinct input snapshots")
    consumed = 0
    for row in result["frames"]:
        offset, size = row["output_offset"], row["output_bytes"]
        if offset != consumed or size <= 0 or row.get("complete") is not True:
            raise ValueError("missing or noncontiguous output snapshot")
        consumed += size
        got = actual[offset:offset + size]
        want = refs[row["input_index"]]
        if len(got) != size or got != want:
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            write(Path(result["local_directory"]) / "first_difference.json", {
                "frame_id": row["frame_id"], "input_index": row["input_index"],
                "byte_offset": first, "actual_size": len(got), "expected_size": len(want)})
            raise ValueError("complete output mismatch: " + result["attempt"])
    if consumed != len(actual):
        raise ValueError("unexpected/truncated deferred output bytes")
    result["correctness_verified"] = True
    result["outputs_verified"] = True
    write(Path(result["local_directory"]) / "output_check.json", {
        "passed": True, "frames": len(result["frames"]), "all_bytes_equal": True,
        "outputs_sha256": sha(Path(result["local_directory"]) / "outputs.bin")})


def check_native_evidence(result):
    from c3_input_early_release_validate import validate_native
    # Recompute the call/lease certificates from raw events, not a status string.
    result.update(validate_native(result))
    if not result.get("affinity_verified", False):
        raise ValueError("actual actor/worker affinity not verified")
    proxy_off_control = (result["mode"] == "proxycheck" and result["variant"] == "L2"
                         and result.get("proxy_enabled") is False)
    if not proxy_off_control and result.get("dynamic_ranges_verified") is not True:
        raise ValueError("actual input/intermediate/packed address separation not verified")
    result["environment_verified"] = True
    write(Path(result["local_directory"]) / "verified.json", result)


def qualify(args):
    out = args.out.resolve()
    prepared = read(out / "prepared.json")
    verify_bindings(prepared)
    state = read(out / "state.json")
    if state["phase"] != "PREPARED" or state["global_stop"]:
        raise ValueError("qualification requires a fresh prepared attempt")
    board = Board(args)
    try:
        before = board.preflight("initial_readonly")
        board.boot_id = before["boot_id"]
        write(out / "boot.json", before)
        board.deploy(read(out / "plan.json"))
        write(out / "state.json", {"phase": "QUALIFICATION_RUNNING", "global_stop": False,
                                   "boot_id": board.boot_id, "remote": board.remote})
        reference = board.invoke("reference", "L1", "reference", 100)
        check_native_evidence(reference)
        qualification = []
        for variant in ("L1", "L2", "C1", "E1"):
            result = board.invoke("qualify_" + variant, variant, "qualify", 32)
            compare_outputs(result, reference)
            check_native_evidence(result)
            qualification.append({"attempt": result["attempt"], "passed": True,
                                  "frames": len(result["frames"])})
        for kind in ("event-before", "event-after"):
            result = board.invoke(kind, "E1", kind, 3)
            compare_outputs(result, reference)
            check_native_evidence(result)
            qualification.append({"attempt": result["attempt"], "passed": True,
                                  "frames": len(result["frames"])})
        from c3_input_early_release_analysis import block_metrics
        proxy = []
        for disabled in (True, False):
            result = board.invoke("proxy_off" if disabled else "proxy_on", "L2", "proxycheck", 16,
                                  proxy_off=disabled)
            compare_outputs(result, reference)
            check_native_evidence(result)
            proxy.append(block_metrics(result, expected_frames=16))
        perturbation = abs(proxy[1]["fps"] / proxy[0]["fps"] - 1)
        write(out / "proxy_calibration.json", {"metrics": proxy, "absolute_fps_change": perturbation,
                                               "passed": perturbation <= .02})
        if perturbation > .02:
            raise ValueError("proxy perturbation exceeds frozen 2% infrastructure gate")
        write(out / "qualification.json", {"passed": True, "reference_frames": 100,
                                           "items": qualification, "proxy_calibration_passed": True,
                                           "boot_id": board.boot_id})
        verify_bindings(prepared)
        write(out / "state.json", {"phase": "QUALIFIED", "global_stop": False,
                                   "boot_id": board.boot_id, "remote": board.remote})
        report(out, "四方法、最后读取事件与代理资格通过，正式计时尚未开始。")
    except BaseException as error:
        stop(out, "qualify", error)
        raise


def measure(args):
    out = args.out.resolve()
    prepared, state = read(out / "prepared.json"), read(out / "state.json")
    verify_bindings(prepared)
    if state["phase"] != "QUALIFIED" or state["global_stop"]:
        raise ValueError("formal measurement requires fully qualified, non-failed attempt")
    board = Board(args)
    board.boot_id = state["boot_id"]
    reference = read(out / "board/reference/collected.json")
    blocks = []
    try:
        write(out / "state.json", {**state, "phase": "MEASUREMENT_RUNNING"})
        for round_index, order in enumerate(ORDER, 1):
            for variant in order:
                name = f"r{round_index}_{variant}"
                result = board.invoke(name, variant, "score", 100)
                compare_outputs(result, reference)
                check_native_evidence(result)
                result.update(round=round_index, qualified=True)
                write(out / "board" / name / "verified.json", result)
                blocks.append(result)
                write(out / "blocks_partial.json", blocks)
        verify_bindings(prepared)
        write(out / "blocks.json", blocks)
        write(out / "state.json", {**state, "phase": "MEASURED"})
        analyze(args)
    except BaseException as error:
        stop(out, "measure", error)
        raise


def analyze(args):
    from c3_input_early_release_analysis import analyze_blocks
    out = args.out.resolve()
    state = read(out / "state.json")
    if state["global_stop"] or state["phase"] not in ("MEASURED", "COMPLETE"):
        report(out, "会话未完整完成，不输出正式排名或性能结论。")
        return
    analysis = analyze_blocks(read(out / "blocks.json"))
    qualification = read(out / "qualification.json")
    proxy = read(out / "proxy_calibration.json")
    if qualification.get("passed") is not True or proxy.get("passed") is not True:
        raise ValueError("campaign prerequisites not passed")
    actual_frames = actual_stages = 0
    attempts = []
    for file in sorted((out / "board").glob("*/verified.json")):
        result = read(file)
        count = len(result["frames"])
        actual_frames += count
        actual_stages += sum(result["stage_call_counts"])
        attempts.append({"name": result["attempt"], "frames": count,
                         "stage_calls": result["stage_call_counts"]})
    if actual_frames != 1956 or actual_stages != 5868 or len(attempts) != 25:
        raise ValueError("complete trial call ledger disagrees with frozen 1956/5868 budget")
    ledger = {"complete_inferences": actual_frames, "stage_calls": actual_stages,
              "attempts": attempts, "passed": True}
    write(out / "call_ledger.json", ledger)
    analysis["campaign_qualification_passed"] = True
    analysis["actual_trial_ledger"] = ledger
    write(out / "analysis.json", analysis)
    verify_bindings(read(out / "prepared.json"))
    write(out / "state.json", {**state, "phase": "COMPLETE", "paused_after_delivery": True})
    report(out, "单启动试点完成，已暂停；不自动追加重启或新候选。", analysis)
    seal_outputs(out)


def stop(out, phase, error):
    previous = read(out / "state.json") if (out / "state.json").exists() else {}
    write(out / "invalid_session.json", {"phase": phase, "error": str(error),
                                        "all_formal_timing_invalid": True,
                                        "no_retry": True, "no_splicing": True})
    write(out / "state.json", {**previous, "phase": "STOPPED", "global_stop": True})
    report(out, "实验停止，未获得可用正式结论：" + str(error))
    seal_outputs(out)


def seal_outputs(out):
    out = Path(out)
    write(out / "OUTPUT_HASHES.json", {str(p.relative_to(out)): sha(p)
          for p in sorted(out.rglob("*")) if p.is_file()
          and p.name != "OUTPUT_HASHES.json" and not p.name.endswith(".pending")})


def report(out, conclusion, analysis=None):
    lines = ["# ResNet18共享输入提前归还试点", "", conclusion, "",
             "固定三阶段ResNet18、224×224；目标是单输入槽接近双槽吞吐。",
             "L1和E1借用原输入池；L2额外分配一槽；C1将外部源复制到私有输入池。",
             "单次Linux启动，正式比较不包含图片读取、预处理、参考计算或磁盘写入。", ""]
    if analysis is not None:
        from c3_input_early_release_analysis import render_report
        lines += [render_report(analysis), "完整逐轮数据、内存与继续门槛见 analysis.json；实际调用账目见 call_ledger.json。", ""]
    lines += ["## 结论边界", "", "边界槽节省不等于整图内存减半；单启动门槛不等于统计显著性。",
              "原有计算二进制、生产runner、GraphExecutor、runtime及历史报告均不修改。", ""]
    (Path(out) / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "qualify", "measure", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    parser.add_argument("--parent-local-attempt", type=Path,
                        help="prepare only: link a preserved pre-board local preparation failure")
    args = parser.parse_args()
    if args.phase != "prepare" and args.parent_local_attempt is not None:
        parser.error("--parent-local-attempt is confined to prepare")
    try:
        globals()[args.phase](args)
    except BaseException as error:
        state_file = args.out.resolve() / "state.json"
        if args.phase != "prepare" and state_file.exists() and read(state_file).get("phase") != "STOPPED":
            stop(args.out.resolve(), args.phase, error)
        raise


if __name__ == "__main__":
    main()
