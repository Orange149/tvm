"""Independently measure result serialization using real reference tensors.

Four CPU-only sessions, zero DNN/FPGA calls, unchanged pipeline executable.
This measures a known subset of collector work, not all control overhead.
"""
from pathlib import Path
import argparse
import json
import os
import shlex
import shutil
import statistics as st
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cpu_vta_partitioning.common import read, write, sha256, digest, inventory, verify_inventory
from cpu_vta_partitioning.transport import SSH
from finite_service.common import freeze
from phase_profile.prepare import link_checked

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/output_control_profile_02"
BUNDLE = OUT / "capsule"
RUN = ROOT / "runs/output_control_profile_02"
BOOT = "6018b3b7-bb29-4f80-92c1-0bfd8c81bb99"
Q = shlex.quote


def specs():
    return [dict(network=n, ring_count=r) for n in ("resnet18", "yolov3_tiny416") for r in (1, 4)]


def prepare():
    if (BUNDLE / "manifest.json").exists():
        return validate()
    BUNDLE.mkdir(parents=True, exist_ok=True)
    frozen = ROOT / "artifacts/scheduler_gap_audit_01/build"
    build = read(frozen / "BUILD.json")
    for path, expected in build["source_files"].items():
        if sha256(path) != expected:
            raise ValueError("frozen executable source changed")
    tc = build["toolchain"]
    if sha256(tc["sdk_setup"]) != tc["sdk_setup_sha256"]:
        raise ValueError("SDK changed")
    for tool in tc["tools"].values():
        if sha256(tool["path"]) != tool["sha256"]:
            raise ValueError("compiler changed")
    parents = {n: ROOT / "artifacts/scheduler_gap_audit_01" / n / "capsule" for n in ("resnet18", "yolov3_tiny416")}
    references = {}
    for network, parent in parents.items():
        m = read(parent / "manifest.json")
        programs = read(parent / "plans/programs.json")
        program = programs["cpu_20_20" if network == "resnet18" else "cpu_13_13"]
        record = read(parent / "plans/inputs.json")["records"][0]
        references[network] = []
        lines = [str(7 if network == "resnet18" else 5) + " " + str(len(program["output_node_ids"]))]
        for node in program["output_node_ids"]:
            ref = record["frontiers"][str(node)]
            relative = "data/" + str(node) + ".bin"
            target = BUNDLE / "inputs" / network / relative
            link_checked(parent / ref["raw"], target, m["files"][ref["raw"]]["sha256"])
            code = 2 if ref["dtype"] == "float32" else 0 if ref["dtype"] == "int32" else None
            if code is None:
                raise ValueError("unexpected output dtype")
            lines.append(" ".join(map(str, [code, 32, len(ref["shape"]), *ref["shape"], relative])))
            references[network].append(dict(**ref, source_bundle=str(parent), source_sha256=sha256(parent / ref["raw"]),
                capsule_path=str(target.relative_to(BUNDLE))))
        (BUNDLE / "inputs" / network / "input.txt").write_text("\n".join(lines) + "\n")
    runtime_parent = parents["resnet18"]
    for name, value in read(runtime_parent / "manifest.json")["files"].items():
        if name.startswith("runtime/"):
            link_checked(runtime_parent / name, BUNDLE / name, value["sha256"])
    plan = dict(specs=specs(), references=references, expected_boot_id=BOOT,
        source_files={str(ROOT / "output_control_profile.cc"): sha256(ROOT / "output_control_profile.cc"),
                      str(Path(__file__)): sha256(__file__), **build["source_files"]},
        frozen_pipeline_executable_sha256=build["runner_sha256"],
        exact_ResultJSON_source_reused=True, original_pipeline_main_not_called=True,
        only_reference_output_buffers_no_graph_module_loading=True,
        warmup=2, scored=8, native_sessions=4, DNN_inferences=0, FPGA_calls=0,
        ring_sizes_declared_before_measurement=[1, 4],
        cache_regimes_are_sensitivity_not_proven_pipeline_cache_equivalence=True,
        measured_subset="ResultJSON and deferred string insertion; no disk I/O",
        unmeasured=["input production", "queue/lock work", "frame allocation/destruction", "result buffer release"],
        no_pipeline_values_as_inputs=True, no_score_or_DP_change=True, no_reboot=True,
        no_automatic_retry_or_extra_samples=True)
    freeze(OUT / "PLAN.json", plan)
    dest = BUNDLE / "runner/output_control_profile"
    dest.parent.mkdir(parents=True, exist_ok=True)
    recipe = ('exec $CXX -std=c++17 -O2 -ffunction-sections -fdata-sections "$1" -o "$2" '
        '-I"$5" -I"$3/include" -I"$3/3rdparty/dlpack/include" -I"$3/3rdparty/dmlc-core/include" '
        '-I"$3/3rdparty/vta-hw/include" -L"$4" -Wl,--gc-sections -Wl,-rpath,\'$ORIGIN/../runtime\' '
        '-ltvm_runtime -lvta -ldl -pthread')
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    t = time.monotonic()
    result = subprocess.run(["bash", "-c", 'source "$1" || exit $?; shift; exec "$@"', "output-sdk", tc["sdk_setup"],
        "bash", "-c", recipe, "output-compile", str(ROOT / "output_control_profile.cc"), str(dest),
        str(ROOT.parents[4]), str(BUNDLE / "runtime"), str(frozen / "sources")],
        env=env, capture_output=True, text=True, timeout=180)
    (OUT / "compile.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ValueError("CPU-only control probe compilation failed; see compile.log")
    write(OUT / "BUILD.json", dict(executable_sha256=sha256(dest), recipe=recipe, toolchain=tc,
        plan_sha256=sha256(OUT / "PLAN.json"), seconds=time.monotonic()-t,
        runtime_recompiled=False, stage_modules_recompiled=False, pipeline_runner_changed=False))
    write(BUNDLE / "plans/board_execution.json", dict(storage_margin_bytes=67108864))
    payload = inventory(BUNDLE)
    write(BUNDLE / "plans/board_payload.json", dict(files=payload, unpacked_bytes=sum(v["bytes"] for v in payload.values())))
    manifest = dict(experiment="isolated_output_control_profile_02", files=inventory(BUNDLE),
        plan_sha256=sha256(OUT / "PLAN.json"), executable_sha256=sha256(dest))
    manifest["bundle_id"] = digest(manifest)[:24]
    write(BUNDLE / "manifest.json", manifest)
    return validate()


def validate():
    m = read(BUNDLE / "manifest.json")
    if m["bundle_id"] != digest({k: v for k, v in m.items() if k != "bundle_id"})[:24]:
        raise ValueError("capsule seal differs")
    verify_inventory(BUNDLE, m["files"])
    if m["plan_sha256"] != sha256(OUT / "PLAN.json"):
        raise ValueError("control-profile plan changed")
    for p, expected in read(OUT / "PLAN.json")["source_files"].items():
        if sha256(p) != expected:
            raise ValueError("control-profile source changed")
    return dict(validated=True, bundle_id=m["bundle_id"], DNN_inferences=0, planned_native_sessions=4)


def validate_output(value, spec, *, native=True):
    if any(value[k] != v for k, v in dict(kind="isolated_output_serialization", DNN_inferences=0,
            FPGA_calls=0, warmup=2, scored=8, ring_count=spec["ring_count"],
            stage_count=7 if spec["network"] == "resnet18" else 5).items()):
        raise ValueError("unexpected CPU-only execution identity")
    if value["affinity"] != "0,1,2,3" or len(value["samples"]) != 8:
        raise ValueError("wrong affinity or sample count")
    refs = read(OUT / "PLAN.json")["references"][spec["network"]]
    if len(value["raw_outputs"]) != len(refs):
        raise ValueError("missing output tensors")
    byte_count = 0
    for i, (actual, ref) in enumerate(zip(value["raw_outputs"], refs)):
        data = (BUNDLE / ref["capsule_path"]).read_bytes()
        h = 1469598103934665603
        for b in data:
            h = ((h ^ b) * 1099511628211) & ((1 << 64)-1)
        if (actual["index"] != i or actual["shape"] != ref["shape"] or actual["dtype"] != ref["dtype"]
            or actual["nbytes"] != len(data) or actual["fnv1a64"] != f"{h:016x}"):
            raise ValueError("output data do not match frozen real references")
        byte_count += len(data)
    if value["output_bytes"] != byte_count:
        raise ValueError("output size mismatch")
    if any(s["CPU_ns"] <= 0 or s["wall_ns"] <= 0 or s["JSON_bytes"] <= 0
           or (native and s["CPU_ns"] > s["wall_ns"] + 20000) for s in value["samples"]):
        raise ValueError("invalid CPU/wall samples")
    return value


def qemu():
    validate()
    for spec in specs():
        key = spec["network"] + "_ring" + str(spec["ring_count"])
        path = OUT / "qemu" / (key + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            r = subprocess.run(["qemu-aarch64-static", str(BUNDLE / "runtime/ld-linux-aarch64.so.1"),
                "--library-path", str(BUNDLE / "runtime"), str(BUNDLE / "runner/output_control_profile"),
                str(BUNDLE / "inputs" / spec["network"] / "input.txt"), str(spec["ring_count"])],
                capture_output=True, text=True, timeout=60)
            (path.with_suffix(".log")).write_text(r.stderr)
            if r.returncode:
                raise ValueError("QEMU CPU-only output replay failed: " + r.stderr[-1200:])
            write(path, json.loads(r.stdout))
        # QEMU virtual thread CPU and wall clocks are not a native timing test.
        validate_output(read(path), spec, native=False)
    freeze(OUT / "QEMU.json", dict(passed=True, files={str(p): sha256(p) for p in (OUT / "qemu").glob("*.json")},
        not_native_performance=True, DNN_inferences=0))
    return dict(passed=True, QEMU_sessions=4, DNN_inferences=0)


def board(*, execute_board=False):
    if not execute_board:
        raise ValueError("--execute-board is required")
    validate()
    if not read(OUT / "QEMU.json")["passed"]:
        raise ValueError("QEMU verification required")
    RUN.mkdir(parents=True, exist_ok=True)
    config = read(ROOT / "artifacts/cross_network_model_01/board.local.json")["board"]
    ssh = SSH(config, execute_board=True)
    boot = ssh.boot_id()
    if boot != BOOT:
        raise ValueError("boot changed; keep batch separate")
    remote = ssh.deploy(BUNDLE, RUN)
    job = config["remote_base"].rstrip("/") + "/zero_dnn/output_control_profile_02"
    for spec in specs():
        key = spec["network"] + "_ring" + str(spec["ring_count"])
        receipt = RUN / (key + ".json")
        if receipt.exists():
            r = read(receipt)
            if r["spec"] != spec or r["boot_id"] != boot or r["plan_sha256"] != sha256(OUT / "PLAN.json"):
                raise ValueError("accepted session identity changed")
            validate_output(r["output"], spec)
            continue
        directory = job + "/" + key
        command = [remote + "/runtime/ld-linux-aarch64.so.1", "--library-path", remote + "/runtime",
            remote + "/runner/output_control_profile", remote + "/inputs/" + spec["network"] + "/input.txt", str(spec["ring_count"])]
        script = "\n".join(["set -eu", "mkdir -p " + Q(directory), "cd " + Q(directory),
            "if test -f done; then sha256sum -c done >/dev/null; else",
            "test ! -e started || { echo 'incomplete prior CPU-only session; no automatic repeat' >&2; exit 94; }",
            "touch started", "env -i PATH=/usr/bin:/bin " + " ".join(map(Q, command)) + " > output.json 2>stderr.log",
            "sha256sum output.json > done", "fi", "cat output.json"])
        raw = ssh.idle_command(script, boot=boot, timeout=60)
        value = validate_output(json.loads(raw), spec)
        freeze(receipt, dict(spec=spec, output=value, output_sha256=digest(value), boot_id=boot,
            bundle_id=read(BUNDLE / "manifest.json")["bundle_id"], plan_sha256=sha256(OUT / "PLAN.json"),
            native_command=command, remote_output=directory + "/output.json", DNN_inferences=0, FPGA_calls=0))
    if ssh.boot_id() != boot:
        raise ValueError("boot changed during CPU-only collection")
    write(RUN / "state.json", dict(status="COMPLETE", boot_id=boot, accepted_sessions=4,
        DNN_inferences=0, FPGA_calls=0), replace=True)
    return analyze()


def analyze():
    validate()
    rows, sources = [], {}
    for spec in specs():
        key = spec["network"] + "_ring" + str(spec["ring_count"])
        path = RUN / (key + ".json")
        r = read(path)
        if r["boot_id"] != BOOT or r["plan_sha256"] != sha256(OUT / "PLAN.json") or r["output_sha256"] != digest(r["output"]):
            raise ValueError("native output provenance changed")
        v = validate_output(r["output"], spec)
        cpus = [s["CPU_ns"] / 1e6 for s in v["samples"]]
        rows.append(dict(**spec, CPU_ms=st.mean(cpus), CPU_min_ms=min(cpus), CPU_max_ms=max(cpus),
            wall_ms=st.mean(s["wall_ns"] for s in v["samples"]) / 1e6,
            output_bytes=v["output_bytes"], output_count=len(v["raw_outputs"]), samples=8))
        sources[str(path)] = sha256(path)
    result = dict(rows=rows, sources=sources, scope="independent subset of collector CPU work",
        DNN_inferences=0, FPGA_calls=0, native_sessions=4, no_score_or_DP_change=True,
        no_total_collector_cost_claim=True, pipeline_cache_state_not_reconstructed=True,
        source_sha256=sha256(__file__), plan_sha256=sha256(OUT / "PLAN.json"))
    write(OUT / "RESULTS.json", result, replace=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "validate", "qemu", "board", "analyze"))
    p.add_argument("--execute-board", action="store_true")
    a = p.parse_args()
    result = board(execute_board=a.execute_board) if a.command == "board" else globals()[a.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
