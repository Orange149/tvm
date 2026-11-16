#!/usr/bin/env python3
"""Persisted, fail-closed single-input preparation and board qualification.

This campaign is a smoke test, never a replacement for the formal stage gates.
Each operation records its actual subprocess costs in the immutable RunStore.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tarfile
import time
import uuid

from thesis_supplement_run import RunStore, sha256, write_new_json

FRONTEND = Path(__file__).resolve().parent
ROOT = FRONTEND.parents[2]
BASE = FRONTEND / "report_out/stage_tile_cotuning/thesis_supplement_experiments"
PREPARATION = BASE / "results/01/20260916T011355Z_completion_0670cfb1"
PYTHON = "/home/orange/miniconda3/envs/vta-resnet/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
       "-o", "StrictHostKeyChecking=yes", "-o", "HostKeyAlgorithms=+ssh-rsa",
       "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa", "root@192.168.1.247"]


def create(retry_of=None, baseline_history=None, reused_evidence_run=None, snapshot_fix_redeployment=False, native_env_redeployment=False):
    image = ROOT / "build_axu_aarch64/cat.png"
    canonical = PREPARATION / "raw/canonical_split_preparation_02/canonical_qir.json"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_single_image_" + uuid.uuid4().hex[:8]
    store = RunStore.create(BASE / "results/06" / run_id, {
        "protocol_version": "thesis_single_image_smoke_v1_discard_first_two",
        "campaign_id": json.loads((BASE / "results/ACTIVE_CAMPAIGN.json").read_text())["campaign_id"],
        "stage": "06", "run_id": run_id, "scope": "single_real_input_smoke_only",
        "retry_of": retry_of,
        "reused_evidence_run": str(Path(reused_evidence_run).resolve()) if reused_evidence_run else None,
        "evidence_reuse_rule": "Only sealed same-graph/input evidence preceding a recovery or identified inspection-only failure; preserve source executable/run/boot identities and paid costs, hash imported files; deployment changes are separately recorded and qualified, never relabel source scores",
        "snapshot_fix_redeployment": bool(snapshot_fix_redeployment),
        "native_env_redeployment": bool(native_env_redeployment),
        "selector_deployment_boundary": "Original three-boot C1 certificate selects graph cuts/threads under its original runner identity. A private managed-slot inspection fix changes final runner identity; all final conditions use that same new runner and are freshly qualified/measured." if snapshot_fix_redeployment else None,
        "formal_stage_completed": False, "accuracy_evidence": False,
        "model": "pretrained MXNet ResNet18_v1@224", "extra_quantize_calls": 0,
        "image_sha256": sha256(image), "image_source": str(image),
        "canonical_qir_sha256": sha256(canonical), "canonical_source": str(canonical),
        "weights_sha256": sha256(Path("/home/orange/.mxnet/models/resnet18_v1-a0666292.params")),
        "baseline_history": str(Path(baseline_history).resolve()) if baseline_history else None,
        "baseline_history_sha256": sha256(baseline_history) if baseline_history else None,
        "A_thread_vector": [4, 1, 4] if baseline_history else [4, 1, 1],
        "preprocessing": "Pillow resize224 default interpolation before RGB conversion; float32 (RGB-[123,117,104])/[58.395,57.12,57.375]; NCHW",
        "input_shape": [1, 3, 224, 224], "seed": None,
        "python": PYTHON, "board_host": "192.168.1.247", "udmabuf_bytes": 201326592,
        "bitstream_sha256": sha256(ROOT / "apps/vta_rpc/firmware/vta_hpc.bit"),
        "completion": {"execution_context_reset": {"required_before_each_new_command_queue": True}},
        "numeric": {"integer": "exact", "copy_shared": "byte_exact", "float_atol": 1e-5,
                    "float_rtol": 1e-4, "finite_required": True},
        "correctness_repeats": 10, "delayed_read_ms": 10,
        "timeouts_seconds": {"correctness": 60, "serial": 60, "window": 600,
                             "host_preparation": 600, "build": 1200},
        "performance": {"scope": "engineering_smoke_one_boot_no_inferential_CI",
                        "orders": [["A", "B", "C", "Ca", "D", "E"], ["E", "D", "Ca", "C", "B", "A"]],
                        "warmup_per_block": 5, "serial_per_block": 15,
                        "window_rows": 302, "discard_first_rows": 2},
        "conditions": {"A": "CPU00..02/VTA03..17/CPU18..20 ordinary copy; threads bound by A_thread_vector",
                       "B": "current qIR frozen C1 selection; no old-package substitute",
                       "C": "B with real K2 sharing", "Ca": "C with exclusive input anchor",
                       "D": "Ca with actual-program Q1", "E": "D with qualified C3 and regenerated Q1; all-rejected E=D"},
        "preparation_run": str(PREPARATION),
        "stop_rule": "timeout, contract drift or numeric failure invalidates this run; no later condition/performance; retry in new directory",
        "prior_costs": "linked original preparation run; interactive discovery cost unknown, not zero",
    })
    shutil.copy2(image, store.directory / "raw/input_image.png")
    shutil.copy2(Path(__file__), store.directory / "raw/executed_controller.py")
    shutil.copy2(FRONTEND / "thesis_e2e_ablation_protocol.py", store.directory / "raw/window_protocol.py")
    (store.directory / "REPORT.md").write_text("# 单图语义一致性与递进消融试运行\n\n状态：准备中。尚无板端推理结果。\n\n本运行不替代正式阶段认证，不报告任务准确率。\n")
    print(store.directory, flush=True)


def checked(store, argv, action, cost_class, timeout, stdin=None):
    result = store.execute(argv, action, cost_class, timeout, stdin)
    if result["status"] != "ok":
        raise RuntimeError(f"{action}: {result['status']}; stderr={result['stderr']}")
    return result


def host_worker(directory):
    import numpy as np
    from PIL import Image
    import tvm
    from tvm.contrib import graph_executor
    store = RunStore(directory)
    r = store.directory
    image = r / "raw/input_image.png"
    if sha256(image) != store.contract["image_sha256"]:
        raise ValueError("Input image changed")
    with Image.open(image) as source:
        values = np.asarray(source.resize((224, 224)).convert("RGB"), dtype="float32")
    values = (values - np.array([123, 117, 104], dtype="float32")) / np.array([58.395, 57.12, 57.375], dtype="float32")
    values = np.ascontiguousarray(values.transpose(2, 0, 1)[None], dtype="float32")
    inputs = r / "raw/inputs"
    inputs.mkdir()
    np.save(inputs / "input.npy", values, allow_pickle=False)
    (inputs / "input.bin").write_bytes(values.tobytes())
    write_new_json(r / "analysis/input_contract.json", {
        "input_sha256": sha256(inputs / "input.npy"), "input_bin_sha256": sha256(inputs / "input.bin"),
        "shape": list(values.shape), "dtype": str(values.dtype), "pillow_version": __import__("PIL").__version__,
        "numpy_version": np.__version__, "label": None, "accuracy_evidence": False,
    })
    source = PREPARATION / "raw/canonical_host_qualification_01"
    source_contract = json.loads((source / "contract.json").read_text())
    if source_contract['canonical_qir_sha256'] != store.contract['canonical_qir_sha256']:
        raise ValueError('Host reference was built from a different canonical qIR')
    manifest = json.loads((source / "artifact_hashes.json").read_text())
    reference = r / "raw/host_reference"
    reference.mkdir()
    executors = {}
    for name in ("canonical_qir", "canonical_boundary_probe"):
        target = reference / name
        target.mkdir()
        for filename in ("module.so", "graph.json", "params.bin"):
            relative = f"compiled/{name}/{filename}"
            path = source / relative
            if sha256(path) != manifest[relative]["sha256"]:
                raise ValueError("Host reference artifact drift: " + relative)
            shutil.copy2(path, target / filename)
        exe = graph_executor.create((target / "graph.json").read_text(),
                                    tvm.runtime.load_module(str(target / "module.so")), tvm.cpu())
        exe.load_params((target / "params.bin").read_bytes())
        exe.set_input("data", values)
        executors[name] = exe
    exe = executors["canonical_qir"]
    output_dir = reference / "outputs"
    output_dir.mkdir()
    expected = None
    observations = []
    for i in range(10):
        exe.run()
        first = exe.get_output(0).numpy()
        time.sleep(0.01)
        delayed = exe.get_output(0).numpy()
        if not np.isfinite(first).all() or first.tobytes() != delayed.tobytes():
            raise ValueError("Host reference output unstable/non-finite")
        if expected is not None and first.tobytes() != expected.tobytes():
            raise ValueError("Host reference repeats differ")
        expected = first
        np.save(output_dir / f"r{i:02}_immediate.npy", first, allow_pickle=False)
        np.save(output_dir / f"r{i:02}_delayed.npy", delayed, allow_pickle=False)
        observations.append({"round": i, "stable": True, "finite": True})
    np.save(reference / "reference_logits.npy", expected, allow_pickle=False)
    from thesis_canonical_graph_split import UnitGraph
    graph = UnitGraph(tvm.ir.load_json(Path(store.contract["canonical_source"]).read_text()))
    edges = sorted({n for b in range(22) for n in graph.frontier(b)}, key=lambda n: graph.ids.get(n, -1))
    probe = executors["canonical_boundary_probe"]
    probe.run()
    snapshots = {f"node_{graph.ids.get(n, -1)}": probe.get_output(i).numpy() for i, n in enumerate(edges)}
    final = snapshots[f"node_{graph.ids[graph.mod['main'].body]}"]
    if final.tobytes() != expected.tobytes():
        raise ValueError("Boundary probe and full reference differ")
    np.savez(reference / "reference_boundaries.npz", **snapshots)
    write_new_json(r / "analysis/host_reference_checks.json", {
        "status": "SINGLE_IMAGE_HOST_REFERENCE_PASS", "repeats": observations,
        "boundary_tensors": len(snapshots), "top5": np.argsort(-expected.reshape(-1), kind="stable")[:5].tolist(),
        "label_available": False, "board_calls": 0, "extra_quantize_calls": 0,
    })
    print("single image host reference: 10 immediate/delayed pairs and complete live frontiers PASS", flush=True)


def bootstrap(store):
    r = store.directory
    source_bindings = json.loads((PREPARATION / "analysis/board_artifact_bindings.json").read_text())
    bindings = {}
    for name, item in source_bindings.items():
        path = Path(item["host_path"])
        if sha256(path) != item["sha256"]:
            raise ValueError("Frozen board artifact changed: " + name)
        bindings[name] = item
    write_new_json(r / "analysis/board_artifact_bindings.json", bindings)
    for p in list((PREPARATION / "raw").glob("*.npy")):
        shutil.copy2(p, r / "raw" / p.name)
    remote = "/var/volatile/thesis_" + r.name
    write_new_json(r / "analysis/board_directory.json", {"path": remote})
    with tarfile.open(r / "raw/board_bundle.tar", "w") as archive:
        for name, item in bindings.items():
            archive.add(item["host_path"], arcname=name)
    script = r / "raw/board_restore.sh"
    script.write_text("""#!/bin/sh
set -eu
remote=""" + shlex.quote(remote) + """
cd "$remote"
tar -xf board_bundle.tar
""" + "\n".join("test \"$(sha256sum " + shlex.quote(name) + " | cut -d ' ' -f 1)\" = " + item["sha256"] for name, item in bindings.items()) + """
chmod 755 tvm_rpc
if ! test -e /dev/udmabuf0; then /sbin/insmod "$remote/u-dma-buf.ko" udmabuf0=201326592; fi
test "$(cat /sys/class/u-dma-buf/udmabuf0/size)" = 201326592
ln -sf "$remote/vta_hpc.bit" /lib/firmware/vta_hpc.bit
printf 'vta_hpc.bit\n' > /sys/class/fpga_manager/fpga0/firmware
test "$(cat /sys/class/fpga_manager/fpga0/state)" = operating
cat /proc/sys/kernel/random/boot_id
cat /sys/class/u-dma-buf/udmabuf0/size /sys/class/u-dma-buf/udmabuf0/phys_addr /sys/class/u-dma-buf/udmabuf0/sync_mode
export LD_LIBRARY_PATH="$remote" LD_PRELOAD="$remote/libvta.so"
export C3_COMPLETION_TRACE=1 TVM_NUM_THREADS=1 VTA_REPLAY_POLICY=disabled VTA_QUEUE_DIAGNOSTICS=1 VTA_RUNTIME_TRACE=0
export AXU5EVB_DRIVER_INFO=1 AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000 AXU5EVB_DRIVER_POLL_SLEEP_NS=1000
unset VTA_INSN_SUBMIT_THRESHOLD_BYTES VTA_INSN_BUFFER_BYTES VTA_UOP_BUFFER_BYTES
./tvm_rpc server --host=0.0.0.0 --port=9090 --port-end=9091 > "$remote/rpc.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$remote/rpc.pid"
sleep 1
kill -0 "$(cat "$remote/rpc.pid")"
""")
    result = checked(store, SSH + ["cat /proc/sys/kernel/random/boot_id"], "board_boot_identity", "one_time_preparation", 30)
    boot = (r / result["stdout"]).read_text().strip()
    uuid.UUID(boot)
    write_new_json(r / "analysis/board_boot.json", {"boot_id": boot})
    checked(store, SSH + ["test ! -d " + shlex.quote(remote) + " && mkdir " + shlex.quote(remote)], "board_ram_directory", "one_time_preparation", 30)
    checked(store, SSH + ["cat > " + shlex.quote(remote + "/board_bundle.tar")], "board_bundle_transfer", "one_time_preparation", 120, r / "raw/board_bundle.tar")
    checked(store, SSH + ["sh -s"], "board_restore", "one_time_preparation", 60, script)
    # Four fresh contexts on this boot; these are diagnostic calls, not a two-boot certificate.
    for mode in ("O", "I", "W", "IW"):
        checked(store, [PYTHON, str(FRONTEND / "thesis_completion_context_reset.py"), str(r),
                        "--boot-index", "1", "--boot-id", boot, "--mode", mode,
                        "--board-directory", remote], "reset_" + mode, "functional_qualification", 60)
        checked(store, [PYTHON, str(FRONTEND / "thesis_completion_board.py"), str(r),
                        "--boot-index", "1", "--boot-id", boot, "--mode", mode,
                        "--board-directory", remote], "completion_" + mode, "functional_qualification", 60)
    checked(store, SSH + ["cat " + shlex.quote(remote + "/rpc.log")], "completion_log_fetch", "functional_qualification", 30)
    write_new_json(r / "analysis/completion_smoke.json", {"boot_id": boot, "modes": ["O", "I", "W", "IW"],
                   "scored_calls": 120, "health_calls": 4, "formal_two_boot_certificate": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("create", "host", "host-worker", "bootstrap"))
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--retry-of")
    p.add_argument("--baseline-history")
    p.add_argument("--reused-evidence-run")
    p.add_argument("--snapshot-fix-redeployment", action="store_true")
    p.add_argument("--native-env-redeployment", action="store_true")
    args = p.parse_args()
    if args.operation == "create":
        return create(args.retry_of, args.baseline_history, args.reused_evidence_run, args.snapshot_fix_redeployment, args.native_env_redeployment)
    if not args.run_dir:
        p.error("--run-dir is required")
    store = RunStore(args.run_dir)
    if (store.directory / "invalid_session.json").exists():
        raise ValueError("Invalid run cannot be resumed; create a new retry run")
    try:
        if args.operation == "host-worker":
            return host_worker(args.run_dir)
        if args.operation == "host":
            checked(store, [PYTHON, str(Path(__file__)), "host-worker", "--run-dir", str(args.run_dir)],
                    "single_image_host_reference", "functional_qualification", 600)
        elif args.operation == "bootstrap":
            bootstrap(store)
    except BaseException as error:
        if not (store.directory / "invalid_session.json").exists():
            write_new_json(store.directory / "invalid_session.json", {"operation": args.operation,
                           "reason": str(error), "all_paid_costs_retained": True})
        store.event("single_image_operation_failed", operation=args.operation, reason=str(error))
        raise


if __name__ == "__main__":
    main()
