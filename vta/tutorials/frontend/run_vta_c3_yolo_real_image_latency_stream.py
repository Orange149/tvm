#!/usr/bin/env python3
"""Measure real-image YOLO-416 latency and safe YOLO-512 streaming throughput."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import statistics
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from tvm import rpc
from tvm.relay.testing import darknet

import run_vta_c3_reuse_board as reuse_board
from c3_evaluation_v5 import read_json, seal_flat, sha256_file, verify_flat, write_json
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_yolo608_graph_boot import load_executor, output_summary
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_416 = GROUPED / "20260926_c3_yolo416_stock_graph_build_run01"
DEFAULT_512 = GROUPED / "20260926_c3_yolo512_evaluation_v5_graph_build_run01"
DEFAULT_FREEZE = GROUPED / "20260926_c3_yolo512_evaluation_v5_freeze_run01"
DEFAULT_COMPLETION = GROUPED / "20260926_c3_yolo512_evaluation_v5_completion_audit_run01"
DEFAULT_STATE = GROUPED / "20260926_c3_mode11_cross_program_state_run06"


def connect(host, port, timeout):
    for _ in range(30):
        try:
            return rpc.connect(host, port, session_timeout=timeout)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable")


def image_input(path: Path, resolution: int) -> np.ndarray:
    data = darknet.load_image(str(path), resolution, resolution)
    if data is None or tuple(data.shape) != (3, resolution, resolution):
        raise ValueError("invalid preprocessed image: " + str(path))
    return np.ascontiguousarray(data[None, :].astype("float32"))


def exact(lhs, rhs) -> bool:
    return len(lhs) == len(rhs) and all(np.array_equal(a, b) for a, b in zip(lhs, rhs))


def summary_identity(summary):
    return tuple((row["sha256"], tuple(row["shape"]), row["dtype"]) for row in summary)


def dataset_inventory(directory: Path, count: int):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_paths = sorted(
        path.resolve() for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if len(all_paths) < count:
        raise ValueError("dataset has only {} images, need {}".format(len(all_paths), count))
    selected = all_paths[:count]
    digest = hashlib.sha256()
    rows = []
    for index, path in enumerate(selected):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("undecodable RGB image: " + str(path))
        file_hash = sha256_file(path)
        relative = str(path.relative_to(directory.resolve()))
        digest.update(relative.encode("utf-8") + b"\0" + file_hash.encode("ascii") + b"\n")
        rows.append({
            "index": index,
            "relative_path": relative,
            "sha256": file_hash,
            "source_shape_hwc": list(image.shape),
            "bytes": path.stat().st_size,
        })
    return all_paths, selected, rows, digest.hexdigest()


def stream_result(count, wall_seconds, graph_ms, preprocess_ms, output_ids, mode):
    return {
        "mode": mode,
        "images": count,
        "wall_seconds": wall_seconds,
        "stream_fps": count / wall_seconds,
        "mean_intercompletion_ms": wall_seconds * 1000.0 / count,
        "graph_median_ms": float(statistics.median(graph_ms)),
        "graph_fps_from_median": 1000.0 / float(statistics.median(graph_ms)),
        "preprocess_median_ms": float(statistics.median(preprocess_ms)),
        "graph_latencies_ms": graph_ms,
        "preprocess_latencies_ms": preprocess_ms,
        "output_identities": output_ids,
    }


def run_stream(executor, timer, paths, resolution, mode, depth, exact_refs, label=""):
    if mode not in ("serial", "pipeline"):
        raise ValueError(mode)
    graph_ms, preprocess_ms, output_ids = [], [], []
    exact_hits = 0

    def preprocess(index):
        started = time.perf_counter()
        data = image_input(paths[index], resolution)
        return data, (time.perf_counter() - started) * 1000.0

    def summarize(outputs):
        return summary_identity(output_summary(outputs))

    wall_started = time.perf_counter()
    if mode == "serial":
        for index in range(len(paths)):
            data, prep_ms = preprocess(index)
            executor.set_input("data", data)
            latency = float(timer(executor.module))
            if not math.isfinite(latency) or latency <= 0:
                raise RuntimeError("invalid graph latency")
            outputs = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
            if index in exact_refs:
                if not exact(outputs, exact_refs[index]):
                    raise RuntimeError("stream output differs from stock at index {}".format(index))
                exact_hits += 1
            output_ids.append(summarize(outputs))
            graph_ms.append(latency)
            preprocess_ms.append(prep_ms)
            if len(paths) >= 100 and ((index + 1) % 25 == 0 or index + 1 == len(paths)):
                print("stream {} {}/{}".format(label or mode, index + 1, len(paths)), flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as producer, \
                concurrent.futures.ThreadPoolExecutor(max_workers=1) as validator:
            pending_preprocess = {
                index: producer.submit(preprocess, index)
                for index in range(min(depth, len(paths)))
            }
            pending_validation = []
            validation_indices = []
            for index in range(len(paths)):
                data, prep_ms = pending_preprocess.pop(index).result()
                next_index = index + depth
                if next_index < len(paths):
                    pending_preprocess[next_index] = producer.submit(preprocess, next_index)
                executor.set_input("data", data)
                latency = float(timer(executor.module))
                if not math.isfinite(latency) or latency <= 0:
                    raise RuntimeError("invalid graph latency")
                outputs = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
                if index in exact_refs:
                    if not exact(outputs, exact_refs[index]):
                        raise RuntimeError("stream output differs from stock at index {}".format(index))
                    exact_hits += 1
                pending_validation.append(validator.submit(summarize, outputs))
                validation_indices.append(index)
                if len(pending_validation) > depth:
                    output_ids.append(pending_validation.pop(0).result())
                    validation_indices.pop(0)
                graph_ms.append(latency)
                preprocess_ms.append(prep_ms)
                if len(paths) >= 100 and ((index + 1) % 25 == 0 or index + 1 == len(paths)):
                    print("stream {} {}/{}".format(label or mode, index + 1, len(paths)), flush=True)
            for future in pending_validation:
                output_ids.append(future.result())
    wall_seconds = time.perf_counter() - wall_started
    if len(output_ids) != len(paths):
        raise RuntimeError("stream output count mismatch")
    return stream_result(len(paths), wall_seconds, graph_ms, preprocess_ms, output_ids, mode), exact_hits


def run(args):
    build416 = Path(args.build416_dir).resolve()
    build512 = Path(args.build512_dir).resolve()
    freeze_dir = Path(args.freeze_dir).resolve()
    completion_dir = Path(args.completion_dir).resolve()
    state_dir = Path(args.state_diagnostic_dir).resolve()
    output = Path(args.output_dir).resolve()
    image_path = Path(args.single_image).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    if not image_path.is_file() or not dataset_dir.is_dir():
        raise FileNotFoundError("image or dataset directory missing")

    manifests = {
        "build416": verify_flat(build416),
        "build512": verify_flat(build512),
        "freeze": verify_flat(freeze_dir),
        "completion": verify_flat(completion_dir),
        "state_diagnostic": verify_flat(state_dir),
    }
    summary416 = read_json(build416 / "summary.json")
    summary512 = read_json(build512 / "summary.json")
    freeze = read_json(freeze_dir / "contract.json")
    policy = read_json(completion_dir / "completion_policy.json")
    state_diagnostic = read_json(state_dir / "summary.json")
    if summary416.get("status") != "C3_YOLO416_STOCK_GRAPH_BUILT":
        raise ValueError("YOLO-416 build is incomplete")
    if summary416.get("input_shape") != [1, 3, 416, 416]:
        raise ValueError("YOLO-416 input contract drift")
    if summary512.get("status") != "C3_EVALUATION_V5_THREE_YOLO512_DEPLOYMENTS_BUILT":
        raise ValueError("YOLO-512 build is incomplete")
    if policy.get("status") != "COMPLETION_POLICY_FROZEN":
        raise ValueError("completion policy is not frozen")
    if state_diagnostic.get("classification") != "GENERAL_CROSS_RPC_FPGA_STATE_CONTAMINATION":
        raise ValueError("clean-deployment requirement is not frozen")

    helper = Path(freeze["helper_binary"])
    if sha256_file(helper) != freeze["helper_sha256"]:
        raise ValueError("helper identity drift")
    runtime = read_json(freeze["runtime_contract"])
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(Path(path)) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime identity drift: " + name)

    all_paths, selected, inventory_rows, dataset_hash = dataset_inventory(
        dataset_dir, args.dataset_count
    )
    if args.dataset_count < 4:
        raise ValueError("dataset-count must be at least 4")
    qualification_indices = (0, args.dataset_count // 2, args.dataset_count - 1)
    output.mkdir(parents=True)
    write_json(output / "dataset_inventory.json", {
        "root": str(dataset_dir),
        "available_images": len(all_paths),
        "selected_images": len(selected),
        "selection": "first N image paths in lexical order",
        "selected_inventory_sha256": dataset_hash,
        "rows": inventory_rows,
    })
    contract = {
        "schema": "c3_yolo_real_image_latency_stream_v1",
        "status": "FROZEN_BEFORE_BOARD_OBSERVATION",
        "manifests": manifests,
        "single_image": {"path": str(image_path), "sha256": sha256_file(image_path)},
        "single_image_resolution": 416,
        "single_image_repeats": args.single_repeats,
        "stream_resolution": 512,
        "dataset_root": str(dataset_dir),
        "dataset_available": len(all_paths),
        "dataset_selected": len(selected),
        "dataset_inventory_sha256": dataset_hash,
        "qualification_indices": list(qualification_indices),
        "pipeline": {
            "stages": ["host JPEG decode plus letterbox", "one serialized VTA graph worker",
                       "host output copy plus exact hash validation"],
            "queue_depth": args.pipeline_depth,
            "multiple_VTA_graphs_in_flight": False,
            "multiple_RPC_processes_concurrent": False,
            "claim": "safe cross-frame software pipeline, not concurrent VTA command queues",
        },
        "full_stream_order": [
            "incumbent:block_A", "selector_v4:block_A",
            "selector_v4:block_B", "incumbent:block_B",
        ],
        "short_serial_order": ["incumbent", "selector_v4"],
        "clean_bitstream_before_every_RPC": True,
        "bitstream_reload_excluded_from_latency_and_throughput": True,
        "completion_mode": policy["mode"],
        "no_transparent_retry": True,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {
        "contract_sha256": sha256_file(output / "contract.json"),
        "dataset_inventory_sha256": sha256_file(output / "dataset_inventory.json"),
    })

    board = ExperimentBoard(args, "/var/volatile/" + output.name, args.port)
    initial = None
    session_index = 0
    reloads = []
    sessions = []
    stock_refs = {}
    block_outputs = {}
    single_result = None
    started = time.monotonic()
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        environment = board.environment()
        reuse_board.RUNTIME = runtime["runtime_sha256"]
        write_json(output / "runtime_attestation.json", board.deploy(runtime["runtime_paths"]))
        write_json(output / "preflight.json", {"state": initial, "environment": environment})

        def one_session(build_dir, variant, purpose, callback):
            nonlocal session_index
            state = board.state()
            reuse_board.validate_state(state, exclusive=True)
            if state["boot_id"] != initial["boot_id"]:
                raise RuntimeError("boot changed before session")
            reload_attestation = board.reload()
            reloads.append({
                "session": session_index,
                "purpose": purpose,
                "variant": variant,
                "attestation": reload_attestation,
            })
            board.port = args.port + 2 * session_index
            board.pid = None
            board.start()
            remote = connect(args.host, board.port, args.session_timeout)
            _, functions = runtime_inventory(remote)
            retained = [load_ephemeral(remote, helper)]
            timer_name = (
                "c3.v5.graph_run_only_timed_once"
                if policy["mode"] == "run_only"
                else "c3.v5.graph_run_sync_timed_once"
            )
            timer = remote.get_function(timer_name)
            reset_counts = remote.get_function("c3.v5.reset_counts")
            helper_counts = remote.get_function("c3.v5.counts")
            reset_counts()
            with tempfile.TemporaryDirectory(prefix="c3_yolo_real_stream_") as temporary:
                executor = load_executor(
                    remote,
                    build_dir,
                    variant,
                    [remote.ext_dev(0), remote.cpu(0)],
                    temporary,
                )
                current = board.environment(require_affinity=True)
                if current["governor"] != environment["governor"]:
                    raise RuntimeError("CPU governor changed")
                result = callback(executor, timer)
                counts = json.loads(str(helper_counts()))
                session = {
                    "session": session_index,
                    "purpose": purpose,
                    "variant": variant,
                    "helper_counts": counts,
                    "clean_bitstream_reload": reload_attestation,
                    "result": result,
                }
                sessions.append(session)
                write_json(output / "session_{:02d}.json".format(session_index), session)
                del executor
            retained.clear()
            remote = None
            gc.collect()
            board.stop()
            board.pid = None
            session_index += 1
            print("session {} {} complete".format(purpose, variant), flush=True)
            return result

        def single_416(executor, timer):
            warm_data = image_input(image_path, 416)
            executor.set_input("data", warm_data)
            executor.run()
            warm_outputs = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
            rows = []
            for repeat in range(args.single_repeats):
                total_started = time.perf_counter()
                prep_started = total_started
                data = image_input(image_path, 416)
                prep_ms = (time.perf_counter() - prep_started) * 1000.0
                executor.set_input("data", data)
                graph_ms = float(timer(executor.module))
                outputs = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
                application_ms = (time.perf_counter() - total_started) * 1000.0
                if not exact(outputs, warm_outputs):
                    raise RuntimeError("YOLO-416 repeated output mismatch")
                rows.append({
                    "repeat": repeat,
                    "preprocess_ms": prep_ms,
                    "graph_ms": graph_ms,
                    "application_ms": application_ms,
                    "outputs": output_summary(outputs),
                })
            return {
                "warmup_calls": 1,
                "timed_calls": len(rows),
                "rows": rows,
                "median_preprocess_ms": float(statistics.median(r["preprocess_ms"] for r in rows)),
                "median_graph_ms": float(statistics.median(r["graph_ms"] for r in rows)),
                "median_application_ms": float(statistics.median(r["application_ms"] for r in rows)),
            }

        single_result = one_session(build416, "stock", "yolo416_single_image", single_416)

        def stock_qualification(executor, _timer):
            rows = []
            for index in qualification_indices:
                data = image_input(selected[index], 512)
                executor.set_input("data", data)
                executor.run()
                outputs = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
                stock_refs[index] = [np.array(value, copy=True) for value in outputs]
                rows.append({
                    "index": index,
                    "relative_path": str(selected[index].relative_to(dataset_dir)),
                    "outputs": output_summary(outputs),
                })
            return {"calls": len(rows), "rows": rows}

        one_session(build512, "stock", "yolo512_stock_qualification", stock_qualification)

        blocks = {
            "A": (0, args.dataset_count // 2),
            "B": (args.dataset_count // 2, args.dataset_count),
        }
        full_order = (
            ("incumbent", "A"), ("selector_v4", "A"),
            ("selector_v4", "B"), ("incumbent", "B"),
        )
        stream_results = {}
        for variant, block in full_order:
            begin, end = blocks[block]
            paths = selected[begin:end]
            local_refs = {
                index - begin: outputs for index, outputs in stock_refs.items()
                if begin <= index < end
            }

            def full_callback(executor, timer, paths=paths, refs=local_refs):
                warm = image_input(paths[0], 512)
                executor.set_input("data", warm)
                executor.run()
                warm_outputs = [executor.get_output(i).numpy()
                                for i in range(executor.get_num_outputs())]
                if 0 in refs and not exact(warm_outputs, refs[0]):
                    raise RuntimeError("stream warmup differs from stock")
                result, exact_hits = run_stream(
                    executor, timer, paths, 512, "pipeline", args.pipeline_depth, refs,
                    "{}:{}".format(variant, block),
                )
                result["warmup_calls"] = 1
                result["stock_exact_checks"] = exact_hits
                return result

            result = one_session(
                build512, variant, "yolo512_pipeline_block_" + block, full_callback
            )
            stream_results[(variant, block)] = result
            block_outputs[(variant, block)] = result["output_identities"]

        for block in ("A", "B"):
            if block_outputs[("incumbent", block)] != block_outputs[("selector_v4", block)]:
                raise RuntimeError("incumbent/selector stream outputs differ in block " + block)

        serial_paths = selected[:args.serial_count]
        serial_refs = {index: outputs for index, outputs in stock_refs.items()
                       if index < args.serial_count}
        serial_results = {}
        for variant in ("incumbent", "selector_v4"):
            def serial_callback(executor, timer, paths=serial_paths, refs=serial_refs):
                warm = image_input(paths[0], 512)
                executor.set_input("data", warm)
                executor.run()
                warm_outputs = [executor.get_output(i).numpy()
                                for i in range(executor.get_num_outputs())]
                if 0 in refs and not exact(warm_outputs, refs[0]):
                    raise RuntimeError("serial warmup differs from stock")
                result, exact_hits = run_stream(
                    executor, timer, paths, 512, "serial", 1, refs, variant + ":serial"
                )
                result["warmup_calls"] = 1
                result["stock_exact_checks"] = exact_hits
                return result

            serial_results[variant] = one_session(
                build512, variant, "yolo512_serial_control", serial_callback
            )
        if serial_results["incumbent"]["output_identities"] \
                != serial_results["selector_v4"]["output_identities"]:
            raise RuntimeError("incumbent/selector serial outputs differ")

        aggregate = {}
        for variant in ("incumbent", "selector_v4"):
            wall = sum(stream_results[(variant, block)]["wall_seconds"] for block in ("A", "B"))
            count = sum(stream_results[(variant, block)]["images"] for block in ("A", "B"))
            graph_values = sum(
                (stream_results[(variant, block)]["graph_latencies_ms"] for block in ("A", "B")),
                [],
            )
            aggregate[variant] = {
                "images": count,
                "wall_seconds": wall,
                "pipeline_stream_fps": count / wall,
                "graph_median_ms": float(statistics.median(graph_values)),
                "graph_fps_from_median": 1000.0 / float(statistics.median(graph_values)),
                "blocks": {block: {
                    "stream_fps": stream_results[(variant, block)]["stream_fps"],
                    "wall_seconds": stream_results[(variant, block)]["wall_seconds"],
                } for block in ("A", "B")},
                "serial_control_fps": serial_results[variant]["stream_fps"],
            }
        final = board.state()
        reuse_board.validate_state(final, exclusive=True)
        final_environment = board.environment()
        if final["boot_id"] != initial["boot_id"]:
            raise RuntimeError("boot changed during experiment")
        if final_environment["governor"] != environment["governor"]:
            raise RuntimeError("CPU governor changed during experiment")

        summary = {
            "schema": contract["schema"],
            "status": "C3_YOLO_REAL_IMAGE_LATENCY_STREAM_COMPLETE",
            "boot_id": initial["boot_id"],
            "single_image_yolo416": {
                "image": str(image_path),
                "resolution": 416,
                "median_graph_ms": single_result["median_graph_ms"],
                "median_application_ms": single_result["median_application_ms"],
                "median_preprocess_ms": single_result["median_preprocess_ms"],
                "repeats": args.single_repeats,
            },
            "stream_yolo512": {
                "dataset_images": args.dataset_count,
                "available_images": len(all_paths),
                "all_incumbent_selector_outputs_equal": True,
                "qualification_stock_exact_indices": list(qualification_indices),
                "variants": aggregate,
                "selector_vs_incumbent_pipeline_throughput_improvement_pct": 100.0 * (
                    aggregate["selector_v4"]["pipeline_stream_fps"]
                    / aggregate["incumbent"]["pipeline_stream_fps"] - 1.0
                ),
                "selector_pipeline_vs_selector_serial_improvement_pct": 100.0 * (
                    aggregate["selector_v4"]["pipeline_stream_fps"]
                    / aggregate["selector_v4"]["serial_control_fps"] - 1.0
                ),
                "claim_boundary": (
                    "host JPEG/letterbox producer and host output validator overlap a single "
                    "serialized VTA graph worker; no concurrent VTA command queues"
                ),
            },
            "sessions": session_index,
            "bitstream_reloads": len(reloads),
            "clean_bitstream_before_every_RPC": len(reloads) == session_index,
            "wall_seconds": time.monotonic() - started,
        }
        write_json(output / "bitstream_reloads.json", {
            "count": len(reloads), "before_every_RPC": True, "reloads": reloads
        })
        write_json(output / "postflight.json", {"state": final, "environment": final_environment})
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write_json(output / "invalid_session.json", {
            "status": "INVALID_NON_SPLICEABLE_REAL_IMAGE_STREAM",
            "message": str(error),
            "type": type(error).__name__,
            "traceback": traceback.format_exc()[-20000:],
            "sessions_completed": session_index,
        })
        raise
    finally:
        try:
            board.stop()
        except Exception:
            pass
        seal_flat(output, (Path(__file__).resolve(),))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build416-dir", default=str(DEFAULT_416))
    parser.add_argument("--build512-dir", default=str(DEFAULT_512))
    parser.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE))
    parser.add_argument("--completion-dir", default=str(DEFAULT_COMPLETION))
    parser.add_argument("--state-diagnostic-dir", default=str(DEFAULT_STATE))
    parser.add_argument("--single-image", default="cat.png")
    parser.add_argument("--dataset-dir", default="n02123597")
    parser.add_argument("--dataset-count", type=int, default=1000)
    parser.add_argument("--single-repeats", type=int, default=10)
    parser.add_argument("--serial-count", type=int, default=20)
    parser.add_argument("--pipeline-depth", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=7200)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
