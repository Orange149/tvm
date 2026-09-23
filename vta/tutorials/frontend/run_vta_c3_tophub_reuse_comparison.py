#!/usr/bin/env python3
"""Fresh same-workload TopHub versus four-mode operator comparison.

The cached TopHub incumbent is re-instantiated through its genuine source
template, not through its historical index. All timings use one isolated,
qualified runtime. Existing sealed experiments are read-only. This adapter
redirects the existing worker protocol in memory; no frozen file is changed.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import run_vta_c3_reuse_supplement as s

BASE = s.HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
PREPARED = BASE / "07_grouped_holdout/20260915_c3_reuse_fixed_driver_prepare_run01"
SPEC = dict(ci=64, co=64, height=56, width=56, kernel=3, stride=1)
ORIGINAL_INSTANTIATE = s.instantiate


def instantiate(row):
    if not row.get("source_tophub"):
        return ORIGINAL_INSTANTIATE(row)
    from tvm import autotvm
    from tvm.autotvm.task.space import ConfigEntity
    import vta
    from tune_resnet18_vta import register_vta_conv2d_template
    register_vta_conv2d_template()
    env = vta.get_env()
    task = autotvm.task.create("conv2d_packed.vta", args=tuple(row["identity"]["workload"][1:]),
                               target=env.target, target_host=env.target_host)
    config = ConfigEntity.from_json_dict(row["source_config"])
    with task.target:
        return task.instantiate(config)


def redirected(function, *args):
    # Helpers spawn their __file__ with an existing JSON worker protocol.
    # Bind this explicitly recorded adapter instead of editing the sealed code.
    old = s.__file__
    s.__file__ = __file__
    previous = os.environ.pop("LD_LIBRARY_PATH", None) if function is s.cross else None
    try:
        return function(*args)
    finally:
        s.__file__ = old
        if previous is not None:
            os.environ["LD_LIBRARY_PATH"] = previous


def tiling(entity):
    return {name: value[-1] if kind == "sp" else value for name, kind, value in entity["entity"]}


def prepare(args, out, start):
    from prepare_vta_c3_resnet18_literature_candidates import make_workload, create_task, candidate_row
    from c3_candidate_identity import normalize_config_entity
    parent = s.verify(PREPARED)
    s.check_bindings(parent["source_bindings"])
    env, fingerprint = s.hardware()
    workload = make_workload(SPEC, env)
    cache = Path("/home/orange/.tvm/tophub/vta_v0.10.log")
    matches = []
    for number, line in enumerate(cache.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        record = json.loads(line)
        inp = record["input"]
        if inp[1] == workload[0] and inp[2] == workload[1:] and record["result"][1] == 0:
            matches.append((statistics.mean(record["result"][0]), number, record))
    if not matches:
        raise ValueError("no exact-workload TopHub record; no substitute shapes allowed")
    _, line_number, official = min(matches, key=lambda r: r[0])
    task = create_task(workload, 0, env)
    domain = [(i, normalize_config_entity(task.config_space.get(i).to_json_dict()))
              for i in range(len(task.config_space))]
    source_knobs = tiling(official["config"])
    same = [(i, entity) for i, entity in domain if tiling(entity) == source_knobs]
    if len(same) != 1:
        raise ValueError("TopHub actual tile not uniquely represented in current ConfigSpace")
    bindings = {**parent["source_bindings"], **s.source_bindings(),
                str(Path(__file__).resolve()): s.file_hash(__file__),
                str(s.HERE / "tune_resnet18_vta.py"): s.file_hash(s.HERE / "tune_resnet18_vta.py"),
                str(cache): s.file_hash(cache)}
    version = s.digest(bindings)
    index, entity = same[0]
    baseline = candidate_row("R18-H1", SPEC, workload, entity, index, "TOPHUB", "original",
                              version, fingerprint, env)
    baseline.update(candidate_id="tophub_source_" + s.digest(official["config"])[:24],
                    source_tophub=True, source_config=official["config"], source_target=official["input"][0],
                    identity={"workload": workload, "complete_config_entity": official["config"]})
    # Finite label-blind alternatives, selected for reuse and SRAM feasibility;
    # not an exhaustive search and never expanded after performance is exposed.
    desired = [(7, 56, 1, 2), (14, 28, 1, 1), (2, 56, 1, 2), (4, 28, 1, 1)]
    families = [("TOPHUB_TILE", index, entity)]
    for h, w, ci, co in desired:
        hits = [(i, e) for i, e in domain if tiling(e) == dict(tile_b=1, tile_h=h, tile_w=w,
                 tile_ci=ci, tile_co=co, oc_nthread=1, h_nthread=1)]
        if len(hits) != 1:
            raise ValueError("requested finite alternative missing or ambiguous")
        families.append(("ALT_" + str(hits[0][0]), *hits[0]))
    rows = [baseline]
    for family, idx, ent in families:
        for mode in s.MODES:
            row = candidate_row("R18-H1", SPEC, workload, ent, idx, family, mode, version, fingerprint, env)
            row["identity"] = {"workload": workload, "complete_config_entity": ent}
            rows.append(row)
    contract = dict(schema="c3_tophub_four_mode_v1", phase="local_prepared", source_bindings=bindings,
                    parent_directory=str(PREPARED), parent_manifest_sha256=s.file_hash(PREPARED / "artifact_hashes.json"),
                    workload=workload, geometry=SPEC, official_record=official, official_line_number=line_number,
                    exact_cache_match_count=len(matches), baseline_candidate_id=baseline["candidate_id"],
                    cache_target_retargeted=True, reference_kind="exact-workload cached PYNQ entity requalified on AXU5EVB",
                    historical_latency_not_used_as_measurement=True, candidate_selection="four fixed label-blind alternative tiles plus exact TopHub tile",
                    buffer_layout=["output", "data", "weight"], correctness_seeds=list(s.SEEDS),
                    timing_rounds=10, all_modes_same_experimental_driver=True,
                    operator_only=True, whole_graph_tested=False, board_contacted=False,
                    performance_labels_collected=False, runtime_paths=parent["runtime_paths"],
                    runtime_sha256=parent["runtime_sha256"], health=parent["health"],
                    generated_register_map_attested=False, trace_during_formal_timing=False)
    s.write(out / "contract.json", contract)
    for row in rows:
        row["local_qualification"] = s.lower_candidate(row, out)
        if row["local_qualification"]["status"] == "passed":
            row["fsim_qualification"] = redirected(s.fsim, row, row["local_qualification"], out,
                                                    str(s.ROOT / "build/libvta_fsim.so"), args.timeout)
            if row["fsim_qualification"]["status"] == "passed":
                row["cross_qualification"] = redirected(s.cross, row, out, args)
        s.append(out / "qualification.jsonl", row)
        print(row["family_id"], row["residence_mode"],
              row["local_qualification"]["status"], row.get("fsim_qualification", {}).get("status"),
              row.get("cross_qualification", {}).get("status"), flush=True)
    s.write(out / "candidates.json", rows)
    passed = [r for r in rows if r.get("cross_qualification", {}).get("status") == "passed"]
    adapters = [r for r in rows if r["family_id"] == "TOPHUB_TILE" and r["implementation_mode"] == 0]
    equal = bool(adapters and adapters[0]["local_qualification"].get("tir_sha256") == baseline["local_qualification"].get("tir_sha256"))
    s.write(out / "summary.json", dict(status="READY" if baseline in passed else "BASELINE_LOCAL_REJECTED",
            qualified_count=len(passed), gross_candidates=len(rows), official_source_equals_mode0_tir=equal,
            wall_seconds=time.monotonic() - start, board_contacted=False))
    s.check_bindings(bindings)


def board_run(args, out, start):
    import run_vta_c3_reuse_board as b
    from diagnose_vta_c3_reuse_completion import DiagnosticBoard
    from run_vta_p7r120_yolo_rpc_board import runtime_inventory, load_ephemeral
    from run_vta_p7r132_y00_search_confirmation import allocate_reusable_buffers
    import tvm
    from tvm import rpc
    directory = Path(args.contract).resolve().parent
    contract = s.verify(directory)
    s.check_bindings(contract["source_bindings"])
    parent = s.verify(PREPARED)
    if contract["parent_manifest_sha256"] != s.file_hash(PREPARED / "artifact_hashes.json"):
        raise ValueError("qualified runtime parent changed")
    candidates = [r for r in s.read(directory / "candidates.json")
                  if r.get("cross_qualification", {}).get("status") == "passed"]
    baseline = next(r for r in candidates if r["candidate_id"] == contract["baseline_candidate_id"])
    b.RUNTIME = contract["runtime_sha256"]
    board = DiagnosticBoard(args, "/var/volatile/" + out.name + "/runtime")
    initial = board.state()
    b.validate_state(initial, exclusive=True)
    s.write(out / "preflight.json", initial)
    contract.update(phase="board", boot_id=initial["boot_id"], board_contacted=True,
                    prepared_directory=str(directory), prepared_manifest_sha256=s.file_hash(directory / "artifact_hashes.json"),
                    runtime_directory=board.runtime)
    s.write(out / "contract.json", contract)
    remote, buffers, modules = None, {}, {}
    observations, passed, rejected = [], [], []
    try:
        s.write(out / "runtime_attestation.json", board.deploy(contract["runtime_paths"]))
        s.write(out / "bitstream_reload.json", board.reload())
        board.start()
        for _ in range(20):
            try:
                remote = rpc.connect(args.host, 9090, session_timeout=7200)
                break
            except Exception:
                time.sleep(.25)
        if remote is None:
            raise RuntimeError("RPC readiness failed")
        t0 = time.monotonic()
        s.append(out / "timeline.jsonl", dict(event="T0", elapsed_seconds=t0 - start))
        inventory, functions = runtime_inventory(remote)
        s.write(out / "runtime_inventory.json", inventory)
        load_ephemeral(remote, PREPARED / "binaries/c3_reuse_timed_once.so")
        timer = remote.get_function("c3.reuse.timed_once")
        device = remote.ext_dev(0)

        def call(row, phase, seed, timed=False, **extras):
            cid, key = row["candidate_id"], row["workload_id"]
            if key not in buffers:
                buffers[key] = allocate_reusable_buffers(device, row["identity"]["workload"])
                s.append(out / "allocations.jsonl", dict(workload_id=key, layout=["output", "data", "weight"], allocation_events=3))
            if cid not in modules:
                binary = (PREPARED if key == "HEALTH" else directory) / "binaries" / (cid + ".so")
                if s.file_hash(binary) != row["cross_qualification"]["binary_sha256"]:
                    raise ValueError("binary hash changed")
                modules[cid] = load_ephemeral(remote, binary)
            s.append(out / "timeline.jsonl", dict(event="dispatch_begin", phase=phase, candidate_id=cid,
                     elapsed_seconds=time.monotonic() - start, **extras))
            begin = time.monotonic()
            result = b.invoke(modules[cid], buffers[key], row["identity"]["workload"], seed, functions, timer, timed)
            result.update(candidate_id=cid, family_id=row["family_id"], residence_mode=row["residence_mode"],
                          phase=phase, host_wall_seconds=time.monotonic() - begin, **extras)
            s.append(out / "results.jsonl", result)
            s.append(out / "timeline.jsonl", dict(event="dispatch_end", phase=phase, candidate_id=cid,
                     correct=result["correct"], elapsed_seconds=time.monotonic() - start))
            return result

        for seed in s.SEEDS:
            if not call(parent["health"], "health", seed)["correct"]:
                raise ValueError("health probe incorrect")
        for row in [baseline] + [r for r in candidates if r is not baseline]:
            valid = True
            for seed in s.SEEDS:
                if not call(row, "correctness", seed)["correct"]:
                    valid = False
                    break
            (passed if valid else rejected).append(row)
            print("FPGA", row["family_id"], row["residence_mode"], valid, flush=True)
            if row is baseline and not valid:
                s.write(out / "summary.json", dict(status="TOPHUB_FPGA_REJECTED", timing_valid=False,
                        conclusion="No speed comparison possible: baseline failed correctness", wall_seconds=time.monotonic()-start))
                return
        for row in passed:
            if not call(row, "warmup", 0)["correct"]:
                raise ValueError("warmup incorrect; whole timing session invalid")
        # TopHub is revisited immediately around EVERY alternative measurement;
        # reverse alternative order on alternating rounds to balance drift.
        alternatives = [r for r in passed if r is not baseline]
        for round_id in range(contract["timing_rounds"]):
            order = alternatives if round_id % 2 == 0 else list(reversed(alternatives))
            seed = s.SEEDS[round_id % len(s.SEEDS)]
            for row in order:
                pair = []
                for item in (baseline, row, baseline):
                    result = call(item, "timing", seed, True, round=round_id, compared_candidate=row["candidate_id"])
                    if not result["correct"]:
                        raise ValueError("timing output incorrect; whole timing session invalid")
                    pair.append(result)
                observations.append(dict(candidate_id=row["candidate_id"], round=round_id,
                    baseline_ms=(pair[0]["latency_ms"]+pair[2]["latency_ms"])/2,
                    candidate_ms=pair[1]["latency_ms"]))
            print("timing round", round_id + 1, "complete", flush=True)
        final = board.state()
        b.validate_state(final)
        if final["boot_id"] != initial["boot_id"]:
            raise ValueError("boot changed: whole session invalid")
        s.write(out / "postflight.json", final)
        results = []
        for row in alternatives:
            pairs = [o for o in observations if o["candidate_id"] == row["candidate_id"]]
            base = statistics.median(o["baseline_ms"] for o in pairs)
            value = statistics.median(o["candidate_ms"] for o in pairs)
            results.append(dict(candidate_id=row["candidate_id"], family_id=row["family_id"],
                residence_mode=row["residence_mode"], knobs=row["visible_features"], baseline_median_ms=base,
                median_ms=value, latency_reduction_percent=100*(1-value/base),
                paired_median_reduction_percent=statistics.median(100*(1-o["candidate_ms"]/o["baseline_ms"]) for o in pairs),
                wins=sum(o["candidate_ms"]<o["baseline_ms"] for o in pairs), rounds=len(pairs)))
        s.write(out / "paired_observations.json", observations)
        s.write(out / "summary.json", dict(status="COMPLETE", timing_valid=True, operator_only=True,
            whole_graph_tested=False, exact_workload=True, reference_kind=contract["reference_kind"],
            passed_identities=len(passed), rejected_identities=len(rejected), timing_calls=len(observations)*3,
            all_timing_calls_correct=True, comparisons=sorted(results,key=lambda r:r["median_ms"]),
            T0_to_T1_seconds=time.monotonic()-t0, W0_to_T1_seconds=time.monotonic()-start))
        s.check_bindings(contract["source_bindings"])
    except BaseException as error:
        s.write(out / "invalid_session.json", dict(message=str(error)[-5000:], all_timing_invalid=True))
        raise
    finally:
        if remote is not None:
            del remote
        if board.pid is not None:
            board.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "board"))
    parser.add_argument("--output-dir")
    parser.add_argument("--contract")
    parser.add_argument("--worker-request")
    parser.add_argument("--cross-request")
    parser.add_argument("--worker-response")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--sdk-environment", default="/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux")
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    args = parser.parse_args()
    s.instantiate = instantiate
    if args.worker_request:
        return s.worker(s.read(args.worker_request), Path(args.worker_response))
    if args.cross_request:
        return s.cross_worker(s.read(args.cross_request), Path(args.worker_response))
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    s.append(out / "timeline.jsonl", dict(event="W0", wall_unix=time.time()))
    try:
        (prepare if args.phase == "prepare" else board_run)(args, out, start)
    finally:
        s.seal(out)


if __name__ == "__main__":
    main()
