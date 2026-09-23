#!/usr/bin/env python3
"""R18-H1 T8/T7 × eight versions. New sealed artifacts; no production mutation."""
import argparse
import contextlib
import ctypes
import csv
import itertools
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import run_vta_c3_reuse_supplement as s
import c3_thread_sync_protocol as protocol
import c3_thread_sync_audit as audit

HERE, ROOT = s.HERE, s.ROOT
BASE = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
PARENT = BASE / "07_grouped_holdout/20260915_c3_reuse_fixed_driver_prepare_run01"
TOPHUB = BASE / "07_grouped_holdout/20260915_c3_tophub_h1_prepare_run02"
SCHEMA = "c3_thread_sync_ablation_v1"
SPEC = dict(ci=64, co=64, height=56, width=56, kernel=3, stride=1)
SDK = "/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux"


def instantiate(row):
    import tvm
    from tvm import autotvm
    from tvm.autotvm.task.space import ConfigEntity
    import vta
    import c3_reuse_experimental_template  # noqa: F401
    import vta.top.vta_conv2d_residency  # noqa: F401
    from tune_resnet18_vta import register_vta_conv2d_template
    env = vta.get_env()
    if row.get("source_tophub"):
        register_vta_conv2d_template()
        name, args = "conv2d_packed.vta", tuple(row["identity"]["workload"][1:])
    else:
        name = "conv2d_packed_reuse_control.vta" if row.get("experimental_reload_control") else "conv2d_packed_residency.vta"
        args = tuple(row["identity"]["workload"][1:]) + (row["implementation_mode"],)
    task = autotvm.task.create(name, args=args, target=env.target, target_host=env.target_host)
    # The index is serialization metadata only: instantiate the complete entity,
    # never look it up in a potentially different ConfigSpace.
    config = ConfigEntity.from_json_dict({"index": -1, "code_hash": None, **row["complete_config_entity"]})
    with task.target:
        return task.instantiate(config)


@contextlib.contextmanager
def adapter():
    old_file, old_instantiate = s.__file__, s.instantiate
    s.__file__, s.instantiate = __file__, instantiate
    old_ld = os.environ.pop("LD_LIBRARY_PATH", None)
    old_env = {k: os.environ.get(k) for k in protocol.QUEUE_ENV}
    os.environ.update(protocol.QUEUE_ENV)
    try:
        yield
    finally:
        s.__file__, s.instantiate = old_file, old_instantiate
        if old_ld is not None:
            os.environ["LD_LIBRARY_PATH"] = old_ld
        for key,value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def bindings():
    result = s.source_bindings()
    names = [Path(__file__), HERE / "c3_thread_sync_audit.py", HERE / "c3_thread_sync_protocol.py",
             HERE / "c3_thread_sync_diagnostic.cc", HERE / "c3_reuse_timed_once.cc",
             HERE / "test_c3_reuse_timed_once.py", HERE / "run_vta_c3_reuse_board.py",
             HERE / "diagnose_vta_c3_reuse_completion.py", HERE / "run_vta_p7r132_y00_search_confirmation.py",
             HERE / "run_vta_p7r120_yolo_rpc_board.py", HERE / "tune_resnet18_vta.py",
             HERE / "audit_resnet18_sram_accounting.py", HERE / "recover_vta_c3_reuse_cross.py",
             HERE / "c3_thread_sync_board.py", HERE / "c3_thread_sync_results.py"]
    return {**result, **{str(p.resolve()): s.file_hash(p) for p in names}}


def make_candidates(env, fingerprint, version):
    from prepare_vta_c3_resnet18_literature_candidates import make_workload, create_task, knobs
    from c3_candidate_identity import normalize_config_entity
    workload = make_workload(SPEC, env)
    official = s.verify(TOPHUB)["official_record"]
    if official["input"][1:3] != [workload[0], workload[1:]]:
        raise ValueError("TopHub workload changed")
    task = create_task(workload, 0, env)
    domain = [normalize_config_entity(task.config_space.get(i).to_json_dict()) for i in range(len(task.config_space))]
    rows = []
    for tile, h in (("T8", 8), ("T7", 7)):
        for label, (mode, threads, reload) in protocol.VERSIONS.items():
            desired = dict(tile_b=1, tile_h=h, tile_w=56, tile_ci=1, tile_co=2, oc_nthread=threads, h_nthread=1)
            matches = [entity for entity in domain if knobs(entity) == desired]
            if len(matches) != 1:
                raise ValueError("complete entity not uniquely represented " + tile + label)
            entity = matches[0]
            row = dict(candidate_id=tile + "_" + label, tile=tile, version=label,
                       family_id=tile, workload_id="R18-H1", implementation_mode=mode,
                       residence_mode={0:"original", 1:"input_stationary", 2:"weight_loop_control", 4:"weight_resident_barrier", 5:"input_weight_resident_barrier"}[mode],
                       experimental_reload_control=reload, complete_config_entity=entity,
                       identity=dict(workload=workload, complete_config_entity=entity),
                       visible_features=desired, hardware_fingerprint=fingerprint,
                       applicability=dict(full_weight_fits_sram=True), schedule_version=version)
            if tile == "T8" and label == "A":
                row.update(source_tophub=True, adapter_config=entity, complete_config_entity=official["config"],
                           identity=dict(workload=workload, complete_config_entity=official["config"]))
            row["identity_sha256"] = s.digest({k:v for k,v in row.items() if k != "candidate_id"})
            rows.append(row)
    return rows


def diagnostic_worker(request, response):
    import numpy as np
    import tvm
    import vta
    from c3_reuse_semantics import capture_lower
    from run_vta_p7r132_y00_search_confirmation import allocate_reusable_buffers, prepare_reused_buffers
    row = request["candidate"]
    backend = ctypes.CDLL(str(ROOT / "build/libvta_fsim.so"), mode=ctypes.RTLD_GLOBAL)
    backend.VTATLSCommandHandle.restype = ctypes.c_void_p
    backend.VTASetDebugMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    schedule, tensors = instantiate(row)
    lowered, _, _ = capture_lower(
        schedule,
        tensors,
        build_config_factory=lambda **kwargs: s.candidate_build_config(row, **kwargs),
    )
    if s.digest(tvm.ir.save_json(lowered)) != s.digest(Path(request["tir"]).read_text()):
        raise ValueError("diagnostic program identity changed")
    with s.candidate_build_config(
        row, disabled_pass={"tir.CommonSubexprElimTIR"}
    ):
        module = vta.build(schedule, tensors, target=tvm.target.Target(vta.get_env().target, host="llvm"), name="main")
    buffers = allocate_reusable_buffers(tvm.device("ext_dev", 0), row["identity"]["workload"])
    results = []
    for phase in ("warmup", "diagnostic"):
        for seed in protocol.SEEDS:
            expected = prepare_reused_buffers(buffers, row["identity"]["workload"], seed)
            backend.VTASetDebugMode(backend.VTATLSCommandHandle(), 6 if phase == "diagnostic" else 0)
            ctypes.CDLL(None).fflush(None)
            print(f"[ABLATION_BEGIN] {phase} {seed}", flush=True)
            tvm.get_global_func("vta.runtime.profiler_clear")()
            module["main"](buffers["data"], buffers["weight"], buffers["output"])
            ctypes.CDLL(None).fflush(None)
            print(f"[ABLATION_END] {phase} {seed}", flush=True)
            if not np.array_equal(buffers["output"].numpy(), expected):
                raise ValueError("diagnostic FSim output incorrect")
            results.append(dict(phase=phase, seed=seed, correct=True,
                                profile=json.loads(tvm.get_global_func("vta.runtime.profiler_status")())))
    backend.VTASetDebugMode(backend.VTATLSCommandHandle(), 0)
    s.write(response, dict(status="passed", calls=results, complete_operator_calls=6))


def diagnostic(row, out, args):
    cid = row["candidate_id"]
    with tempfile.TemporaryDirectory(prefix="c3_thread_diag_") as temp:
        req, resp = Path(temp)/"request.json", Path(temp)/"response.json"
        s.write(req, dict(candidate=row, tir=str(out/"tir"/(cid+".json"))))
        completed = subprocess.run([sys.executable, __file__, "--diagnostic-request", str(req), "--worker-response", str(resp)],
                                   capture_output=True, text=True, timeout=args.timeout)
        (out/"logs"/(cid+".diagnostic.stdout.log")).write_text(completed.stdout)
        (out/"logs"/(cid+".diagnostic.stderr.log")).write_text(completed.stderr)
        if completed.returncode or not resp.exists():
            raise ValueError("diagnostic subprocess failed: " + completed.stderr[-2000:])
        result = s.read(resp)
        result["instruction_audits"] = {}
        for seed in protocol.SEEDS:
            begin, end = f"[ABLATION_BEGIN] diagnostic {seed}\n", f"[ABLATION_END] diagnostic {seed}"
            text = completed.stdout.split(begin, 1)[1].split(end, 1)[0]
            parsed = audit.instruction_dump(text)
            profile = next(c["profile"] for c in result["calls"] if c["phase"] == "diagnostic" and c["seed"] == seed)
            queued = row["fsim_qualification"]["command_features"]["values"]
            if parsed["submission_count"] != queued["submissions"] or sum(
                    block["instructions"] * 16 for block in parsed["submissions"]) != queued["totals"]["insn_bytes"]:
                raise ValueError("diagnostic submission count mismatch")
            if profile["synchronize_insns"] != sum(block["instructions"] for block in parsed["submissions"]):
                raise ValueError("diagnostic/runtime instruction count mismatch")
            result["instruction_audits"][str(seed)] = parsed
        return result


def build_helper(args, out):
    from recover_vta_c3_reuse_cross import sdk_environment
    dest = out/"binaries/c3_thread_sync_helper.so"
    command = ["bash", "-c", 'source "$1" >/dev/null && shift && exec $CXX "$@"', "ablation-helper", args.sdk_environment,
               "-std=c++17", "-shared", "-fPIC", str(HERE/"c3_reuse_timed_once.cc"), str(HERE/"c3_thread_sync_diagnostic.cc"),
               "-I"+str(ROOT/"include"), "-I"+str(ROOT/"3rdparty/dlpack/include"), "-I"+str(ROOT/"3rdparty/dmlc-core/include"),
               "-L"+str(ROOT/"build_axu_aarch64"), "-ltvm_runtime", "-ldl", "-Wl,-rpath,$ORIGIN", "-o", str(dest)]
    completed = subprocess.run(command, env=sdk_environment(), capture_output=True, text=True, timeout=args.timeout)
    s.write(out/"helper_build.json", dict(command=command, returncode=completed.returncode, stderr=completed.stderr))
    if completed.returncode:
        raise ValueError("helper cross compilation failed")
    return s.file_hash(dest)


def prepare(args, out):
    import numpy as np
    import tvm
    from audit_resnet18_sram_accounting import domain_metrics
    from qualify_vta_residency_fsim import reference_data_from_workload, array_sha256
    parent = s.verify(PARENT)
    # Attest archived runtime artifacts, not current unrelated source paths against an old snapshot.
    for name, path in parent["runtime_paths"].items():
        if s.file_hash(path) != parent["runtime_sha256"][name]:
            raise ValueError("qualified runtime changed: " + name)
    s.verify(Path(parent["micro_qualification_directory"]))
    env, fingerprint = s.hardware()
    source = bindings()
    rows = make_candidates(env, fingerprint, s.digest(source))
    for folder in ("logs", "tir", "binaries", "address_audit"):
        (out/folder).mkdir()
    # Exact helper behavior is tested through RPC, with no FPGA involved.
    checked = subprocess.run([sys.executable, "-m", "pytest", "-q", str(HERE/"test_c3_reuse_timed_once.py")],
                             capture_output=True, text=True, timeout=args.timeout)
    (out/"timer_test.log").write_text(checked.stdout + checked.stderr)
    if checked.returncode:
        raise ValueError("one-call RPC timer contract test failed")
    helper_sha = build_helper(args, out)
    expected = [reference_data_from_workload(rows[0]["identity"]["workload"], seed)[2] for seed in protocol.SEEDS]
    different = {f"{protocol.SEEDS[i]}/{protocol.SEEDS[j]}": int(np.count_nonzero(expected[i] != expected[j]))
                 for i,j in itertools.combinations(range(3), 2)}
    if not all(different.values()) or any(not np.all((x >= 0) & (x <= 127)) for x in expected):
        raise ValueError("seed discrimination/sentinel contract failed")
    contract = dict(schema=SCHEMA, phase="prepare", source_bindings=source,
                    candidates_before_labels=[r["identity_sha256"] for r in rows],
                    parent_directory=str(PARENT), parent_manifest_sha256=s.file_hash(PARENT/"artifact_hashes.json"),
                    runtime_paths=parent["runtime_paths"], runtime_sha256=parent["runtime_sha256"],
                    micro_qualification_directory=parent["micro_qualification_directory"],
                    helper_sha256=helper_sha, bitstream_sha256=s.BITSTREAM,
                    queue_environment=protocol.QUEUE_ENV,
                    timer_test_passed=True, timer_test_sha256=s.file_hash(out/"timer_test.log"),
                    reference_hashes={str(seed):array_sha256(x) for seed,x in zip(protocol.SEEDS,expected)},
                    seed_pair_mismatches=different, williams=protocol.williams(), seeds=list(protocol.SEEDS),
                    affinity_cpu=3, host_policy="record governor, do not modify; no foreign-service interruption",
                    buffer_protocol="one shared pool; output,data,weight; fixed handles; input+weight update; output=-113; exact check",
                    warmup_per_candidate=3, timing_invocations_per_observation=1, hidden_warmup=0,
                    statistical_rule="median of eight per-round ratios; >=2% and >=6/8 wins in BOTH boots",
                    operator_only=True, board_contacted=False)
    s.write(out/"contract.json", contract)
    snapshots = out/"source_snapshot"
    snapshots.mkdir()
    for path in source:
        original = Path(path)
        if original.suffix in (".py", ".cc", ".h", ".json") and original.is_relative_to(ROOT):
            target = snapshots/original.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    modules, addresses = {}, {}
    with adapter():
        for row in rows:
            cid = row["candidate_id"]
            row["local_qualification"] = s.lower_candidate(row, out)
            try:
                if row["local_qualification"]["status"] != "passed":
                    raise ValueError("lowering rejected")
                module = tvm.ir.load_json((out/"tir"/(cid+".json")).read_text())
                address = audit.final_tir_audit(module, row["local_qualification"]["semantics"])
                s.write(out/"address_audit"/(cid+".json"), address)
                row["address_status"] = "passed"
                row["arenas"] = domain_metrics(row["local_qualification"]["storage_arenas"]["allocations"], fingerprint)
                if any(value is None for key,value in row["arenas"].items() if key.endswith("_arena_capacity_pct")) or any(
                        value > 100 for key,value in row["arenas"].items() if key.endswith("_arena_capacity_pct")):
                    raise ValueError("arena legality ambiguous or exceeds capacity")
                modules[cid], addresses[cid] = module, address
                row["fsim_qualification"] = s.fsim(row, row["local_qualification"], out, str(ROOT/"build/libvta_fsim.so"), args.timeout)
                if row["fsim_qualification"]["status"] != "passed":
                    raise ValueError("FSim rejected")
                row["command_qualification"] = diagnostic(row, out, args)
                # A legacy qualification import reintroduces LD_LIBRARY_PATH.
                # Sanitize immediately before the SDK child, not just at phase entry.
                with adapter():
                    row["cross_qualification"] = s.cross(row, out, args)
                if row["cross_qualification"]["status"] != "passed":
                    raise ValueError("cross compilation rejected")
                row["status"] = "passed"
            except Exception as error:
                row.update(status="rejected", rejection=str(error))
            s.append(out/"qualification.jsonl", row)
            print(cid, row["status"], row.get("rejection", ""), flush=True)
        baseline = rows[0]
        alias = {**baseline, "source_tophub":False, "candidate_id":"T8_A_adapter", "complete_config_entity":baseline["adapter_config"]}
        alias["identity"] = {**alias["identity"], "complete_config_entity":alias["complete_config_entity"]}
        alias_result = s.lower_candidate(alias, out)
        equivalent = alias_result.get("tir_sha256") == baseline["local_qualification"].get("tir_sha256") and alias_result["status"] == "passed"
        s.write(out/"adapter_equivalence.json", dict(passed=equivalent, adapter=alias_result))
    controls = {}
    by_id = {r["candidate_id"]:r for r in rows}
    for tile in ("T8", "T7"):
        for a,b,kind in protocol.COMPARISONS:
            if kind not in ("sync", "residency"):
                continue
            left, right = tile+"_"+a, tile+"_"+b
            key = tile+":"+a+"/"+b
            if not all(by_id[c]["status"] == "passed" for c in (left,right)):
                controls[key] = dict(matched=False, reason="candidate unqualified")
                continue
            match = audit.control_match(addresses[left], addresses[right], modules[left], modules[right], kind)
            if kind == "residency":
                lq,rq = by_id[left]["local_qualification"],by_id[right]["local_qualification"]
                match["checks"]["explicit_sync_structure"] = lq["sync_multiplicity"] == rq["sync_multiplicity"]
                match["checks"]["reduced_weight_bytes"] = lq["semantics"]["totals"]["load_buffer_2d_wgt_bytes"] > rq["semantics"]["totals"]["load_buffer_2d_wgt_bytes"]
                match["matched"] = all(match["checks"].values())
            controls[key] = match
    s.write(out/"controls.json", controls)
    s.write(out/"candidates.json", rows)
    s.write(out/"summary.json", dict(status="PREPARED" if equivalent else "ADAPTER_REJECTED",
             gross_candidates=16, passed=[r["candidate_id"] for r in rows if r["status"]=="passed"],
             controls_matched={k:v["matched"] for k,v in controls.items()}, board_contacted=False))
    s.check_bindings(source)


def freeze(args, out):
    directory = Path(args.contract).resolve().parent
    contract = s.verify(directory)
    if contract["schema"] != SCHEMA or contract["phase"] != "prepare":
        raise ValueError("freeze requires this experiment's prepare")
    s.check_bindings(contract["source_bindings"])
    rows = s.read(directory/"candidates.json")
    if len(rows) != 16 or {r["candidate_id"] for r in rows} != {t+"_"+v for t in ("T8","T7") for v in protocol.LETTERS}:
        raise ValueError("incomplete attempted pool")
    if not s.read(directory/"adapter_equivalence.json")["passed"] or not contract["timer_test_passed"]:
        raise ValueError("baseline/timer equivalence failed")
    accepted = [r["candidate_id"] for r in rows if r["status"] == "passed"]
    if "T8_A" not in accepted:
        raise ValueError("genuine baseline unqualified")
    for cid in accepted:
        row = next(r for r in rows if r["candidate_id"] == cid)
        if any(row[key]["status"] != "passed" for key in ("local_qualification","fsim_qualification","command_qualification","cross_qualification")) or row["address_status"] != "passed":
            raise ValueError("qualification incomplete")
    contract = {**contract, "phase":"freeze", "prepared_directory":str(directory),
                "prepared_manifest_sha256":s.file_hash(directory/"artifact_hashes.json"),
                "accepted":accepted, "timing_schedule":protocol.schedule(accepted),
                "balance":{t:protocol.balance([x[-1] for x in accepted if x.startswith(t)]) for t in ("T8","T7")}}
    s.write(out/"contract.json", contract)
    s.write(out/"summary.json", dict(status="FROZEN", accepted=accepted, timing_samples=len(contract["timing_schedule"]), board_contacted=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare","freeze","board","analyze"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--contract")
    parser.add_argument("--previous-boot-dir", type=Path)
    parser.add_argument("--boot-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--execute-board", action="store_true")
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    parser.add_argument("--sdk-environment", default=SDK)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--worker-request")
    parser.add_argument("--cross-request")
    parser.add_argument("--diagnostic-request")
    parser.add_argument("--worker-response")
    args = parser.parse_args()
    with adapter():
        if args.worker_request:
            s.worker(s.read(args.worker_request), Path(args.worker_response)); return
        if args.cross_request:
            s.cross_worker(s.read(args.cross_request), Path(args.worker_response)); return
        if args.diagnostic_request:
            diagnostic_worker(s.read(args.diagnostic_request), Path(args.worker_response)); return
    if not args.phase or not args.output_dir:
        parser.error("phase and output-dir required")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        if args.phase == "prepare": prepare(args, out)
        elif args.phase == "freeze": freeze(args, out)
        elif args.phase == "board":
            from c3_thread_sync_board import run
            run(args, out)
        else:
            from c3_thread_sync_results import analyze
            analyze(args, out)
    except BaseException as error:
        s.write(out/"invalid_session.json", dict(type=type(error).__name__, message=str(error),
                  all_timing_invalid=True, wall_seconds=time.monotonic()-start))
        raise
    finally:
        s.write(out/"phase_cost.json", dict(wall_seconds=time.monotonic()-start))
        s.seal(out)


if __name__ == "__main__":
    main()
