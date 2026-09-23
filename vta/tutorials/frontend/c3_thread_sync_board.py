"""Exclusive two-session board protocol. Never reboot or stop foreign RPCs."""
import json
from pathlib import Path
import shlex
import subprocess
import time

import c3_thread_sync_protocol as p
import c3_thread_sync_audit as audit
import run_vta_c3_reuse_supplement as s
import run_vta_c3_reuse_board as b
from diagnose_vta_c3_reuse_completion import DiagnosticBoard


class ExperimentBoard(DiagnosticBoard):
    def __init__(self, args, runtime, port):
        super().__init__(args, runtime)
        self.port = port

    def start(self):
        path = shlex.quote(self.runtime)
        self.ssh("test -d /sys/devices/system/cpu/cpu3 && command -v taskset >/dev/null && "
                 "taskset -c 3 true")
        script = ("(cd " + path + " && exec taskset -c 3 env LD_LIBRARY_PATH=" + path +
                  " LD_PRELOAD=" + path + "/libvta.so TVM_NUM_THREADS=1 "
                  + " ".join(k+"="+shlex.quote(v) for k,v in p.QUEUE_ENV.items()) +
                  " VTA_QUEUE_DIAGNOSTICS=0 "
                  "./tvm_rpc server --host=0.0.0.0 --port=" + str(self.port) +
                  " --port-end=" + str(self.port + 1) + ") >" +
                  path + "/rpc.log 2>&1 < /dev/null & echo $!")
        self.pid = int(self.ssh(script).strip())
        return self.pid

    def environment(self, require_affinity=False):
        # Only inspect processes whose executable and cwd are this experiment's.
        path = shlex.quote(self.runtime)
        raw = self.ssh(
            "cat /proc/sys/kernel/random/boot_id; cat /sys/devices/system/cpu/online; "
            "cat /proc/loadavg; "
            "for f in /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor "
            "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq; do "
            "test ! -r \"$f\" || { echo \"$f\"; cat \"$f\"; }; done; "
            "for ablation_pid in $(pidof tvm_rpc); do "
            "test \"$(readlink /proc/$ablation_pid/cwd)\" = " + path + " || continue; "
            "test \"$(readlink /proc/$ablation_pid/exe)\" = " + path + "/tvm_rpc || continue; "
            "for f in /proc/$ablation_pid/task/*/status; do "
            "awk '/^Cpus_allowed_list:/ {print \"AFFINITY \" $2}' \"$f\"; done; done; "
            "ps")
        masks = [line.split()[1] for line in raw.splitlines() if line.startswith("AFFINITY ")]
        if require_affinity and (not masks or any(mask != "3" for mask in masks)):
            raise ValueError("RPC process/thread CPU3 affinity mismatch")
        governor = "unknown_unavailable"
        lines = raw.splitlines()
        for i,line in enumerate(lines[:-1]):
            if line.endswith("/scaling_governor"):
                governor = lines[i+1]
        return dict(raw=raw, governor=governor, masks=masks, boot_id=lines[0])


def validate_profile(row, result, require_warmed_signature=True):
    observed = result["runtime_profile_complete"]
    expected = row["local_qualification"]["semantics"]["totals"]
    if any(observed.get(key) != value for key,value in expected.items()):
        raise ValueError("board/static data requests differ: " + row["candidate_id"])
    expected_sig = row["command_qualification"]["calls"][-1]["profile"]
    if require_warmed_signature and p.logical_signature(observed) != p.logical_signature(expected_sig):
        raise ValueError("board/local warmed command signature differs: " + row["candidate_id"])
    # FSim driver_run_calls is zero by design, not a submission observation.
    expected_submits = row["command_qualification"]["instruction_audits"]["0"]["submission_count"]
    if observed["driver_run_calls"] != expected_submits:
        raise ValueError("board/local submission count differs")


def requires_local_warmed_signature(phase):
    return phase == "diagnostic"


def run(args, out):
    if not args.execute_board:
        raise ValueError("board requires --execute-board")
    import numpy as np
    from tvm import rpc
    from run_vta_p7r120_yolo_rpc_board import runtime_inventory, load_ephemeral
    from run_vta_p7r132_y00_search_confirmation import allocate_reusable_buffers, prepare_reused_buffers
    from qualify_vta_residency_fsim import array_sha256
    frozen = Path(args.contract).resolve().parent
    contract = s.verify(frozen)
    if contract.get("phase") != "freeze" or contract.get("schema") != "c3_thread_sync_ablation_v1":
        raise ValueError("board requires frozen ablation contract")
    s.check_bindings(contract["source_bindings"])
    prepared = Path(contract["prepared_directory"])
    s.verify(prepared)
    if s.file_hash(prepared/"artifact_hashes.json") != contract["prepared_manifest_sha256"]:
        raise ValueError("preparation changed")
    for name,path in contract["runtime_paths"].items():
        if s.file_hash(path) != contract["runtime_sha256"][name]:
            raise ValueError("runtime changed")
    previous = None
    if args.previous_boot_dir:
        s.verify(args.previous_boot_dir)
        previous = s.read(args.previous_boot_dir/"summary.json")
        prev_contract = s.read(args.previous_boot_dir/"contract.json")
        if not previous.get("timing_valid") or prev_contract["frozen_manifest_sha256"] != s.file_hash(frozen/"artifact_hashes.json"):
            raise ValueError("invalid/different first boot")
        if prev_contract["boot_index"] != 1:
            raise ValueError("previous session must be boot 1")
    rows = {r["candidate_id"]:r for r in s.read(prepared/"candidates.json")}
    selected = list(contract["accepted"])
    run_contract = dict(schema=contract["schema"], phase="board", frozen_directory=str(frozen),
                        frozen_manifest_sha256=s.file_hash(frozen/"artifact_hashes.json"),
                        boot_index=2 if previous else 1,
                        previous_boot_directory=str(args.previous_boot_dir) if previous else None,
                        source_bindings=contract["source_bindings"], hidden_extra_calls=0,
                        invocation_per_sample=1, no_transparent_retry=True,
                        software_policy=dict(cpu=3, threads=1, governor="record_only", replay="disabled",
                                             session_ports={"diagnostic": 9090, "timing": 9091}))
    s.write(out/"contract.json", run_contract)
    (out/"diagnostic_dumps").mkdir()
    boot_id, governor, diagnostic_signatures = None, None, {}
    timing, calls = [], []
    b.RUNTIME = contract["runtime_sha256"]
    for session in ("diagnostic", "timing"):
        port = {"diagnostic": 9090, "timing": 9091}[session]
        board = ExperimentBoard(args, "/var/volatile/" + out.name + "_" + session, port)
        remote, modules, retained = None, {}, []
        try:
            state = board.state()
            b.validate_state(state, exclusive=True)
            environment = board.environment()
            s.write(out/(session+"_preflight.json"), dict(state=state, environment=environment))
            if boot_id is not None and state["boot_id"] != boot_id:
                raise ValueError("boot changed between diagnostic and timing")
            boot_id = state["boot_id"]
            if previous and previous["boot_id"] == boot_id:
                raise ValueError("user reboot required; current boot is unchanged")
            if previous and previous["governor"] != environment["governor"]:
                raise ValueError("CPU governor policy changed between boots")
            if governor is not None and governor != environment["governor"]:
                raise ValueError("CPU governor policy changed between sessions")
            governor = environment["governor"]
            board.deploy(contract["runtime_paths"])
            s.write(out/(session+"_programming.json"), board.reload())
            board.start()
            # Connection retries do not invoke an operator.
            for _ in range(20):
                try:
                    remote = rpc.connect(args.host, port, session_timeout=7200)
                    break
                except Exception:
                    time.sleep(.25)
            if remote is None:
                raise RuntimeError("RPC unavailable")
            inventory, functions = runtime_inventory(remote)
            s.write(out/(session+"_inventory.json"), inventory)
            helper = prepared/"binaries/c3_thread_sync_helper.so"
            if s.file_hash(helper) != contract["helper_sha256"]:
                raise ValueError("timer helper identity mismatch")
            retained.append(load_ephemeral(remote, helper))
            timer = remote.get_function("c3.reuse.timed_once")
            diagnostic = remote.get_function("c3.thread_sync.diagnostic_once")
            buffers = allocate_reusable_buffers(remote.ext_dev(0), rows["T8_A"]["identity"]["workload"])
            s.write(out/(session+"_affinity.json"), board.environment(require_affinity=True))

            def call(cid, seed, phase, spec=None):
                row = rows[cid]
                if cid not in modules:
                    binary = prepared/"binaries"/(cid+".so")
                    if s.file_hash(binary) != row["cross_qualification"]["binary_sha256"]:
                        raise ValueError("operator binary changed")
                    modules[cid] = load_ephemeral(remote, binary)
                record = dict(candidate_id=cid, seed=seed, phase=phase, session=session,
                              invocation_index=len(calls), tile=row["tile"], version=row["version"])
                s.append(out/"dispatch.jsonl", {**record, "event":"begin"})
                if session == "diagnostic":
                    expected = prepare_reused_buffers(buffers, row["identity"]["workload"], seed)
                    functions["vta.runtime.profiler_clear"]()
                    dump = str(diagnostic(modules[cid], buffers["data"], buffers["weight"], buffers["output"]))
                    actual = buffers["output"].numpy()
                    profile = json.loads(functions["vta.runtime.profiler_status"]())
                    result = dict(correct=bool(np.array_equal(actual, expected)), mismatch_count=int(np.count_nonzero(actual != expected)),
                                  actual_sha256=array_sha256(actual), expected_sha256=array_sha256(expected),
                                  runtime_profile_complete=profile, complete_operator_calls=1, hidden_warmup_calls=0)
                    name = f"{len(calls):04d}_{cid}_{phase}_{seed}.txt"
                    (out/"diagnostic_dumps"/name).write_text(dump)
                    result["instruction_audit"] = audit.instruction_dump(dump)
                    if result["instruction_audit"]["submission_count"] != profile["driver_run_calls"]:
                        raise ValueError("board dump/submission mismatch")
                    result["dump_file"] = "diagnostic_dumps/" + name
                else:
                    result = b.invoke(modules[cid], buffers, row["identity"]["workload"], seed,
                                      functions, timer, timed=phase=="timing")
                record.update(result)
                if spec:
                    record.update(spec)
                calls.append(record)
                s.append(out/"calls.jsonl", record)
                s.append(out/"dispatch.jsonl", dict(invocation_index=record["invocation_index"], event="end"))
                if not record["correct"]:
                    return record
                # The first call after loading a module can contain a different
                # pipeline NOP count.  It is retained as first-call evidence,
                # but only the post-warmup diagnostic calls are compared to
                # the locally captured warmed command signature.  A timing
                # mismatch is retained below, as required by the protocol;
                # it prevents using the dump to explain that timing but does
                # not invalidate an otherwise correct single invocation.
                validate_profile(row, record, requires_local_warmed_signature(phase))
                if session == "diagnostic" and phase == "diagnostic":
                    diagnostic_signatures[cid,seed] = p.logical_signature(record["runtime_profile_complete"])
                if phase == "timing":
                    record["diagnostic_signature_matches"] = diagnostic_signatures.get((cid,seed)) == p.logical_signature(record["runtime_profile_complete"])
                    timing.append(record)
                    s.append(out/"timing.jsonl", record)
                return record

            for seed in p.SEEDS:
                if not call("T8_A", seed, "health")["correct"]:
                    raise ValueError("baseline health output incorrect")
            if session == "diagnostic":
                accepted = []
                for cid in selected:
                    correct = True
                    for seed in p.SEEDS:
                        result = call(cid,seed,"correctness_warmup")
                        correct = correct and result["correct"]
                    if correct:
                        for seed in p.SEEDS:
                            result = call(cid,seed,"diagnostic")
                            correct = correct and result["correct"]
                    if correct:
                        accepted.append(cid)
                    print("board qualification", cid, correct, flush=True)
                if "T8_A" not in accepted:
                    raise ValueError("genuine baseline board-rejected")
                selected = accepted
                s.write(out/"board_pool.json", dict(accepted=selected, rejected=[x for x in contract["accepted"] if x not in selected],
                      timing_schedule=p.schedule(selected),
                      balance={t:p.balance([x[-1] for x in selected if x.startswith(t)]) for t in ("T8","T7")}))
            else:
                for cid in selected:
                    for seed in p.SEEDS:
                        if not call(cid,seed,"warmup")["correct"]:
                            raise ValueError("formal warmup output incorrect")
                for spec in p.schedule(selected):
                    if not call(spec["candidate_id"], spec["seed"], "timing", spec)["correct"]:
                        raise ValueError("timed output incorrect; whole session invalid")
                    if spec["ordinal"] % 48 == 47:
                        s.append(out/"environment.jsonl", board.environment(require_affinity=True))
                        print("timing calls", spec["ordinal"]+1, flush=True)
                p.validate_timing(timing, selected)
            final = board.state()
            b.validate_state(final)
            final_env = board.environment(require_affinity=True)
            if final["boot_id"] != boot_id or final_env["governor"] != governor:
                raise ValueError("boot/environment changed")
            s.write(out/(session+"_postflight.json"), dict(state=final, environment=final_env))
            s.check_bindings(contract["source_bindings"])
        finally:
            # Release every remote module while the connection remains alive.
            modules.clear()
            retained.clear()
            remote = None
            if board.pid is not None:
                board.stop()
    s.write(out/"summary.json", dict(status="BOOT2_COMPLETE" if previous else "BOOT1_COMPLETE_WAIT_USER_REBOOT",
              timing_valid=True, boot_id=boot_id, governor=governor, accepted=selected,
              timing_samples=len(timing), timed_operator_calls=len(timing),
              formal_warmup_calls=sum(x["session"]=="timing" and x["phase"]=="warmup" for x in calls),
              all_completed_operator_calls=len(calls), hidden_extra_calls=0,
              phase_counts={phase:sum(x["phase"]==phase for x in calls) for phase in {x["phase"] for x in calls}},
              all_timing_correct=True, all_diagnostic_signatures_match=all(x["diagnostic_signature_matches"] for x in timing)))
