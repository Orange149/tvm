#!/usr/bin/env python3
"""Local freeze and explicit board entry for the fixed C3 aggregation pool.

Import/plan/freeze perform no board access. Qualify/diagnose/formal require
--execute-board and a new, completely frozen local pool and ARM index.
Historical runners remain unchanged; only their narrow methods are cloned.
"""

from collections import Counter
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import shlex
import sys
import textwrap
import time
import traceback
from types import MethodType, SimpleNamespace


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCHEMA = "c3_valid_window_five_program_board_v1"
LABELS = ("Q1","Q2","Q4","Q8","TPS")
SEEDS = (0,20250901,20260910)
ORDERS = (("Q1","Q2","Q8","Q4","TPS"),("Q2","Q8","Q4","TPS","Q1"),
          ("Q8","Q4","TPS","Q1","Q2"),("Q4","TPS","Q1","Q2","Q8"),
          ("TPS","Q1","Q2","Q8","Q4"),("TPS","Q4","Q8","Q2","Q1"))
SEED_ORDERS = tuple(SEEDS[r%3:]+SEEDS[:r%3] for r in range(6))
BOARD_RUNTIME_SHA256 = "43aaf6321eb6cf685ce8b69ee2ef469bcf708d57afd74525e13287d77cbb36ef"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")


def plan(aliases=None):
    aliases = {name:(aliases or {}).get(name,name) for name in LABELS}
    if aliases["TPS"] != "TPS" or any(aliases[n] not in LABELS[:4] for n in LABELS[:4]):
        raise ValueError("aliases must only deduplicate Q labels; TPS remains sealed")
    if any(aliases[aliases[n]] != aliases[n] for n in LABELS):
        raise ValueError("aliases must point directly to canonical programs")
    programs = tuple(dict.fromkeys(aliases[n] for n in LABELS))
    orders = [list(dict.fromkeys(aliases[n] for n in order)) for order in ORDERS]
    if any(set(row) != set(programs) for row in orders):
        raise ValueError("deduplicated order does not preserve the entire fixed pool")
    count = 6*len(programs)*3
    diagnostic_count = 3*len(programs)*3
    return {"schema":SCHEMA,"programs":list(programs),"aliases":aliases,"orders":orders,
        "seed_orders":[list(row) for row in SEED_ORDERS],"nominal_orders":[list(row) for row in ORDERS],
        "expected_calls":{"correctness":3*len(programs),"new_Q_weight_ABA":3*(len(programs)-1),
            "diagnostic_prefix":diagnostic_count,"diagnostic_positions_4_5_6":diagnostic_count,
            "formal_explicit_warmup":count,"formal_timing":count},
        "diagnostic_first_call_is_prefix_position_1":True,
        "diagnostic_rotations":3,"formal_round_to_diagnostic_rotation":[0,1,2,0,1,2],
        "dedup_rule":"keep each canonical program's first occurrence in each original row; no replacement or reordered pool",
        "analysis_scope":"Q8 versus Q1 primary; Q2/Q4 grain description; sealed TPS practical reference; no fit/selection/graph/reboot",
        "no_automatic_retry":True,"board_contacted":False}


def _base():
    sys.path[:0] = [str(REPO / "python"),str(REPO / "vta/python")]
    import c3_residency_tradeoff_board
    return c3_residency_tradeoff_board


def execution_sources():
    names = ("c3_valid_window_board.py","c3_valid_window_uop_values.py","c3_valid_window.py","c3_valid_window_audit.py",
        "c3_valid_window_analysis.py","c3_valid_window_analysis_review.py",
        "c3_banerjee_uop_queue.h","build_c3_banerjee_runtime.py",
        "c3_residency_tradeoff_board.py","c3_residency_tradeoff.py","c3_residency_tradeoff_evidence.py",
        "c3_thread_sync_board.py","c3_thread_sync_protocol.py","c3_thread_sync_audit.py",
        "c3_five_tradeoff_commands.py","diagnose_vta_c3_reuse_completion.py",
        "run_vta_c3_reuse_board.py","run_vta_c3_reuse_supplement.py",
        "run_vta_c3_mode11_graph_context_diagnostic.py","run_vta_p7r120_yolo_rpc_board.py",
        "run_vta_c3_node24_partial_board.py","run_vta_c3_node24_partial_local.py",
        "c3_node34_fused_timed_once.cc","c3_node24_fused_diagnostic_once.cc")
    return {str(HERE/name):digest(HERE/name) for name in names}


def command_fingerprint(text):
    """All encoded command descriptors; normalize one DDR object per domain.

    The frozen four-buffer ABI maps INP/WGT/ACC-load/OUT-store to X/W/ADD/Y.
    UOP's host serialization has its own base. Necessary stage/token fields,
    shapes, padding, SRAM ranges, GEMM/ALU loops and command order remain.
    Unprinted ALU immediate is certified by final TIR/LLVM, separately.
    """
    if len(re.findall(r"There are \d+ instructions",text)) != 1:
        raise ValueError("normalized command support requires one complete submit")
    blocks = [block for block in re.split(r"(?=INSTRUCTION \d+:)",text[text.index("INSTRUCTION 0:"):]) if block]
    declared = int(re.search(r"There are (\d+) instructions",text).group(1))
    if len(blocks) != declared or [int(re.match(r"INSTRUCTION (\d+):",block).group(1)) for block in blocks] != list(range(declared)):
        raise ValueError("incomplete command dump")
    records,bases = [],{}
    for block in blocks:
        lines = block.splitlines()
        description = re.match(r"INSTRUCTION (\d+): (.*)",lines[0]).group(2)
        prefixes = ("dep -","DRAM:","y:","x:","reset_out:","range ","outer loop -","inner loop -")
        detail = [line.strip() for line in lines[1:] if line.strip().startswith(prefixes)]
        address = re.search(r"DRAM: 0x([0-9a-f]+), SRAM:0x([0-9a-f]+)",block)
        domain = description if description.startswith("LOAD ") else "STORE OUT" if description.startswith("STORE") and address else None
        if address:
            if domain is None:
                raise ValueError("unrecognized memory address descriptor")
            value = int(address[1],16)
            bases[domain] = min(bases.get(domain,value),value)
            record = {"description":description,"detail":detail,"domain":domain,"dram":value}
        else:
            record = {"description":description,"detail":detail}
        records.append(record)
    for record in records:
        if "dram" in record:
            relative = record.pop("dram")-bases[record.pop("domain")]
            record["detail"] = [re.sub(r"DRAM: 0x[0-9a-f]+", "DRAM_RELATIVE_ELEMENTS: %d" % relative,line) for line in record["detail"]]
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()


def expected_kernels(record):
    trace = read(record["expected_trace"]["path"])
    if isinstance(trace,list):
        if trace and isinstance(trace[0],dict) and "kind" in trace[0]:
            return [event for event in trace if event["kind"] == "kernel"]
        return trace  # Legacy list of per-kernel UOP rows.
    if "kernels" in trace:
        return trace["kernels"]
    if "events" in trace:
        return [event for event in trace["events"] if event["kind"] == "kernel"]
    raise ValueError("expected trace lacks a kernel list")


def _tir_and_trace_binding(record):
    """JSON byte identities stay separate from structural/trace equivalence."""
    import tvm
    from c3_valid_window import trace_function
    final_path = Path(record["directory"])/"node24_final.tir.json"
    arm = tvm.ir.load_json(final_path.read_text())
    if record.get("source_sim_directory"):
        sim_path = Path(record["source_sim_directory"])/"node24_final.tir.json"
        if digest(sim_path) != record["source_sim_tir_sha256"]:
            raise ValueError("source simulation final TIR changed")
        sim = tvm.ir.load_json(sim_path.read_text())
        if not tvm.ir.structural_equal(arm,sim,map_free_vars=False):
            raise ValueError("ARM and simulation final TIR differ structurally")
    expected = read(record["expected_trace"]["path"])
    if isinstance(expected,dict):
        expected = expected.get("events")
    if not isinstance(expected,list) or not expected or not isinstance(expected[0],dict) or "kind" not in expected[0]:
        raise ValueError("board freeze requires an independently frozen full event trace")
    actual = trace_function(arm)
    normalize = lambda value:json.dumps(value,sort_keys=True)
    if normalize(actual) != normalize(expected):
        raise ValueError("re-expanded ARM final TIR event trace differs from frozen simulation trace")
    return {"status":"ARM_SIM_STRUCTURAL_AND_REEXPANDED_TRACE_EQUAL",
        "arm_final_tir_sha256":digest(final_path),"event_count":len(actual),
        "trace_sha256":hashlib.sha256(normalize(actual).encode()).hexdigest()}


def _restore_evidence(root):
    directories = [root.parent/"20261014_c3_valid_window_volatile_restore_run02",
                   root.parent/"20261014_c3_valid_window_taskset_restore_run01"]
    expected = ("VOLATILE_PREREQUISITES_RESTORED","VOLATILE_TASKSET_RESTORED")
    records = []
    for directory,status in zip(directories,expected):
        contract,summary,manifest = [read(directory/name) for name in ("contract.json","summary.json","artifact_hashes.json")]
        if summary["status"] != status or contract["boot_id"] != summary["boot_id"] or summary["SD_writes"] or contract["SD_writes"]:
            raise ValueError("restoration evidence is incomplete or violates the RAM-only contract")
        for rel,sha in manifest["artifacts"].items():
            if digest(directory/rel) != sha:
                raise ValueError("restoration artifact changed")
        records.append({"directory":str(directory),"boot_id":contract["boot_id"],
            "artifacts":{name:{"path":str(directory/name),"sha256":digest(directory/name)} for name in ("contract.json","summary.json","artifact_hashes.json")}})
    if records[0]["boot_id"] != records[1]["boot_id"]:
        raise ValueError("restoration records are from different Linux boots")
    if "201326592" not in read(directories[0]/"summary.json")["state"] or "0x0000000068500000" not in read(directories[0]/"summary.json")["state"]:
        raise ValueError("restored DMA size/physical base does not match the frozen contract")
    return records


def freeze(root,pool_path,arm_path,runtime_path,document):
    """Only local reads/new writes. An actual ARM index is mandatory."""
    root = Path(root).resolve()
    pool_path,arm_path,runtime_path,document = map(lambda p:Path(p).resolve(),(pool_path,arm_path,runtime_path,document))
    pool,arms,runtime = read(pool_path),read(arm_path),read(runtime_path)
    if pool["status"] != "Q_LOCAL_POOL_FROZEN_BOARD_NOT_QUALIFIED" or pool["TPS_modified"]:
        raise ValueError("the fixed, fully gated local Q pool is required")
    schedule = plan(pool["aliases"])
    if pool["round_order"] != schedule["orders"] or pool["seed_orders"] != schedule["seed_orders"]:
        raise ValueError("local pool order or seed rotations differ from the fixed plan")
    if pool["formal_warmups"] != schedule["expected_calls"]["formal_explicit_warmup"] or pool["formal_timings"] != schedule["expected_calls"]["formal_timing"]:
        raise ValueError("local pool warmup/timing ledger differs")
    local = {row["program"]:row for row in pool["programs"]}
    if any(row["status"] != "ALL_LOCAL_WINDOW_GATES_PASSED" for row in local.values()):
        raise ValueError("local Q native/version/raw-convolution gates are incomplete")
    records = {}
    for name in schedule["programs"]:
        record = arms[name]
        for field in ("directory","binary","binary_sha256","tir_sha256","request_totals","expected_trace","expectation","local_native_signature","local_dumps"):
            if field not in record:
                raise ValueError("ARM record missing required field %s: %s" % (field,name))
        directory = Path(record["directory"])
        if digest(record["binary"]) != record["binary_sha256"] or digest(directory/"node24_final.tir.json") != record["tir_sha256"]:
            raise ValueError("ARM binary/final TIR changed: "+name)
        if digest(record["expected_trace"]["path"]) != record["expected_trace"]["sha256"]:
            raise ValueError("expected final trace changed: "+name)
        if name != "TPS" and record["source_sim_tir_sha256"] != local[name]["final_tir_sha256"]:
            raise ValueError("ARM source simulation final TIR identity differs: "+name)
        exp = record["expectation"]
        if any(field not in exp for field in ("mac","reset","total_gemm_push","wgt_loads","wgt_payload_bytes")):
            raise ValueError("incomplete expectation: "+name)
        if exp["total_gemm_push"] != exp["mac"]+exp["reset"]:
            raise ValueError("GEMM push count inconsistent: "+name)
        if name != "TPS" and (exp["mac"] != local[name]["MAC_GEMM"] or exp["reset"] != local[name]["RESET_GEMM"]):
            raise ValueError("ARM/local actual GEMM signature differs: "+name)
        paths = [Path(record["local_dumps"]["first"]),*[Path(p) for p in record["local_dumps"]["warmed"]]]
        if len(paths) != 4:
            raise ValueError("one cold and three warmed local dumps required")
        copied = dict(record)
        copied["command_fingerprints"] = {"first":command_fingerprint(paths[0].read_text()),
            "warmed":[command_fingerprint(p.read_text()) for p in paths[1:]]}
        copied["local_dump_sha256"] = {str(p):digest(p) for p in paths}
        records[name] = copied
    if runtime["runtime_sha256"]["libvta.so"] != BOARD_RUNTIME_SHA256:
        raise ValueError("wrong common experimental board DSO")
    for name,path in runtime["runtime_paths"].items():
        if digest(path) != runtime["runtime_sha256"][name]:
            raise ValueError("frozen runtime binary changed: "+name)
    runtime_sources = [REPO/rel for rel in ("vta/runtime/runtime.cc","3rdparty/vta-hw/config/vta_config.json",
        "3rdparty/vta-hw/include/vta/hw_spec.h","3rdparty/vta-hw/include/vta/hw_spec_const.h",
        "3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc")]
    runtime_source_bindings = {str(path):runtime["source_bindings"][str(path)] for path in runtime_sources}
    for path,sha in runtime_source_bindings.items():
        if digest(path) != sha:
            raise ValueError("frozen runtime/hardware source changed")
    base = _base()
    for record in records.values():
        record["arm_sim_trace_binding"] = _tir_and_trace_binding(record)
    restored = _restore_evidence(root)
    protocol = {**schedule,"status":"LOCAL_PROTOCOL_FROZEN_BOARD_NOT_STARTED",
        "local_pool_path":str(pool_path),"local_pool_sha256":digest(pool_path),
        "arm_index_path":str(arm_path),"arm_index_sha256":digest(arm_path),
        "runtime_contract_path":str(runtime_path),"runtime_contract_sha256":digest(runtime_path),
        "runtime_sha256":runtime["runtime_sha256"],"document":str(document),"document_sha256":digest(document),
        "runtime_source_bindings":runtime_source_bindings,
        "inherited_unrelated_source_bindings_not_used_for_this_pool":[path for path in runtime["source_bindings"] if path not in runtime_source_bindings],
        "source_sha256":execution_sources(),"helpers":base.helper_certificate(),
        "program_records_sha256":hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
        "restoration_evidence":restored,"required_boot_id":restored[0]["boot_id"],
        "expected_udmabuf_bytes":201326592,"expected_udmabuf_phys":0x68500000,
        "timer_boundary":"one complete four-parameter fused call; organization, all submissions and existing completion inside; upload/RPC/readback/validation outside",
        "diagnostics":"independent fresh RPC per each of three distinct seed rotations/program; first dump is prefix position 1, two further prefix calls, then positions 4/5/6; same module/cache definitions; rotations cover formal rounds 1/4,2/5,3/6",
        "no_reboot_no_SD_writes_no_automatic_infrastructure_restore":True}
    write_new(root/"board_protocol.json",protocol)
    write_new(root/"board_prepared.json",{"schema":SCHEMA,"status":"LOCAL_POOL_AND_ARM_BOUND_BOARD_NOT_QUALIFIED",
        "programs":records,"protocol_sha256":digest(root/"board_protocol.json")})
    print(json.dumps({"status":protocol["status"],"board_contacted":False,"programs":schedule["programs"]}))


def _connect_once(host,port,timeout):
    from tvm import rpc
    return rpc.connect(host,port,session_timeout=timeout)


def _verify_maps(text,runtime_dir):
    for library in ("libvta.so","libtvm_runtime.so"):
        lines = [line for line in text.splitlines() if library in line]
        if not lines or any(runtime_dir+"/"+library not in line for line in lines):
            raise ValueError("actual mapped runtime DSO differs from this tmpfs session")


def _verify_dma_state(board,expected_phys):
    lines = board.ssh("stat -f -c %T /tmp; cat /sys/class/u-dma-buf/udmabuf0/phys_addr").strip().splitlines()
    if len(lines) != 2 or lines[0] != "tmpfs" or int(lines[1],0) != expected_phys:
        raise ValueError("diagnostic /tmp or current DMA physical base differs from the RAM contract")


def _wait_owned_rpc(board,boot_id,limit_seconds=8.0,evidence=None):
    """Read-only readiness polls, never an RPC connect or target-call retry."""
    started,polls = time.monotonic(),[]
    evidence = {} if evidence is None else evidence
    evidence.update(status="OWNED_RPC_STARTUP_PENDING",boot_id=boot_id,port=board.port,
        limit_seconds=limit_seconds,polls=polls,connect_attempts_so_far=0,target_calls_so_far=0)
    runtime = shlex.quote(board.runtime)
    executable = shlex.quote(board.runtime+"/tvm_rpc")
    port = "%04X" % board.port
    command = ("printf 'BOOT '; cat /proc/sys/kernel/random/boot_id; "
        "for c3_ready_pid in $(pidof tvm_rpc 2>/dev/null); do "
        "test \"$(readlink /proc/$c3_ready_pid/cwd)\" = "+runtime+" || continue; "
        "test \"$(readlink /proc/$c3_ready_pid/exe)\" = "+executable+" || continue; "
        "printf 'OWNED %s\\n' \"$c3_ready_pid\"; "
        "for c3_ready_inode in $(awk '$2 ~ /:"+port+"$/ && $4 == \"0A\" {print $10}' /proc/net/tcp); do "
        "for c3_ready_fd in /proc/$c3_ready_pid/fd/*; do "
        "test \"$(readlink \"$c3_ready_fd\")\" = \"socket:[$c3_ready_inode]\" || continue; "
        "printf 'LISTEN %s %s\\n' \"$c3_ready_pid\" \"$c3_ready_inode\"; done; done; done")
    while True:
        remaining = limit_seconds-(time.monotonic()-started)
        if remaining <= 0:
            evidence.update(status="OWNED_RPC_LISTEN_TIMEOUT",elapsed_seconds=time.monotonic()-started)
            raise TimeoutError("owned RPC did not reach LISTEN before readiness deadline")
        try:
            raw = board.ssh(command,timeout=remaining)
        except Exception as error:
            evidence.update(status="OWNED_RPC_READINESS_READ_FAILED",elapsed_seconds=time.monotonic()-started,error=str(error))
            raise
        lines = raw.strip().splitlines()
        if not lines or lines[0] != "BOOT "+boot_id:
            evidence.update(status="OWNED_RPC_BOOT_CHANGED",elapsed_seconds=time.monotonic()-started,observed_boot=lines[0] if lines else None)
            raise ValueError("Linux boot changed during owned RPC readiness")
        owned = [line.split()[1] for line in lines if line.startswith("OWNED ")]
        listening = [line.split()[1:] for line in lines if line.startswith("LISTEN ")]
        elapsed = time.monotonic()-started
        polls.append({"poll":len(polls)+1,"elapsed_seconds":elapsed,"owned_pids":owned,"listeners":listening})
        evidence["elapsed_seconds"] = elapsed
        if any(pid in owned for pid,inode in listening):
            evidence["status"] = "OWNED_RPC_LISTEN_READY"
            return evidence
        time.sleep(min(0.1,max(0,limit_seconds-elapsed)))


def _verify_inventory(inventory,functions):
    for name in ("vta.runtime.profiler_clear","vta.runtime.profiler_status",
                 "vta.runtime.queue_capacity_status","vta.runtime.replay_status"):
        if name not in functions:
            raise ValueError("required common runtime introspection function unavailable")
    capacity = inventory["queue_capacity_initial"]
    expected = {"insn_capacity_bytes":33554432,"uop_capacity_bytes":33554432,
        "insn_submit_threshold_bytes":33554432,"insn_submit_threshold_configured":True,
        "command_manifest_id":"","replay_policy":"disabled","submissions":0}
    if any(capacity.get(key) != value for key,value in expected.items()):
        raise ValueError("actual fresh runtime queue/replay initialization differs from the frozen protocol")
    replay = inventory["replay_initial"]
    if replay.get("mode") != "disabled" or replay.get("template_count") != 0 or inventory["profiler_initial"].get("driver_run_calls") != 0:
        raise ValueError("fresh runtime already contains replay or completed target work")


def _cleanup_owned(board,data,out,sessions,runtime_dir,root_name,append):
    """A prefix is not ownership; failed/partial deployment is retained."""
    if not data.get("owned_runtime_directory",False):
        data["cleanup_status"] = "SKIPPED_UNPROVEN_DIRECTORY_OWNERSHIP_RETAINED"
        append(out/"sessions.jsonl",data)
        sessions.append(data)
        return
    try:
        board.stop()
        data["rpc_log"] = board.ssh("test ! -f "+shlex.quote(runtime_dir+"/rpc.log")+
                                   " || cat "+shlex.quote(runtime_dir+"/rpc.log"))
        if not runtime_dir.startswith("/var/volatile/c3_tradeoff_"+root_name+"_"):
            raise ValueError("unvalidated owned cleanup target")
        target = shlex.quote(runtime_dir)
        board.ssh("test -z \"$(pidof tvm_rpc 2>/dev/null || true)\" && test -d "+target+
                  " && test ! -L "+target+" && rm -r "+target)
        data["cleanup_status"] = "OWNED_DIRECTORY_REMOVED_AFTER_LOG_RETENTION"
    except Exception as error:
        data["cleanup_error"] = str(error)
        append(out/"sessions.jsonl",data)
        sessions.append(data)
        raise
    append(out/"sessions.jsonl",data)
    sessions.append(data)


class Runner:
    def __init__(self,args):
        if not args.execute_board or args.phase not in ("qualify","diagnose","formal"):
            raise ValueError("board runner requires an explicit execution phase/flag")
        self.args,self.root = args,args.output_dir.resolve()
        self.prepared,self.protocol = read(self.root/"board_prepared.json"),read(self.root/"board_protocol.json")
        if self.prepared["schema"] != SCHEMA or self.protocol["schema"] != SCHEMA or self.prepared["protocol_sha256"] != digest(self.root/"board_protocol.json"):
            raise ValueError("invalid aggregation-specific board preparation")
        self.programs = tuple(self.protocol["programs"])
        self.runtime = read(self.protocol["runtime_contract_path"])
        self._base = base = _base()
        self.ensure_identity()
        base.reuse.RUNTIME = self.runtime["runtime_sha256"]
        proxy = SimpleNamespace(**vars(base.p))
        proxy.PROGRAMS,proxy.SCHEMA = self.programs,SCHEMA
        proxy.expectations = lambda name:self.prepared["programs"][name]["expectation"]
        proxy.verify_call_profile = self.verify_profile
        def clone(method,replacements=()):
            source = textwrap.dedent(inspect.getsource(method))
            for old,new in replacements:
                if source.count(old) != 1:
                    raise ValueError("sealed narrow method anchor changed")
                source = source.replace(old,new)
            namespace = dict(base.__dict__,p=proxy,connect=_connect_once,_verify_maps=_verify_maps,
                _verify_dma_state=_verify_dma_state,_wait_owned_rpc=_wait_owned_rpc,
                _verify_inventory=_verify_inventory,_cleanup_owned=_cleanup_owned)
            result = {}
            exec(compile(source,__file__+":"+method.__name__,"exec"),namespace,result)
            return MethodType(result[method.__name__],self)
        for name in ("start_phase","buffers","final_state"):
            setattr(self,name,clone(getattr(base.Runner,name)))
        cleanup_source = '''        try:
            board.stop()
            # Retain all logs before removing only this created runtime copy.
            data["rpc_log"] = board.ssh("test ! -f " + shlex.quote(runtime_dir + "/rpc.log") +
                                       " || cat " + shlex.quote(runtime_dir + "/rpc.log"))
            if not runtime_dir.startswith("/var/volatile/c3_tradeoff_" + self.root.name + "_"):
                raise ValueError("unvalidated cleanup target")
            target = shlex.quote(runtime_dir)
            board.ssh("test -z \\"$(pidof tvm_rpc 2>/dev/null || true)\\" && test -d " + target +
                      " && test ! -L " + target + " && rm -r " + target)
        except Exception as cleanup:
            data["cleanup_error"] = str(cleanup)
            append(self.out / "sessions.jsonl", data)
            self.sessions.append(data)
            raise
        append(self.out / "sessions.jsonl", data)
        self.sessions.append(data)
'''
        self._session = clone(base.Runner.session,((
            '"runtime_directory": runtime_dir, "start_state": state}',
            '"runtime_directory": runtime_dir, "start_state": state, "owned_runtime_directory": False}'),(
            "        lib = load_ephemeral(remote, Path(binary))",
            "        _verify_maps(data['loaded_maps'], runtime_dir)\n        lib = load_ephemeral(remote, Path(binary))"),(
            '        data["runtime_attestation"] = board.deploy(self.runtime["runtime_paths"])',
            '        _verify_dma_state(board,self.protocol["expected_udmabuf_phys"])\n        data["runtime_attestation"] = board.deploy(self.runtime["runtime_paths"])\n        data["owned_runtime_directory"] = True'),(
            '        board.start()',
            '        board.start()\n        _wait_owned_rpc(board,self.boot,evidence=data.setdefault("rpc_readiness",{}))'),(
            '        inventory, functions = runtime_inventory(remote)',
            '        inventory, functions = runtime_inventory(remote)\n        _verify_inventory(inventory,functions)'),(
            cleanup_source,'        _cleanup_owned(board,data,self.out,self.sessions,runtime_dir,self.root.name,append)\n')))
        self._raw_call = clone(base.Runner.call,(("row[\"model_features\"]","row[\"command_features\"]"),))
        self.libraries = {name:Path(self.prepared["programs"][name]["binary"]) for name in self.programs}
        self.totals = {name:self.prepared["programs"][name]["request_totals"] for name in self.programs}
        self.x = {s:base.np.load(base.OLD/"board_qualify03"/("X_%d.npy" % s)) for s in SEEDS}
        self.y = {s:base.np.load(base.OLD/"board_qualify03"/("Y24_%d.npy" % s)) for s in SEEDS}
        self.weight,self.weight_b,self.add,self.y_b = [base.np.load(base.OLD/"board_qualify03"/name) for name in ("W_A.npy","W_B.npy","ADD.npy","Y_B.npy")]
        if any(not base.same(base._reference(self.x[s],self.weight,self.add),self.y[s]) for s in SEEDS) or not base.same(base._reference(self.x[0],self.weight_b,self.add),self.y_b) or base.same(self.y[0],self.y_b):
            raise ValueError("full-fusion references or A/B distinguishing outputs failed")
        abis = []
        for name in self.programs:
            function = base.function(Path(self.prepared["programs"][name]["directory"]))
            abis.append([{"shape":[int(v) for v in function.buffer_map[p].shape],"dtype":str(function.buffer_map[p].dtype),
                          "strides":[str(v) for v in function.buffer_map[p].strides],"element_offset":str(function.buffer_map[p].elem_offset)} for p in function.params])
        if any(abi != abis[0] for abi in abis) or len(abis[0]) != 4:
            raise ValueError("complete four-buffer ABI differs")
        self.data_contract = {"abi_order":["X","W","ADD","Y"],"buffers":abis[0],
            "input_sha256":{str(s):base.array_sha(self.x[s]) for s in SEEDS},
            "reference_sha256":{str(s):base.array_sha(self.y[s]) for s in SEEDS},
            "weight_A_sha256":base.array_sha(self.weight),"weight_B_sha256":base.array_sha(self.weight_b),"add_sha256":base.array_sha(self.add)}
        probe = base.ExperimentBoard(args,"/var/volatile/c3_tradeoff_readonly_preflight",args.port)
        self.initial = probe.state()
        base.reuse.validate_state(self.initial,exclusive=True)
        # The unchanged diagnostic helper uses libc tmpfile(); verify /tmp is RAM.
        _verify_dma_state(probe,self.protocol["expected_udmabuf_phys"])
        self.boot = self.initial["boot_id"]
        if self.boot != self.protocol["required_boot_id"]:
            raise ValueError("Linux boot differs from the frozen restoration contract")
        self.sessions,self.rows,self.out = [],[],None
        self._cache_definitions = {}
        self._defer_record,self._pending_record = False,None

    def ensure_identity(self):
        if any(self.root.rglob("invalid_attempt.json")):
            raise ValueError("failed attempt retained; no automatic retry or sample splicing")
        for key in ("local_pool","arm_index","runtime_contract"):
            if digest(self.protocol[key+"_path"]) != self.protocol[key+"_sha256"]:
                raise ValueError("frozen prerequisite changed: "+key)
        if digest(self.protocol["document"]) != self.protocol["document_sha256"] or execution_sources() != self.protocol["source_sha256"]:
            raise ValueError("execution document/source identity changed")
        if hashlib.sha256(json.dumps(self.prepared["programs"],sort_keys=True).encode()).hexdigest() != self.protocol["program_records_sha256"]:
            raise ValueError("board-prepared records differ from the frozen protocol")
        for evidence in self.protocol["restoration_evidence"]:
            for record in evidence["artifacts"].values():
                if digest(record["path"]) != record["sha256"]:
                    raise ValueError("restoration contract or evidence changed")
        for name,record in self.prepared["programs"].items():
            if digest(record["binary"]) != record["binary_sha256"] or digest(Path(record["directory"])/"node24_final.tir.json") != record["tir_sha256"] or digest(record["expected_trace"]["path"]) != record["expected_trace"]["sha256"]:
                raise ValueError("frozen program or expected trace changed: "+name)
            if any(digest(path) != sha for path,sha in record["local_dump_sha256"].items()):
                raise ValueError("local native dump identity changed")
        if self._base.helper_certificate() != self.protocol["helpers"]:
            raise ValueError("frozen helper changed")
        for name,path in self.runtime["runtime_paths"].items():
            if digest(path) != self.runtime["runtime_sha256"][name]:
                raise ValueError("common runtime binary changed")
        for path,sha in self.protocol["runtime_source_bindings"].items():
            if digest(path) != sha:
                raise ValueError("common runtime/hardware source binding changed")

    def session(self,name,binary):
        self.ensure_identity()
        return self._session(name,binary)

    def record(self,row):
        row.setdefault("latency_ms",None)
        row.update(boot_id=self.boot,invocation_id=self.out.name+":%06d" % len(self.rows))
        if self._defer_record:
            if self._pending_record is not None:
                raise ValueError("more than one record for a single explicit invocation")
            row["status"],row["native_gate_status"] = "PENDING_FINAL_GATES","PENDING"
            self._pending_record = row
            return row
        return self._commit_record(row)

    def _commit_record(self,row):
        row.setdefault("status","PASS" if row.get("correct") and row.get("arguments_unchanged") else "FAIL")
        self._base.append(self.out/"calls.jsonl",row)
        self._base.append(self.out/"programs"/(row["program"]+".jsonl"),row)
        self.rows.append(row)
        return row

    def verify_profile(self,name,profile,signature):
        exp = self.prepared["programs"][name]["expectation"]
        if any(signature[key] != exp[key] for key in ("mac","reset","wgt_loads","wgt_payload_bytes")) or profile.get("push_gemm_op_calls") != exp["total_gemm_push"] or profile.get("driver_timeout_calls",0):
            raise ValueError("actual program commands or driver result differ")
        return {key:signature[key] for key in ("mac","reset","uop_loads","uop_payload_bytes","instructions","submissions")}

    def call(self,*args,**kwargs):
        if self._defer_record or self._pending_record is not None:
            raise ValueError("nested/unfinished invocation is outside the fixed protocol")
        self._defer_record = True
        try:
            row = self._raw_call(*args,**kwargs)
            if row.get("dump_file"):
                from c3_valid_window_uop_values import certify_uop_values
                record = self.prepared["programs"][row["program"]]
                text = (self.out/row["dump_file"]).read_text()
                key = (row["session_id"],row["program"])
                certificate = certify_uop_values(text,expected_kernels(record),self._cache_definitions.get(key))
                if not certificate["full_cache_contract_bound_across_calls"]:
                    raise ValueError("strict v2 same-session cache definitions required")
                self._cache_definitions.setdefault(key,certificate["cache_definitions"])
                first = row["call_position"] == 1
                expected = record["command_fingerprints"]["first" if first else "warmed"]
                actual = command_fingerprint(text)
                matches = actual == expected if first else actual in expected
                if not matches:
                    raise ValueError("complete normalized actual command description differs from frozen local program")
                self._base.append(self.out/"native_certificates.jsonl",{"invocation_id":row["invocation_id"],
                    "program":row["program"],"session_id":row["session_id"],"dump_sha256":row["dump_sha256"],
                    "command_fingerprint":actual,"uop_certificate":certificate,"board_contacted":True})
                row["native_gate_status"] = "ACTUAL_UOP_AND_COMMAND_FIELDS_PASSED"
            else:
                row["native_gate_status"] = "PROFILE_AND_OUTPUT_PASSED_NO_DUMP"
            row["status"] = "PASS"
            return self._commit_record(row)
        except BaseException as error:
            if self._pending_record is not None:
                row = self._pending_record
                row.update(status="FAIL_NATIVE_OR_EXECUTION",native_gate_status="FAIL",error=str(error))
                self._commit_record(row)
            raise
        finally:
            self._pending_record,self._defer_record = None,False

    def prerequisite(self,phase,status):
        summary = read(self.root/phase/"summary.json")
        if summary["schema"] != SCHEMA or summary["status"] != status or summary["boot_id"] != self.boot or summary["prepared_sha256"] != digest(self.root/"board_prepared.json") or summary["protocol_sha256"] != digest(self.root/"board_protocol.json"):
            raise ValueError("stale/incomplete prerequisite phase: "+phase)
        return summary

    def begin(self,phase):
        self.start_phase(phase)
        write_new(self.out/"contract.json",{"schema":SCHEMA,"boot_id":self.boot,"protocol":self.protocol,
            "prepared_sha256":digest(self.root/"board_prepared.json"),"data_contract":self.data_contract,
            "sources":execution_sources(),"helpers":self.protocol["helpers"]})

    def end(self,status,expected):
        if Counter(row["call_kind"] for row in self.rows) != expected:
            raise ValueError("phase target-call ledger mismatch")
        state = self.final_state()
        self.ensure_identity()
        write_new(self.out/"summary.json",{"schema":SCHEMA,"status":status,"boot_id":self.boot,
            "call_counts":dict(expected),"target_calls":len(self.rows),"all_correct":True,"final_state":state,
            "prepared_sha256":digest(self.root/"board_prepared.json"),"protocol_sha256":digest(self.root/"board_protocol.json")})

    def qualify(self):
        self.begin("qualification")
        base = self._base
        for name in self.programs:
            with self.session(name,self.libraries[name]) as (remote,lib,funcs,refs,sid):
                refs.append(base.load_ephemeral(remote,base.TIMER))
                timer,reset,counts = [remote.get_function(n) for n in ("c3.node34.fused_timed_once","c3.node34.reset_counts","c3.node34.counts")]
                reset()
                buffers = self.buffers(remote,SEEDS[0])
                for position,seed in enumerate(SEEDS,1):
                    row = self.call(lib,funcs,name,buffers,seed,"correctness",sid,timer=timer,
                        call_position=position,first_cache_build=position==1,
                        cost_scope="normal complete qualification call plus raw profiler; cold/hot difference is not isolated initializer cost")
                    if int(counts()) != position:
                        raise ValueError("unexpected qualification timer target invocation count")
                    base.append(self.out/"first_cache_costs.jsonl",{"invocation_id":row["invocation_id"],
                        "program":name,"call_position":position,"first_cache_build":position==1,
                        "complete_call_ms":row["latency_ms"],
                        "raw_push_gemm_op_us":row["runtime_profile"].get("push_gemm_op_us"),
                        "raw_push_alu_op_us":row["runtime_profile"].get("push_alu_op_us"),
                        "not_isolated_initializer_cost":True,"excluded_from_formal_samples":True})
                if name != "TPS":
                    handles = base.buffer_handles(buffers)
                    for label,weight,reference in (("A",self.weight,self.y[0]),("B",self.weight_b,self.y_b),("A_return",self.weight,self.y[0])):
                        self.call(lib,funcs,name,buffers,0,"weight_ABA",sid,weight=weight,expected=reference,aba_label=label)
                        if base.buffer_handles(buffers) != handles:
                            raise ValueError("same-module A/B/A changed parameter handles")
        self.end("FIXED_POOL_FPGA_QUALIFIED",{"correctness":3*len(self.programs),"weight_ABA":3*(len(self.programs)-1)})

    def diagnose(self):
        self.prerequisite("qualification","FIXED_POOL_FPGA_QUALIFIED")
        self.begin("diagnostic")
        signatures = {}
        base = self._base
        for r,order in enumerate(self.protocol["orders"][:3]):
            seeds = self.protocol["seed_orders"][r]
            signatures[str(r)] = {}
            for name in order:
                with self.session(name,self.libraries[name]) as (remote,lib,funcs,refs,sid):
                    refs.append(base.load_ephemeral(remote,base.DIAGNOSTIC))
                    diag = remote.get_function("c3.node24.fused_diagnostic_once")
                    buffers = self.buffers(remote,seeds[0])
                    for pos,seed in enumerate(seeds):
                        self.call(lib,funcs,name,buffers,seed,"diagnostic_prefix",sid,
                            diagnostic=diag if pos == 0 else None,round=r+1,call_position=pos+1)
                    signatures[str(r)][name] = []
                    for pos,seed in enumerate(seeds):
                        row = self.call(lib,funcs,name,buffers,seed,"warmed_diagnostic",sid,
                            diagnostic=diag,round=r+1,call_position=pos+4)
                        signatures[str(r)][name].append(row["command_signature"])
        write_new(self.out/"diagnostic_signatures.json",signatures)
        count = self.protocol["expected_calls"]["diagnostic_prefix"]
        self.end("ALL_ROUND_PREFIX_AND_UOP_DIAGNOSTICS_QUALIFIED",{"diagnostic_prefix":count,"warmed_diagnostic":count})

    def formal(self):
        self.prerequisite("qualification","FIXED_POOL_FPGA_QUALIFIED")
        self.prerequisite("diagnostic","ALL_ROUND_PREFIX_AND_UOP_DIAGNOSTICS_QUALIFIED")
        signatures = read(self.root/"diagnostic/diagnostic_signatures.json")
        self.begin("board")
        base = self._base
        for r,order in enumerate(self.protocol["orders"]):
            seeds = self.protocol["seed_orders"][r]
            for name in order:
                with self.session(name,self.libraries[name]) as (remote,lib,funcs,refs,sid):
                    refs.append(base.load_ephemeral(remote,base.TIMER))
                    timer,reset,counts = [remote.get_function(n) for n in ("c3.node34.fused_timed_once","c3.node34.reset_counts","c3.node34.counts")]
                    reset()
                    buffers = self.buffers(remote,seeds[0])
                    for pos,seed in enumerate(seeds):
                        self.call(lib,funcs,name,buffers,seed,"warmup",sid,round=r+1,seed_position=pos,call_position=pos+1)
                    for pos,seed in enumerate(seeds):
                        row = self.call(lib,funcs,name,buffers,seed,"timing",sid,timer=timer,
                        signature=signatures[str(r%3)][name][pos],round=r+1,seed_position=pos,call_position=pos+4)
                        if int(counts()) != pos+1:
                            raise ValueError("unexpected extra or missing timed target invocation")
                        base.append(self.out/"helper_counts.jsonl",{"invocation_id":row["invocation_id"],"count":pos+1})
        count = self.protocol["expected_calls"]["formal_timing"]
        self.end("FORMAL_FIXED_POOL_ACQUISITION_COMPLETE",{"warmup":count,"timing":count})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("plan","freeze","qualify","diagnose","formal"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--pool",type=Path)
    parser.add_argument("--arm-index",type=Path)
    parser.add_argument("--runtime-contract",type=Path)
    parser.add_argument("--document",type=Path)
    parser.add_argument("--host",default="192.168.1.247")
    parser.add_argument("--port",type=int,default=9090)
    parser.add_argument("--timeout",type=int,default=900)
    parser.add_argument("--known-hosts",default="/home/orange/.ssh/known_hosts")
    parser.add_argument("--execute-board",action="store_true")
    args = parser.parse_args()
    if args.phase == "plan":
        write_new(args.output_dir/"board_protocol_design.json",plan())
    elif args.phase == "freeze":
        if any(v is None for v in (args.pool,args.arm_index,args.runtime_contract,args.document)):
            parser.error("freeze requires --pool --arm-index --runtime-contract --document")
        freeze(args.output_dir,args.pool,args.arm_index,args.runtime_contract,args.document)
    else:
        if not args.execute_board:
            parser.error("acquisition requires --execute-board after local freeze and execution authorization")
        runner = None
        try:
            runner = Runner(args)
            getattr(runner,args.phase)()
        except BaseException as error:
            destination = runner.out if runner is not None and runner.out is not None else args.output_dir
            write_new(destination/"invalid_attempt.json",{"schema":SCHEMA,"status":"INVALID_STOP_ALL_BOARD_WORK_NO_RETRY_OR_SPLICE",
                "phase":args.phase,"error":str(error),"traceback":traceback.format_exc(),"all_timing_invalid":True,
                "completed_calls":len(runner.rows) if runner else 0})
            raise


if __name__ == "__main__":
    main()
