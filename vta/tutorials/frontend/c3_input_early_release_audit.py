"""Read-only, fail-closed certificate for the frozen ResNet A input lease.

This is deliberately an identity-qualified audit, not a generic CPU-kernel
classifier.  It never loads an ARM module, executes a graph, or contacts a board.
Physical pointer/range, loaded-library and actor-thread checks remain board gates.
"""

import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[3]
TARGET = "tvmgen_default_fused_reshape_transpose_multiply_transpose_reshape_round_clip"
PACK = "tvmgen_default_fused_cast_reshape_transpose"
B = 802816
STAGE1_SHA256 = "692b3af8856cbd96e959de6e3aaaf25e12aa0773ff6ce9c53ae66d5febc6fc1b"
TVM_SHA256 = "04e894baf305311315fd0c79d03ec2a6375b9591916e9c6f429cf3697c4457b2"
VTA_SHA256 = "e9a67cd4365597aba03db0b564045524d4edb891d12a04cf7952d5aedd493226"
POOL_LAUNCH = "_ZN3tvm7runtime10ThreadPool6LaunchEPFiiP19TVMParallelGroupEnvPvES4_ii"
SDK_OBJDUMP = Path("/home/orange/alinx_plsdk/install/sysroots/x86_64-petalinux-linux/usr/bin/"
                   "aarch64-xilinx-linux/aarch64-xilinx-linux-objdump")
INSTRUCTION = re.compile(r"^[ \t]*([0-9a-f]+):[ \t]+([0-9a-f]{8})[ \t]+(\S+)[ \t]*([^\r\n]*)$", re.M)


class AuditError(ValueError):
    """Necessary static or identity evidence did not establish this certificate."""


def require(condition, reason):
    if not condition:
        raise AuditError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
        return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def entry_id(graph, node, output=0):
    row = graph.get("node_row_ptr", [])
    nodes = graph.get("nodes", [])
    require(len(row) == len(nodes) + 1 and row[0] == 0, "ENTRY_ROW_PTR_INVALID")
    require(all(type(x) is int for x in row) and all(a < b for a, b in zip(row, row[1:])),
            "ENTRY_ROW_PTR_INVALID")
    require(type(node) is int and 0 <= node < len(nodes) and type(output) is int and output >= 0,
            "ENTRY_INDEX_INVALID")
    require(output < row[node + 1] - row[node], "ENTRY_OUTPUT_INVALID")
    return row[node] + output


def audit_graph(graph):
    """Check all graph consumers and storage aliases, not just the next node."""
    nodes = graph.get("nodes", [])
    require(len(nodes) >= 4, "GRAPH_TOO_SMALL")
    count = graph.get("node_row_ptr", [0])[-1]
    arrays = {}
    for key in ("shape", "dltype", "storage_id", "device_index"):
        value = graph.get("attrs", {}).get(key)
        require(isinstance(value, list) and len(value) == 2 and isinstance(value[1], list)
                and len(value[1]) == count, "GRAPH_ATTRIBUTE_INVALID:" + key)
        arrays[key] = value[1]
    ids = [entry_id(graph, node) for node in (0, 1, 2)]
    require(all(graph["node_row_ptr"][n + 1] - graph["node_row_ptr"][n] == 1 for n in (0, 1, 2)),
            "TARGET_MULTI_OUTPUT_UNSUPPORTED")
    require(nodes[0].get("op") == "null" and nodes[0].get("name") == "data0"
            and nodes[0].get("inputs") == [] and 0 in graph.get("arg_nodes", []),
            "ENTRY_NOT_PLAIN_DATA_ARGUMENT")
    for node, symbol in ((1, TARGET), (2, PACK)):
        attrs = nodes[node].get("attrs", {})
        require(nodes[node].get("op") == "tvm_op" and attrs.get("func_name") == symbol,
                "FUNCTION_IDENTITY_INVALID")
        require(attrs.get("num_inputs") == "1" and attrs.get("num_outputs") == "1"
                and attrs.get("flatten_data") == "0", "TARGET_ABI_INVALID")
        require(nodes[node].get("inputs") == [[node - 1, 0, 0]], "TARGET_INPUT_CHAIN_INVALID")
    require(sum(n.get("attrs", {}).get("func_name") == TARGET for n in nodes) == 1,
            "TARGET_SYMBOL_HAS_MULTIPLE_CALLS")
    consumers = []
    for number, node in enumerate(nodes):
        for port, operand in enumerate(node.get("inputs", [])):
            require(isinstance(operand, list) and len(operand) == 3 and operand[2] == 0,
                    "UNKNOWN_GRAPH_EDGE")
            identifier = entry_id(graph, operand[0], operand[1])
            if identifier == ids[0]:
                consumers.append({"node": number, "port": port})
    require(consumers == [{"node": 1, "port": 0}], "ENTRY_HAS_OTHER_READERS")
    heads = set()
    for head in graph.get("heads", []):
        require(isinstance(head, list) and len(head) == 3 and head[2] == 0, "UNKNOWN_GRAPH_HEAD")
        heads.add(entry_id(graph, head[0], head[1]))
    require(heads and not heads.intersection(ids), "ENTRY_OR_CONVERSION_IS_GRAPH_HEAD")
    sids = [arrays["storage_id"][identifier] for identifier in ids]
    require(sids[0] == 0 and len(set(sids)) == 3, "CONVERSION_STORAGE_ALIAS")
    require([i for i, sid in enumerate(arrays["storage_id"]) if sid == sids[0]] == [ids[0]],
            "ENTRY_STORAGE_HAS_ALIASES")
    expected = (([1, 64, 56, 56], "float32"), ([1, 64, 56, 56], "float32"),
                ([1, 4, 56, 56, 1, 16], "int8"))
    entries = []
    for node, identifier, (shape, dtype) in zip((0, 1, 2), ids, expected):
        require(arrays["shape"][identifier] == shape and arrays["dltype"][identifier] == dtype,
                "CONVERSION_TENSOR_CONTRACT_INVALID")
        require(arrays["device_index"][identifier] == 12, "STORAGE_DEVICE_CONTRACT_INVALID")
        entries.append({"node": node, "entry": identifier, "storage_id": arrays["storage_id"][identifier],
                        "shape": shape, "dtype": dtype, "device_type": 12})
    return {"status": "FROZEN_GRAPH_LAST_READER_PASSED", "entries": entries,
            "boundary_bytes": B, "last_reader_node": 1, "last_reader_symbol": TARGET,
            "target_call_count_per_stage_run": 1, "consumers": consumers,
            "node1_output_is_packed": False,
            "release_boundary": "AFTER_SUCCESSFUL_SYNCHRONOUS_CPU_FUNCTION_RETURN",
            "abi": {"argument_count": 2, "order": ["data0", "node1_output"],
                    "strides": "compact; exact actual DLTensor values require runtime checking",
                    "byte_offset": 0, "alias_requirement": "input/output physical ranges disjoint"}}


def verify_package(package, expected):
    package = Path(package).resolve()
    required = {"manifest.json", "libtvm_runtime.so", "libvta.so"}
    required.update(f"stages/stage{i}/{name}" for i in range(3)
                    for name in ("graph.json", "graphlib.so", "params.params"))
    require(isinstance(expected, dict) and required.issubset(expected), "INCOMPLETE_A_PACKAGE_MANIFEST")
    observed = {}
    for name, frozen in sorted(expected.items()):
        path = (package / name).resolve()
        require(path.is_relative_to(package) and path.is_file(), "PACKAGE_PATH_INVALID:" + name)
        require(isinstance(frozen, str) and re.fullmatch("[0-9a-f]{64}", frozen), "PACKAGE_SHA_INVALID")
        observed[name] = sha(path)
        require(observed[name] == frozen, "PACKAGE_SHA_MISMATCH:" + name)
    require(observed["stages/stage1/graphlib.so"] == STAGE1_SHA256, "UNREVIEWED_STAGE1_BINARY")
    require(observed["libtvm_runtime.so"] == TVM_SHA256, "UNREVIEWED_PACKAGED_TVM_RUNTIME")
    return observed


def _arm_elf(path):
    with Path(path).open("rb") as stream:
        header = stream.read(64)
    require(len(header) == 64 and header[:6] == b"\x7fELF\x02\x01", "EXPECTED_ELF64_LITTLE_ENDIAN")
    require(struct.unpack_from("<H", header, 18)[0] == 183, "EXPECTED_AARCH64_ELF")


def instructions(text):
    return [{"address": int(a, 16), "word": w, "mnemonic": m, "operands": o}
            for a, w, m, o in INSTRUCTION.findall(text)]


def direct_calls(text):
    return [re.search(r"<([^>]+)>", row["operands"]).group(1).split("@", 1)[0]
            for row in instructions(text) if row["mnemonic"] == "bl"
            and re.search(r"<([^>]+)>", row["operands"])]


def certify_wait_loop(text):
    """Recognize the reviewed inlined WaitForJobs acquire-load/Yield loop."""
    rows = instructions(text)
    by_address = {row["address"]: row for row in rows}
    for first, second in zip(rows, rows[1:]):
        if first["mnemonic"] != "ldar" or second["mnemonic"] != "cbnz":
            continue
        match = re.fullmatch(r"(w\d+),\s*([0-9a-f]+)\s+.*", second["operands"])
        if not match or first["operands"].split(",", 1)[0] != match.group(1):
            continue
        target = int(match.group(2), 16)
        start = by_address.get(target, {})
        if target < first["address"] and start.get("mnemonic") == "bl" and "Yield" in start.get("operands", ""):
            require(any(r["mnemonic"] == "ret" and r["address"] > second["address"] for r in rows),
                    "WAIT_LOOP_HAS_NO_REVIEWED_RETURN")
            return {"status": "REVIEWED_PENDING_JOBS_WAIT_LOOP", "load_address": first["address"],
                    "backedge_address": second["address"], "yield_address": target,
                    "meaning": "acquire-load pending jobs; nonzero loops through Yield before return"}
    raise AuditError("SYNCHRONOUS_WAIT_LOOP_NOT_ESTABLISHED")


def certify_cpu_path(wrapper, compute, leaf, relocations):
    """Structural checks only; callers must first enforce the exact ELF identity."""
    require(direct_calls(wrapper) == ["VTATLSCommandHandle", TARGET + "_compute_"],
            "TARGET_WRAPPER_CALL_GRAPH_CHANGED")
    require(direct_calls(compute) == ["VTABufferCPUPtr", "VTABufferCPUPtr"],
            "TARGET_COMPUTE_CALL_GRAPH_CHANGED")
    compute_rows = instructions(compute)
    require(sum(r["mnemonic"] == "blr" for r in compute_rows) == 1
            and "__TVMBackendParallelLaunch" in relocations,
            "PARALLEL_LAUNCH_BINDING_UNRESOLVED")
    require("#0xd90" in compute and "15000" in compute, "PARALLEL_LAMBDA_ADDRESS_CHANGED")
    leaf_rows = instructions(leaf)
    require(leaf_rows and leaf_rows[0]["address"] == 0x15D90 and leaf_rows[-1]["address"] == 0x15F64
            and leaf_rows[-1]["mnemonic"] == "ret", "PARALLEL_LAMBDA_CLOSURE_CHANGED")
    require(not any(r["mnemonic"] in ("bl", "blr", "br", "svc", "hvc", "smc") for r in leaf_rows),
            "PARALLEL_LAMBDA_HAS_UNKNOWN_CALL_OR_EFFECT")
    forbidden = ("VTALoadBuffer2D", "VTAStoreBuffer2D", "VTAPushGEMMOp", "VTAPushALUOp",
                 "VTASynchronize", "VTADeviceRun")
    for label, text in (("wrapper", wrapper), ("compute", compute), ("lambda", leaf)):
        require(not any(name in text for name in forbidden), "CPU_PATH_HAS_DEVICE_COMMAND:" + label)
    return {"status": "REVIEWED_EXACT_ELF_CPU_PATH", "lambda_start": 0x15D90,
            "lambda_end_exclusive": 0x15F68, "compute_cpu_pointer_resolutions": 2,
            "compute_parallel_launches": 1,
            "qualification": "only valid together with the frozen stage1 ELF hash"}


def _source_excerpt(path, needle, before=0, after=12):
    lines = Path(path).read_text().splitlines()
    positions = [i for i, line in enumerate(lines) if needle in line]
    require(len(positions) == 1, "SOURCE_EXPLANATION_UNRESOLVED:" + needle)
    start, end = max(0, positions[0] - before), min(len(lines), positions[0] + after + 1)
    return {"path": str(path), "sha256": sha(path), "first_line": start + 1,
            "text": "\n".join(lines[start:end]),
            "scope": "current explanatory source; frozen ELF, not recompilation, establishes binary identity"}


def native_certificate(stage1, tvm_runtime, vta_runtime, *, objdump=None, source_root=ROOT):
    stage1, tvm_runtime, vta_runtime = map(Path, (stage1, tvm_runtime, vta_runtime))
    for path in (stage1, tvm_runtime, vta_runtime):
        _arm_elf(path)
    require(sha(stage1) == STAGE1_SHA256, "UNREVIEWED_STAGE1_BINARY")
    require(sha(tvm_runtime) == TVM_SHA256, "UNREVIEWED_ACTUAL_TVM_RUNTIME")
    require(sha(vta_runtime) == VTA_SHA256, "UNREVIEWED_ACTUAL_VTA_RUNTIME")
    tool = Path(objdump) if objdump else SDK_OBJDUMP
    if not tool.is_file() and objdump is None:
        fallback = shutil.which("aarch64-linux-gnu-objdump")
        require(fallback is not None, "AARCH64_OBJDUMP_UNAVAILABLE")
        tool = Path(fallback)
    evidence = {}

    def run(key, path, *options):
        command = [str(tool), *options, str(path)]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
        require(completed.returncode == 0, "OBJDUMP_FAILED:" + key)
        evidence[key] = {"command": command, "text": completed.stdout,
                         "text_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest()}
        return completed.stdout

    symbols = run("graph_symbols", stage1, "-t")
    require("__tvm_lookup_linked_param" not in symbols, "LINKED_PARAMETER_MODULE_UNSUPPORTED")
    wrapper = run("target_wrapper", stage1, "-d", "--disassemble=" + TARGET)
    compute = run("target_compute", stage1, "-d", "--disassemble=" + TARGET + "_compute_")
    # This unnamed lambda's bounds are reviewed for the exact allowlisted ELF.
    # They are not a general search heuristic for another compiler output.
    leaf = run("target_parallel_lambda", stage1, "-d", "--start-address=0x15d90", "--stop-address=0x15f68")
    parallel = run("backend_parallel_launch", tvm_runtime, "-d", "--disassemble=TVMBackendParallelLaunch")
    pool = run("thread_pool_launch", tvm_runtime, "-d", "--disassemble=" + POOL_LAUNCH)
    cpu_ptr = run("buffer_cpu_ptr", vta_runtime, "-d", "--disassemble=VTABufferCPUPtr")
    relocations = run("graph_relocations", stage1, "-R")
    cpu_path = certify_cpu_path(wrapper, compute, leaf, relocations)
    require(not any(name in cpu_ptr for name in ("VTALoadBuffer2D", "VTAStoreBuffer2D", "VTAPushGEMMOp",
                                                "VTAPushALUOp", "VTASynchronize", "VTADeviceRun")),
            "CPU_POINTER_RESOLUTION_HAS_DEVICE_COMMAND")
    require(any(POOL_LAUNCH in symbol for symbol in direct_calls(parallel)),
            "BACKEND_PARALLEL_POOL_CALL_UNRESOLVED")
    wait = certify_wait_loop(pool)
    data = stage1.read_bytes()
    require(b"num_args == 2" in data and b"p0.dtype is expected to be float32" in data
            and b"clipB.dtype is expected to be float32" in data,
            "COMPILED_TARGET_ABI_ASSERTIONS_MISSING")
    require(b"byte_offset has an unsatisfied constraint: T.uint64(0)" in data,
            "COMPILED_ZERO_BYTE_OFFSET_ASSERTION_MISSING")
    source_root = Path(source_root)
    thread_source = source_root / "src/runtime/thread_pool.cc"
    runtime_source = source_root / "vta/runtime/runtime.cc"
    return {"status": "EXACT_FROZEN_ARM_CPU_RETURN_CERTIFIED", "linked_parameters": False,
            "tool": {"path": str(tool), "sha256": sha(tool)}, "evidence": evidence,
            "cpu_path": cpu_path, "wait_for_jobs": wait,
            "source_explanations": [_source_excerpt(thread_source, "int WaitForJobs()", after=14),
                                    _source_excerpt(thread_source, "int res = launcher->WaitForJobs();", before=2, after=2),
                                    _source_excerpt(runtime_source, "void* VTABufferCPUPtr(", after=8)],
            "pointer_abi": "ext_dev DLTensor.data is a DataBuffer handle, not a CPU pointer; "
                           "VTABufferCPUPtr resolves registered handles and returns raw CPU pointers unchanged. "
                           "Qualification must use the frozen runtime CPU-view API, never cast a handle.",
            "indirect_call_scope": "only the exact reviewed ELF is accepted; device selection/error paths "
                                   "and the ParallelLaunch function-pointer binding are not generalized to unknown ELFs",
            "identities": {"stage1_sha256": sha(stage1), "actual_tvm_sha256": sha(tvm_runtime),
                           "actual_vta_sha256": sha(vta_runtime)}}


def audit(package, a_json, tvm_runtime, vta_runtime, *, objdump=None, source_root=ROOT):
    """Return a local certificate; raise AuditError for unknown/failed evidence.

    ``a_json`` accepts the saved A.json path or its decoded dictionary.  Original
    package files, including its old libvta, are verified but never overwritten.
    The separate C2 experimental driver is verified independently.
    """
    package, source_root = Path(package), Path(source_root)
    if isinstance(a_json, dict):
        original = a_json
        a_identity = {"content_sha256": digest(original)}
    else:
        a_path = Path(a_json)
        original = json.loads(a_path.read_text())
        a_identity = {"path": str(a_path), "sha256": sha(a_path), "content_sha256": digest(original)}
    require(isinstance(original, dict), "A_AUDIT_RECORD_INVALID")
    package_hashes = verify_package(package, original.get("package_sha256"))
    graph_path = package / "stages/stage1/graph.json"
    graph = audit_graph(json.loads(graph_path.read_text()))
    native = native_certificate(package / "stages/stage1/graphlib.so", tvm_runtime, vta_runtime,
                                objdump=objdump, source_root=source_root)
    build_path = source_root / "vta/tutorials/frontend/report_out/stage_tile_cotuning/c3s_buffer_reuse/build_s1_run2/build_manifest.json"
    record_path = build_path.parent.parent / "s1_board_run1/preregistered.json"
    require(build_path.is_file() and record_path.is_file(), "FROZEN_DRIVER_BUILD_EVIDENCE_MISSING")
    build, historical = json.loads(build_path.read_text()), json.loads(record_path.read_text())
    require(build.get("artifacts", {}).get("libvta.so") == VTA_SHA256
            and historical.get("build", {}).get("artifacts", {}).get("libvta.so") == VTA_SHA256,
            "FROZEN_DRIVER_BUILD_IDENTITY_INVALID")
    certificate = {
        "schema": "c3-input-early-release-local-certificate-v1",
        "status": "LOCAL_STATIC_GATES_PASSED_BOARD_QUALIFICATION_REQUIRED",
        "passed": True, "local_static_passed": True, "board_qualified": False,
        "audit_source": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "scope": "frozen ResNet A stage1 input lease; no new graph compilation or board invocation",
        "a_audit_identity": a_identity, "package": str(package.resolve()),
        "original_package_sha256": package_hashes, "graph": graph, "native": native,
        "frozen_driver_build": {"path": str(build_path), "sha256": sha(build_path),
                                "source_sha256": build.get("source_sha256", {}),
                                "historical_record": str(record_path), "historical_record_sha256": sha(record_path),
                                "scope": "saved build-source identities, not claims that current source matches old build"},
        "dynamic_qualification": {"status": "PENDING", "passed": False,
                                  "required": ["loaded runtime/helper/module identities", "actual DLTensor ABI and byte offsets",
                                               "sid0/sid1/sid2 virtual AND physical nonoverlap and ext_dev owner lifetime",
                                               "same stage actor calls target exactly once and releases after successful return",
                                               "generation/exactly-once lease state and immutable bound arguments",
                                               "all actual worker CPU masks and output correctness"]},
        "board_calls": 0,
    }
    certificate["certificate_sha256"] = digest(certificate)
    return certificate
