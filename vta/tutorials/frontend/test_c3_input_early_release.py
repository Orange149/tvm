"""Local-only last-reader certificate tests; no board or graph execution."""

import copy
from pathlib import Path

import pytest

import c3_input_early_release_audit as audit


def graph_fixture():
    return {"nodes": [
        {"op": "null", "name": "data0", "inputs": []},
        {"op": "tvm_op", "name": audit.TARGET,
         "attrs": {"func_name": audit.TARGET, "num_inputs": "1", "num_outputs": "1", "flatten_data": "0"},
         "inputs": [[0, 0, 0]]},
        {"op": "tvm_op", "name": audit.PACK,
         "attrs": {"func_name": audit.PACK, "num_inputs": "1", "num_outputs": "1", "flatten_data": "0"},
         "inputs": [[1, 0, 0]]},
        {"op": "tvm_op", "name": "output", "attrs": {"func_name": "output"}, "inputs": [[2, 0, 0]]},
    ], "node_row_ptr": [0, 1, 2, 3, 4], "arg_nodes": [0], "heads": [[3, 0, 0]],
        "attrs": {"shape": ["list_shape", [[1, 64, 56, 56], [1, 64, 56, 56], [1, 4, 56, 56, 1, 16], [1]]],
                  "dltype": ["list_str", ["float32", "float32", "int8", "int8"]],
                  "storage_id": ["list_int", [0, 1, 2, 3]],
                  "device_index": ["list_int", [12, 12, 12, 12]]}}


def test_positive_last_reader_distinguishes_float_intermediate_and_pack():
    result = audit.audit_graph(graph_fixture())
    assert result["last_reader_node"] == 1
    assert result["boundary_bytes"] == 802816
    assert result["node1_output_is_packed"] is False
    assert [row["storage_id"] for row in result["entries"]] == [0, 1, 2]
    assert result["abi"]["argument_count"] == 2


@pytest.mark.parametrize("change,reason", [
    (lambda g: g["nodes"][3]["inputs"].append([0, 0, 0]), "OTHER_READERS"),
    (lambda g: g["attrs"]["storage_id"][1].__setitem__(1, 0), "STORAGE_ALIAS"),
    (lambda g: g["attrs"]["storage_id"][1].__setitem__(3, 0), "STORAGE_HAS_ALIASES"),
    (lambda g: g["heads"].append([0, 0, 0]), "GRAPH_HEAD"),
    (lambda g: g["heads"].__setitem__(0, [1, 0, 0]), "GRAPH_HEAD"),
    (lambda g: g["nodes"][1]["attrs"].__setitem__("num_inputs", "2"), "ABI_INVALID"),
    (lambda g: g["nodes"][1]["attrs"].__setitem__("flatten_data", "1"), "ABI_INVALID"),
    (lambda g: g["attrs"]["dltype"][1].__setitem__(1, "int8"), "TENSOR_CONTRACT"),
    (lambda g: g["attrs"]["device_index"][1].__setitem__(0, 1), "DEVICE_CONTRACT"),
    (lambda g: g["nodes"][3]["attrs"].__setitem__("func_name", audit.TARGET), "MULTIPLE_CALLS"),
    (lambda g: g["nodes"][2]["inputs"].__setitem__(0, [0, 0, 0]), "INPUT_CHAIN"),
    (lambda g: g["nodes"][3]["inputs"].__setitem__(0, [2, 0, 1]), "UNKNOWN_GRAPH_EDGE"),
])
def test_graph_refusals(change, reason):
    graph = graph_fixture()
    change(graph)
    with pytest.raises(audit.AuditError, match=reason):
        audit.audit_graph(graph)


def test_entry_ids_are_computed_not_node_numbers():
    graph = {"nodes": [{}, {}, {}], "node_row_ptr": [0, 2, 3, 5]}
    assert audit.entry_id(graph, 1) == 2
    assert audit.entry_id(graph, 2, 1) == 4
    with pytest.raises(audit.AuditError, match="OUTPUT_INVALID"):
        audit.entry_id(graph, 1, 1)


def test_unknown_row_pointer_is_refused():
    graph = graph_fixture()
    graph["node_row_ptr"][2] = 0
    with pytest.raises(audit.AuditError, match="ROW_PTR_INVALID"):
        audit.audit_graph(graph)


WAIT = """
 e74b8: 97fdbc0a bl 564e0 <threading5YieldEv@plt>
 e74bc: 91009001 add x1, x0, #0x24
 e74d4: 88dffc21 ldar w1, [x1]
 e74d8: 35ffff01 cbnz w1, e74b8 <ThreadPoolLaunch+0x280>
 e7788: d65f03c0 ret
"""


def test_exact_wait_loop_accepts_acquire_load_backedge():
    proof = audit.certify_wait_loop(WAIT)
    assert proof["yield_address"] == 0xE74B8


@pytest.mark.parametrize("text", [WAIT.replace("ldar w1", "ldr w1"),
                                 WAIT.replace("cbnz w1", "cbnz w2"),
                                 WAIT.replace("e74b8 <ThreadPool", "e7788 <ThreadPool"),
                                 WAIT.replace("threading5YieldEv", "OtherOperation"),
                                 WAIT.replace("e7788: d65f03c0 ret", "")])
def test_wait_loop_unknown_or_no_completion_is_refused(text):
    with pytest.raises(audit.AuditError):
        audit.certify_wait_loop(text)


WRAPPER = f"""
 a010: 97ffdc60 bl 1190 <VTATLSCommandHandle@plt>
 a218: 94002ec7 bl 15d34 <{audit.TARGET}_compute_>
 a228: d65f03c0 ret
"""
COMPUTE = """
 15d48: 97ffacf6 bl 1120 <VTABufferCPUPtr@plt>
 15d58: 97ffacf2 bl 1120 <VTABufferCPUPtr@plt>
 15d70: 90000000 adrp x0, 15000
 15d74: 91364000 add x0, x0, #0xd90
 15d7c: d63f0100 blr x8
 15d8c: d65f03c0 ret
"""
LEAF = """
 15d90: b9400829 ldr w9, [x1, #8]
 15d94: 4e218863 frinta v3.4s, v3.4s
 15f64: d65f03c0 ret
"""


def test_positive_cpu_closure_requires_no_device_enqueue_or_unknown_leaf_call():
    result = audit.certify_cpu_path(WRAPPER, COMPUTE, LEAF, "__TVMBackendParallelLaunch")
    assert result["compute_parallel_launches"] == 1
    assert result["compute_cpu_pointer_resolutions"] == 2


@pytest.mark.parametrize("wrapper,compute,leaf,relocation,reason", [
    (WRAPPER + " a230: 94000000 bl 1000 <VTASynchronize@plt>\n", COMPUTE, LEAF,
     "__TVMBackendParallelLaunch", "CALL_GRAPH_CHANGED"),
    (WRAPPER, COMPUTE + " 15d80: 94000000 bl 1000 <UnknownClosure@plt>\n", LEAF,
     "__TVMBackendParallelLaunch", "CALL_GRAPH_CHANGED"),
    (WRAPPER, COMPUTE, LEAF.replace("15d94: 4e218863 frinta v3.4s, v3.4s",
                                   "15d94: 94000000 bl 1000 <UnknownClosure>"),
     "__TVMBackendParallelLaunch", "UNKNOWN_CALL_OR_EFFECT"),
    (WRAPPER, COMPUTE, LEAF, "__TVMFuncCall", "BINDING_UNRESOLVED"),
    (WRAPPER, COMPUTE.replace("#0xd90", "#0xd94"), LEAF,
     "__TVMBackendParallelLaunch", "LAMBDA_ADDRESS_CHANGED"),
    (WRAPPER, COMPUTE, LEAF.replace("15f64", "15f68"),
     "__TVMBackendParallelLaunch", "CLOSURE_CHANGED"),
])
def test_unknown_cpu_closure_is_refused(wrapper, compute, leaf, relocation, reason):
    with pytest.raises(audit.AuditError, match=reason):
        audit.certify_cpu_path(wrapper, compute, leaf, relocation)


def test_incomplete_package_and_path_escape_refused(tmp_path):
    with pytest.raises(audit.AuditError, match="INCOMPLETE"):
        audit.verify_package(tmp_path, {})
    # A manifest cannot authorize reads outside its package even with a valid SHA.
    expected = {"manifest.json": "0" * 64, "libtvm_runtime.so": "0" * 64, "libvta.so": "0" * 64}
    expected.update({f"stages/stage{i}/{name}": "0" * 64 for i in range(3)
                     for name in ("graph.json", "graphlib.so", "params.params")})
    expected["../outside"] = "0" * 64
    with pytest.raises(audit.AuditError, match="PACKAGE_PATH_INVALID"):
        audit.verify_package(tmp_path, expected)


BASE = audit.ROOT / "vta/tutorials/frontend/report_out"
PACKAGE = BASE / "resource_aware_maxplus/v1_p7_top20_board_20260904/packages/01_cpu-00-02_t4__vta-03-17_t1__cpu-18-20_t1/package"
A_JSON = BASE / "stage_tile_cotuning/c3s_buffer_reuse/s0_static_run1/A.json"
VTA = BASE / "stage_tile_cotuning/c3s_buffer_reuse/build_s1_run2/libvta.so"


@pytest.mark.skipif(not PACKAGE.is_dir() or not VTA.is_file(), reason="frozen local artifacts unavailable")
def test_frozen_package_native_certificate_is_local_only_and_dynamic_pending():
    result = audit.audit(PACKAGE, A_JSON, PACKAGE / "libtvm_runtime.so", VTA)
    assert result["status"] == "LOCAL_STATIC_GATES_PASSED_BOARD_QUALIFICATION_REQUIRED"
    assert result["passed"] is True and result["local_static_passed"] is True
    assert result["board_qualified"] is False
    assert result["dynamic_qualification"]["status"] == "PENDING"
    assert result["dynamic_qualification"]["passed"] is False
    assert result["board_calls"] == 0
    assert result["native"]["linked_parameters"] is False
    assert result["native"]["identities"]["actual_vta_sha256"] == audit.VTA_SHA256
    saved = copy.deepcopy(result)
    certificate_sha = saved.pop("certificate_sha256")
    assert certificate_sha == audit.digest(saved)


def test_source_does_not_load_modules_recompile_or_contact_board():
    source = Path(audit.__file__).read_text()
    for forbidden in ("import tvm", "tvm.build(", "LoadFromFile(", "ssh ", "scp ", "time_evaluator("):
        assert forbidden not in source
