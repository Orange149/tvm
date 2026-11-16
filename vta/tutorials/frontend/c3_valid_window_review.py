#!/usr/bin/env python3
"""Independent small semantic counterexamples for the bounded window checker."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import tvm
from tvm import tir

from c3_valid_window import certify_trace, expanded_uops, statement_sequence, trace_function
from test_c3_valid_window import fixture


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attempt(function):
    try:
        return dict(accepted=True, result=function())
    except Exception as error:
        return dict(accepted=False, error=str(error))


def mixed_alu_function():
    """One kernel requiring two different ALU opcodes: runtime rejects this."""
    function = fixture(1)
    scope = 0

    def post(node):
        nonlocal scope
        if isinstance(node, tir.AttrStmt) and node.attr_key == "coproc_uop_scope":
            scope += 1
            if scope != 2:
                return None
            pushes = [tir.Evaluate(tir.call_intrin("int32", "tir.vta.uop_push",
                1, 0, dst, dst, 0, 2 if dst == 0 else 3, 1, 1)) for dst in range(4)]
            return tir.AttrStmt(node.node, node.attr_key, tir.StringImm("VTAPushALUOp"),
                                statement_sequence(pushes))
    return function.with_body(tir.stmt_functor.ir_transform(function.body, None, post))


def int64_signature_function():
    """The only free kernel variable has an eight-byte closure ABI."""
    function = fixture(1)
    key = tir.Var("independent_cache_key", "int64")

    def post(node):
        if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            call = node.value
            if call.op.name == "tir.vta.uop_push" and int(call.args[1]) == 0:
                args = list(call.args)
                args[4] = key
                return tir.Evaluate(tir.call_intrin("int32", "tir.vta.uop_push", *args))
    body = tir.stmt_functor.ir_transform(function.body, None, post)
    return function.with_body(tir.For(key, tir.IntImm("int64", 0), tir.IntImm("int64", 2),
                                     tir.ForKind.SERIAL, body))


def wrong_dispatch_function(dispatch):
    function = fixture(1)
    scope = 0

    def post(node):
        nonlocal scope
        if isinstance(node, tir.AttrStmt) and node.attr_key == "coproc_uop_scope":
            scope += 1
            if scope == 2:
                return tir.AttrStmt(node.node, node.attr_key, tir.StringImm(dispatch), node.body)
    return function.with_body(tir.stmt_functor.ir_transform(function.body, None, post))


def alu_kernel(dst, src, opcode, immediate, imm, scope):
    return dict(kind="kernel", scope=scope, signature=(), mode="VTAPushALUOp",
                mode_reset=[1, 0],
                uops=[dict(args=[1, 0, dst, src, 0, opcode, immediate, imm], loops=[])])


def run(out):
    core = Path(__file__).with_name("c3_valid_window.py")
    tests = Path(__file__).with_name("test_c3_valid_window.py")
    source_before = {str(p): sha(p) for p in (core, tests)}
    findings = {}

    mixed = attempt(lambda: len(trace_function(mixed_alu_function())))
    findings["ALU_context_uniformity"] = dict(
        requirement="runtime UopKernel::Push requires opcode/use_imm/imm identical throughout one ALU kernel",
        observation=mixed, risk="false support-domain acceptance" if mixed["accepted"] else "rejected")

    signature = attempt(lambda: [e["signature"] for e in trace_function(int64_signature_function())
                                 if e["kind"] == "kernel"])
    findings["cache_signature_integer_width"] = dict(
        requirement="frozen runtime accepts only nbytes 0 or sizeof(int)=4",
        observation=signature,
        risk="int64 free variable accepted by static trace" if signature["accepted"] else "rejected")

    wrong_dispatch = attempt(lambda: len(trace_function(wrong_dispatch_function("VTAPushALUOp"))))
    unknown_dispatch = attempt(lambda: len(trace_function(wrong_dispatch_function("UnknownUopDispatch"))))
    findings["scope_dispatch_matches_kernel_mode"] = dict(
        requirement="coproc_uop_scope dispatch selects actual runtime GEMM/ALU instruction family",
        wrong_dispatch=wrong_dispatch, unknown_dispatch=unknown_dispatch,
        risk="false instruction-family interpretation" if wrong_dispatch["accepted"] or unknown_dispatch["accepted"]
        else "rejected")

    base = trace_function(fixture(1))
    mac_index = next(i for i, e in enumerate(base) if e["kind"] == "kernel" and e["mode_reset"] == [0, 0])
    store_index = next(i for i, e in enumerate(base) if e["kind"] == "store")
    after_store = deepcopy(base)
    late_mac = deepcopy(base[mac_index])
    late_mac["scope"] = 99
    after_store.insert(store_index + 1, late_mac)
    late = attempt(lambda: {k: certify_trace(after_store)[k] for k in
                           ("packed_product_count", "maximum_packed_terms_per_output",
                            "output_product_postprocess_sha256")})
    findings["STORE_closes_ACC_observation"] = dict(
        requirement="later MAC must not race prior STORE or mutate the stored output's term snapshot",
        observation=late, expected_terms_at_STORE=1,
        risk="post-STORE MAC accepted and appended to old output snapshot" if late["accepted"] else "rejected")

    add_other = alu_kernel(0, 1, 2, 0, 0, 100)
    shift_other = alu_kernel(1, 1, 3, 1, 1, 101)
    first = deepcopy(base)
    second = deepcopy(base)
    first[mac_index + 1:mac_index + 1] = [add_other, shift_other]
    second[mac_index + 1:mac_index + 1] = [shift_other, add_other]
    a = attempt(lambda: certify_trace(first)["output_product_postprocess_sha256"])
    b = attempt(lambda: certify_trace(second)["output_product_postprocess_sha256"])
    equal = a["accepted"] and b["accepted"] and a["result"] == b["result"]
    findings["ALU_other_ACC_version"] = dict(
        requirement="an ALU source stage must identify the value/version observed at the read",
        add_before_shift=a, shift_before_add=b, hashes_equal=equal,
        concrete_example=dict(initial_ACC0=2, initial_ACC1=6, add_before_shift_output0=8,
                              shift_before_add_output0=5),
        risk="distinct observed arithmetic hashes equal" if equal else "rejected or distinguished")

    sample = dict(uops=[dict(args=[0, 0, 0, 10, 20, 0, 0, 0],
                            loops=[[2, 4, 8, 12], [3, 1, 2, 3]]),
                       dict(args=[0, 0, 7, 17, 27, 0, 0, 0],
                            loops=[[2, 4, 8, 12], [3, 1, 2, 3]])])
    actual = list(expanded_uops(sample))
    expected = []
    # Independent order follows the actual FSim RunGEMM's y/x/uindex loops.
    for outer in range(2):
        for inner in range(3):
            for dst, inp, wgt in ((0, 10, 20), (7, 17, 27)):
                expected.append([0, 0, dst + outer * 4 + inner,
                                 inp + outer * 8 + inner * 2,
                                 wgt + outer * 12 + inner * 3, 0, 0, 0])
    if actual != expected:
        raise AssertionError("expanded UOP order does not follow real hardware instruction loops")
    findings["actual_hardware_execution_order"] = dict(passed=True, expanded_steps=len(actual),
        order="outer hardware loop, inner hardware loop, UOP index",
        source="3rdparty/vta-hw/src/sim/sim_driver.cc:403")

    source_after = {str(p): sha(p) for p in (core, tests)}
    unresolved = [key for key, finding in findings.items()
                  if finding.get("risk", "").startswith(("false", "int64", "post-STORE", "distinct"))]
    result = dict(schema="c3_valid_window_independent_review_v1",
        source_before=source_before, source_after=source_after,
        stable_source=source_before == source_after, unresolved=unresolved, findings=findings,
        modified_core=False, board_contacted=False,
        scope="small static support-domain counterexamples; no extra Tile/program/board candidate")
    out.mkdir(parents=True, exist_ok=True)
    (out / "review.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(stable_source=result["stable_source"], unresolved=unresolved), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())
