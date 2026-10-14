"""Bounded, local-only reproduction inputs for Banerjee's double buffering.

The author schedule is evaluated in a private namespace, not registered in
TOPI and not installed in the production pipeline. This is a functional
adaptation of the published mechanism, not the author's complete stack.
"""

import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tvm
from tvm import autotvm, te, topi, tir
from tvm.autotvm.task.space import ConfigEntity
import vta

from c3_residency_tradeoff import digest, write_new
from run_vta_c3_node24_partial_local import frozen_fusion, route, SYMBOL, TILE


COMMIT = "9bc19a8a90b6fef53ef05a4320032414e545f22f"
AUTHOR_HASHES = {
    "inject_virtual_thread.cc": "3be1306bec4c24e6bdf352c15fccbac4468ad9f3d2b9e415c4571b4f6490e014",
    "vta_conv2d.py": "c8428db2c98a82e55cf38e737b0d5ce1c29265e0a18991d7b51ba182e56361eb",
    "runtime.cc": "98527054f20b7682fedc131fab3deba812cbd2e1081006f25ef285b52de80187",
    "coproc_sync.cc": "b55a26d97bd0f0b90330f512ce7b151f42cabfaf0d01f8d06aae3849159ac62d",
}


def author_namespace(directory, safe_uop_order=False):
    directory = Path(directory)
    for name, sha in AUTHOR_HASHES.items():
        if digest(directory / name) != sha:
            raise ValueError("author source identity drift: " + name)
    parsed = ast.parse((directory / "vta_conv2d.py").read_text())
    functions = [x for x in parsed.body if isinstance(x, ast.FunctionDef)
                 and x.name in ("conv2d_packed", "schedule_conv2d_packed")]
    if len(functions) != 2:
        raise ValueError("author compute/schedule pair missing")
    for function in functions:
        function.decorator_list = []  # Never replace global production TOPI registrations.
    if safe_uop_order:
        changes = 0
        for node in ast.walk(functions[1]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "reorder" \
                    and [getattr(x, "id", None) for x in node.args] == \
                    ["x_bo", "k_o", "x_j", "d_j", "d_i", "x_co", "x_i", "x_bi", "x_ci", "k_i"]:
                names = ["x_bo", "k_o", "d_j", "d_i", "x_co", "x_i", "x_j", "x_bi", "x_ci", "k_i"]
                node.args = [ast.Name(id=name, ctx=ast.Load()) for name in names]
                changes += 1
        if changes != 1:
            raise ValueError("common safe UOP reorder not uniquely located")
    env = vta.get_env()
    class EnvironmentView:
        # Local TE recognizes cthread, not the author's dthread extension.
        # The ordinary control uses this mapping only; reuse is a separate pass.
        VTHREAD_NAME = "cthread"
        def __getattr__(self, name):
            return getattr(env, name)
    namespace = {"np": np, "tvm": tvm, "te": te, "topi": topi,
                 "get_env": lambda: EnvironmentView(),
                 "is_packed_layout": lambda value: value == "NCHW1n16c"}
    tree = ast.Module(body=functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(directory / "vta_conv2d.py"), "exec"), namespace)
    namespace["shared_safe_uop_order"] = safe_uop_order
    return namespace


def complete_compute(namespace):
    """Four-buffer TE equation bound to the actual frozen Relay semantics."""
    relay = frozen_fusion()
    text = relay.astext()
    if "right_shift" not in text or "a_min=-127" not in text or "a_max=127" not in text:
        raise ValueError("actual node24 fusion arithmetic changed")
    x = te.placeholder((1, 8, 14, 14, 1, 16), "int8", name="X")
    w = te.placeholder((16, 8, 3, 3, 16, 16), "int8", name="W")
    add = te.placeholder((16, 1, 1, 1, 16), "int32", name="ADD")
    # Reuse only the complete Tile entity; the private author schedule below
    # controls execution. No residency dispatch or Input schedule is installed.
    config = route("input_stationary", TILE)["identity"]["complete_config_entity"]
    config = json.loads(json.dumps(config))
    for row in config["entity"]:
        if row[0] == "oc_nthread":
            row[2] = 2
    cfg = ConfigEntity.from_json_dict(dict(config, index=-1))
    conv = namespace["conv2d_packed"](cfg, x, w, (1, 1), (1, 1), (1, 1), "NCHW1n16c", "int32")
    # Separate min/max stages as in Relay's legalised VTA fusion, rather than
    # one nested scalar expression unsupported by the existing ALU injector.
    shifted = topi.right_shift(topi.add(conv, add), 7)
    out = topi.cast(topi.maximum(topi.minimum(shifted, 127), -127), "int8")
    schedule = namespace["schedule_conv2d_packed"](cfg, [out])
    return schedule, [x, w, add, out], {
        "frozen_relay_sha256": hashlib.sha256(tvm.ir.save_json(relay).encode()).hexdigest(),
        "abi_order": ["X", "W", "ADD", "Y"],
        "buffers": [{"shape": list(map(int, t.shape)), "dtype": t.dtype} for t in (x, w, add, out)],
        "compute": "int32 convolution, broadcast ADD, arithmetic >>7, clip(-127,127), int8 cast",
        "tile": TILE, "oc_nthread": 2, "h_nthread": 1,
        "config_entity": config, "author_thread_tag_adaptation": "VTHREAD_NAME -> cthread",
        "not_exact_author_binary_or_graph_route": True,
        "shared_safe_uop_order": namespace["shared_safe_uop_order"],
        "safe_order_scope": "both controls: spatial output traversal moved inside KH/KW; each output's reduction order unchanged",
    }


@contextmanager
def private_author_schedule(namespace):
    """One isolated worker's compiler hook; restored even on failure.

    The production file and TOPI registry are untouched. This context may
    only wrap the single frozen node24 Relay module, never a production graph.
    """
    from vta.top import vta_conv2d
    original = vta_conv2d._schedule_conv2d_packed_impl
    def scheduled(cfg, outs, residency_mode="original", **kwargs):
        if residency_mode not in ("original", 0) or kwargs:
            raise ValueError("private author schedule may only compile the Original control route")
        return namespace["schedule_conv2d_packed"](cfg, outs)
    vta_conv2d._schedule_conv2d_packed_impl = scheduled
    try:
        yield
    finally:
        vta_conv2d._schedule_conv2d_packed_impl = original


def SpecializeNode24Contexts():
    """Common host-context specialisation, without changing virtual lanes.

    Kept separate from reuse so both controls avoid the frozen runtime's
    two-int microkernel cache-key restriction in exactly the same way.
    """
    @tir.transform.prim_func_pass(opt_level=0, name="tir.c3.SpecializeNode24Contexts")
    def transform(f, _mod, _ctx):
        found = []
        def post(node):
            if isinstance(node, tir.For) and str(node.loop_var).startswith("cthread"):
                if int(node.extent) != 2 or int(node.min) != 0:
                    raise ValueError("common specialisation requires exactly two contexts")
                found.append(str(node.loop_var))
                return tir.SeqStmt([tir.stmt_functor.substitute(node.body,
                                   {node.loop_var: tir.IntImm("int32", lane)}) for lane in range(2)])
        body = tir.stmt_functor.ir_transform(f.body, None, post)
        if not found:
            raise ValueError("no original two-context loops to specialise")
        return f.with_body(body).with_attr("c3_common_context_specialisation", len(found))
    return transform


def BanerjeeNode24Reuse():
    """Paper-pattern adaptation after the local cthread injection.

    Original: I0/W0 and I1/W1 contain identical input for one CI slice.
    Adapted: both MACs read I[k%2], while W keeps its CO-thread bank.
    The duplicate input LOAD is deleted; all original WGT and ACC scopes
    and their tokens remain. Two-lane host loops are specialised to avoid
    the author's two-int runtime cache-key extension. No new drain/serial.

    This is not a mechanical port of the old author C++ implementation.
    """
    analyzer = tvm.arith.Analyzer()

    @tir.transform.prim_func_pass(opt_level=0, name="tir.c3.BanerjeeReuse")
    def transform(f, _mod, _ctx):
        changes = {"input_load_kept": 0, "input_load_deleted": 0, "uop_input_remapped": 0}

        def walk(node, reduction=None):
            if isinstance(node, tir.For) and str(node.loop_var) == "k_o_outer":
                if int(node.extent) != 8 or reduction is not None:
                    raise ValueError("fixed node24 reduction changed")
                return tir.For(node.loop_var, node.min, node.extent, node.kind,
                               walk(node.body, node.loop_var), node.thread_binding, node.annotations, node.span)
            if isinstance(node, tir.For) and str(node.loop_var).startswith("cthread"):
                if int(node.extent) != 2 or int(node.min) != 0:
                    raise ValueError("expected two output-channel contexts")
                return tir.SeqStmt([walk(tir.stmt_functor.substitute(node.body,
                                      {node.loop_var: tir.IntImm("int32", lane)}), reduction) for lane in range(2)])
            if isinstance(node, tir.For):
                return tir.For(node.loop_var, node.min, node.extent, node.kind, walk(node.body, reduction),
                               node.thread_binding, node.annotations, node.span)
            if isinstance(node, tir.SeqStmt):
                return tir.SeqStmt([walk(x, reduction) for x in node.seq])
            if isinstance(node, tir.AttrStmt):
                return tir.AttrStmt(node.node, node.attr_key, node.value, walk(node.body, reduction), node.span)
            if isinstance(node, tir.Allocate):
                return tir.Allocate(node.buffer_var, node.dtype, node.extents, node.condition,
                                    walk(node.body, reduction), node.annotations, node.span)
            if isinstance(node, tir.LetStmt):
                return tir.LetStmt(node.var, node.value, walk(node.body, reduction), node.span)
            if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
                return node
            call = node.value
            if call.op.name == "tir.call_extern" and call.args[0].value == "VTALoadBuffer2D" \
                    and int(call.args[-1]) == 2:
                if reduction is None:
                    raise ValueError("input LOAD outside expected reduction")
                dest = call.args[11]
                if not isinstance(dest, tir.Call) or dest.op.name != "tir.tvm_access_ptr":
                    raise ValueError("reuse must precede device-storage lowering")
                offset = analyzer.simplify(dest.args[2])
                if not isinstance(offset, tir.IntImm) or int(offset) not in (0, 1024):
                    raise ValueError("unexpected input bank offset")
                if int(offset) == 1024:
                    changes["input_load_deleted"] += 1
                    return tir.Evaluate(0)
                args, ptr = list(call.args), list(dest.args)
                ptr[2] = tir.indexmod(reduction, 2) * 1024
                args[11] = tir.Call(dest.dtype, dest.op, ptr, dest.span)
                changes["input_load_kept"] += 1
                return tir.Evaluate(tir.Call(call.dtype, call.op, args, call.span))
            if call.op.name == "tir.vta.uop_push" and int(call.args[0]) == 0 and int(call.args[1]) == 0:
                if reduction is None:
                    raise ValueError("MAC outside expected reduction")
                ptr, args = list(call.args[3].args), list(call.args)
                if call.args[3].op.name != "tir.tvm_access_ptr":
                    raise ValueError("missing INP access pointer")
                ptr[2] = tir.indexmod(reduction, 2) * 1024 + tir.indexmod(ptr[2], 1024)
                args[3] = tir.Call(call.args[3].dtype, call.args[3].op, ptr, call.args[3].span)
                changes["uop_input_remapped"] += 1
                return tir.Evaluate(tir.Call(call.dtype, call.op, args, call.span))
            return node

        result = f.with_body(walk(f.body))
        if changes != {"input_load_kept": 1, "input_load_deleted": 1, "uop_input_remapped": 2}:
            raise ValueError("paper-pattern adaptation did not touch exact paired scopes: " + str(changes))
        return result.with_attr("c3_banerjee_functional_adaptation", 1)

    return transform


@tvm.instrument.pass_instrument
class Archive:
    def __init__(self, output):
        self.output, self.rows = Path(output), []

    def save(self, mod, info, prefix):
        if not any(x in info.name for x in ("StorageFlatten", "InjectVirtualThread", "InjectDMAIntrin",
                    "InjectCoProcSync", "FoldUopLoop", "CPUAccessRewrite", "BanerjeeReuse", "SpecializeNode24Contexts")):
            return
        name = "%03d_%s_%s" % (len(self.rows), prefix, info.name.replace(".", "_"))
        write_new(self.output / (name + ".json"), json.loads(tvm.ir.save_json(mod)))
        # Script is a generated compiler artifact, not handwritten source.
        (self.output / (name + ".tir")).write_text(mod.script())
        self.rows.append({"file": name, "pass": info.name, "when": prefix})

    def run_before_pass(self, mod, info):
        self.save(mod, info, "before")

    def run_after_pass(self, mod, info):
        self.save(mod, info, "after")

