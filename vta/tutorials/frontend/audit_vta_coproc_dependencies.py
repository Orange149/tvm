"""Audit lowered TIR for VTA coprocessor dependency-topology violations.

The VTA runtime accepts dependencies only between adjacent pipeline stages:
load(1) <-> compute(2) and compute(2) <-> store(3).  This module provides a
read-only IRModule API and a small JSON-in/JSON-out command-line interface.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import tvm
from tvm import tir
import vta  # pylint: disable=unused-import  # registers tir.vta dependency ops


SCHEMA = "vta_coproc_dependency_audit_v1"
PUSH_OP = "tir.vta.coproc_dep_push"
POP_OP = "tir.vta.coproc_dep_pop"
SUPPORTED_EDGES = frozenset({(1, 2), (2, 1), (2, 3), (3, 2)})
FORBIDDEN_DIRECT_EDGES = frozenset({(1, 3), (3, 1)})
STAGE_NAMES = {1: "load", 2: "compute", 3: "store"}


def _functions(ir: Any) -> Iterable[Tuple[str, tir.PrimFunc]]:
    if isinstance(ir, tir.PrimFunc):
        return (("main", ir),)
    if not isinstance(ir, tvm.IRModule):
        raise TypeError("expected tvm.IRModule or tvm.tir.PrimFunc")
    functions = []
    for global_var, function in ir.functions.items():
        if isinstance(function, tir.PrimFunc):
            functions.append((global_var.name_hint, function))
    return tuple(sorted(functions, key=lambda item: item[0]))


def _constant_int(expr: tir.PrimExpr):
    return int(expr.value) if isinstance(expr, tir.IntImm) else None


def audit_vta_coproc_dependencies(ir: Any) -> Dict[str, Any]:
    """Return a deterministic, JSON-serializable dependency audit.

    Push/pop balance is checked per function and directed edge using static call
    sites.  Calls inside a loop are counted once structurally; this audit does not
    attempt symbolic path-feasibility or loop-trip-count analysis.
    """

    calls = []
    counts = defaultdict(lambda: {"push": 0, "pop": 0})
    errors = []
    function_names = []

    for function_name, function in _functions(ir):
        function_names.append(function_name)

        def visit(node):
            if not isinstance(node, tir.Call) or not isinstance(node.op, tvm.ir.Op):
                return
            if node.op.name not in (PUSH_OP, POP_OP):
                return
            kind = "push" if node.op.name == PUSH_OP else "pop"
            callsite = {
                "index": len(calls),
                "function": function_name,
                "kind": kind,
                "from": None,
                "to": None,
            }
            if len(node.args) != 2:
                callsite["argument_count"] = len(node.args)
                calls.append(callsite)
                errors.append(
                    {
                        "code": "malformed_dependency_call",
                        "function": function_name,
                        "callsite_index": callsite["index"],
                        "detail": "expected exactly two stage arguments",
                    }
                )
                return
            source = _constant_int(node.args[0])
            target = _constant_int(node.args[1])
            callsite.update({"from": source, "to": target})
            calls.append(callsite)
            if source is None or target is None:
                errors.append(
                    {
                        "code": "nonconstant_stage_endpoint",
                        "function": function_name,
                        "callsite_index": callsite["index"],
                        "detail": "dependency stages must be compile-time integers",
                    }
                )
                return
            counts[(function_name, source, target)][kind] += 1

        tir.stmt_functor.post_order_visit(function.body, visit)

    edges = []
    topology_valid = True
    balanced = True
    for (function_name, source, target), edge_counts in sorted(counts.items()):
        edge = (source, target)
        allowed = edge in SUPPORTED_EDGES
        forbidden_direct = edge in FORBIDDEN_DIRECT_EDGES
        edge_balanced = edge_counts["push"] == edge_counts["pop"]
        edges.append(
            {
                "function": function_name,
                "from": source,
                "from_name": STAGE_NAMES.get(source, "unknown"),
                "to": target,
                "to_name": STAGE_NAMES.get(target, "unknown"),
                "push_count": edge_counts["push"],
                "pop_count": edge_counts["pop"],
                "balanced": edge_balanced,
                "allowed": allowed,
                "forbidden_direct_load_store": forbidden_direct,
            }
        )
        if not allowed:
            topology_valid = False
            errors.append(
                {
                    "code": (
                        "forbidden_direct_load_store"
                        if forbidden_direct
                        else "unsupported_stage_edge"
                    ),
                    "function": function_name,
                    "from": source,
                    "to": target,
                }
            )
        if not edge_balanced:
            balanced = False
            errors.append(
                {
                    "code": "push_pop_imbalance",
                    "function": function_name,
                    "from": source,
                    "to": target,
                    "push_count": edge_counts["push"],
                    "pop_count": edge_counts["pop"],
                }
            )

    if any(error["code"] in ("malformed_dependency_call", "nonconstant_stage_endpoint") for error in errors):
        topology_valid = False
        balanced = False

    push_count = sum(call["kind"] == "push" for call in calls)
    pop_count = sum(call["kind"] == "pop" for call in calls)
    return {
        "schema": SCHEMA,
        "valid": not errors,
        "semantics": {
            "balance_unit": "static_callsites_per_function_and_directed_edge",
            "loop_multiplicity": "not_expanded",
            "path_sensitive": False,
        },
        "functions": function_names,
        "supported_edges": [list(edge) for edge in sorted(SUPPORTED_EDGES)],
        "summary": {
            "function_count": len(function_names),
            "call_count": len(calls),
            "push_count": push_count,
            "pop_count": pop_count,
            "edge_count": len(edges),
            "only_supported_edges": topology_valid,
            "push_pop_balanced": balanced,
            "forbidden_1_3_present": any(
                edge["forbidden_direct_load_store"] for edge in edges
            ),
        },
        "edges": edges,
        "callsites": calls,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-json", required=True, help="Path produced by tvm.ir.save_json")
    parser.add_argument("--output", help="Write the audit JSON to this path")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()
    module = tvm.ir.load_json(Path(args.ir_json).read_text(encoding="utf-8"))
    result = audit_vta_coproc_dependencies(module)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.fail_on_invalid and not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
