"""Offline kernel-by-kernel LLVM initializer binding for the official study.

CodeGenCPU::CreateStaticInit appends one static handle while visiting each
coproc_uop_scope.  Its call names the initializer for that handle.  We use the
global declaration order, rather than optimized call order (loop peeling may
repeat calls), to bind the final TIR scopes to those initializer bodies.

This is a bounded compiler-output gate.  It does not execute a target or read
device SRAM; the separate native certificate checks addresses, loops and UOPs.
"""
from __future__ import annotations

import hashlib
import re

import tvm
from tvm import tir


_SYMBOL = r'@(?:"(?:[^"\\]|\\.)*"|[-a-zA-Z$._0-9]+)'
_HANDLE_PREFIX = "@__tvm_static_handle"
_INITIALIZER_PREFIX = "@__tvm_static_init_lambda"
_FIELD_INDEX = (0, 1, 5, 6, 7)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _unquote(symbol):
    return "@" + symbol[2:-1] if symbol.startswith('@"') else symbol


def _without_comments(text):
    # Generated LLVM symbols/strings can contain semicolons.  Strip comments
    # only outside quoted strings; keep line breaks for definition boundaries.
    result = []
    for line in text.splitlines():
        quote, escaped, end = False, False, len(line)
        for index, char in enumerate(line):
            if escaped:
                escaped = False
            elif char == "\\" and quote:
                escaped = True
            elif char == '"':
                quote = not quote
            elif char == ";" and not quote:
                end = index
                break
        result.append(line[:end])
    return "\n".join(result)


def _split_arguments(text):
    parts, begin, stack, quote, escaped = [], 0, [], False, False
    pairs = {")": "(", "]": "[", "}": "{", ">": "<"}
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\" and quote:
            escaped = True
        elif char == '"':
            quote = not quote
        elif not quote:
            if char in "([{<":
                stack.append(char)
            elif char in ")]}>":
                _require(stack and stack.pop() == pairs[char], "UNSUPPORTED_LLVM_ARGUMENT_SYNTAX")
            elif char == "," and not stack:
                parts.append(text[begin:index].strip())
                begin = index + 1
    _require(not quote and not stack, "UNSUPPORTED_LLVM_ARGUMENT_SYNTAX")
    parts.append(text[begin:].strip())
    return parts


def _calls(text, name_pattern):
    pattern = re.compile(r"\bcall\b[^\n]*?@(" + name_pattern + r")\(")
    result = []
    for match in pattern.finditer(text):
        depth, quote, escaped, end = 1, False, False, None
        for index in range(match.end(), len(text)):
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\" and quote:
                escaped = True
            elif char == '"':
                quote = not quote
            elif not quote:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        end = index
                        break
        _require(end is not None, "UNSUPPORTED_LLVM_CALL_SYNTAX")
        result.append((match[1], _split_arguments(text[match.end():end])))
    return result


def _function_bodies(llvm):
    pattern = re.compile(r"^define\b[^\n]*?(" + _SYMBOL + r")\([^\n]*\)[^\n]*\{\s*\n(.*?)^\}", re.M | re.S)
    result = {}
    for match in pattern.finditer(llvm):
        symbol = _unquote(match[1])
        _require(symbol not in result, "DUPLICATE_LLVM_FUNCTION_DEFINITION")
        result[symbol] = match[2]
    return result


def _constant_fields(arguments):
    _require(len(arguments) == 8, "UNSUPPORTED_LLVM_UOP_ARITY")
    values = []
    for index in _FIELD_INDEX:
        value = re.fullmatch(r"i32\s+(-?\d+)", arguments[index])
        _require(value is not None, "UNSUPPORTED_LLVM_UOP_MODE_IMMEDIATE")
        values.append(int(value[1]))
    return values


def _direct_global_pointer(argument):
    # These are the opaque/typed pointer spellings emitted by CreateStaticInit.
    # A GEP or arbitrary expression mentioning the same global is not proof of
    # passing that handle/callback and is deliberately unsupported.
    pointer_type = r"(?:ptr|i8\*\*|i32\s*\(\s*i8\*\s*\)\*)"
    attributes = r"(?:(?:nonnull|noundef|nocapture|readonly)\s+)*"
    match = re.fullmatch(pointer_type + r"\s+" + attributes + "(" + _SYMBOL + ")", argument)
    _require(match is not None, "UNSUPPORTED_LLVM_STATIC_INIT_POINTER")
    return _unquote(match[1])


def _source_kernels(function):
    kernels = []

    def walk(node):
        if isinstance(node, tir.AttrStmt):
            if node.attr_key == "coproc_uop_scope":
                _require(isinstance(node.value, tir.StringImm) and node.value.value in
                         ("VTAPushGEMMOp", "VTAPushALUOp"), "UNKNOWN_SOURCE_UOP_DISPATCH")
                rows = []

                def collect(child):
                    if isinstance(child, tir.AttrStmt) and child.attr_key == "coproc_uop_scope":
                        raise ValueError("NESTED_SOURCE_UOP_SCOPE")
                    if not isinstance(child, tir.Call):
                        return
                    if child.op.name == "tir.vta.uop_push":
                        arguments = list(child.args)
                    elif child.op.name == "tir.call_extern" and child.args and isinstance(child.args[0], tir.StringImm) and child.args[0].value == "VTAUopPush":
                        arguments = list(child.args[1:])
                    else:
                        return
                    _require(len(arguments) == 8, "UNSUPPORTED_SOURCE_UOP_ARITY")
                    values = [arguments[index] for index in _FIELD_INDEX]
                    _require(all(isinstance(value, tir.IntImm) for value in values),
                             "UNSUPPORTED_NONCONSTANT_SOURCE_UOP_MODE_IMMEDIATE")
                    rows.append([int(value) for value in values])

                tir.stmt_functor.post_order_visit(node.body, collect)
                _require(rows, "EMPTY_SOURCE_UOP_SCOPE")
                _require(all(row == rows[0] for row in rows), "MIXED_SOURCE_KERNEL_MODE_OR_ALU_SETTINGS")
                mode, reset, opcode, use_imm, imm = rows[0]
                _require(mode in (0, 1) and reset in (0, 1), "UNSUPPORTED_SOURCE_MODE_RESET")
                _require(node.value.value == ("VTAPushGEMMOp" if mode == 0 else "VTAPushALUOp"),
                         "SOURCE_DISPATCH_MODE_MISMATCH")
                _require((mode == 0 and [opcode, use_imm, imm] == [0, 0, 0]) or
                         (mode == 1 and reset == 0 and opcode in (0, 1, 2, 3) and use_imm in (0, 1)
                          and -32768 <= imm <= 32767), "UNSUPPORTED_SOURCE_KERNEL_AUXILIARY_FIELDS")
                free = list(tir.analysis.undefined_vars(node.body))
                _require(len(free) <= 1 and all(str(var.dtype) == "int32" for var in free),
                         "UNSUPPORTED_SOURCE_CACHE_SIGNATURE")
                kernels.append({"dispatch": node.value.value, "fields": rows,
                                "signature_bytes": len(free) * 4})
            else:
                walk(node.body)
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                walk(child)
        elif isinstance(node, (tir.For, tir.LetStmt, tir.Allocate, tir.AssertStmt)):
            walk(node.body)
        elif isinstance(node, tir.IfThenElse):
            walk(node.then_case)
            if node.else_case is not None:
                walk(node.else_case)
        else:
            _require(isinstance(node, tir.Evaluate), "UNSUPPORTED_SOURCE_STATEMENT_ORDER")

    walk(function.body)
    _require(kernels, "EMPTY_SOURCE_KERNEL_SEQUENCE")
    return kernels


def llvm_alu_contract(function, llvm):
    """Bind all GEMM/ALU initializer auxiliary fields to each final TIR scope.

    The function is the exact post-CPU final PrimFunc used to produce `llvm`.
    Extra/missing handles, inconsistent callbacks, unknown syntax, unsupported
    signatures and cross-kernel field swaps fail closed.
    """
    source = _source_kernels(function)
    clean = _without_comments(llvm)
    handles = []
    for match in re.finditer(r"^(" + _SYMBOL + r")\s*=([^\n]*)", clean, re.M):
        symbol = _unquote(match[1])
        if symbol.startswith(_HANDLE_PREFIX):
            _require(re.search(r"\bprivate\s+global\b.*\bnull\b", match[2]) is not None,
                     "UNSUPPORTED_STATIC_HANDLE_DECLARATION")
            _require(symbol not in handles, "DUPLICATE_STATIC_HANDLE_DECLARATION")
            handles.append(symbol)
    _require(len(handles) == len(source), "LLVM_STATIC_HANDLE_SOURCE_SCOPE_CENSUS")
    bodies = _function_bodies(clean)
    mappings = {}
    calls = _calls(clean, r"VTAPush(?:GEMM|ALU)Op")
    for dispatch, arguments in calls:
        _require(len(arguments) == 4, "UNSUPPORTED_LLVM_STATIC_INIT_ARITY")
        handle, initializer = [_direct_global_pointer(argument) for argument in arguments[:2]]
        _require(handle in handles and initializer.startswith(_INITIALIZER_PREFIX),
                 "LLVM_STATIC_INIT_POINTER_NOT_BOUND")
        size = re.fullmatch(r"i32\s+(\d+)", arguments[3])
        _require(size is not None and int(size[1]) in (0, 4), "UNSUPPORTED_LLVM_CACHE_SIGNATURE")
        record = (dispatch, initializer, int(size[1]))
        if handle in mappings:
            _require(mappings[handle]["contract"] == record, "STATIC_HANDLE_CHANGES_INITIALIZER_OR_DISPATCH")
            mappings[handle]["call_sites"] += 1
        else:
            mappings[handle] = {"contract": record, "call_sites": 1}
    _require(set(mappings) == set(handles), "LLVM_STATIC_HANDLE_DISPATCH_CENSUS")
    _require(len({value["contract"][1] for value in mappings.values()}) == len(handles),
             "SHARED_INITIALIZER_BETWEEN_STATIC_HANDLES_UNSUPPORTED")
    all_initializers = {name for name in bodies if name.startswith(_INITIALIZER_PREFIX)}
    _require(all_initializers == {value["contract"][1] for value in mappings.values()},
             "LLVM_INITIALIZER_DEFINITION_CENSUS")
    _require(len(_calls(clean, "VTAUopPush")) ==
             sum(len(_calls(bodies[name], "VTAUopPush")) for name in all_initializers),
             "LLVM_UOP_PUSH_OUTSIDE_BOUND_INITIALIZER")
    bindings = []
    for index, (kernel, handle) in enumerate(zip(source, handles)):
        entry = mappings[handle]
        dispatch, initializer, signature = entry["contract"]
        _require(dispatch == kernel["dispatch"], "LLVM_KERNEL_DISPATCH_DIFFERS_FROM_SOURCE_SCOPE")
        _require(signature == kernel["signature_bytes"], "LLVM_KERNEL_SIGNATURE_DIFFERS_FROM_SOURCE_SCOPE")
        actual = [_constant_fields(arguments) for _, arguments in _calls(bodies[initializer], "VTAUopPush")]
        _require(actual, "LLVM_KERNEL_UOP_PUSH_CENSUS_DIFFERS_FROM_SOURCE_SCOPE")
        # LLVM unrolls host loops in GEMM initializers (the real R18 RESET
        # has one TIR push site but eight LLVM sites). These sites have the
        # same mode/reset/auxiliary fields. Executed UOP values/counts are
        # checked by the native cold-initializer certificate, not a count of
        # source statements. Keep the stricter ALU static-site census for
        # the currently supported final-TIR tails.
        if kernel["fields"][0][0] == 1:
            _require(len(actual) == len(kernel["fields"]), "LLVM_KERNEL_UOP_PUSH_CENSUS_DIFFERS_FROM_SOURCE_SCOPE")
        _require(all(row == kernel["fields"][0] for row in actual),
                 "LLVM_KERNEL_AUXILIARY_FIELDS_DIFFER_FROM_SOURCE_SCOPE")
        bindings.append({"source_scope_ordinal": index, "handle": handle, "initializer": initializer,
                         "dispatch": dispatch, "signature_bytes": signature, "dispatch_call_sites": entry["call_sites"],
                         "source_uop_push_sites": len(kernel["fields"]), "llvm_uop_push_sites": len(actual),
                         "mode_reset_opcode_use_imm_imm": actual[0]})
    return {"status": "LLVM_KERNEL_ALU_FIELDS_BOUND_TO_STATIC_INITIALIZERS",
            "kernel_scopes": len(bindings), "dispatch_call_sites": len(calls), "bindings": bindings,
            "fields": [row for kernel in source for row in kernel["fields"]],
            "llvm_sha256": hashlib.sha256(llvm.encode()).hexdigest(),
            "binding_basis": "Final TIR scope traversal -> CodeGenCPU static-handle declaration order -> actual dispatch callback -> initializer body",
            "addresses_separately_bound_by_actual_initializers": True,
            "static_push_site_counts_not_executed_uop_counts": True,
            "unprinted_immediates_not_direct_device_observation": True,
            "limits": ["Requires the frozen CodeGenCPU CreateStaticInit traversal/declaration-order contract",
                       "Uniform settings within each kernel; unknown LLVM syntax or optimizer scope elimination rejected",
                       "Does not independently prove machine-code execution or direct SRAM contents"]}
