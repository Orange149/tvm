"""Restricted two-bank notification migration; no tile/data rearrangement.

Transforms are local diagnostic proposals until the caller certifies complete
products, actual native FIFO/token bindings, hardware and cache lifetimes.
This module cannot authorize FPGA execution.
"""
import hashlib
import tvm
from tvm import tir

from c3_valid_window import call_info, integer, statement_sequence, expanded_uops
from c3_official_window_method import _candidate, _merge


def _dependency(stmt, name, lane):
    info = call_info(stmt)
    return info is not None and info[0] == name and tuple(integer(x) for x in info[1]) == lane


def _bank(node):
    unit, reason = _candidate(node, None)
    if unit is None:
        return None
    if len(unit['prefix']) != 1 or len(unit['suffix']) != 1:
        return None
    if not _dependency(unit['prefix'][0], 'tir.vta.coproc_dep_pop', (1,2)) or \
       not _dependency(unit['suffix'][0], 'tir.vta.coproc_dep_push', (2,1)):
        return None
    return unit


def _compatible(left, right):
    if len(left['begins']) != len(right['begins']):
        return False
    return all(tvm.ir.structural_equal(a,b) for a,b in zip(left['begins'], right['begins']))


def _outer(unit, body):
    old = unit['outer']
    return tir.AttrStmt(old.node, old.attr_key, old.value, body, old.span)


def migrate(function, variant, first_only=False, bank_loop_source_verified=False):
    """Return a diagnostic N or G proposal and a deterministic source ledger.

    Direct pairs need no host-loop expansion. A bank For is accepted only
    when the caller separately proves its virtual-thread provenance. The N
    variant retains the original kernel AttrStmt object beneath that For.
    """
    if variant not in ('B','N','G'):
        raise ValueError('UNKNOWN_VARIANT')
    if variant == 'B':
        return function, []
    decisions = []

    def maybe_first(old, new, host):
        if not first_only:
            return new
        condition = tir.const(True, 'bool')
        for loop in host:
            condition = tir.And(condition, loop.loop_var == loop.min)
        return tir.IfThenElse(condition, new, old)

    def form(units, host, original, bank_for=None):
        if not _compatible(*units):
            return None
        if sum(u['count'] for u in units) > 8192:
            decisions.append({'status':'PRESERVE_BOUNDARY','reason':'UOP_CAPACITY'})
            return None
        ready = [u['prefix'][0] for u in units]
        release = [u['suffix'][0] for u in units]
        if bank_for is not None:
            # N preserves the one initializer handle and its two integer keys.
            original_unit = _bank(bank_for.body)
            if original_unit is None:
                return None
            ready = [original_unit['prefix'][0]]*2
            release = [original_unit['suffix'][0]]*2
            compute = tir.For(bank_for.loop_var, bank_for.min, bank_for.extent,
                              bank_for.kind, original_unit['kernel'],
                              bank_for.thread_binding, bank_for.annotations, bank_for.span)
        else:
            compute = statement_sequence([u['kernel'] for u in units])
        if variant == 'G':
            # Strip control prefix/suffix before joining; insert them exactly
            # once below. Original _merge would otherwise duplicate controls.
            naked = [dict(u, outer=None, prefix=[], suffix=[]) for u in units]
            compute = _merge(naked)
        body = statement_sequence([*ready, compute, *release])
        new = _outer(units[0], body)
        decisions.append({'status':'LOCAL_DIAGNOSTIC_PROPOSAL',
            'variant':variant, 'source_kind':'verified_bank_axis' if bank_for is not None else 'adjacent_compute_pair',
            'source_uops':[u['count'] for u in units],
            'joined_uops':sum(u['count'] for u in units) if variant=='G' else None,
            'host_axes':len(host),
            'original_sha256':hashlib.sha256(tvm.ir.save_json(original).encode()).hexdigest(),
            'no_board_authorization':True})
        if variant=='N' and first_only and bank_for is None:
            # Branching the WHOLE pair duplicates each initializer's lexical
            # definition and hence LLVM static handles. Only the notifications
            # may be conditional in the one-window diagnostic N.
            condition=tir.const(True,'bool')
            for loop in host:
                condition=tir.And(condition,loop.loop_var==loop.min)
            left,right=units
            left_body=statement_sequence([left['prefix'][0],
                tir.IfThenElse(condition,right['prefix'][0],None),left['kernel'],
                tir.IfThenElse(tir.Not(condition),left['suffix'][0],None)])
            right_body=statement_sequence([
                tir.IfThenElse(tir.Not(condition),right['prefix'][0],None),right['kernel'],
                tir.IfThenElse(condition,left['suffix'][0],None),right['suffix'][0]])
            return statement_sequence([_outer(left,left_body),_outer(right,right_body)])
        return maybe_first(original, new, host)

    def walk(node, host):
        if isinstance(node, tir.SeqStmt):
            items = list(node.seq)
            result, at = [], 0
            while at < len(items):
                if at+1 < len(items):
                    left, right = _bank(items[at]), _bank(items[at+1])
                    if left and right:
                        original = statement_sequence(items[at:at+2])
                        formed = form([left,right], host, original)
                        if formed is not None:
                            result.append(formed)
                            at += 2
                            continue
                result.append(walk(items[at],host))
                at += 1
            return statement_sequence(result)
        if isinstance(node, tir.For):
            unit = _bank(node.body)
            if unit and integer(node.min)==0 and integer(node.extent)==2:
                if bank_loop_source_verified:
                    units = []
                    for bank in (0,1):
                        specialized = tir.stmt_functor.substitute(node.body, {node.loop_var:tir.IntImm(node.loop_var.dtype,bank)})
                        units.append(_bank(specialized))
                    if all(units):
                        formed = form(units,host,node,bank_for=node)
                        if formed is not None:
                            return formed
                else:
                    decisions.append({'status':'SUPPORT_UNRESOLVED','reason':'BANK_AXIS_PROVENANCE_REQUIRED'})
            return tir.For(node.loop_var,node.min,node.extent,node.kind,walk(node.body,[*host,node]),
                           node.thread_binding,node.annotations,node.span)
        if isinstance(node, tir.AttrStmt):
            if node.attr_key=='coproc_uop_scope':
                return node
            return tir.AttrStmt(node.node,node.attr_key,node.value,walk(node.body,host),node.span)
        if isinstance(node,tir.LetStmt):
            return tir.LetStmt(node.var,node.value,walk(node.body,host),node.span)
        if isinstance(node,tir.IfThenElse):
            return tir.IfThenElse(node.condition,walk(node.then_case,host),
                walk(node.else_case,host) if node.else_case is not None else None,node.span)
        return node
    return function.with_body(walk(function.body,[])), decisions


def acc_spacing(events, minimum_gap):
    """Expanded issue distances, including both loop carries and join seams.

    `minimum_gap` must be issued by a bound hardware proof, not a guessed
    latency. A diagnostic result with None cannot qualify a candidate.
    """
    rows=[]
    for eid,event in enumerate(events):
        if event['kind']!='kernel' or event['mode_reset'] != [0,0]:
            continue
        last,closest={},None
        for ordinal,args in enumerate(expanded_uops(event)):
            dst=args[2]
            if dst in last:
                gap=ordinal-last[dst]
                closest=gap if closest is None else min(closest,gap)
                if minimum_gap is not None and gap < minimum_gap:
                    raise ValueError('ACC_PIPELINE_DISTANCE_VIOLATION')
            last[dst]=ordinal
        rows.append({'event':eid,'minimum_repeated_acc_issue_distance':closest,
                     'expanded_steps':ordinal+1})
    return {'status':'DIAGNOSTIC_DISTANCE_ONLY' if minimum_gap is None else 'ACC_DISTANCE_PASSED',
            'verified_hardware_gap':minimum_gap,'kernels':rows}

