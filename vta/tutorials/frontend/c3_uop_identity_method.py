"""Finite-domain MAC cache-signature quotient, with no schedule changes.

Every equivalence class uses an ORIGINAL signature as representative. Handles
remain separate. Unknown domains are unresolved, not an absence of opportunity.
The executable LLVM gate is intentionally narrower than the finite TIR method:
it currently accepts zero-based power-of-two modulo projections only.
"""
from copy import deepcopy
import json
import re

from c3_uop_identity_applicability import inspect_function, require


def representative_map(row):
    domain = row['domain']
    require(domain and domain == list(range(domain[0], domain[-1] + 1)), 'NONCONTIGUOUS_DOMAIN')
    result = {}
    for group in row['classes']:
        rep = group['representative']
        require(len(rep) == 1 and rep in group['members'], 'REPRESENTATIVE_NOT_ORIGINAL')
        for value in group['members']:
            require(len(value) == 1 and value[0] not in result, 'INVALID_CLASS_PARTITION')
            result[value[0]] = rep[0]
    require(sorted(result) == domain and all(0 <= v <= 98 for v in result.values()), 'INCOMPLETE_OR_BAD_KEY_MAP')
    return result


def projection(counter, mapping):
    """Use modulo only when it EXACTLY implements the enumerated class map."""
    from tvm import tir
    keys = sorted(mapping)
    for period in range(1, len(keys)):
        if all(mapping[k] == keys[0] + (k - keys[0]) % period for k in keys):
            expr = keys[0] + tir.floormod(counter - keys[0], period)
            return expr, {'kind': 'modulo', 'origin': keys[0], 'period': period}
    expr = tir.IntImm('int32', mapping[keys[-1]])
    for k in reversed(keys[:-1]):
        expr = tir.Select(counter == k, tir.IntImm('int32', mapping[k]), expr)
    return expr, {'kind': 'finite_select', 'domain': keys}


def transform(function):
    import tvm
    from tvm import tir
    census = inspect_function(function)
    require(census['status'] == 'FINITE_DOMAIN_CONTENT_SCREEN_COMPLETE', 'DOMAIN_SUPPORT_UNRESOLVED')
    rows = {r['scope']: r for r in census['scopes']}
    ordinal = 0
    changes = []
    inverse = {}

    def visit(node):
        nonlocal ordinal
        if isinstance(node, tir.AttrStmt) and node.attr_key == 'coproc_uop_scope':
            index = ordinal
            ordinal += 1
            row = rows[index]
            if row['mode_reset'] != [0, 0] or not row['redundant_identities']:
                return node
            require(row['closure_arity'] == 1, 'EXPECTED_SINGLE_INTEGER_CLOSURE')
            counter, = tir.analysis.undefined_vars(node.body)
            mapping = representative_map(row)
            expr, form = projection(counter, mapping)
            representative = tir.Var('c3_uop_representative_' + str(index), 'int32')
            body = tir.stmt_functor.substitute(node.body, {counter: representative})
            free = list(tir.analysis.undefined_vars(body))
            require(len(free) == 1 and free[0].same_as(representative), 'PROJECTED_CLOSURE_ARITY')
            scope = tir.AttrStmt(node.node, node.attr_key, node.value, body, node.span)
            # Undo just the variable rename to prove all other statements and
            # every initializer expression were preserved. Full finite content
            # equivalence, not this inverse alone, proves the changed value safe.
            inverse[representative] = counter
            changes.append({'scope': index, 'domain': row['domain'],
                            'mapping': [[k, mapping[k]] for k in sorted(mapping)],
                            'form': form, 'original_identities': row['signatures'],
                            'representatives': sorted(set(mapping.values())),
                            'content_classes': row['content_classes']})
            return tir.LetStmt(representative, expr, scope)
        if isinstance(node, tir.SeqStmt):
            return tir.SeqStmt([visit(x) for x in node.seq])
        if isinstance(node, tir.For):
            return tir.For(node.loop_var, node.min, node.extent, node.kind, visit(node.body),
                           node.thread_binding, node.annotations, node.span)
        if isinstance(node, tir.AttrStmt):
            return tir.AttrStmt(node.node, node.attr_key, node.value, visit(node.body), node.span)
        if isinstance(node, tir.LetStmt):
            return tir.LetStmt(node.var, node.value, visit(node.body), node.span)
        if isinstance(node, tir.Allocate):
            return tir.Allocate(node.buffer_var, node.dtype, node.extents, node.condition,
                                visit(node.body), node.annotations, node.span)
        if isinstance(node, tir.AssertStmt):
            return tir.AssertStmt(node.condition, node.message, visit(node.body), node.span)
        return node

    candidate = function.with_body(visit(function.body))
    require(ordinal == census['static_scopes'], 'SCOPE_TRAVERSAL_MISMATCH')

    def undo(node):
        if isinstance(node, tir.LetStmt) and node.var in inverse:
            return tir.stmt_functor.substitute(node.body, {node.var: inverse[node.var]})
    restored = tir.stmt_functor.ir_transform(candidate.body, None, undo, ['tir.LetStmt'])
    require(tvm.ir.structural_equal(restored, function.body), 'RENAME_INVERSE_CHANGED_PROGRAM')
    return candidate, {'status': 'FULL_FINITE_CONTENT_AND_RENAME_INVERSE_PASSED',
                       'changes': changes, 'census': census,
                       'DMA_control_and_non_MAC_body_restored_exactly': True,
                       'cross_handle_merging': False}


def event_contract(before, after, proof):
    # Trace tuples and their persisted JSON arrays have the same ordered
    # meaning. Canonicalize this transport representation only, no fields.
    before = json.loads(json.dumps(before))
    after = json.loads(json.dumps(after))
    require(len(before) == len(after), 'LOGICAL_EVENT_CENSUS_CHANGED')
    maps = {r['scope']: dict(r['mapping']) for r in proof['changes']}
    identities = {}
    changed = 0
    for old, new in zip(before, after):
        a, b = deepcopy(old), deepcopy(new)
        if old['kind'] == 'kernel':
            for field in ('cache_identity', 'signature', 'source_scope_sha256'):
                a.pop(field, None); b.pop(field, None)
            mapping = maps.get(old['scope'])
            expected = [mapping[old['signature'][0]]] if mapping is not None else old['signature']
            require(new['signature'] == expected, 'REPRESENTATIVE_SIGNATURE_MISMATCH')
            changed += expected != old['signature']
            key = (new['scope'], tuple(new['signature']))
            from c3_uop_identity_applicability import full_content
            content = full_content(new)
            require(key not in identities or identities[key] == content, 'NEW_IDENTITY_CONTENT_MUTATION')
            identities[key] = content
        require(a == b, 'COMPUTATION_TENSOR_REQUEST_OR_TOKEN_CHANGED')
    return {'status': 'EXACT_ORDERED_EVENT_CONTENT_EQUAL_EXCEPT_DECLARED_CACHE_IDENTITY',
            'candidate_cache_identities': len(identities), 'ordered_events': len(before),
            'changed_dynamic_signatures': changed, 'cross_handle_merging': False}


def llvm_signature_provenance(llvm, proof):
    """Check every final nonempty MAC closure store, not just TIR or counts.

    This bounded SSA gate fails closed for Select, non-power-of-two remainder,
    mixed projected/unprojected MAC sites and other optimizer representations.
    The TIR exact-event proof and every native reference remain required.
    """
    from c3_official_window_llvm import _function_bodies, _calls
    changes = proof['changes']
    require(changes, 'NO_PROJECTION_FOR_LLVM_GATE')
    expected = []
    for change in changes:
        form = change['form']
        require(form['kind'] == 'modulo' and form['origin'] == 0 and
                form['period'] > 1 and form['period'] & (form['period'] - 1) == 0,
                'LLVM_PROJECTION_SUPPORT_UNRESOLVED')
        expected.append(form['period'] - 1)
    rows = []
    for function, body in _function_bodies(llvm).items():
        for _, fields in _calls(body, 'VTAPushGEMMOp'):
            if fields[-1] == 'i32 0': continue
            require(len(rows) < len(expected) and len(fields) == 4 and fields[-1] == 'i32 4',
                    'FINAL_MAC_SIGNATURE_SITE_OR_ABI')
            ptr = fields[2].split()[-1]
            stores = re.findall(r'\bstore i32 (%[-\w.$]+), ptr ' + re.escape(ptr) + r'(?=[,\s])', body)
            require(stores, 'FINAL_MAC_SIGNATURE_STORE_NOT_FOUND')
            roots = []
            for value in stores:
                definition = re.search(r'^\s*' + re.escape(value) +
                    r' = and i32 (%[-\w.$]+), ' + str(expected[len(rows)]) + r'\s*$', body, re.M)
                require(definition is not None, 'FINAL_MAC_SIGNATURE_IS_NOT_REPRESENTATIVE')
                roots.append(definition[1])
            rows.append({'scope': changes[len(rows)]['scope'], 'host_function': function,
                         'handle': fields[0], 'initializer': fields[1], 'signature_storage': ptr,
                         'stores': stores, 'raw_counters': roots, 'mask': expected[len(rows)]})
    require(len(rows) == len(expected), 'FINAL_MAC_PROJECTION_SITE_COUNT')
    return {'status': 'FINAL_LLVM_REPRESENTATIVE_SIGNATURE_PASSED', 'sites': rows,
            'known_SSA_support_only': True, 'mapping_derived_from_full_finite_content': True}
