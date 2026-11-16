"""Restricted TPS notification release, independent of kernel representation.

Only the second iteration of one two-iteration reduction loop is changed.
The initializer statements are retained, not cloned into condition branches.
All returned transformations remain proposals until the caller certifies the
full logical function, actual native cache contents and frozen hardware.
"""
from collections import Counter, defaultdict
import hashlib
import json

import tvm
from tvm import tir

from c3_valid_window import call_info, dma_cells, expanded_uops, integer, statement_sequence
from c3_official_window_method import _limits, _trace
from c3_sync_release_audit import Graph, Obligations, access_model, requirements
from c3_five_tradeoff_commands import decode_submissions
from c3_tps_window_progress import LANES, certify_progress


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _dependency(node, kind, lane):
    info = call_info(node)
    return info is not None and info[0] == 'tir.vta.coproc_dep_' + kind and \
        tuple(integer(x) for x in info[1]) == lane


def _load(node, domain):
    info = call_info(node)
    return info is not None and info[0] == 'VTALoadBuffer2D' and \
        len(info[1]) == 12 and integer(info[1][11]) == domain


def _candidate(node):
    if not isinstance(node, tir.AttrStmt) or node.attr_key != 'coproc_scope' or integer(node.value) != 1:
        return None
    parts = list(node.body.seq) if isinstance(node.body, tir.SeqStmt) else [node.body]
    if len(parts) != 4 or not _dependency(parts[0], 'pop', (2, 1)) or \
            not _dependency(parts[-1], 'push', (1, 2)) or not _load(parts[2], 1):
        return None
    prefix = parts[1]
    if not isinstance(prefix, tir.For) or integer(prefix.min) != 0 or integer(prefix.extent) != 4 or \
            prefix.kind != tir.ForKind.SERIAL or prefix.thread_binding is not None or prefix.annotations or \
            not _load(prefix.body, 2):
        return None
    return parts


def _event_positions(events):
    """Control identity is lane/ordinal; material identity is ordered position."""
    materials, controls, counts = [], {}, Counter()
    last_compute = None
    for eid, event in enumerate(events):
        kind = event['kind']
        if kind in ('load', 'store', 'kernel'):
            materials.append(event)
            if kind == 'kernel' or kind == 'load' and event['domain'] == 3:
                last_compute = len(materials) - 1
        elif kind in ('push', 'pop'):
            lane = tuple(event['pair'])
            base = kind, lane
            key = kind, lane, counts[base]
            counts[base] += 1
            controls[key] = {'event': eid, 'cursor': len(materials), 'last_compute': last_compute}
        elif kind == 'sync':
            key = kind, (), counts[(kind, ())]
            counts[(kind, ())] += 1
            controls[key] = {'event': eid, 'cursor': len(materials), 'last_compute': last_compute}
        else:
            raise ValueError('UNKNOWN_LOGICAL_EVENT')
    return materials, controls


def verify_event_delta(reference_events, candidate_events, expected_moves=2):
    """Require the exact full event delta and the two independent input ranges.

    This does not replace the full version/product or native physical proof.
    It binds moved waits to MAC producers through FIFO token ordinals, not
    old native instruction indices or names of loop variables.
    """
    old, old_controls = _event_positions(reference_events)
    new, new_controls = _event_positions(candidate_events)
    require(old == new, 'MATERIAL_OR_KERNEL_CACHE_IDENTITY_CHANGED')
    require([e for e in reference_events if e['kind'] in ('push', 'pop', 'sync')] ==
            [e for e in candidate_events if e['kind'] in ('push', 'pop', 'sync')],
            'ORDERED_CONTROL_SEQUENCE_CHANGED')
    require(old_controls.keys() == new_controls.keys(), 'CONTROL_CENSUS_OR_LANE_CHANGED')
    moved = []
    for key, before in old_controls.items():
        after = new_controls[key]
        if before['cursor'] == after['cursor']:
            continue
        kind, lane, ordinal = key
        require(kind == 'pop' and lane == (2, 1), 'UNEXPECTED_CONTROL_MOVEMENT')
        start, stop = before['cursor'], after['cursor']
        require(stop == start + 4 and stop < len(old), 'WAIT_MUST_CROSS_EXACTLY_FOUR_INPUT_LOADS')
        prefix = old[start:stop]
        require(all(e['kind'] == 'load' and e['domain'] == 2 for e in prefix),
                'WAIT_CROSSES_NON_INPUT_MATERIAL')
        weight = old[stop]
        require(weight['kind'] == 'load' and weight['domain'] == 1, 'WAIT_NOT_IMMEDIATELY_BEFORE_WEIGHT')
        require(not any(k != key and v['event'] > before['event'] and v['cursor'] <= stop and
                        v['event'] < next((j for j in range(before['event'] + 1, len(reference_events))
                                         if reference_events[j] is weight), len(reference_events))
                        for k, v in old_controls.items()), 'WAIT_CROSSES_ANOTHER_CONTROL')
        push = old_controls.get(('push', lane, ordinal))
        require(push is not None and push['last_compute'] is not None, 'WAIT_PRODUCER_UNRESOLVED')
        producer_ordinal = push['last_compute']
        producer = old[producer_ordinal]
        require(producer['kind'] == 'kernel' and producer['mode_reset'] == [0, 0],
                'ONLY_MAC_PRODUCED_WAIT_MAY_MOVE')
        used_input, used_weight = set(), set()
        for args in expanded_uops(producer):
            used_input.add(args[3])
            used_weight.add(args[4])
        written_input = set().union(*({a for a, _ in dma_cells(e)} for e in prefix))
        written_weight = {a for a, _ in dma_cells(weight)}
        require(not used_input.intersection(written_input), 'NEW_INPUT_OVERWRITES_ACTIVE_MAC_INPUT')
        require(bool(used_weight.intersection(written_weight)), 'FOLLOWING_WEIGHT_WAIT_HAS_NO_MATCHING_MAC_USE')
        moved.append({'lane': list(lane), 'ordinal': ordinal,
                      'producer_material': producer_ordinal,
                      'old_consumer_material': start, 'new_consumer_material': stop,
                      'released_input_materials': list(range(start, stop)),
                      'current_mac_input_range': [min(used_input), max(used_input) + 1],
                      'released_input_range': [min(written_input), max(written_input) + 1],
                      'protected_weight_range': [min(written_weight), max(written_weight) + 1]})
    require(len(moved) == expected_moves, 'EXPECTED_EXACTLY_TWO_DYNAMIC_WAIT_MOVES')
    return {'status': 'EXACT_LOGICAL_WAIT_DELTA_PASSED', 'dynamic_moves': moved,
            'unchanged_material_events': len(old), 'unchanged_control_count': len(old_controls),
            'kernel_content_and_cache_identity_shared_relations_unchanged': True,
            'joint_native_physical_certificate_required': True}


def transform(function):
    """Return (PrimFunc, decisions); fail closed outside one frozen TPS pattern."""
    proposals = []

    def walk(node, host):
        parts = _candidate(node)
        if parts is not None:
            require(host, 'REDUCTION_LOOP_NOT_FOUND')
            reduction = host[-1]
            require(integer(reduction.min) == 0 and integer(reduction.extent) == 2 and
                    reduction.kind == tir.ForKind.SERIAL and reduction.thread_binding is None and
                    not reduction.annotations, 'REDUCTION_MUST_BE_TWO_SERIAL_ITERATIONS')
            first = reduction.loop_var == reduction.min
            second = reduction.loop_var == reduction.min + 1
            body = statement_sequence([tir.IfThenElse(first, parts[0], None), parts[1],
                                       tir.IfThenElse(second, parts[0], None), parts[2], parts[3]])
            proposals.append({'static_scope': len(proposals), 'reduction_extent': 2,
                              'first_iteration_wait_preserved': True,
                              'second_iteration_input_load_count': 4,
                              'condition_contains_only_notification': True})
            return tir.AttrStmt(node.node, node.attr_key, node.value, body, node.span)
        if isinstance(node, tir.AttrStmt):
            if node.attr_key == 'coproc_uop_scope':
                return node
            return tir.AttrStmt(node.node, node.attr_key, node.value, walk(node.body, host), node.span)
        if isinstance(node, tir.SeqStmt):
            return statement_sequence([walk(x, host) for x in node.seq])
        if isinstance(node, tir.For):
            return tir.For(node.loop_var, node.min, node.extent, node.kind, walk(node.body, host + [node]),
                           node.thread_binding, node.annotations, node.span)
        if isinstance(node, tir.LetStmt):
            return tir.LetStmt(node.var, node.value, walk(node.body, host), node.span)
        if isinstance(node, tir.IfThenElse):
            return tir.IfThenElse(node.condition, walk(node.then_case, host),
                                  walk(node.else_case, host) if node.else_case is not None else None, node.span)
        return node

    result = function.with_body(walk(function.body, []))
    require(len(proposals) == 1, 'EXPECTED_ONE_STATIC_SHARED_INPUT_LOAD_SCOPE')
    before = _trace(function, _limits(None))[0]
    after = _trace(result, _limits(None))[0]
    delta = verify_event_delta(before, after)
    return result, [{**proposals[0], **delta, 'status': 'LOCAL_SYNC_RELEASE_PROPOSAL',
                     'reference_function_sha256': hashlib.sha256(tvm.ir.save_json(function).encode()).hexdigest(),
                     'candidate_function_sha256': hashlib.sha256(tvm.ir.save_json(result).encode()).hexdigest()}]


def _material_indices(certificate):
    return [m['instruction'] for m in sorted(certificate['material_map'], key=lambda m: m['tir_event'])]


def certify_native_pair(reference_text, reference_certificate, candidate_text, candidate_certificate,
                        reference_events, candidate_events, command_depths, token_depths,
                        max_states=1_000_000):
    """Check both wait moves together on their actual emitted instruction DAG.

    The caller first obtains each native cache/value certificate from the frozen
    runtime emulator. This comparison retains all original non-UOP physical
    orders, requires each actual UOP stream's live orders, and reports existing
    unobserved physical read races without declaring the baseline erroneous.
    """
    delta = verify_event_delta(reference_events, candidate_events)
    certificates = [reference_certificate, candidate_certificate]
    records = []
    for text, cert in zip((reference_text, candidate_text), certificates):
        require(hashlib.sha256(text.encode()).hexdigest() == cert['dump_sha256'], 'NATIVE_DUMP_IDENTITY_CHANGED')
        blocks = decode_submissions(text)
        require(len(blocks) == 1 and cert['submission_count'] == 1, 'UNEXPECTED_MULTIPLE_SUBMISSIONS')
        records.append(blocks[0])
    return _certify_joint(records, certificates, delta, command_depths, token_depths, max_states)


def certify_joint(before_records, before_native, after_records, after_native,
                  command_depths=None, token_depths=None, max_states=1_000_000):
    """Records API after caller has bound dump hashes and logical event delta.

    The default FIFO depths are the frozen experiment hardware's 512/1024.
    Callers with any other hardware identity must pass its certified depths.
    """
    return _certify_joint([before_records, after_records], [before_native, after_native], None,
                          command_depths or {stage: 512 for stage in (1, 2, 3)},
                          token_depths or {lane: 1024 for lane in LANES}, max_states)


def _certify_joint(records, certificates, delta, command_depths, token_depths, max_states):
    reference_certificate, candidate_certificate = certificates
    graphs, accesses, indices = [], [], []
    for rows, cert in zip(records, certificates):
        require(cert['submission_count'] == 1, 'UNEXPECTED_MULTIPLE_SUBMISSIONS')
        graph = Graph(rows)
        access = access_model(rows, cert)
        token_by_identity = {(tuple(t['lane']), t['ordinal']): t for t in graph.tokens}
        for control in cert['dependency_map']:
            token = token_by_identity.get((tuple(control['lane']), control['ordinal']))
            require(token is not None and token['producer' if control['kind'] == 'push' else 'consumer'] ==
                    control['instruction'], 'NATIVE_DEPENDENCY_CERTIFICATE_BINDING_CHANGED')
        Obligations(requirements(access), graph.ancestors, True)
        graphs.append(graph)
        accesses.append(access)
        indices.append(_material_indices(cert))
    require(len(indices[0]) == len(indices[1]), 'NATIVE_MATERIAL_CENSUS_CHANGED')
    old_index, new_index = indices
    old_to_new = dict(zip(old_index, new_index))
    old_material = {index: ordinal for ordinal, index in enumerate(old_index)}
    new_material = {index: ordinal for ordinal, index in enumerate(new_index)}
    for field in ('definitions', 'contracts'):
        require(reference_certificate['cache_definitions'][field] == candidate_certificate['cache_definitions'][field],
                'NATIVE_KERNEL_CACHE_CONTENT_OR_SHARING_CHANGED')
    require([x['kernel_identity'] for x in reference_certificate['uop_values']['checks']] ==
            [x['kernel_identity'] for x in candidate_certificate['uop_values']['checks']],
            'NATIVE_KERNEL_CACHE_CALL_SEQUENCE_CHANGED')
    for i, j in zip(old_index, new_index):
        for field in ('reads', 'writes', 'dead_reads'):
            before = {k: v for k, v in getattr(accesses[0][i], field).items() if k != 'UOP'}
            after = {k: v for k, v in getattr(accesses[1][j], field).items() if k != 'UOP'}
            require(before == after, 'NATIVE_MATERIAL_ACCESS_CHANGED')
    old_graph, new_graph = graphs
    original_unordered = []
    retained = 0
    for obligation in requirements(accesses[0], include_dead=True):
        if obligation['domain'] == 'UOP':
            continue
        source, target = obligation['source'], obligation['target']
        require(source in old_to_new and target in old_to_new, 'PHYSICAL_ACCESS_WITHOUT_MATERIAL_BINDING')
        if not old_graph.ancestors[target] >> source & 1:
            original_unordered.append(obligation)
            continue
        require(new_graph.ancestors[old_to_new[target]] >> old_to_new[source] & 1,
                'JOINT_MOVE_LOSES_PHYSICAL_ORDER:' + json.dumps(obligation, sort_keys=True))
        retained += 1
    old_tokens = {(tuple(t['lane']), t['ordinal']): t for t in old_graph.tokens}
    new_tokens = {(tuple(t['lane']), t['ordinal']): t for t in new_graph.tokens}
    require(old_tokens.keys() == new_tokens.keys(), 'NATIVE_TOKEN_CENSUS_CHANGED')

    def anchor(graph, material_map, index):
        if index in material_map:
            return 'material', material_map[index]
        # Runtime-only commands are bound by their adjacent same-stage materials.
        stage = graph.records[index]['stage']
        previous = next((material_map[i] for i in range(index - 1, -1, -1)
                         if i in material_map and graph.records[i]['stage'] == stage), None)
        following = next((material_map[i] for i in range(index + 1, len(graph.records))
                          if i in material_map and graph.records[i]['stage'] == stage), None)
        return 'runtime', graph.records[index]['description'], stage, previous, following

    if delta is None:
        moves = []
        for key, old in old_tokens.items():
            new = new_tokens[key]
            if anchor(old_graph, old_material, old['consumer']) == anchor(new_graph, new_material, new['consumer']):
                continue
            require(key[0] == (2, 1), 'UNEXPECTED_NATIVE_TOKEN_CONSUMER_CHANGE')
            producer = old['producer']
            require(old_graph.records[producer]['description'] == 'GEMM' and
                    'reset_out: 0' in old_graph.records[producer]['detail'], 'ONLY_MAC_PRODUCED_WAIT_MAY_MOVE')
            start, stop = old_material.get(old['consumer']), new_material.get(new['consumer'])
            require(start is not None and stop == start + 4, 'WAIT_MUST_CROSS_EXACTLY_FOUR_INPUT_LOADS')
            require(all(old_graph.records[old_index[i]]['description'] == 'LOAD INP' for i in range(start, stop)),
                    'WAIT_CROSSES_NON_INPUT_MATERIAL')
            require(old_graph.records[old_index[stop]]['description'] == 'LOAD WGT',
                    'WAIT_NOT_IMMEDIATELY_BEFORE_WEIGHT')
            require(not any(t['lane'] == [2, 1] and old['consumer'] < t['consumer'] <= old_index[stop]
                            for t in old_graph.tokens), 'WAIT_CROSSES_ANOTHER_CONTROL')
            prefix_mask = 0
            for i in range(start, stop):
                prefix_mask |= accesses[0][old_index[i]].writes.get('INP', 0)
            require(not prefix_mask & accesses[0][producer].reads.get('INP', 0),
                    'NEW_INPUT_OVERWRITES_ACTIVE_MAC_INPUT')
            require(accesses[0][old_index[stop]].writes.get('WGT', 0) & accesses[0][producer].reads.get('WGT', 0),
                    'FOLLOWING_WEIGHT_WAIT_HAS_NO_MATCHING_MAC_USE')
            moves.append({'lane': [2, 1], 'ordinal': key[1], 'producer_material': old_material[producer],
                          'old_consumer_material': start, 'new_consumer_material': stop,
                          'released_input_materials': list(range(start, stop))})
        require(len(moves) == 2, 'EXPECTED_EXACTLY_TWO_DYNAMIC_WAIT_MOVES')
        delta = {'dynamic_moves': moves}
    moved = {(tuple(r['lane']), r['ordinal']): r for r in delta['dynamic_moves']}

    native_moves = []
    for key, old in old_tokens.items():
        new = new_tokens[key]
        require(anchor(old_graph, old_material, old['producer']) == anchor(new_graph, new_material, new['producer']),
                'NATIVE_TOKEN_PRODUCER_CHANGED')
        if key in moved:
            row = moved[key]
            require(old_material.get(old['producer']) == row['producer_material'] and
                    new_material.get(new['producer']) == row['producer_material'] and
                    old_material.get(old['consumer']) == row['old_consumer_material'] and
                    new_material.get(new['consumer']) == row['new_consumer_material'],
                    'NATIVE_WAIT_MOVE_NOT_BOUND_TO_EXPECTED_MATERIAL')
            native_moves.append({**row, 'reference_producer_instruction': old['producer'],
                                 'reference_consumer_instruction': old['consumer'],
                                 'candidate_producer_instruction': new['producer'],
                                 'candidate_consumer_instruction': new['consumer']})
        else:
            require(anchor(old_graph, old_material, old['consumer']) == anchor(new_graph, new_material, new['consumer']),
                    'UNEXPECTED_NATIVE_TOKEN_CONSUMER_CHANGE')
    # The per-program native certificates explain LOAD merging and cache state.
    # A new runtime-only instruction position still needs an explicit explanation.
    runtime_anchors = [[anchor(g, material, i) for i in range(len(g.records)) if i not in material]
                       for g, material in zip(graphs, (old_material, new_material))]
    require(runtime_anchors[0] == runtime_anchors[1], 'RUNTIME_ENCODING_CHANGE_REQUIRES_SEPARATE_EXPLANATION')
    progress = certify_progress(new_graph.records, command_depths, token_depths, max_states)
    require(progress['status'] == 'FINITE_FIFO_PROGRESS_PROVED', 'FINITE_FIFO_PROGRESS_UNRESOLVED:' + progress['status'])
    return {'status': 'JOINT_NATIVE_SYNC_RELEASE_PASSED', 'native_moves': native_moves,
            'original_ordered_physical_conflicts_retained': retained,
            'original_unordered_dead_read_conflicts': original_unordered,
            'all_candidate_live_and_uop_orders_checked': True,
            'kernel_identity_content_and_sharing_unchanged': True,
            'runtime_only_instruction_positions_unchanged': True, 'progress': progress,
            'proves_performance': False, 'proves_general_sync_optimization': False}
