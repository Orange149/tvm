"""Static RESET opportunity triage. Never builds or invokes a target function.

An edge-move hypothesis is not rewritten TIR or a native FIFO certificate.
Compare all newly unordered material events, including preceding STOREs.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from c3_yolo224_sync_structure import HOLD, read, write, sha, require, paired_graph, project, input_cells

CENSUS = HOLD / '20261016_c3_yolo224_sync_structure_readonly03'


def ancestors(pred, target):
    seen, todo = set(), [target]
    while todo:
        i = todo.pop()
        for p, _ in pred[i]:
            if p not in seen:
                seen.add(p)
                todo.append(p)
    return seen


def moved_graph(pred, rows):
    result = list(pred)
    touched = set()
    for r in rows:
        source, old, new = r['push'], r['pop'], r['following_weight']
        require(new is not None and source < old < new, 'NOT_A_FORWARD_MOVE')
        for index in (old, new):
            if index not in touched:
                result[index] = list(result[index]); touched.add(index)
        require((source, 'token') in result[old], 'MISSING_TOKEN')
        result[old].remove((source, 'token'))
        result[new].append((source, 'token'))
    return result


def label(e):
    if e['kind'] != 'kernel':
        return e['kind'].upper()
    return 'RESET' if e['mode_reset'] == [0, 1] else 'MAC' if e['mode_reset'] == [0, 0] else 'ALU'


def work(e):
    return len(e['uops']) * math.prod(loop[0] for loop in e['uops'][0]['loops'])


def summarize_loss(events, lost, inp_indices):
    from c3_valid_window import expanded_uops
    writes = set().union(*(input_cells(events[i]) for i in inp_indices))
    mats = sorted(i for i in lost if events[i]['kind'] in ('load', 'store', 'kernel'))
    conflicts, records = [], []
    for i in mats:
        e = events[i]; record = {'event': i, 'kind': label(e)}
        if e['kind'] == 'kernel':
            record['vector_uop_steps'] = work(e)
            if e['mode_reset'][0] == 0:
                overlap = writes & {u[3] for u in expanded_uops(e)}
                if overlap:
                    conflicts.append({'event': i, 'kind': label(e), 'INP_address': min(overlap)})
        elif e['kind'] == 'store':
            record.update(bytes=e['x'] * e['y'] * 16, sram=e['sram'], tensor=e['tensor'])
            require(all(e['tensor'] != events[j]['tensor'] for j in inp_indices), 'STORE_INPUT_DDR_ALIAS')
        elif e['kind'] == 'load' and e['domain'] == 2:
            overlap = writes & input_cells(e)
            if overlap: conflicts.append({'event': i, 'kind': 'INP_LOAD', 'INP_address': min(overlap)})
        records.append(record)
    return {'materials': records, 'kinds': dict(Counter(r['kind'] for r in records)),
            'new_INP_physical_conflicts': conflicts}


def analyze(events, rows):
    pred, _ = paired_graph(events)
    joint = moved_graph(pred, rows)
    result = []
    for row in rows:
        require(row['producer_kind'] == 'RESET', 'NOT_RESET')
        require(all(events[i]['kind'] == 'load' and events[i]['domain'] == 2 for i in row['input_prefix']), 'NOT_INPUT_PREFIX')
        require(events[row['following_weight']]['kind'] == 'load' and events[row['following_weight']]['domain'] == 1, 'NOT_WEIGHT_TARGET')
        single = moved_graph(pred, [row])
        base, alone, together = set(), set(), set()
        for target in row['input_prefix']:
            b = ancestors(pred, target)
            a = ancestors(single, target)
            c = ancestors(joint, target)
            # Aggregate only losses for each input, not differences of unions.
            base |= b; alone |= b - a; together |= b - c
        a = summarize_loss(events, alone, row['input_prefix'])
        j = summarize_loss(events, together, row['input_prefix'])
        producer = events[row['producer_kernel']]
        inputs = [events[i] for i in row['input_prefix']]
        reset_addresses = [u[2] for u in __import__('c3_valid_window').expanded_uops(producer)]
        previous_mac = any(events[i]['kind'] == 'kernel' and events[i]['mode_reset'] == [0, 0]
                           for i in range(row['producer_kernel']))
        result.append({'source_wait': row, 'phase': 'LATER_OUTPUT_BLOCK' if previous_mac else 'FUNCTION_START',
                       'reset_ACC_range': [min(reset_addresses), max(reset_addresses) + 1],
                       'input_ranges': [[min(input_cells(e)), max(input_cells(e)) + 1] for e in inputs],
                       'input_DDR_bytes': sum(e['x'] * e['y'] * 16 for e in inputs),
                       'input_padding': [e['pad'] for e in inputs],
                       'producer_vector_steps': work(producer),
                       'single_move': a, 'all_reset_moves': j,
                       'single_and_joint_material_loss_equal': a == j,
                       'structural_class': '+'.join(sorted(j['kinds']))})
    return result


def worker(row_path, output):
    import tvm, vta
    entry = read(row_path)
    sealed = read(CENSUS / 'objects' / (entry['object_id'] + '.json'))
    require(sha(entry['tir']) == entry['sha256'], 'SOURCE_CHANGED')
    events, counts, settings = project(tvm.ir.load_json(Path(entry['tir']).read_text()))
    require(len(events) == sealed['event_count'] and counts == sealed['counts'], 'TRACE_DIFFERS_FROM_CENSUS')
    candidates = [r for r in sealed['waits'] if r['producer_kind'] == 'RESET' and
                  r['classification'] == 'ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE']
    rows = analyze(events, candidates)
    write(output, {'source': entry, 'counts': counts, 'runtime_settings': settings,
                   'candidate_count': len(rows), 'rows': rows,
                   'structural_classes': dict(Counter(r['structural_class'] for r in rows)),
                   'phases': dict(Counter(r['phase'] for r in rows)),
                   'direct_physical_conflict_count': sum(len(r['all_reset_moves']['new_INP_physical_conflicts']) for r in rows),
                   'released_MAC_count': sum(r['all_reset_moves']['kinds'].get('MAC', 0) for r in rows),
                   'native_fifo_or_runtime_qualification': False, 'target_calls': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--row', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); worker(args.row, args.output)
