"""Certify unchanged legacy serial fragments and the two replacement kernels.

Reuse the sealed UOP simulator and material binder. Forced-serial fragments
use actual FIFO/token ancestors for every data-version and ACC check.
Unrelated LOAD/STORE events need not have a total order; frozen runtime
RewriteForceSerial deliberately does not connect a LOAD-to-STORE transition. Empty Syncs
must contain only NOPs/FINISH and still close their actual finite FIFOs.
"""
from collections import defaultdict
import re
import types

from c3_tps_sync_release_local import require
from c3_five_tradeoff_commands import decode_submissions
from c3_official_window_native_binding import native_binding_details, material_ancestor_payloads
from c3_yolo416_fifo_certificate import certify_action_graph
import c3_tps_window_native as core


def submit_signature(ledger, mode):
    require(ledger['schema'] == 'c3_official_window_submit_ledger_v1' and ledger['mode'] == mode and
            ledger['overflow'] is False, 'GRAPH_LEDGER_MODE_OR_OVERFLOW')
    rows = ledger['submits']
    require(0 < len(rows) <= 32 and ledger['attempted_submits'] == ledger['recorded_submits'] == len(rows),
            'GRAPH_LEDGER_CENSUS')
    for i, row in enumerate(rows, 1):
        require(row['ordinal'] == i and row['completion'] is True and row['timeout'] == 0 and
                row['reason'] == 'explicit_sync' and row['insns'] > 0 and
                row['insn_bytes'] == row['insns']*16 and row['serialized_uop_bytes'] >= 0 and
                row['serialized_uop_bytes'] % 4 == 0, 'GRAPH_SUBMIT_NOT_COMPLETE')
    return rows


def force_dependencies(events, records):
    occurrences = defaultdict(list)
    for row in records:
        stage = row['stage']
        for flag, kind, lane in zip(row['bits'], ('pop','pop','push','push'),
                ((stage-1,stage),(stage+1,stage),(stage,stage-1),(stage,stage+1))):
            if flag: occurrences[kind,lane].append(row['index'])
    pairs = []
    for lane in {key[1] for key in occurrences}:
        a, b = occurrences['push',lane], occurrences['pop',lane]
        require(len(a) == len(b), 'SERIAL_NATIVE_TOKEN_IMBALANCE')
        pairs += [{'lane': list(lane), 'ordinal': i, 'producer': p, 'consumer': q}
                  for i,(p,q) in enumerate(zip(a,b))]
    return [], [{'origin': 'frozen RewriteForceSerial; all material predecessors checked'}], pairs


def strict_material_order(native):
    earlier = 0
    for row in native['material_map']:
        event = row['tir_event']
        require(native['ancestors'][event] & earlier == earlier, 'SERIAL_MATERIAL_ORDER_NOT_PROVED')
        earlier |= 1 << event


def forced_binding(events, text):
    # No module globals are mutated: only the known dependency-binder argument
    # is replaced, with exact material binding and actual DAG propagation kept.
    binder = types.FunctionType(native_binding_details.__code__,
        {**native_binding_details.__globals__, 'bind_dependencies': force_dependencies})
    native = binder(events, text)
    return native


def control_only(records, row):
    require(row['serialized_uop_bytes'] == 0, 'EMPTY_SYNC_HAS_UOP_SOURCE')
    allowed = {'NOP-MEMORY-STAGE','NOP-COMPUTE-STAGE','NOP-STORE-STAGE','FINISH'}
    require(records and records[-1]['description'] == 'FINISH' and
            all(r['description'] in allowed for r in records), 'EMPTY_SYNC_HAS_MATERIAL_EFFECT')
    material_ancestor_payloads(records, [0]*len(records))


def frames(text, expected):
    pattern = r'\[C3_GRAPH_BEGIN\] (\S+)\n(.*?)\[C3_GRAPH_END\] \1\n'
    rows = re.findall(pattern, text, re.S)
    require([symbol for symbol, _ in rows] == expected and
            text.count('[C3_GRAPH_BEGIN]') == text.count('[C3_GRAPH_END]') == len(rows),
            'GRAPH_DIAGNOSTIC_FRAME_ORDER_OR_COMPLETENESS')
    return rows


def certify_function(events, text, rows, mode, force_serial, definitions, command_depths, token_depths):
    from c3_valid_window import certify_trace
    fragments = [];pending = []
    for event in events:
        pending.append(event)
        if event['kind'] == 'sync': fragments.append(pending);pending = []
    require(not pending and len(fragments) == len(rows), 'FUNCTION_SYNC_CENSUS')
    marks = list(re.finditer(r'There are (\d+) instructions\n', text))
    require(len(marks) == len(rows), 'FUNCTION_DUMP_SUBMIT_CENSUS')
    material_fragments = [i for i, es in enumerate(fragments) if any(e['kind'] in ('kernel','load','store') for e in es)]
    require(len(material_fragments) == 1, 'FIXED_ONE_MATERIAL_FRAGMENT_PER_FUNCTION')
    observed_initializers = core._initializers(text)
    certs = []
    for index,(events1,row) in enumerate(zip(fragments,rows)):
        start = 0 if index == 0 else marks[index].start()
        end = marks[index+1].start() if index+1 < len(marks) else len(text)
        piece = text[start:end]
        records = decode_submissions(piece)
        require(len(records) == 1 and len(records[0]) == row['insns'], 'FUNCTION_DUMP_LEDGER')
        progress = certify_action_graph(records[0], command_depths, token_depths)
        require(progress['status'] == 'FINITE_FIFO_PROGRESS_PROVED', 'FUNCTION_FIFO_PROGRESS')
        if not any(e['kind'] in ('kernel','load','store') for e in events1):
            control_only(records[0], row)
            certs.append({'kind': 'control_only', 'completion': True, 'progress': progress})
            continue
        one_row = {**row, 'ordinal': 1}
        # A leading control-only Sync prints before the host creates kernels.
        # Its textual dump block therefore contains those later initializer
        # declarations. Relocate only these observed declarations in the proof
        # view; preserve every instruction and keep the raw graph dump intact.
        present = core._initializers(piece)
        require(present == observed_initializers or not present, 'PARTIAL_INITIALIZER_PLACEMENT_UNRESOLVED')
        moved = bool(observed_initializers and not present)
        if moved:
            prefix = ''
            for values in observed_initializers:
                prefix += 'There are %d uops\n' % len(values)
                prefix += ''.join('[%d] acc=%d, inp=%d, wgt=%d\n' % (j,*value) for j,value in enumerate(values))
            piece = prefix+piece
        ledger = {'schema': 'c3_official_window_submit_ledger_v1', 'mode': mode, 'overflow': False,
                  'attempted_submits': 1, 'recorded_submits': 1, 'submits': [one_row]}
        checker = core.certify_native
        if force_serial:
            checker = types.FunctionType(checker.__code__, {**checker.__globals__, 'native_binding_details': forced_binding})
        native = checker(events1, piece, ledger, definitions)
        definitions = native['cache_definitions']
        certify_trace(events1, ancestors=native['ancestors'])
        certs.append({'kind': 'material', 'native': native, 'progress': progress,
                      'forced_serial_actual_data_dependencies_proved': force_serial,
                      'observed_initializers_relocated_in_certificate_view': moved})
    return {'status': 'ALL_FUNCTION_FRAGMENTS_CERTIFIED', 'submissions': certs}, definitions


def certify_graph(graph, functions, traces, text, ledger, definitions, command_depths, token_depths):
    rows = submit_signature(ledger, ledger['mode'])
    calls = [(i,n['attrs']['func_name']) for i,n in enumerate(graph['nodes'])
             if n.get('attrs',{}).get('func_name') in functions]
    chunks = frames(text, [s for _,s in calls])
    require(len(calls) == 12 and len(rows) == 22, 'FIXED_GRAPH_CALLS_AND_SUBMISSIONS')
    offset = 0;certificates = [];definitions = dict(definitions or {})
    for (node,symbol), (_,piece) in zip(calls,chunks):
        count = sum(e['kind'] == 'sync' for e in traces[symbol])
        certificate, cached = certify_function(traces[symbol], piece, rows[offset:offset+count],
            ledger['mode'], bool(functions[symbol]['runtime_flags']), definitions.get(symbol), command_depths, token_depths)
        definitions[symbol] = cached
        certificates.append({'node': node, 'symbol': symbol, 'first_submission': offset+1,
                             'submission_count': count, **certificate})
        offset += count
    require(offset == len(rows), 'UNEXPLAINED_GRAPH_SUBMISSION')
    return {'status': 'ALL_22_SUBMISSIONS_AND_12_FUNCTION_CALLS_CERTIFIED',
            'calls': certificates, 'submission_count': offset}, definitions
