"""Certificates for the already frozen carrier rewrite, including FPGA dumps.

The pre-encoding queue is captured but never executed. Its exact UOP lifetime
proof transfers through unchanged ordered payloads to the actual queue. Actual
driver completion belongs only to the post-encoding submission.
"""
from copy import deepcopy
import hashlib

from c3_five_tradeoff_commands import decode_submissions
from c3_tps_notification_probe import certify_rewrite, require
from c3_tps_window_native import certify_native
from c3_official_window_native_binding import native_binding_details
from c3_tps_aggregation_capacity import organization
from c3_thread_sync_audit import instruction_dump

BEGIN = 'C3_CARRIER_PRE_ENCODING_BEGIN\n'
END = 'C3_CARRIER_PRE_ENCODING_END\n'


def split_capture(dump):
    require(dump.count(BEGIN) == dump.count(END) == 1, 'CARRIER_EXPECTED_ONE_CAPTURED_SUBMIT')
    prefix, rest = dump.split(BEGIN)
    pre, actual = rest.split(END)
    require(len(decode_submissions(prefix + pre)) == len(decode_submissions(actual)) == 1,
            'CARRIER_UNEXPLAINED_SUBMIT')
    return prefix + pre, actual


def certify_call(events, dump, ledger, definitions, command_depths, token_depths):
    before_text, actual_text = split_capture(dump)
    before, actual = decode_submissions(before_text)[0], decode_submissions(actual_text)[0]
    require(ledger['attempted_submits'] == ledger['recorded_submits'] == len(ledger['submits']) == 1,
            'CARRIER_LEDGER_CENSUS')
    row = ledger['submits'][0]
    require(row['insns'] == len(actual) and row['insn_bytes'] == 16*len(actual)
            and row['completion'] is True and row['timeout'] == 0, 'CARRIER_ACTUAL_COMPLETION')
    rewrite = certify_rewrite(before, actual, command_depths, token_depths)
    structural_ledger = deepcopy(ledger)
    structural_ledger['submits'][0]['insns'] = len(before)
    structural_ledger['submits'][0]['insn_bytes'] = 16*len(before)
    pre_native = certify_native(events, before_text, structural_ledger, definitions)
    binding = native_binding_details(events, actual_text)
    require(binding['ancestors'] == pre_native['ancestors'], 'CARRIER_ACTUAL_DEPENDENCIES_CHANGED')
    projection = organization(before_text, pre_native)
    actual_counts = instruction_dump(actual_text)['submissions'][0]
    semantic = actual_counts['semantic']
    for key, value in [('MAC_GEMM', semantic.get('MAC_GEMM', 0)),
                       ('RESET_GEMM', semantic.get('RESET_GEMM', 0)),
                       ('ALU', semantic.get('ALU', 0)), ('UOP_LOAD', semantic.get('LOAD_UOP', 0)),
                       ('UOP_LOAD_bytes', actual_counts['load_bytes'].get('UOP', 0))]:
        require(projection[key] == value, 'CARRIER_PAYLOAD_ACCOUNTING_CHANGED:' + key)
    projection.update(status='CERTIFIED_CARRIER_CALL_DIAGNOSTIC_PROJECTION',
                      all_instructions=len(actual), compute_NOP=semantic.get('NOP-COMPUTE-STAGE', 0),
                      scope='actual post-encoding counts; cache allocation preserved from pre-encoding proof')
    return {'status': 'ACTUAL_CARRIER_FIFO_UOP_AND_PROGRESS_CERTIFIED',
            'capture_sha256': hashlib.sha256(dump.encode()).hexdigest(),
            'rewrite': rewrite, 'actual_binding': binding, 'actual_submit_ledger': ledger,
            'organization': projection, 'pre_encoding_native': pre_native,
            'pre_encoding_scope': 'unexecuted captured queue; not a second target or driver submission',
            'uop_value_proof_transfer': 'all ordered non-NOP payloads exactly preserved',
            'cache_definitions': pre_native['cache_definitions'], 'ancestors': binding['ancestors']}


def for_storage(certificate):
    from c3_official_window_workflow import native_certificate_for_storage
    result = dict(certificate)
    result['pre_encoding_native'] = native_certificate_for_storage(result['pre_encoding_native'])
    result['ancestors'] = [hex(a) for a in result['ancestors']]
    result['actual_binding'] = dict(result['actual_binding'])
    result['actual_binding']['ancestors'] = [hex(a) for a in result['actual_binding']['ancestors']]
    return result
