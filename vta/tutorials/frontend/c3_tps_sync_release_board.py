"""Frozen R/S board execution: 120 calls, no retries, joint actual-stream proof."""
import ctypes
import json
import math
from pathlib import Path
import shlex
import time

from c3_tps_aggregation_board import sha, write, append, verify_bindings, submit_signature, make_cases
from c3_tps_aggregation_transport_v2 import TransportBoard as Board

def execute(args, artifacts, ledger, out, bindings):
    import numpy as np
    import tvm
    import gc
    from c3_official_window_reference import array_sha
    from c3_official_window_method import audit_function
    from c3_valid_window import certify_trace
    from c3_tps_window_native import certify_native
    from c3_tps_window_progress import certify_progress, LANES
    from c3_five_tradeoff_commands import decode_submissions
    from c3_official_window_workflow import native_certificate_for_storage

    from c3_tps_sync_release_local import initialize_helper, capture_helper_call
    from c3_tps_sync_release_method import certify_native_pair
    from c3_tps_aggregation_capacity import organization
    from c3_tps_sync_release_support import validate_completion, accounting, verify_native_storage
    out.mkdir(exist_ok=False)
    all_results, warmed, begun = [], {}, 0
    originals = {}
    board = Board(args, '/var/volatile/c3_tps_' + out.parent.name, artifacts['bitstream']['sha256'])
    remote, session_out = None, None
    try:
        verify_bindings(bindings)
        write(out / 'contract.json', {'ledger_sha256': sha(out.parent / 'call_manifest.json'),
             'bindings': bindings, 'host': args.host, 'known_hosts_sha256': sha(args.known_hosts),
             'target_calls_planned': 120, 'no_retries': True, 'root': board.root})
        cases, case_ids = make_cases(json.loads(Path(artifacts['programs']['R']['reference_contract']).read_text()))
        write(out / 'reference_identities.json', case_ids)
        analyses = {}
        for program, descriptor in artifacts['programs'].items():
            parent = tvm.ir.load_json((Path(descriptor['qualified_directory']) / 'parent.json').read_text())
            events = audit_function(parent[parent.get_global_var(descriptor['symbol'])])['events']
            analyses[program] = (events, certify_trace(events)['output_terms'])
        hardware = json.loads(Path(artifacts['hardware_archive']['path']).read_text())
        depths = hardware['fifo_depths']
        command_depths = {1: depths['load_queue'], 2: depths['gemm_queue'], 3: depths['store_queue']}
        token_depths = dict(zip(LANES, [depths[k] for k in ('l2g_queue','g2l_queue','g2s_queue','s2g_queue')]))
        initial = board.preflight()
        write(out / 'preflight.json', initial)
        boot = initial['boot_id']
        write(out / 'deployment.json', board.deploy(artifacts))
        for ordinal, session in enumerate(ledger['sessions'], 1):
            rows = [r for r in ledger['calls'] if r['session_id'] == session['session_id']]
            if len(rows) != 6:
                raise ValueError('SESSION_NOT_SIX_CALLS')
            first = rows[0]
            program, mode = first['program'], first['collection']
            descriptor = artifacts['programs'][program]
            session_out = out / ('session_%02d' % ordinal)
            session_out.mkdir()
            verify_bindings(bindings)
            write(session_out / 'preflight.json', board.preflight(boot))
            write(session_out / 'start.json', board.start(ordinal, mode))
            if args.transport_contract['sessions'][ordinal-1]['session_id'] != session['session_id']:
                raise ValueError('FROZEN_TRANSPORT_SESSION_CHANGED')
            remote = board.connect(session_out)
            retained = []
            expected_files = {'libtvm_runtime.so': artifacts['runtime_tvm']['sha256'],
                              'libvta.so': artifacts['runtime']['sha256']}
            for name, local in [('once_helper.so', artifacts['helper']['path']),
                                ('storage_helper.so', artifacts['storage_helper']['path']),
                                ('target.so', descriptor['arm_module'])]:
                remote.upload(local, target=name)
                retained.append(remote.load_module(name))
                expected_files[name] = sha(local)
            module = retained[-1]
            get = remote.get_function
            timer, diagnostic = (get('c3.official_window.fused_' + kind + '_once') for kind in ('timed','diagnostic'))
            metadata, counts = (get('c3.official_window.' + key) for key in ('metadata','counts'))
            reset = get('c3.official_window.reset_counts')
            clear, get_ledger, collection = (get('vta.runtime.official_' + key) for key in
                                             ('submit_ledger_clear','submit_ledger','collection_status'))
            storage_query = get('c3.tps_window.buffer_storage')
            inventory = {key: json.loads(get('vta.runtime.' + key)()) for key in
                         ('queue_capacity_status','replay_status')}
            write(session_out / 'inventory.json', inventory)
            capacity = inventory['queue_capacity_status']
            if any(capacity.get(k) != 33554432 for k in
                   ('insn_capacity_bytes', 'uop_capacity_bytes', 'insn_submit_threshold_bytes')) or \
                    capacity.get('replay_policy') != 'disabled' or capacity.get('submissions') != 0:
                raise ValueError('ACTUAL_QUEUE_CONFIGURATION_MISMATCH')
            if inventory['replay_status'].get('mode') != 'disabled' or \
                    inventory['replay_status'].get('template_count') != 0:
                raise ValueError('ACTUAL_REPLAY_NOT_DISABLED')
            status = json.loads(collection())
            if status['mode'] != mode or status['debug_flag'] != 0:
                raise ValueError('INITIAL_COLLECTION_MISMATCH')
            buffers = tuple(tvm.nd.array(a, remote.ext_dev(0)) for a in cases[first['seed'], first['weight']][0])
            handles = [int(ctypes.cast(b.handle, ctypes.c_void_p).value) for b in buffers]
            storage = json.loads(storage_query(*buffers))
            if len(storage) != 4 or any(s['byte_offset'] != 0 or not s['data_address'] for s in storage):
                raise ValueError('PARAMETER_STORAGE_UNEXPLAINED')
            write(session_out / 'actual_environment.json', board.environment(boot, expected_files))
            initialize_helper(reset, counts, metadata,
                lambda value: write(session_out/'helper_initial_state.json', value))
            definitions, backend_bases = None, None
            events, products = analyses[program]
            for row in rows:
                if row['call_index'] != len(all_results) + 1:
                    raise ValueError('CALL_NOT_NEXT_FROZEN_ROW')
                arrays, expected = cases[row['seed'], row['weight']]
                for buffer, value in zip(buffers, arrays):
                    buffer.copyfrom(value)
                if not np.all(buffers[-1].numpy() != expected):
                    raise ValueError('OUTPUT_SENTINEL_NOT_WRITTEN')
                clear()
                begin = {**row, 'start_unix': time.time(), 'target_calls_planned': 1,
                         'buffer_handles': handles, 'storage': storage,
                         'reference_identity': case_ids[str(row['seed']) + ':' + row['weight']]}
                append(out / 'begins.jsonl', begin)
                write(session_out / ('call_%02d_begin.json' % row['position']), begin)
                begun += 1
                dump, elapsed = None, None
                if row['phase'].startswith('diagnostic_'):
                    dump, execution = capture_helper_call(row['position'],
                        lambda: diagnostic(module, descriptor['symbol'], *buffers),
                        counts, metadata, get_ledger,
                        lambda value: write(session_out / ('call_%02d_execution.json' % row['position']), value))
                else:
                    elapsed, execution = capture_helper_call(row['position'],
                        lambda: timer(module, descriptor['symbol'], *buffers),
                        counts, metadata, get_ledger,
                        lambda value: write(session_out / ('call_%02d_execution.json' % row['position']), value))
                    elapsed = float(elapsed)
                    if not math.isfinite(elapsed) or elapsed <= 0:
                        raise ValueError('NONFINITE_OR_NONPOSITIVE_TIME')
                returned = time.time()
                validate_completion(execution, row['position'],
                                    'diagnostic' if dump is not None else 'timer')
                submit, meta = execution['submits'], execution['metadata']
                status = json.loads(collection())
                actual = buffers[-1].numpy()
                current_storage = json.loads(storage_query(*buffers))
                result = {**row, 'target_calls': 1, 'elapsed_ms': elapsed,
                          'start_unix': begin['start_unix'], 'end_unix': returned,
                          'helper_metadata': meta, 'collection_status': status, 'submit_ledger': submit,
                          'buffer_handles': handles, 'storage': current_storage,
                          'reference_identity': begin['reference_identity'], 'output_sha256': array_sha(actual),
                          'mismatch_count': int(np.count_nonzero(actual != expected)), 'status': 'PENDING_CHECKS'}
                # Persist the completed invocation even if an outside-timer gate fails.
                write(session_out / ('call_%d_returned.json' % row['position']), result)
                if not np.array_equal(actual, expected):
                    raise ValueError('FPGA_FULL_FUSION_OUTPUT_MISMATCH')
                if any(not np.array_equal(buf.numpy(), val) for buf, val in zip(buffers[:-1], arrays[:-1])):
                    raise ValueError('READONLY_PARAMETER_CHANGED')
                if storage != current_storage or handles != [int(ctypes.cast(b.handle, ctypes.c_void_p).value) for b in buffers]:
                    raise ValueError('PARAMETER_STORAGE_CHANGED')
                if meta['started_calls'] != row['position'] or meta['completed_calls'] != row['position'] or \
                        int(counts()) != row['position'] or meta['parameter_count'] != 4 or \
                        meta['packed_calls_per_invocation'] != 1 or meta['adds_synchronize']:
                    raise ValueError('HIDDEN_OR_MISSING_COMPLETE_CALL')
                if status['mode'] != mode or status['debug_flag'] != 0 or \
                        (mode == 'minimal' and status['detailed_clock_reads'] != 0):
                    raise ValueError('COLLECTION_MODE_CHANGED_OR_DETAIL_CLOCKS_RETAINED')
                signature = submit_signature(submit, mode)
                if len(signature) != 1:
                    raise ValueError('UNEXPECTED_MULTI_SUBMISSION')
                if dump is not None:
                    (session_out / ('native_%d.txt' % row['position'])).write_text(dump)
                    native = certify_native(events, dump, submit, definitions)
                    definitions = native['cache_definitions']
                    verify_native_storage(native, current_storage)
                    if backend_bases is None:
                        backend_bases = native['tensor_domain_dram_bases']
                    if native['tensor_domain_dram_bases'] != backend_bases:
                        raise ValueError('ACTUAL_BACKEND_STORAGE_CHANGED')
                    pair_key = (row['rotation'], row['position'])
                    if program == 'R':
                        originals[pair_key] = (dump, native_certificate_for_storage(native))
                    else:
                        reference_dump, reference_native = originals[pair_key]
                        joint = certify_native_pair(reference_dump, reference_native, dump, native,
                            analyses['R'][0], events, command_depths, token_depths)
                        write(session_out / ('joint_%d.json' % row['position']), joint)
                    if certify_trace(events, ancestors=native['ancestors'])['output_terms'] != products:
                        raise ValueError('NATIVE_DEPENDENCY_SEMANTICS_CHANGED')
                    progress = [certify_progress(records, command_depths, token_depths)
                                for records in decode_submissions(dump)]
                    if any(p['status'] != 'FINITE_FIFO_PROGRESS_PROVED' for p in progress):
                        raise ValueError('FINITE_FIFO_PROGRESS_UNPROVEN')
                    write(session_out / ('native_%d.json' % row['position']), native_certificate_for_storage(native))
                    write(session_out / ('progress_%d.json' % row['position']), progress)
                    write(session_out / ('organization_%d.json' % row['position']), organization(dump, native))
                    warmed[program, row['rotation'], row['position']] = signature
                elif row['phase'] in ('warmup', 'formal'):
                    key = program, row['rotation'], row['position']
                    if key not in warmed or warmed[key] != signature:
                        raise ValueError('ACTUAL_SUBMIT_PATH_DIFFERS_FROM_OWN_DIAGNOSTIC')
                else:
                    local = json.loads(Path(descriptor['local_calls']).read_text())['calls']
                    if signature != submit_signature(local[row['position']-1]['submit_ledger'], 'detailed'):
                        raise ValueError('QUALIFICATION_SUBMIT_PATH_CHANGED')
                result['status'] = 'PASS'
                write(session_out / ('call_%02d_end.json' % row['position']), result)
                all_results.append(result)
                append(out / 'results.jsonl', result)
                print(json.dumps({'call': row['call_index'], 'program': program, 'phase': row['phase'],
                                  'seed': row['seed'], 'status': 'PASS'}), flush=True)
            write(session_out / 'final_environment.json', board.environment(boot, expected_files))
            # Release every RPC-backed object before explicit transport shutdown.
            # Loop variable buffer also retains the last NDArray after zip().
            buffer = None
            buffers, retained, module = (), [], None
            timer = diagnostic = metadata = counts = reset = None
            clear = get_ledger = collection = storage_query = get = None
            gc.collect()
            board.close_connection(remote, session_out)
            remote = None
            gc.collect()
            board.stop()
            write(session_out / 'stopped_sockets.json', board.sockets())
            (session_out / 'rpc.log').write_text(board.ssh('cat ' + shlex.quote(board.session + '/rpc.log')))
            board.session = None
        verify_bindings(bindings)
        write(out / 'final_state.json', board.preflight(boot))
        if begun != 120 or len(all_results) != 120:
            raise ValueError('TARGET_CALL_BUDGET_NOT_COMPLETE')
        summary = {'status': 'COMPLETE_SUCCESS', 'begun_calls': begun, 'passed_calls': len(all_results),
                   'formal_samples': sum(r['timed'] for r in all_results), 'sessions': 20,
                   'boot_id': boot, 'call_accounting': accounting(out), 'paused_after_attempt': True}
        write(out / 'summary.json', summary)
        return summary
    except BaseException as error:
        failure = {'status': 'STOP_ATTEMPT_FAILED', 'error_type': type(error).__name__, 'error': str(error),
                   'registered_attempts': begun, 'passed_calls': len(all_results),
                   'call_accounting': accounting(out), 'performance_conclusion_allowed': False,
                   'no_retry_or_sample_splicing': True, 'paused_after_attempt': True}
        write(out / 'summary.json', failure)
        raise
    finally:
        if board.session is not None:
            try:
                board.stop()
                if session_out is not None and not (session_out / 'rpc.log').exists():
                    (session_out / 'rpc.log').write_text(board.ssh('cat ' + shlex.quote(board.session + '/rpc.log')))
            except Exception as error:
                write(out / 'cleanup_warning.json', {'error': str(error)})
