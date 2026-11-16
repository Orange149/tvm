"""TPS-only board executor. A frozen row permits exactly one target call.

No retries of target calls, no production runtime edits, no SD writes. Only
processes started in this attempt's tmpfs directory may be stopped.
"""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()


def verify_bindings(bindings):
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError('FROZEN_IDENTITY_CHANGED:' + path)


def submit_signature(ledger, mode):
    if (ledger.get('schema') != 'c3_official_window_submit_ledger_v1' or
            ledger.get('mode') != mode or ledger.get('overflow') is not False):
        raise ValueError('SUBMIT_LEDGER_MODE_OR_OVERFLOW')
    rows = ledger.get('submits', [])
    if not 1 <= len(rows) <= 16 or ledger.get('attempted_submits') != len(rows) or \
            ledger.get('recorded_submits') != len(rows):
        raise ValueError('SUBMIT_LEDGER_INCOMPLETE')
    for i, row in enumerate(rows, 1):
        if row.get('ordinal') != i or row.get('completion') is not True or row.get('timeout') != 0:
            raise ValueError('SUBMIT_NOT_COMPLETED')
        if any(type(row.get(k)) is not int or row[k] <= 0 for k in
               ('insns', 'insn_bytes', 'serialized_uop_bytes')) or row['serialized_uop_bytes'] % 4:
            raise ValueError('SUBMIT_SIZE_INVALID')
        if row.get('reason') not in ('explicit_sync', 'auto_sync') or \
                row.get('insn_bytes') != 16 * row.get('insns', -1):
            raise ValueError('SUBMIT_UNEXPLAINED')
    return rows


class Board:
    def __init__(self, args, root, bitstream):
        self.args, self.root, self.bitstream = args, root, bitstream
        self.session = None
        self.opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                     '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + args.known_hosts,
                     '-o', 'HostKeyAlgorithms=+ssh-rsa', '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa']

    def ssh(self, command, timeout=30):
        return subprocess.run(['ssh', *self.opts, 'root@' + self.args.host, command],
                              check=True, capture_output=True, text=True, timeout=timeout).stdout

    def preflight(self, boot=None):
        # The first six lines are an intentionally fixed machine-readable prefix.
        raw = self.ssh('cat /proc/sys/kernel/random/boot_id && '
            'cat /sys/class/fpga_manager/fpga0/state && '
            'cat /sys/class/u-dma-buf/udmabuf0/size && '
            'sha256sum /lib/firmware/vta_hpc.bit && '
            'readlink -f /tmp && '
            'cat /sys/devices/system/cpu/online && '
            'command -v taskset >/dev/null && taskset -c 3 true && '
            'mount && ps && dmesg')
        lines = raw.splitlines()
        state = dict(boot_id=lines[0], fpga_state=lines[1], dma_bytes=int(lines[2]),
                     bitstream_sha256=lines[3].split()[0], tmp_realpath=lines[4], raw=raw)
        if boot is not None and state['boot_id'] != boot:
            raise ValueError('BOOT_CHANGED')
        if state['fpga_state'] != 'operating' or state['dma_bytes'] != 201326592 or \
                state['bitstream_sha256'] != self.bitstream:
            raise ValueError('FROZEN_HARDWARE_OR_192_MIB_DMA_UNAVAILABLE')
        if 'tmpfs on /var/volatile type tmpfs' not in raw:
            raise ValueError('DESTINATION_NOT_TMPFS')
        # std::tmpfile in the existing diagnostic helper uses /tmp.
        if not (state['tmp_realpath'].startswith('/var/volatile/') or
                'tmpfs on /tmp type tmpfs' in raw):
            raise ValueError('DIAGNOSTIC_TMPFILE_NOT_PROVEN_RAM')
        if re.search(r'^\s*\d+.*(?:/|\s)tvm_rpc(?:\s|$)', raw, re.M):
            raise ValueError('EXTERNAL_RPC_PRESENT_DO_NOT_INTERRUPT')
        if re.search(r'EXT4-fs.*(?:error|abort)|mmc.*(?:I/O error|timeout|error -)', raw, re.I):
            raise ValueError('BOARD_FILESYSTEM_ERROR_REQUIRES_DIAGNOSIS')
        return state

    def deploy(self, artifacts):
        root = shlex.quote(self.root)
        self.ssh('test ! -e ' + root + ' && mkdir ' + root)
        paths = {name: artifacts[key] for name, key in
                 [('tvm_rpc', 'rpc_server'), ('libtvm_runtime.so', 'runtime_tvm'), ('libvta.so', 'runtime')]}
        for name, item in paths.items():
            if sha(item['path']) != item['sha256']:
                raise ValueError('DEPLOY_IDENTITY_CHANGED:' + name)
            subprocess.run(['scp', '-O', *self.opts, item['path'],
                            'root@' + self.args.host + ':' + self.root + '/' + name],
                           capture_output=True, text=True, check=True, timeout=60)
        observed = self.ssh('cd ' + root + ' && sha256sum tvm_rpc libtvm_runtime.so libvta.so')
        actual = {line.split()[1]: line.split()[0] for line in observed.splitlines()}
        if actual != {name: item['sha256'] for name, item in paths.items()}:
            raise ValueError('REMOTE_RUNTIME_HASH_MISMATCH')
        return actual

    def start(self, ordinal, mode):
        self.session = self.root + '/session_%02d' % ordinal
        path, root = shlex.quote(self.session), shlex.quote(self.root)
        self.ssh('test ! -e ' + path + ' && mkdir ' + path)
        self.ssh('test "$(sha256sum /lib/firmware/vta_hpc.bit | cut -d\' \' -f1)" = ' +
                 shlex.quote(self.bitstream) +
                 ' && echo vta_hpc.bit > /sys/class/fpga_manager/fpga0/firmware' +
                 ' && test "$(cat /sys/class/fpga_manager/fpga0/state)" = operating')
        env = {'LD_LIBRARY_PATH': self.root, 'LD_PRELOAD': self.root + '/libvta.so',
               'TMPDIR': self.session, 'TVM_NUM_THREADS': '1', 'VTA_REPLAY_POLICY': 'disabled',
               'VTA_INSN_BUFFER_BYTES': '33554432', 'VTA_UOP_BUFFER_BYTES': '33554432',
               'VTA_INSN_SUBMIT_THRESHOLD_BYTES': '33554432', 'VTA_QUEUE_DIAGNOSTICS': '0',
               'VTA_RUNTIME_TRACE': '0', 'VTA_QUEUE_BOUNDARY_AUDIT': '0', 'VTA_QUEUE_DRAIN_AUDIT': '0',
               'C3_OFFICIAL_WINDOW_COLLECTION': mode}
        command = '(cd ' + path + ' && exec taskset -c 3 env ' + ' '.join(
            key + '=' + shlex.quote(value) for key, value in env.items()) + \
            ' ' + root + '/tvm_rpc server --host=0.0.0.0 --port=9090 --port-end=9091 --work-dir=' + path + \
            ') >' + path + '/rpc.log 2>&1 < /dev/null & echo $!'
        return {'pid': int(self.ssh(command).strip()), 'environment': env, 'workdir': self.session}

    def environment(self, boot, expected_files):
        path, exe = shlex.quote(self.session), shlex.quote(self.root + '/tvm_rpc')
        raw = self.ssh('cat /proc/sys/kernel/random/boot_id; cat /proc/loadavg; '
            'for f in /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor '
            '/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq; do '
            'if test -r "$f"; then echo "$f"; cat "$f"; fi; done; '
            'for c3_pid in $(pidof tvm_rpc); do '
            'test "$(readlink /proc/$c3_pid/cwd)" = ' + path + ' || continue; '
            'test "$(readlink /proc/$c3_pid/exe)" = ' + exe + ' || continue; '
            'echo "PID $c3_pid"; '
            'for f in /proc/$c3_pid/task/*/status; do '
            'awk \'/^Cpus_allowed_list:/ {print "AFFINITY " $2}\' "$f"; done; '
            'cat /proc/$c3_pid/maps; done')
        if raw.splitlines()[0] != boot:
            raise ValueError('BOOT_CHANGED')
        masks = [line.split()[1] for line in raw.splitlines() if line.startswith('AFFINITY ')]
        if not masks or any(mask != '3' for mask in masks):
            raise ValueError('ACTUAL_RPC_THREADS_NOT_CPU3')
        maps = {line.split()[-1] for line in raw.splitlines() if re.match(r'^[0-9a-f]+-[0-9a-f]+ ', line)}
        for name, digest in expected_files.items():
            candidates = {p for p in maps if p.endswith('/' + name)}
            if len(candidates) != 1:
                raise ValueError('ACTUAL_DSO_MAPPING_AMBIGUOUS:' + name)
            filename = candidates.pop()
            if not filename.startswith(self.root + '/'):
                raise ValueError('ACTUAL_DSO_OUTSIDE_ATTEMPT:' + filename)
            if self.ssh('sha256sum ' + shlex.quote(filename)).split()[0] != digest:
                raise ValueError('ACTUAL_DSO_HASH_MISMATCH:' + filename)
        return {'raw': raw, 'masks': masks, 'actual_mapped_sha256': expected_files}

    def stop(self):
        if self.session is None:
            return
        path, exe = shlex.quote(self.session), shlex.quote(self.root + '/tvm_rpc')
        self.ssh('for c3_pid in $(pidof tvm_rpc); do '
            'case $c3_pid in \'\'|*[!0-9]*) exit 9;; esac; '
            'if test "$(readlink /proc/$c3_pid/cwd)" = ' + path +
            ' && test "$(readlink /proc/$c3_pid/exe)" = ' + exe +
            '; then kill $c3_pid; fi; done')
        for _ in range(20):
            remaining = self.ssh('for c3_pid in $(pidof tvm_rpc); do '
                'if test "$(readlink /proc/$c3_pid/cwd)" = ' + path +
                ' && test "$(readlink /proc/$c3_pid/exe)" = ' + exe +
                '; then echo "$c3_pid"; fi; done')
            if not remaining.strip():
                return
            time.sleep(.1)
        raise ValueError('OWN_RPC_DID_NOT_EXIT_AFTER_TERM')


def make_cases(reference):
    """Independent references only: does not invoke a compiled target."""
    from c3_official_window_reference import from_workload, array_sha, output_sentinel
    from c3_tps_aggregation_protocol import SEEDS
    spec = from_workload(['conv2d_packed.vta', ['TENSOR', reference['input_shape'], 'int8'],
                         ['TENSOR', reference['weight_shape'], 'int8'], reference['strides'],
                         reference['padding'], reference['dilation'], 'NCHW1n16c', 'int32'],
                         reference['abi'], reference['postprocess'])
    if spec.contract() != reference:
        raise ValueError('REFERENCE_CONTRACT_CHANGED')
    cases = {}
    for seed in SEEDS:
        arrays = spec.arrays(seed)
        expected = spec.reference(arrays)
        arrays[-1] = output_sentinel(expected)
        cases[seed, 'W0'] = (arrays, expected)
    arrays = [value.copy() for value in cases[SEEDS[0], 'W0'][0]]
    arrays[1], expected = spec.observable_weights(arrays)
    arrays[-1] = output_sentinel(expected)
    cases[SEEDS[0], 'W1'] = (arrays, expected)
    identities = {str(seed) + ':' + weight: {'parameters': [array_sha(a) for a in values[:-1]],
                  'reference': array_sha(expected), 'sentinel': array_sha(values[-1])}
                  for (seed, weight), (values, expected) in cases.items()}
    return cases, identities


def execute(args, artifacts, ledger, out, bindings):
    import numpy as np
    import tvm
    from tvm import rpc
    from c3_official_window_reference import array_sha
    from c3_official_window_method import audit_function
    from c3_valid_window import certify_trace
    from c3_tps_window_native import certify_native
    from c3_tps_window_progress import certify_progress, LANES
    from c3_five_tradeoff_commands import decode_submissions
    from c3_official_window_workflow import native_certificate_for_storage

    out.mkdir(exist_ok=False)
    all_results, warmed, begun = [], {}, 0
    board = Board(args, '/var/volatile/c3_tps_' + out.parent.name, artifacts['bitstream']['sha256'])
    remote, session_out = None, None
    try:
        verify_bindings(bindings)
        write(out / 'contract.json', {'ledger_sha256': sha(out.parent / 'call_manifest.json'),
             'bindings': bindings, 'host': args.host, 'known_hosts_sha256': sha(args.known_hosts),
             'target_calls_planned': 180, 'no_retries': True, 'root': board.root})
        cases, case_ids = make_cases(json.loads(Path(artifacts['programs']['B']['reference_contract']).read_text()))
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
            # Readiness retries do not open a target and are explicitly recorded.
            readiness = []
            for _ in range(20):
                try:
                    remote = rpc.connect(args.host, 9090, session_timeout=1800)
                    readiness.append({'status': 'CONNECTED'})
                    break
                except Exception as error:
                    readiness.append({'status': 'NOT_READY', 'error': str(error)})
                    time.sleep(.25)
            write(session_out / 'readiness.json', readiness)
            if remote is None:
                raise RuntimeError('RPC_UNAVAILABLE_BEFORE_TARGET')
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
            reset()
            if int(counts()) != 0:
                raise ValueError('HELPER_NOT_FRESH')
            definitions = None
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
                begun += 1
                dump, elapsed = None, None
                if row['phase'].startswith('diagnostic_'):
                    dump = diagnostic(module, descriptor['symbol'], *buffers)
                else:
                    elapsed = float(timer(module, descriptor['symbol'], *buffers))
                    if not math.isfinite(elapsed) or elapsed <= 0:
                        raise ValueError('NONFINITE_OR_NONPOSITIVE_TIME')
                returned = time.time()
                submit = json.loads(get_ledger())
                meta, status = json.loads(metadata()), json.loads(collection())
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
                if dump is not None:
                    (session_out / ('native_%d.txt' % row['position'])).write_text(dump)
                    native = certify_native(events, dump, submit, definitions)
                    definitions = native['cache_definitions']
                    if certify_trace(events, ancestors=native['ancestors'])['output_terms'] != products:
                        raise ValueError('NATIVE_DEPENDENCY_SEMANTICS_CHANGED')
                    progress = [certify_progress(records, command_depths, token_depths)
                                for records in decode_submissions(dump)]
                    if any(p['status'] != 'FINITE_FIFO_PROGRESS_PROVED' for p in progress):
                        raise ValueError('FINITE_FIFO_PROGRESS_UNPROVEN')
                    write(session_out / ('native_%d.json' % row['position']), native_certificate_for_storage(native))
                    write(session_out / ('progress_%d.json' % row['position']), progress)
                    warmed[program, row['rotation'], row['position']] = signature
                elif row['phase'] in ('warmup', 'formal'):
                    key = program, row['rotation'], row['position']
                    if key not in warmed or warmed[key] != signature:
                        raise ValueError('ACTUAL_SUBMIT_PATH_DIFFERS_FROM_OWN_DIAGNOSTIC')
                else:
                    local = json.loads((Path(descriptor['qualified_directory']) / 'calls.json').read_text())
                    if signature != submit_signature(local[row['position']-1]['submit_ledger'], 'detailed'):
                        raise ValueError('QUALIFICATION_SUBMIT_PATH_CHANGED')
                result['status'] = 'PASS'
                all_results.append(result)
                append(out / 'results.jsonl', result)
                print(json.dumps({'call': row['call_index'], 'program': program, 'phase': row['phase'],
                                  'seed': row['seed'], 'status': 'PASS'}), flush=True)
            write(session_out / 'final_environment.json', board.environment(boot, expected_files))
            # Release remote objects while connection is valid, then close the server we own.
            buffers, retained, module = (), [], None
            remote = None
            board.stop()
            (session_out / 'rpc.log').write_text(board.ssh('cat ' + shlex.quote(board.session + '/rpc.log')))
            board.session = None
        verify_bindings(bindings)
        write(out / 'final_state.json', board.preflight(boot))
        if begun != 180 or len(all_results) != 180:
            raise ValueError('TARGET_CALL_BUDGET_NOT_COMPLETE')
        summary = {'status': 'COMPLETE_SUCCESS', 'begun_calls': begun, 'passed_calls': len(all_results),
                   'formal_samples': sum(r['timed'] for r in all_results), 'sessions': 30,
                   'boot_id': boot, 'paused_after_attempt': True}
        write(out / 'summary.json', summary)
        return summary
    except BaseException as error:
        failure = {'status': 'STOP_ATTEMPT_FAILED', 'error_type': type(error).__name__, 'error': str(error),
                   'begun_calls': begun, 'passed_calls': len(all_results), 'performance_conclusion_allowed': False,
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
