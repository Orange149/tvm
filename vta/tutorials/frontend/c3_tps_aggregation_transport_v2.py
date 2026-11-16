"""Fixed one-port-per-session transport; leaves sealed V1 and target code intact."""
import json
from pathlib import Path
import shlex
import time

from c3_tps_aggregation_board import Board, write


def transport_contract(ledger, base_port=9100):
    if type(base_port) is not int or not 1024 <= base_port <= 32000:
        raise ValueError('PORT_RANGE_OUTSIDE_FROZEN_SUPPORT')
    sessions = ledger['sessions']
    if len(sessions) != 30 or len({s['session_id'] for s in sessions}) != 30:
        raise ValueError('TRANSPORT_REQUIRES_30_UNIQUE_SESSIONS')
    return {'schema': 'c3_tps_aggregation_transport_v2', 'base_port': base_port,
            'sessions': [{'ordinal': i, 'session_id': s['session_id'], 'port': base_port+i-1}
                         for i, s in enumerate(sessions, 1)],
            'rpc_connect_attempts_per_session': 1, 'target_retries': 0,
            'port_reuse_within_attempt': False,
            'shutdown': 'release remote arrays/modules/functions, explicit CloseRPCConnection, then stop owned server',
            'listener_identity': 'socket inode held by process with this session cwd and frozen executable'}


def parse_sockets(raw):
    rows = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < 10 or not fields[0].rstrip(':').isdigit():
            continue
        try:
            rows.append({'local': fields[1], 'port': int(fields[1].rsplit(':', 1)[1], 16),
                         'remote': fields[2], 'state': fields[3], 'inode': int(fields[9])})
        except (ValueError, IndexError) as error:
            raise ValueError('SOCKET_TABLE_UNPARSEABLE') from error
    return rows


class TransportBoard(Board):
    def __init__(self, args, root, bitstream):
        super().__init__(args, root, bitstream)
        self.transport = args.transport_contract
        self.port = None
        self.reprogram = getattr(args, 'reprogram', True)

    def sockets(self):
        raw = self.ssh('cat /proc/net/tcp /proc/net/tcp6')
        return {'raw': raw, 'entries': parse_sockets(raw)}

    def owned_socket_inodes(self):
        path, exe = shlex.quote(self.session), shlex.quote(self.root + '/tvm_rpc')
        raw = self.ssh('for c3_pid in $(pidof tvm_rpc); do '
            'test "$(readlink /proc/$c3_pid/cwd)" = ' + path + ' || continue; '
            'test "$(readlink /proc/$c3_pid/exe)" = ' + exe + ' || continue; '
            'for c3_fd in /proc/$c3_pid/fd/*; do '
            'c3_link=$(readlink "$c3_fd") || continue; '
            'case "$c3_link" in socket:*) echo "$c3_pid $c3_link";; esac; done; done')
        import re
        return {'raw': raw, 'inodes': [int(v) for v in re.findall(r'socket:\[(\d+)\]', raw)]}

    def start(self, ordinal, mode):
        spec = self.transport['sessions'][ordinal-1]
        if spec['ordinal'] != ordinal or mode not in ('detailed', 'minimal'):
            raise ValueError('TRANSPORT_SESSION_MISMATCH')
        self.port = spec['port']
        before = self.sockets()
        if any(row['port'] == self.port for row in before['entries']):
            raise ValueError('FROZEN_PORT_ALREADY_HAS_SOCKET_NO_FALLBACK')
        self.session = self.root + '/session_%02d' % ordinal
        path, root = shlex.quote(self.session), shlex.quote(self.root)
        self.ssh('test ! -e ' + path + ' && mkdir ' + path)
        if self.reprogram:
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
            k + '=' + shlex.quote(v) for k, v in env.items()) + ' ' + root + \
            '/tvm_rpc server --host=0.0.0.0 --port=' + str(self.port) + \
            ' --port-end=' + str(self.port+1) + ' --work-dir=' + path + \
            ') >' + path + '/rpc.log 2>&1 < /dev/null & echo $!'
        return {'pid': int(self.ssh(command).strip()), 'environment': env,
                'workdir': self.session, 'port': self.port, 'socket_state_before': before}

    def connect(self, output):
        from tvm import rpc
        observations = []
        deadline = time.monotonic() + 10
        while True:
            snapshot = self.sockets()
            active = [r for r in snapshot['entries'] if r['port'] == self.port and r['state'] == '0A']
            observations.append({'unix': time.time(), 'socket_state': snapshot})
            if active:
                owners = self.owned_socket_inodes()
                observations[-1]['owned_socket_inodes'] = owners
                if len(active) != 1 or active[0]['inode'] not in owners['inodes']:
                    write(output / 'readiness.json', observations)
                    raise ValueError('LISTENER_NOT_UNIQUELY_OWNED')
                break
            if time.monotonic() >= deadline:
                write(output / 'readiness.json', observations)
                raise RuntimeError('OWN_LISTENER_NOT_READY_NO_RPC_RETRY')
            time.sleep(.1)
        write(output / 'readiness.json', observations)
        # Exactly one connection attempt after proving our listener is ready.
        remote = rpc.connect(self.args.host, self.port, session_timeout=1800)
        write(output / 'connection.json', {'port': self.port, 'connect_attempts': 1,
                                          'status': 'CONNECTED', 'target_calls': 0})
        return remote

    def close_connection(self, remote, output):
        remote._remote_funcs.clear()
        remote._sess.get_function('CloseRPCConnection')()
        # This call is a transport shutdown, not a target function or VTA sync.
        deadline = time.monotonic() + 10
        while True:
            log = self.ssh('cat ' + shlex.quote(self.session + '/rpc.log'))
            if 'Finish serving ' in log and 'Socket Connection Closed' in log:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('RPC_CLEAN_CLOSE_NOT_ACKNOWLEDGED')
            time.sleep(.1)
        write(output / 'connection_closed.json', {'status': 'EXPLICIT_SHUTDOWN_ACKNOWLEDGED',
              'port': self.port, 'target_calls': 0, 'socket_state': self.sockets()})


def smoke(args, artifacts, ledger, output):
    """Thirty consecutive sessions with upload/download only, no VTA calls."""
    output.mkdir(exist_ok=False)
    from types import SimpleNamespace
    import gc
    contract = transport_contract(ledger, 9200)
    board = TransportBoard(SimpleNamespace(**vars(args), transport_contract=contract, reprogram=False),
                           '/var/volatile/c3_tps_transport_' + output.name, artifacts['bitstream']['sha256'])
    remote = None
    passed = 0
    try:
        write(output / 'transport_contract.json', contract)
        initial = board.preflight()
        write(output / 'preflight.json', initial)
        write(output / 'deployment.json', board.deploy(artifacts))
        for ordinal in range(1, 31):
            folder = output / ('session_%02d' % ordinal)
            folder.mkdir()
            write(folder / 'start.json', board.start(ordinal, 'detailed' if ordinal <= 12 else 'minimal'))
            remote = board.connect(folder)
            token = ('session-%02d-no-target' % ordinal).encode()
            remote.upload(bytearray(token), target='transport_probe.txt')
            if bytes(remote.download('transport_probe.txt')) != token:
                raise ValueError('TRANSPORT_SESSION_TOKEN_MISMATCH')
            write(folder / 'environment.json', board.environment(initial['boot_id'], {
                'libvta.so': artifacts['runtime']['sha256'], 'libtvm_runtime.so': artifacts['runtime_tvm']['sha256']}))
            board.close_connection(remote, folder)
            remote = None
            gc.collect()
            board.stop()
            (folder / 'rpc.log').write_text(board.ssh('cat ' + shlex.quote(board.session + '/rpc.log')))
            write(folder / 'stopped_sockets.json', board.sockets())
            board.session = None
            passed += 1
            print(json.dumps({'transport_session': ordinal, 'status': 'PASS', 'target_calls': 0}), flush=True)
        write(output / 'final_state.json', board.preflight(initial['boot_id']))
        result = {'status': 'THIRTY_TRANSPORT_SESSIONS_PASSED', 'sessions': passed,
                  'target_calls': 0, 'FPGA_reprogrammed': False, 'boot_id': initial['boot_id']}
        write(output / 'summary.json', result)
        return result
    except BaseException as error:
        write(output / 'summary.json', {'status': 'TRANSPORT_SMOKE_FAILED', 'sessions': passed,
                                        'target_calls': 0, 'error': str(error)})
        raise
    finally:
        if board.session is not None:
            board.stop()
