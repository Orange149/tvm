"""One frozen TCP port per YOLO416 comparison unit; no board action on import."""
import shlex
import subprocess
from c3_tps_aggregation_transport_v2 import TransportBoard
from c3_yolo416_comparison_protocol import manifest,canonical,require


def transport_contract(ledger,base_port=18000):
    require(type(base_port) is int and 1024<=base_port<=32000,'INVALID_FROZEN_PORT_BASE')
    require(canonical(ledger)==canonical(manifest(ledger['attempt'],ledger['pool'])),
            'INVALID_COMPARISON_LEDGER')
    sessions=ledger['sessions']
    require(len(sessions)<=240 and base_port+len(sessions)<65536,'SESSION_OR_PORT_LIMIT')
    require(len({s['session_id'] for s in sessions})==len(sessions),'DUPLICATE_SESSION')
    return {'schema':'c3_yolo416_comparison_transport_v1','base_port':base_port,
        'sessions':[{'ordinal':i,'session_id':s['session_id'],'port':base_port+i-1}
                    for i,s in enumerate(sessions,1)],
        'rpc_connect_attempts_per_session':1,'target_retries':0,'port_reuse_within_attempt':False,
        'shutdown':'release every RPC-backed reference, close connection explicitly, stop owned server only'}


class ComparisonBoard(TransportBoard):
    def ssh(self,command,timeout=30):
        try:return super().ssh(command,timeout=timeout)
        except subprocess.CalledProcessError as error:
            raise RuntimeError('SSH_FAILED_%d: %s'%(error.returncode,(error.stderr or '').strip())) from error

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
               'TMPDIR': self.session, 'TVM_NUM_THREADS': '1', 'TVM_BIND_THREADS': '0', 'VTA_REPLAY_POLICY': 'disabled',
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


    def preflight(self,boot=None):
        state=super().preflight(boot)
        # Check actual open mappings as well as the inherited tvm_rpc process
        # check. Merely seeing no RPC process does not prove VTA is unused.
        raw=self.ssh('for c3_proc in /proc/[0-9]*; do '
            'test -r "$c3_proc/maps" || continue; '
            'c3_mapped=$(awk \'/\\/dev\\/(u-dma-buf\\/udmabuf[0-9]+|udmabuf[0-9]+|uio[0-9]+|mem)([[:space:]]|$)/ '
            '{print $NF}\' "$c3_proc/maps" 2>/dev/null); '
            'for c3_fd in "$c3_proc"/fd/*; do '
            'c3_link=$(readlink "$c3_fd" 2>/dev/null) || continue; '
            'case "$c3_link" in /dev/u-dma-buf/*|/dev/udmabuf*|/dev/uio[0-9]*|/dev/mem) '
            'c3_mapped="$c3_mapped $c3_link";; esac; done; '
            'if test -n "$c3_mapped"; then '
            'echo "$c3_proc $c3_mapped"; fi; done',timeout=30)
        if raw.strip():raise ValueError('EXTERNAL_ACCELERATOR_MAPPING_PRESENT_DO_NOT_INTERRUPT:'+raw)
        state['accelerator_occupancy_scan']=raw
        return state
