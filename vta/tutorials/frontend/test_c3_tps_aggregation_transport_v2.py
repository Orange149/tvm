"""Transport lifecycle regression checks without a device or target call."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from c3_tps_aggregation_protocol import manifest
from c3_tps_aggregation_transport_v2 import transport_contract, parse_sockets, TransportBoard


class TransportTests(unittest.TestCase):
    def contract(self):
        return transport_contract(manifest('test', {p: {'id': p} for p in 'BNG'}))

    def instance(self):
        args = SimpleNamespace(host='invalid', known_hosts='/tmp/no-host', transport_contract=self.contract())
        return TransportBoard(args, '/var/volatile/test', 'f'*64)

    def test_thirty_unique_ports_with_same_frozen_session_order(self):
        contract = self.contract()
        self.assertEqual([s['port'] for s in contract['sessions']], list(range(9100, 9130)))
        self.assertEqual(len({s['session_id'] for s in contract['sessions']}), 30)
        self.assertEqual(contract['rpc_connect_attempts_per_session'], 1)
        smoke = transport_contract(manifest('test', {p: {} for p in 'BNG'}), 9200)
        self.assertFalse({s['port'] for s in smoke['sessions']} & {s['port'] for s in contract['sessions']})

    def test_ipv4_ipv6_and_timewait_are_parsed(self):
        rows = parse_sockets('sl local_address rem_address st tx_queue rx_queue tr tm retr uid inode\n'
            '0: 00000000:238C 00000000:0000 0A 0:0 00:0 0 0 0 123\n'
            '1: 00000000000000000000000000000000:238D 0:0000 06 0:0 00:0 0 0 0 0')
        self.assertEqual([(r['port'], r['state'], r['inode']) for r in rows], [(9100,'0A',123),(9101,'06',0)])

    def test_existing_port_refused_without_start_or_fallback(self):
        board = self.instance()
        board.sockets = Mock(return_value={'entries': [{'port':9100,'state':'06'}]})
        board.ssh = Mock()
        with self.assertRaisesRegex(ValueError, 'FROZEN_PORT_ALREADY'):
            board.start(1, 'detailed')
        board.ssh.assert_not_called()

    def test_start_binds_only_its_frozen_port(self):
        board = self.instance()
        board.sockets = Mock(return_value={'entries': []})
        board.ssh = Mock(side_effect=['','','123'])
        info = board.start(2, 'minimal')
        command = board.ssh.call_args.args[0]
        self.assertIn('--port=9101 --port-end=9102', command)
        self.assertNotIn('--port=9090', command)
        self.assertEqual(info['port'], 9101)

    def test_foreign_listener_rejected_before_rpc_connect(self):
        board = self.instance()
        board.port = 9100
        board.sockets = Mock(return_value={'entries':[{'port':9100,'state':'0A','inode':3}]})
        board.owned_socket_inodes = Mock(return_value={'inodes':[4]})
        rpc = SimpleNamespace(connect=Mock())
        with TemporaryDirectory() as temp, patch.dict('sys.modules', {'tvm': SimpleNamespace(rpc=rpc)}):
            with self.assertRaisesRegex(ValueError, 'LISTENER_NOT_UNIQUELY_OWNED'):
                board.connect(Path(temp))
        rpc.connect.assert_not_called()

    def test_explicit_close_before_owned_server_stop(self):
        board = self.instance()
        board.session, board.port = '/var/volatile/test/session_01', 9100
        board.ssh = Mock(return_value='Finish serving client\nSocket Connection Closed')
        board.sockets = Mock(return_value={'entries': []})
        events = []
        remote = SimpleNamespace(_remote_funcs=SimpleNamespace(clear=lambda: events.append('clear')),
            _sess=SimpleNamespace(get_function=lambda name: lambda: events.append(name)))
        with TemporaryDirectory() as temp:
            board.close_connection(remote, Path(temp))
        self.assertEqual(events, ['clear','CloseRPCConnection'])
        self.assertNotIn('kill ', board.ssh.call_args.args[0])

    def test_target_call_loop_unchanged_from_frozen_v1(self):
        here = Path(__file__).parent
        def target_loop(path):
            tree = ast.parse(path.read_text())
            return next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                        and isinstance(node.target, ast.Name) and node.target.id == 'row'
                        and isinstance(node.iter, ast.Name) and node.iter.id == 'rows')
        self.assertEqual(ast.dump(target_loop(here/'c3_tps_aggregation_board.py')),
                         ast.dump(target_loop(here/'c3_tps_aggregation_board_v2.py')))


if __name__ == '__main__':
    unittest.main()
