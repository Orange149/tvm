import unittest
import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from c3_yolo416_comparison_protocol import manifest
from c3_yolo416_board_transport import transport_contract,ComparisonBoard
from test_c3_yolo416_comparison_protocol import pool


class TransportTests(unittest.TestCase):
    def test_full_pool_uses_240_unique_frozen_ports(self):
        m=manifest('attempt',pool());t=transport_contract(m)
        self.assertEqual(len(t['sessions']),240)
        self.assertEqual([r['port'] for r in t['sessions']],list(range(18000,18240)))
        self.assertEqual([r['session_id'] for r in t['sessions']],[r['session_id'] for r in m['sessions']])
        self.assertEqual(t['target_retries'],0);self.assertEqual(t['rpc_connect_attempts_per_session'],1)

    def test_modified_ledger_and_unfrozen_port_range_reject(self):
        m=manifest('a',pool(0))
        for port in (True,0,65000):
            with self.assertRaises(ValueError):transport_contract(m,port)
        m['sessions'][0]['target_calls']=7
        with self.assertRaises(ValueError):transport_contract(m)

    def test_CPU3_launch_adds_only_thread_rebinding_disable(self):
        root=Path(__file__).resolve().parent
        def method(filename,cls):
            tree=ast.parse((root/filename).read_text())
            c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
            return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='start')
        before=method('c3_tps_aggregation_transport_v2.py','TransportBoard')
        after=method('c3_yolo416_board_transport.py','ComparisonBoard')
        for n in ast.walk(after):
            if isinstance(n,ast.Dict):
                pairs=[(k,v) for k,v in zip(n.keys,n.values) if not isinstance(k,ast.Constant) or k.value!='TVM_BIND_THREADS']
                n.keys=[x[0] for x in pairs];n.values=[x[1] for x in pairs]
        self.assertEqual(ast.dump(before),ast.dump(after))
        m=manifest('a',pool(0));args=SimpleNamespace(transport_contract=transport_contract(m),reprogram=False,
                                                  host='fake',known_hosts='/fake/known_hosts')
        board=ComparisonBoard(args,'/var/volatile/fake','sha');commands=[]
        board.sockets=lambda:{'entries':[]}
        def fake_ssh(command,**kwargs):
            commands.append(command);return '777' if command.endswith('echo $!') else ''
        board.ssh=fake_ssh
        result=board.start(1,'minimal')
        self.assertEqual(result['environment']['TVM_BIND_THREADS'],'0')
        self.assertIn('TVM_BIND_THREADS=0',commands[-1]);self.assertIn('taskset -c 3',commands[-1])

    def test_transport_failure_preserves_reason_and_does_not_retry(self):
        args=SimpleNamespace(transport_contract=transport_contract(manifest('a',pool(0))),
                             host='fake',known_hosts='/fake')
        board=ComparisonBoard(args,'/var/volatile/fake','sha')
        failure=subprocess.CalledProcessError(255,['ssh'],stderr='network blocked')
        with patch('c3_tps_aggregation_transport_v2.TransportBoard.ssh',side_effect=failure) as call:
            with self.assertRaisesRegex(RuntimeError,'SSH_FAILED_255: network blocked'):board.ssh('readonly')
            self.assertEqual(call.call_count,1)


if __name__=='__main__':unittest.main()
