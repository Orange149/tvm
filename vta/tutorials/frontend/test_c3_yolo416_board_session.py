"""Six-call control-flow integration with CPU fakes; no network or DNN calls."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from c3_tps_sync_release_local import sha
from c3_yolo416_comparison_protocol import manifest
from c3_yolo416_board_session import run
from c3_yolo416_board_execute import accounting
from test_c3_yolo416_comparison_protocol import pool


class FakeState:
    def __init__(self,fail_at=None):self.started=self.completed=0;self.fail_at=fail_at;self.stops=0;self.issued=False
    def metadata(self):
        return json.dumps({'schema':'c3_official_window_once_helper_v1','started_calls':self.started,
            'completed_calls':self.completed,'last_kind':'timer','packed_calls_per_invocation':1,
            'parameter_count':4,'adds_synchronize':False})
    def submit(self):
        if not self.issued:
            return json.dumps({'schema':'c3_official_window_submit_ledger_v1','mode':'detailed','overflow':False,
                               'attempted_submits':0,'recorded_submits':0,'submits':[]})
        return json.dumps({'schema':'c3_official_window_submit_ledger_v1','mode':'detailed','overflow':False,
            'attempted_submits':1,'recorded_submits':1,'submits':[{'ordinal':1,'completion':True,'timeout':0,
                'insns':10,'insn_bytes':160,'serialized_uop_bytes':16,'reason':'explicit_sync'}]})


class FakeRemote:
    def __init__(self,state):self.state=state
    def get_function(self,name):
        def result():
            if name.endswith('submit_ledger_clear'):self.state.issued=False;return None
            if name.endswith('submit_ledger'):return self.state.submit()
            if name.endswith('collection_status'):return json.dumps({'mode':'detailed','debug_flag':0,'detailed_clock_reads':0})
            if name.endswith('replay_status'):return '{"mode":"disabled","template_count":0}'
            if name.endswith('queue_capacity_status'):return json.dumps({'insn_capacity_bytes':33554432,
                'uop_capacity_bytes':33554432,'insn_submit_threshold_bytes':33554432,
                'submissions':0,'replay_policy':'disabled'})
            raise AssertionError(name)
        return result


def fake_classes(state):
    class Board:
        def __init__(self,args,root,bitstream):self.root=root;self.session=None
        def preflight(self,boot):return {'boot_id':boot}
        def start(self,ordinal,mode):self.session=self.root+'/session_%02d'%ordinal;return {'fake':True}
        def connect(self,out):return FakeRemote(state)
        def environment(self,boot,files):return {'boot_id':boot,'masks':['3']}
        def close_connection(self,remote,out):pass
        def stop(self):state.stops+=1
        def sockets(self):return {'entries':[]}
        def ssh(self,*args,**kwargs):return 'CPU fake log'
    class Adapter:
        def __init__(self,*args,**kwargs):
            self.expected_files={};self.counts=lambda:state.completed;self.metadata=state.metadata
        def prepare_call(self,row):return {'fake':True}
        def invoke(self,diagnostic):
            if diagnostic:raise AssertionError('wrong fake phase')
            state.started+=1
            state.issued=True
            if state.started==state.fail_at:raise RuntimeError('deliberate fake helper failure')
            state.completed+=1;return 1.0
        def check(self,row):return {'correct':True}
        def local_signature(self,position):return json.loads(state.submit())
        def release(self):pass
    return Board,Adapter


class SessionIntegrationTests(unittest.TestCase):
    def fixture(self,root):
        ledger=manifest('fake',pool(1));key=ledger['calls'][0]['program_key']
        def put(path,value):path.write_text(json.dumps(value))
        put(root/'freeze.json',{'status':'TH_P_OPERATORS_AND_GRAPH_READY_FPGA_PENDING','bindings':{},
                                'host':'fake','known_hosts':'/fake'})
        put(root/'call_manifest.json',ledger);put(root/'transport_contract.json',{})
        put(root/'hardware.json',{'fifo_depths':{k:1024 for k in
            ('load_queue','gemm_queue','store_queue','l2g_queue','g2l_queue','g2s_queue','s2g_queue')}})
        put(root/'artifacts.json',{'bitstream':{'sha256':'b'},'runtime':{'sha256':'r'},
            'runtime_tvm':{'sha256':'t'},'hardware_archive':{'path':str(root/'hardware.json')},
            'programs':{key:{'kind':'operator'}}})
        (root/'board').mkdir();put(root/'board/preflight.json',{'boot_id':'fake_boot'})

    def test_six_frozen_rows_invoke_exactly_six_cpu_fakes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);state=FakeState();board,adapter=fake_classes(state)
            with patch('c3_yolo416_board_session.ComparisonBoard',board),patch(
                    'c3_yolo416_operator_board_adapter.OperatorAdapter',adapter):
                self.assertEqual(run(root,1,sha(root/'freeze.json')),0)
            self.assertEqual((state.started,state.completed,state.stops),(6,6,1))
            account=accounting(root/'board')
            self.assertEqual(account['registered_attempts'],6);self.assertEqual(account['certified'],6)
            self.assertEqual(account['helper_completed_observed_lower_bound'],6)

    def test_first_fake_failure_stops_without_retry_or_later_row(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);state=FakeState(fail_at=3);board,adapter=fake_classes(state)
            with patch('c3_yolo416_board_session.ComparisonBoard',board),patch(
                    'c3_yolo416_operator_board_adapter.OperatorAdapter',adapter):
                with self.assertRaises(Exception):run(root,1,sha(root/'freeze.json'))
            self.assertEqual((state.started,state.completed,state.stops),(3,2,1))
            account=accounting(root/'board')
            self.assertEqual(account['registered_attempts'],3);self.assertEqual(account['certified'],2)
            self.assertEqual(account['helper_started_observed_lower_bound'],3)
            self.assertEqual(account['helper_completed_observed_lower_bound'],2)


if __name__=='__main__':unittest.main()
