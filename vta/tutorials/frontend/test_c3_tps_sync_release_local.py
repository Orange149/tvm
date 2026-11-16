"""Budget/identity checks only; these tests never execute a fused target."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from c3_tps_sync_release_local import (
    HELPER, RUNTIME, call_accounting, capture_helper_call, initialize_helper,
    ledger, read, sha, validate_calls, validate_helper_completion, write,
)


class CallContract(unittest.TestCase):
    def passed(self):
        return [{**r,'status':'PASS','correct':True} for r in ledger()]

    def test_exact_budget_and_order(self):
        planned=ledger()
        self.assertEqual(len(planned),28)
        self.assertEqual([(r['program'],r['mode']) for r in planned[::7]],
                         [('R','detailed'),('R','minimal'),('S','detailed'),('S','minimal')])
        self.assertTrue(validate_calls(planned,self.passed()))

    def test_missing_and_extra_rejected(self):
        for rows in (self.passed()[:-1],self.passed()+self.passed()[:1]):
            with self.assertRaisesRegex(ValueError,'MISSING_OR_EXTRA_CALL'):
                validate_calls(ledger(),rows)

    def test_duplicate_reordered_rejected(self):
        rows=self.passed();rows[1]=copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'CALL_IDENTITY_OR_ORDER_CHANGED'):
            validate_calls(ledger(),rows)

    def test_identity_and_hidden_call_rejected(self):
        for key,value in (('program','O'),('mode','other'),('target_calls',2),('seed',1)):
            rows=self.passed();rows[0][key]=value
            with self.assertRaisesRegex(ValueError,'CALL_IDENTITY_OR_ORDER_CHANGED'):
                validate_calls(ledger(),rows)

    def test_failed_output_not_qualification(self):
        rows=self.passed();rows[0]['correct']=False
        with self.assertRaisesRegex(ValueError,'CALL_NOT_PASSED'):
            validate_calls(ledger(),rows)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);worker=root/'R_detailed';worker.mkdir()
            self.assertEqual(call_accounting(root)['started_calls']['exact'],0)
            write(worker/'call_01_begin.json',ledger()[0])
            counts=call_accounting(root)
            self.assertIsNone(counts['started_calls']['exact'])
            self.assertEqual(counts['started_calls']['upper_bound'],1)
            self.assertEqual(counts['fully_certified'],0)
            write(worker/'call_01_execution.json',{'metadata':{'started_calls':1,'completed_calls':0},'counts':0})
            counts=call_accounting(root)
            self.assertEqual(counts['started_calls']['exact'],1)
            self.assertEqual(counts['completed_calls']['exact'],0)
            self.assertEqual(counts['fully_certified'],0)

    @staticmethod
    def metadata(index):
        return {'schema':'c3_official_window_once_helper_v1','started_calls':index,
                'completed_calls':index,'parameter_count':4,'last_kind':'diagnostic',
                'adds_synchronize':False,'packed_calls_per_invocation':1}

    def test_frozen_helper_reset_contract_without_target_call(self):
        os.environ['TVM_LIBRARY_PATH']=str(RUNTIME)
        import tvm
        from vta.testing import simulator
        from unittest.mock import Mock

        self.assertEqual(Path(simulator.LIBS[0]._name).resolve(),(RUNTIME/'libvta_fsim.so').resolve())
        build=read(HELPER.with_name('build.json'))
        self.assertEqual(sha(HELPER),build['binary_sha256'])
        self.assertEqual(sha(build['source']),build['source_sha256'])
        helper=tvm.runtime.load_module(str(HELPER))
        # Only these three non-target entry points are retrieved or called.
        reset=Mock(wraps=tvm.get_global_func('c3.official_window.reset_counts'))
        counts=Mock(wraps=tvm.get_global_func('c3.official_window.counts'))
        metadata=Mock(wraps=tvm.get_global_func('c3.official_window.metadata'))
        observed=[]
        state=initialize_helper(reset,counts,metadata,observed.append)
        self.assertEqual(state,{'metadata':{},'counts':0})
        self.assertEqual(observed,[state])
        reset.assert_called_once_with();counts.assert_called_once_with();metadata.assert_called_once_with()
        self.assertIsNotNone(helper)

    def test_initial_state_rejects_nonfresh_or_malformed_metadata(self):
        for count,meta in ((1,'{}'),(0,json.dumps(self.metadata(0))),(0,'[]')):
            observed=[]
            with self.assertRaisesRegex(ValueError,'HELPER_NOT_FRESH'):
                initialize_helper(lambda:None,lambda:count,lambda:meta,observed.append)
            self.assertEqual(len(observed),1)
        with self.assertRaises(json.JSONDecodeError):
            initialize_helper(lambda:None,lambda:0,lambda:'invalid')

    def test_first_exception_empty_metadata_keeps_started_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);worker=root/'R_detailed';worker.mkdir()
            write(worker/'call_01_begin.json',ledger()[0])
            attempts=[]
            def fail():
                attempts.append(1)
                raise RuntimeError('synthetic target exception')
            with self.assertRaisesRegex(RuntimeError,'synthetic target exception'):
                capture_helper_call(1,fail,lambda:0,lambda:'{}',lambda:'{}',
                    lambda e:write(worker/'call_01_execution.json',e))
            self.assertEqual(attempts,[1])
            execution=read(worker/'call_01_execution.json')
            self.assertFalse(execution['helper_returned'])
            self.assertEqual(execution['metadata'],{})
            account=call_accounting(root)
            self.assertEqual(account['started_calls'],{'exact':None,'lower_bound':0,'upper_bound':1})
            self.assertEqual(account['completed_calls']['exact'],0)
            self.assertEqual(account['fully_certified'],0)

    def test_later_exception_stale_metadata_keeps_started_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);worker=root/'R_minimal';worker.mkdir()
            write(worker/'call_01_begin.json',ledger()[7])
            _,execution=capture_helper_call(1,lambda:'synthetic dump',lambda:1,
                lambda:json.dumps(self.metadata(1)),lambda:'{}',
                lambda e:write(worker/'call_01_execution.json',e))
            self.assertTrue(validate_helper_completion(execution,1))
            write(worker/'call_02_begin.json',ledger()[8])
            def fail():raise RuntimeError('second synthetic exception')
            with self.assertRaisesRegex(RuntimeError,'second synthetic exception'):
                capture_helper_call(2,fail,lambda:1,lambda:json.dumps(self.metadata(1)),lambda:'{}',
                    lambda e:write(worker/'call_02_execution.json',e))
            account=call_accounting(root)
            self.assertEqual(account['started_calls'],{'exact':None,'lower_bound':1,'upper_bound':2})
            self.assertEqual(account['completed_calls']['exact'],1)
            self.assertEqual(account['fully_certified'],0)
            # No snapshot at all after another begin remains unknown for both counters.
            write(worker/'call_03_begin.json',ledger()[9])
            account=call_accounting(root)
            self.assertEqual(account['completed_calls'],{'exact':None,'lower_bound':1,'upper_bound':3})

    def test_capture_preserves_invocation_error_when_observation_fails(self):
        records=[]
        def fail():raise RuntimeError('original exception')
        with self.assertRaisesRegex(RuntimeError,'original exception'):
            capture_helper_call(1,fail,lambda:0,lambda:'invalid',lambda:'invalid',records.append)
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['counts'],0)
        self.assertEqual(set(records[0]['observation_errors']),{'metadata','submits'})
        self.assertEqual(records[0]['invocation_error']['message'],'original exception')

    def test_success_requires_fresh_complete_metadata(self):
        records=[];invocations=[]
        def success():
            invocations.append(1)
            return 'synthetic dump'
        dump,execution=capture_helper_call(1,success,lambda:1,
            lambda:json.dumps(self.metadata(1)),lambda:'{}',records.append)
        self.assertEqual(invocations,[1]);self.assertEqual(dump,'synthetic dump')
        self.assertEqual(records,[execution]);self.assertTrue(validate_helper_completion(execution,1))
        for meta in ({},{'started_calls':1},self.metadata(0),self.metadata(2)):
            with self.assertRaises(ValueError):validate_helper_completion({**execution,'metadata':meta},1)
        with self.assertRaisesRegex(ValueError,'HELPER_DID_NOT_RETURN'):
            validate_helper_completion({**execution,'helper_returned':False},1)
        with self.assertRaisesRegex(ValueError,'HELPER_OBSERVATION_FAILED'):
            validate_helper_completion({**execution,'observation_errors':{'submits':'failed'}},1)

    def test_unexpected_counter_has_no_budget_upper_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);worker=root/'R_detailed';worker.mkdir()
            write(worker/'call_01_begin.json',ledger()[0])
            write(worker/'call_01_execution.json',{'metadata':self.metadata(2),'counts':2})
            account=call_accounting(root)
            self.assertEqual(account['started_calls'],{'exact':None,'lower_bound':2,'upper_bound':None})
            self.assertIsNone(account['completed_calls']['exact'])
            self.assertTrue(account['units'][0]['evidence_issues'])


if __name__=='__main__':unittest.main()
