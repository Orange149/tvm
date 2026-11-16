import json
from copy import deepcopy
from collections import Counter
from types import SimpleNamespace
from unittest import TestCase,main
from unittest.mock import patch
from c3_tps_cost_protocol import manifest,check_manifest,transport,analyse,validate_results
from c3_tps_cost_probe import split_boundary
from c3_tps_cost_transport import CostBoard,TransportBoard


class ProtocolTests(TestCase):
    def setUp(self):
        self.ledger=manifest('test',{'off':{'sha':'same'},'on':{'sha':'same'}})
        self.rows=[]
        for row in self.ledger['calls']:
            h=dict(begin_ns=1000000,end_ns=4000000,started_calls=row['position'],completed_calls=row['position'])
            d=dict(mode=row['program'],begins=1,ends=1,timeout=0,
                   clock_reads=2 if row['program']=='on' else 0,
                   driver_begin_ns=1500000 if row['program']=='on' else 0,
                   driver_end_ns=3500000 if row['program']=='on' else 0)
            elapsed=None if row['phase'].startswith('diagnostic_') else 3.
            self.rows.append(dict(row,status='PASS',target_calls=1,elapsed_ms=elapsed,helper_boundary=h,
                                  driver_boundary=d,boundary_split=split_boundary(h,d,row['program'],row['position'],elapsed)))

    def test_budget_and_balanced_order(self):
        check_manifest(self.ledger)
        self.assertEqual(len(self.ledger['sessions']),12)
        self.assertEqual(len(self.rows),72)
        self.assertEqual(Counter(r['program'] for r in self.rows),{'off':36,'on':36})
        self.assertEqual(sum(r['timed'] for r in self.rows),24)
        self.assertEqual([s['program'] for s in self.ledger['sessions'][4:]],['off','on','on','off','on','off','off','on'])
        self.assertEqual([r['port'] for r in transport(self.ledger)['sessions']],list(range(9600,9612)))

    def test_each_formal_seed_once_and_warm_path(self):
        for s in self.ledger['sessions'][4:]:
            rows=[r for r in self.rows if r['session_id']==s['session_id']]
            self.assertEqual([r['seed'] for r in rows[:3]],[0,20250901,20260910])
            self.assertEqual([r['seed'] for r in rows[3:]],[0,20250901,20260910])

    def test_weight_storage_check_order(self):
        for s in self.ledger['sessions'][:2]:
            rows=[r for r in self.rows if r['session_id']==s['session_id']]
            self.assertEqual([r['weight'] for r in rows[3:]],['W0','W1','W0'])
            self.assertEqual([r['seed'] for r in rows[3:]],[0,0,0])

    def test_missing_duplicate_reordered_rows(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[-1]],self.rows[1:]+self.rows[:1]):
            with self.assertRaises(ValueError):validate_results(self.ledger,rows)

    def test_identity_unexpected_call_or_clock_rejected(self):
        for update in ({'program_identity':{}},{'target_calls':2},{'elapsed_ms':float('nan')},
                       {'status':'FAIL'},{'boundary_mode':'on'}):
            rows=deepcopy(self.rows);rows[0].update(update)
            with self.assertRaises(ValueError):validate_results(self.ledger,rows)

    def test_statistics_use_formal_samples_and_paired_parts(self):
        s=analyse(self.ledger,self.rows)
        self.assertTrue(s['instrument_sensitivity_gate_passed'])
        self.assertEqual(s['T_ms'],{'on':3.,'off':3.})
        self.assertAlmostEqual(s['on_formal_parts']['driver_run_and_wait_ms']['median_percent'],200/3)
        self.assertFalse(s['pure_hardware_compute_measured'])

    def test_instrument_sensitivity_failure_preserved(self):
        rows=deepcopy(self.rows)
        for r in rows:
            if r['timed'] and r['program']=='on':
                r['helper_boundary']['end_ns']+=90000;r['elapsed_ms']=3.09
                r['boundary_split']=split_boundary(r['helper_boundary'],r['driver_boundary'],'on',r['position'],3.09)
        s=analyse(self.ledger,rows)
        self.assertFalse(s['instrument_sensitivity_gate_passed'])
        self.assertAlmostEqual(s['median_on_minus_off_percent'],3.)


class TransportTests(TestCase):
    def test_only_owned_launch_receives_boundary_mode(self):
        b=CostBoard(SimpleNamespace(host='unused',known_hosts='unused',transport_contract={}),'/tmp/test','sha')
        b._cost_launch_mode='on';b._cost_launch_injections=0
        seen=[]
        with patch.object(TransportBoard,'ssh',lambda self,command,timeout=30:seen.append(command) or ''):
            b.ssh('cat /proc/version')
            b.ssh('(cd /tmp/test && exec taskset -c 3 env OTHER=x /tmp/test/tvm_rpc server)')
        self.assertEqual(seen[0],'cat /proc/version')
        self.assertIn('env C3_TPS_COST_MODE=on OTHER=x',seen[1])
        self.assertEqual(b._cost_launch_injections,1)


if __name__=='__main__':main()
