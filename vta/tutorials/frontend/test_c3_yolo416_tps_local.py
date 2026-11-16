import json
from pathlib import Path
import tempfile
import unittest
from c3_yolo416_tps_local import accounting,objects,ledger
from c3_tps_sync_release_local import HOLD,validate_calls


class ProtocolTests(unittest.TestCase):
    def test_fixed_coverage_and_budget(self):
        rows=objects(HOLD/'20261017_c3_yolo416_tps_prepare_run02')
        self.assertEqual(len(rows),22)
        self.assertEqual([x['variant'] for x in rows],['P']*11+['TH']*11)
        calls=[r for r in ledger() if r['program']=='R']
        self.assertEqual(len(calls),14)
        self.assertEqual(len(rows)*len(calls),308)
        self.assertEqual({r['mode'] for r in calls},{'detailed','minimal'})

    def test_no_execution_is_not_completion(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);p=r/'objects/00/R_detailed';p.mkdir(parents=True)
            (p/'call_01_begin.json').write_text(json.dumps({'call_index':1}))
            a=accounting(r)
            self.assertEqual(a['registered'],1)
            self.assertEqual(a['helper_completed_observed_lower_bound'],0)
            self.assertEqual(a['execution_count_unresolved_workers'],['00_detailed'])
            self.assertEqual(a['all_certificates_passed'],0)

    def test_cumulative_counts_are_not_summed_per_call(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);p=r/'objects/00/R_detailed';p.mkdir(parents=True)
            for i in (1,2):
                (p/('call_%02d_begin.json'%i)).write_text(json.dumps({'call_index':i}))
                (p/('call_%02d_execution.json'%i)).write_text(json.dumps({
                    'counts':i,'metadata':{'started_calls':i},'helper_returned':True}))
                (p/('call_%02d_end.json'%i)).write_text('{}')
            a=accounting(r)
            self.assertEqual(a['helper_completed_observed_lower_bound'],2)
            self.assertEqual(a['helper_started_observed_lower_bound'],2)
            self.assertEqual(a['all_certificates_passed'],2)

    def test_missing_duplicate_and_wrong_identity_rejected(self):
        planned=[x for x in ledger() if x['program']=='R']
        good=[{**x,'status':'PASS','correct':True} for x in planned]
        self.assertTrue(validate_calls(planned,good))
        for bad in (good[:-1],good+[good[-1]],[{**good[0],'program':'S'},*good[1:]]):
            with self.assertRaises(ValueError):validate_calls(planned,bad)


if __name__=='__main__':unittest.main()
