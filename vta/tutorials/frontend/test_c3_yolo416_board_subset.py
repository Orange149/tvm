from pathlib import Path
import tempfile
import unittest
from c3_tps_sync_release_local import write, read, sha
from c3_yolo416_comparison_protocol import manifest
from c3_yolo416_board_subset import prepare
from test_c3_yolo416_comparison_protocol import pool


class SubsetTests(unittest.TestCase):
    def source(self, root):
        p=root/'parent';p.mkdir()
        write(p/'artifacts.json',{'test_fixture_only':True})
        write(p/'call_manifest.json',manifest('parent',pool()))
        write(p/'freeze.json',{'status':'TH_P_OPERATORS_AND_GRAPH_READY_FPGA_PENDING',
            'bindings':{str(p/'artifacts.json'):sha(p/'artifacts.json'),
                        str(p/'call_manifest.json'):sha(p/'call_manifest.json')},
            'host':'test_fixture','known_hosts':'unused'})
        return p

    def test_graph_stratum_preserves_programs_and_has_120_calls(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=self.source(r);out=r/'graph'
            value=prepare(out,p,'graph')
            self.assertEqual(value['pool'],[read(p/'call_manifest.json')['pool'][-1]])
            self.assertEqual((value['total_target_calls'],len(value['sessions']),value['formal_samples']),
                             (120,20,36))
            self.assertEqual(read(out/'artifacts.json'),read(p/'artifacts.json'))
            self.assertFalse((p/'board').exists())

    def test_operator_stratum_keeps_all_11_without_timing_selection(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=self.source(r);value=prepare(r/'operators',p,'operator')
            self.assertEqual(value['pool'],read(p/'call_manifest.json')['pool'][:-1])
            self.assertEqual(value['total_target_calls'],1320)

    def test_refuses_performance_exposure_or_changed_binding(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=self.source(r);(p/'board').mkdir()
            with self.assertRaisesRegex(ValueError,'PERFORMANCE_ALREADY_EXPOSED'):
                prepare(r/'bad',p,'graph')
            self.assertFalse((r/'bad').exists())
            (p/'board').rmdir();(p/'artifacts.json').write_text('{"changed":true}')
            with self.assertRaises(Exception):prepare(r/'bad_binding',p,'graph')
            self.assertFalse((r/'bad_binding').exists())


if __name__=='__main__':unittest.main()
