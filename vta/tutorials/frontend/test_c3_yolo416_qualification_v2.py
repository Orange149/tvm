"""Zero-target integration gates for the new qualification proof contract."""
import ast
from pathlib import Path
import unittest
from c3_yolo416_fifo_certificate import certify_action_graph
from c3_tps_window_progress import LANES


HERE = Path(__file__).resolve().parent


def function(filename, name):
    return next(n for n in ast.parse((HERE/filename).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


class IntegrationTests(unittest.TestCase):
    def test_worker_changes_only_fifo_proof_import(self):
        before = function('c3_tps_sync_release_local.py', 'qualification_worker')
        after = function('c3_yolo416_qualification_v2.py', 'qualification_worker')
        before.body = [n for n in before.body if not isinstance(n, ast.ImportFrom)
                       or n.module != 'c3_tps_window_progress']
        after.body = [n for n in after.body if not isinstance(n, ast.ImportFrom)
                      or n.module not in ('c3_tps_window_progress','c3_yolo416_fifo_certificate')]
        self.assertEqual(ast.dump(before), ast.dump(after))
        actual = function('c3_yolo416_qualification_v2.py', 'qualification_worker')
        matches = [n for n in actual.body if isinstance(n, ast.ImportFrom)
                   and n.module == 'c3_yolo416_fifo_certificate']
        self.assertEqual(len(matches), 1)
        self.assertEqual([(a.name,a.asname) for a in matches[0].names],
                         [('certify_action_graph','certify_progress')])

    def test_graph_worker_changes_only_fifo_proof_import(self):
        before = function('c3_yolo416_graph_local.py','qualify_worker')
        after = function('c3_yolo416_graph_local_v2.py','qualify_worker')
        before.body = [n for n in before.body if not isinstance(n, ast.ImportFrom)
                       or n.module != 'c3_tps_window_progress']
        after.body = [n for n in after.body if not isinstance(n, ast.ImportFrom)
                      or n.module not in ('c3_tps_window_progress','c3_yolo416_fifo_certificate')]
        self.assertEqual(ast.dump(before),ast.dump(after))

    def test_unresolved_cycle_still_fails_existing_acceptance(self):
        rows = [(1,(0,1,0,1)),(2,(1,0,1,0))]
        rs = [{'index':i,'stage':s,'bits':b} for i,(s,b) in enumerate(rows)]
        status = certify_action_graph(rs,{s:2 for s in (1,2,3)},
                                      {lane:2 for lane in LANES})['status']
        self.assertNotEqual(status,'FINITE_FIFO_PROGRESS_PROVED')

    def test_controllers_keep_fresh_directory_and_target_budget(self):
        source=(HERE/'c3_yolo416_tps_local_v2.py').read_text()
        self.assertIn('root.mkdir(exist_ok=False)',source)
        self.assertIn("'max_target_calls':308",source)
        self.assertIn("'old_attempt_adopted_calls':0",source)
        self.assertIn('from c3_yolo416_qualification_v2 import qualification_worker',source)


if __name__ == '__main__': unittest.main()
