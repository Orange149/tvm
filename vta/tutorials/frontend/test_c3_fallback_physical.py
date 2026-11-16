"""Physical proof retains original obligations and never enumerates candidates."""
import ast
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from c3_tps_sync_release_local import HERE,HOLD,read
from c3_five_tradeoff_commands import decode_submissions
from c3_fallback_physical import certify_physical
from c3_sync_release_audit import analyze,Access


class PhysicalTests(unittest.TestCase):
    def test_qualification_body_only_changes_analysis_import(self):
        before=ast.parse((HERE/'c3_yolo416_qualification_v2.py').read_text())
        after=ast.parse((HERE/'c3_fallback_qualification.py').read_text())
        left=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name=='qualification_worker')
        right=copy.deepcopy(next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='qualification_worker'))
        imports=[n for n in ast.walk(right) if isinstance(n,ast.ImportFrom) and n.module=='c3_fallback_physical']
        self.assertEqual(len(imports),1)
        imports[0].module='c3_sync_release_audit'
        imports[0].names=[ast.alias(name='analyze')]
        self.assertEqual(ast.dump(left,include_attributes=False),ast.dump(right,include_attributes=False))

    def test_full_original_obligations_equal_sealed_small_case(self):
        unit=HOLD/'20261017_c3_node34_tps_local_run02/R_detailed'
        records=decode_submissions((unit/'native_call_01.txt').read_text())[0]
        certificate=read(unit/'native_certificate_01.json')
        old=analyze(records,certificate)
        with patch('c3_sync_release_audit.Graph.hypothesis',side_effect=AssertionError('outside scope')):
            new=certify_physical(records,certificate)
        for field in ('instructions','live_obligations','physical_obligations',
                      'original_dead_read_unordered_count','original_dead_read_unordered_examples'):
            self.assertEqual(new[field],old[field])
        self.assertEqual(new['sync_hypotheses_enumerated'],0)

    def test_unordered_live_operand_still_rejected(self):
        records=[{'index':0,'stage':1,'bits':[0,0,0,0]}, {'index':1,'stage':2,'bits':[0,0,0,0]}]
        with patch('c3_fallback_physical.access_model',return_value=[Access(writes={'INP':1}),Access(reads={'INP':1})]):
            with self.assertRaisesRegex(ValueError,'ORIGINAL_LIVE_MEMORY_ORDER_UNRESOLVED'):
                certify_physical(records,{})


if __name__=='__main__':unittest.main()
