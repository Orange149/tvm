"""Read-only real ARM graph LLVM checks; zero target calls."""
from pathlib import Path
import unittest
import tvm,vta
from c3_tps_sync_release_local import HOLD,read
from c3_yolo416_graph_binding import project,bind_graph


SOURCE=HOLD/'20261017_c3_yolo416_tps_prepare_run02'


class BindingTests(unittest.TestCase):
    def functions(self,variant):
        result={}
        for row in read(SOURCE/variant/'functions.json'):
            m=tvm.ir.load_json(Path(row['final']).read_text())
            result[row['symbol']]=list(m.functions.values())[0]
        return result

    def test_real_TH_all_function_ownership(self):
        result=bind_graph((SOURCE/'TH/module.ll').read_text(),self.functions('TH'))
        self.assertEqual(len(result['functions']),11)
        self.assertGreater(result['distinct_handles'],11)

    def test_real_P_all_function_ownership(self):
        result=bind_graph((SOURCE/'P/module.ll').read_text(),self.functions('P'))
        self.assertEqual(len(result['functions']),11)
        self.assertGreater(result['distinct_handles'],11)

    def test_missing_symbol_rejected(self):
        with self.assertRaisesRegex(ValueError,'ENTRY_MISSING'):
            project((SOURCE/'P/module.ll').read_text(),'not_a_real_entry')

    def test_missing_function_cannot_hide_other_handles(self):
        f=self.functions('P');f.pop(next(iter(f)))
        with self.assertRaisesRegex(ValueError,'UNOWNED'):
            bind_graph((SOURCE/'P/module.ll').read_text(),f)


if __name__=='__main__':unittest.main()
