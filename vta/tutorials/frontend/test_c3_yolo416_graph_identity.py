from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
import tvm,vta
from tvm import tir
from c3_tps_sync_release_local import HOLD,read,write
from c3_yolo416_graph_identity import compare


class GraphIdentityTests(unittest.TestCase):
    old=HOLD/'20261017_c3_yolo416_graph_prepare_run01'
    new=HOLD/'20261017_c3_yolo416_graph_prepare_run02'

    def copy_inputs(self,d):
        root=Path(d)
        for variant in ('TH','P'):
            (root/variant).mkdir()
            for name in ('codegen_input.json','graph.json','params.bin','CPU_manifest.json',
                         'module.ll','LLVM_initializer_binding.json'):
                shutil.copyfile(self.new/variant/name,root/variant/name)
        return root

    def test_real_rebuild_has_equal_executable_and_different_serialization(self):
        rows=compare(self.old,self.new)
        self.assertEqual(len(rows),2)
        self.assertTrue(all(x['normalized_entire_LLVM_equal'] for x in rows))
        self.assertTrue(any(not x['IR_serialized_bytes_equal'] for x in rows))

    def test_actual_uop_destination_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.copy_inputs(d);p=root/'TH/codegen_input.json'
            mod=tvm.ir.load_json(p.read_text());changed=False
            for gv,fn in list(mod.functions.items()):
                def mutate(n):
                    nonlocal changed
                    if not changed and isinstance(n,tir.Call) and n.op.name=='tir.vta.uop_push':
                        args=list(n.args);args[2]=args[2]+1;changed=True
                        return tir.Call(n.dtype,n.op,args,n.span)
                    return n
                body=tir.stmt_functor.ir_transform(fn.body,None,mutate,['tir.Call'])
                mod[gv]=fn.with_body(body)
                if changed:break
            self.assertTrue(changed,'real VTA initializer not found')
            p.write_text(tvm.ir.save_json(mod))
            with self.assertRaisesRegex(ValueError,'EXECUTABLE_IR_CHANGED'):compare(self.old,root)

    def test_real_instruction_or_cache_identity_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.copy_inputs(d);p=root/'TH/module.ll'
            text=p.read_text();self.assertIn('@VTAPushGEMMOp',text)
            p.write_text(text.replace('@VTAPushGEMMOp','@ChangedGEMMOp',1))
            with self.assertRaisesRegex(ValueError,'EXECUTABLE_LLVM_CHANGED'):compare(self.old,root)
        with tempfile.TemporaryDirectory() as d:
            root=self.copy_inputs(d);p=root/'TH/LLVM_initializer_binding.json';data=read(p)
            data['distinct_handles']+=1;p.write_text(__import__('json').dumps(data))
            with self.assertRaisesRegex(ValueError,'INITIALIZER_OR_CACHE_SHARING_CHANGED'):compare(self.old,root)

    def test_parameter_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.copy_inputs(d);p=root/'P/params.bin';p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaisesRegex(ValueError,'GRAPH_PARAMETERS_OR_CPU_CHANGED'):compare(self.old,root)


if __name__=='__main__':unittest.main()
