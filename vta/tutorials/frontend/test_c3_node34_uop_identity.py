"""Frozen-parent, no-target regressions for the sole identity variant."""
from copy import deepcopy
import os
from pathlib import Path
import unittest
import tvm,vta
from tvm import tir

import c3_node34_uop_identity as probe


class IdentityTests(unittest.TestCase):
    def function(self):
        m=tvm.ir.load_json((probe.BASELINE/'prepare/R/parent.json').read_text())
        return list(m.functions.values())[0]

    def test_real_parent_inverse_closures_and_DMA_counter(self):
        f=self.function();g,proof=probe.transform(f)
        free=[];loads=[]
        def visit(n):
            if isinstance(n,tir.AttrStmt) and n.attr_key=='coproc_uop_scope':
                free.append([str(v) for v in tir.analysis.undefined_vars(n.body)])
            if isinstance(n,tir.Call) and n.op.name=='tir.call_extern' and n.args[0].value=='VTALoadBuffer2D':loads.append(str(n))
        tir.stmt_functor.post_order_visit(g.body,visit)
        self.assertEqual(free[:4],[[],[],['c3_uop_bank_0'],['c3_uop_bank_1']])
        self.assertTrue(any('k_o_outer * 49' in s for s in loads))
        self.assertTrue(any('k_o_outer * 9' in s for s in loads))
        self.assertTrue(proof['DMA_control_and_non_MAC_body_restored_exactly'])

    def test_raw_counter_extra_use_is_rejected(self):
        f=self.function()
        def change(n):
            if isinstance(n,tir.Call) and n.op.name=='tir.vta.uop_push' and int(n.args[0])==0 and int(n.args[1])==0:
                vars=list(tir.analysis.undefined_vars(tir.Evaluate(n)))
                counter=next(v for v in vars if str(v)=='k_o_outer')
                args=list(n.args);args[2]=args[2]+counter
                return tir.Call(n.dtype,n.op,args,n.span)
        altered=f.with_body(tir.stmt_functor.ir_transform(f.body,None,change,['tir.Call']))
        with self.assertRaisesRegex(ValueError,'RAW_COUNTER_REMAINS'):probe.transform(altered)

    def test_wrong_reduction_domain_is_rejected(self):
        f=self.function()
        def change(n):
            if isinstance(n,tir.For) and str(n.loop_var)=='k_o_outer':
                return tir.For(n.loop_var,n.min,31,n.kind,n.body,n.thread_binding,n.annotations,n.span)
        altered=f.with_body(tir.stmt_functor.ir_transform(f.body,None,change,['tir.For']))
        with self.assertRaisesRegex(ValueError,'COUNTER_DOMAIN_NOT_FROZEN'):probe.transform(altered)

    def test_event_proof_rejects_material_change(self):
        events=probe.read(probe.BASELINE/'prepare/R/events.json');candidate=deepcopy(events)
        for e in candidate:
            if e['kind']=='kernel' and e['mode_reset']==[0,0]:
                e['signature']=[e['signature'][0]%2];e['cache_identity']['signature']=e['signature']
        self.assertEqual(probe.event_contract(events,candidate)['candidate_cache_identities'],14)
        next(e for e in candidate if e['kind']=='load' and e['domain']==1)['offset']+=1
        with self.assertRaisesRegex(ValueError,'COMPUTATION_TENSOR_REQUEST'):probe.event_contract(events,candidate)

    def test_native_pair_rejects_missing_material(self):
        # Use the real sealed native trace to exercise the shared binding API.
        from c3_five_tradeoff_commands import decode_submissions
        d=probe.BASELINE/'R_detailed'
        rows=decode_submissions((d/'native_call_01.txt').read_text())[0]
        cert=probe.read(d/'native_certificate_01.json')
        good=probe.native_pair(rows,rows,cert,cert)
        self.assertGreater(good['retained_obligations'],0)
        bad=deepcopy(cert);bad['material_map'][0]['tir_event']=-1
        with self.assertRaisesRegex(ValueError,'ACTUAL_MATERIAL_BINDING'):probe.native_pair(rows,rows,cert,bad)

    def test_sealed_failed_LLVM_rejected_before_target_call(self):
        old=probe.HOLD/'20261017_c3_node34_uop_identity_local_run01/prepare/R/module.ll'
        with self.assertRaisesRegex(ValueError,'FINAL_MAC_SIGNATURE_IS_NOT_PROJECTED_BANK'):
            probe.llvm_signature_provenance(old.read_text())

    def test_real_compilation_preserves_late_signature_projection(self):
        from c3_official_window_baseline import normalize_llvm
        from c3_tps_sync_release_prepare import complete_function,compiled_module_contract
        before=tvm.ir.load_json((probe.BASELINE/'prepare/R/final.json').read_text())
        with vta.Environment(dict(vta.get_env().cfg_dict,TARGET='sim')):
            target=tvm.target.Target(vta.get_env().target,host=vta.get_env().target_host)
            with tvm.transform.PassContext(opt_level=3,disabled_pass={'AlterOpLayout','tir.CommonSubexprElimTIR'}):
                lowered=tvm.lower(before)
                projected,proof=probe.project_lowered_module(lowered)
                # Deliberately show that the ordinary IRModule build undoes it.
                reverted=tvm.lower(projected)
                _,fn=complete_function(reverted,probe.SYMBOL)
                closures=[]
                tir.stmt_functor.post_order_visit(fn.body,lambda n:closures.append(
                    [str(v) for v in tir.analysis.undefined_vars(n.body)])
                    if isinstance(n,tir.AttrStmt) and n.attr_key=='coproc_uop_scope' else None)
                self.assertEqual(closures[2:4],[['k_o_outer'],['k_o_outer']])
                normal_b=tvm.build(before,target=target)
                map_b=tvm.build({target:lowered})
                lib=tvm.build({target:projected})
                restored=tvm.ir.load_json(tvm.ir.save_json(projected))
                rtlib=tvm.build({target:restored})
        bllvm=normal_b.get_source('ll');llvm=lib.get_source('ll')
        self.assertEqual(normalize_llvm(bllvm),normalize_llvm((probe.BASELINE/'prepare/R/module.ll').read_text()))
        self.assertEqual(normalize_llvm(bllvm),normalize_llvm(map_b.get_source('ll')))
        self.assertNotEqual(normalize_llvm(llvm),normalize_llvm(bllvm))
        self.assertEqual(normalize_llvm(llvm),normalize_llvm(rtlib.get_source('ll')))
        contract=probe.llvm_signature_provenance(llvm)
        self.assertEqual(len(contract['sites']),2)
        bad=llvm.replace(' = and i32 ', ' = or i32 ')
        with self.assertRaisesRegex(ValueError,'FINAL_MAC_SIGNATURE_IS_NOT_PROJECTED_BANK'):
            probe.llvm_signature_provenance(bad)
        compiled=compiled_module_contract(projected,probe.SYMBOL,llvm)
        directory=os.environ.get('C3_IDENTITY_FIX_ARTIFACT_DIR')
        if directory:
            out=Path(directory);out.mkdir(exist_ok=False)
            (out/'B.ll').write_text(bllvm);(out/'A.ll').write_text(llvm)
            (out/'codegen_input_A.json').write_text(tvm.ir.save_json(projected))
            (out/'codegen_input_A.tir').write_text(projected.script())
            lib.export_library(str(out/'A.so'))
            probe.write(out/'compiled_contract.json',compiled)
            probe.write(out/'final_signature_provenance.json',contract)
            probe.write(out/'result.json',{'status':'REAL_COMPILE_AND_FINAL_SIGNATURE_PASSED','target_calls':0,
                'normal_B_equals_sealed_B':True,'target_map_B_equals_normal_B':True,
                'candidate_differs_from_B':True,'projected_roundtrip_LLVM_equal':True,
                'old_inlining_failure_reproduced':True,'projection':proof})


if __name__=='__main__':unittest.main()
