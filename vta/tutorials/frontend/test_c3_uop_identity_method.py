"""Zero-target tests including actual node46 compilation and final LLVM gate."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import unittest
import tvm, vta
from tvm import tir
import c3_uop_identity_method as method
import c3_node46_uop_identity_local as probe
from c3_uop_identity_applicability import classify


def function(root):
    mod = tvm.ir.load_json((root/'prepare/R/parent.json').read_text())
    return list(mod.functions.values())[0]


class MethodTests(unittest.TestCase):
    def test_tuple_list_trace_transport_is_equal(self):
        row=next(e for e in probe.read(probe.BASELINE/'prepare/R/events.json') if e['kind']=='kernel')
        memory=deepcopy(row);memory['signature']=tuple(memory['signature'])
        memory['mode_reset']=tuple(memory['mode_reset'])
        self.assertEqual(method.event_contract([row],[memory],{'changes':[]})['candidate_cache_identities'],1)

    def test_original_representatives_not_class_numbers(self):
        from test_c3_uop_identity_applicability import event
        row = classify([event(k,b) for k,b in zip(range(4,8),(1,2,2,3))])[0]
        row['domain'] = list(range(4,8))
        mapping = method.representative_map(row)
        self.assertEqual(mapping, {4:4,5:5,6:5,7:7})
        k = tir.Var('unrelated_name','int32')
        expr, form = method.projection(k,mapping)
        self.assertEqual(form['kind'],'finite_select')
        for value,expected in mapping.items():
            self.assertEqual(int(tvm.arith.Analyzer().simplify(
                tir.stmt_functor.substitute(expr,{k:tir.IntImm('int32',value)}))),expected)

    def test_period_three_not_hardcoded_bank_two(self):
        from test_c3_uop_identity_applicability import event
        row = classify([event(k) for k in range(12)])[0];row['domain']=list(range(12))
        mapping=method.representative_map(row)
        _,form=method.projection(tir.Var('any','int32'),mapping)
        self.assertEqual(form,{'kind':'modulo','origin':0,'period':3})

    def test_incomplete_partition_rejected(self):
        row={'domain':[4,5], 'classes':[{'representative':[4],'members':[[4]]}]}
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):method.representative_map(row)

    def test_two_real_domains_use_same_method_and_preserve_events(self):
        from c3_official_window_method import audit_function
        for name,n in [('20261017_c3_node34_tps_local_run02',32),
                       ('20261017_c3_node46_tps_local_run01',24)]:
            root=probe.HOLD/name;old=function(root);new,proof=method.transform(old)
            self.assertEqual(len(proof['changes']),2)
            self.assertEqual(proof['changes'][0]['domain'],list(range(n)))
            events=audit_function(new)['events']
            delta=method.event_contract(probe.read(root/'prepare/R/events.json'),events,proof)
            self.assertEqual(delta['candidate_cache_identities'],14)
            if n==32:
                sealed=probe.read(probe.HOLD/'20261017_c3_node34_uop_identity_local_run02/prepare/R/events.json')
                self.assertEqual(method.event_contract(sealed,events,{'changes':[]})['changed_dynamic_signatures'],0)

    def test_no_redundancy_TPS24_is_exact_noop(self):
        old=function(probe.HOLD/'20261016_c3_tps_sync_release_local_run04')
        new,proof=method.transform(old)
        self.assertEqual(proof['changes'],[])
        self.assertTrue(tvm.ir.structural_equal(old,new))

    def test_official_program_is_exact_noop(self):
        from c3_uop_identity_applicability import inventory
        row=next(r for r in inventory() if r['family']=='official_reinstantiated')
        from c3_tps_sync_release_prepare import complete_function
        mod=tvm.ir.load_json(Path(row['path']).read_text());_,old=complete_function(mod,row['symbol'])
        new,proof=method.transform(old)
        self.assertFalse(proof['changes']);self.assertTrue(tvm.ir.structural_equal(old,new))

    def test_unknown_branch_is_unresolved(self):
        old=function(probe.BASELINE)
        old=old.with_body(tir.IfThenElse(tir.IntImm('bool',1),old.body,None))
        with self.assertRaisesRegex(ValueError,'DOMAIN_SUPPORT_UNRESOLVED'):method.transform(old)

    def test_material_or_signature_change_rejected(self):
        old=probe.read(probe.BASELINE/'prepare/R/events.json');_,proof=method.transform(function(probe.BASELINE))
        candidate=deepcopy(old)
        for e in candidate:
            if e['kind']=='kernel' and e['mode_reset']==[0,0]:
                e['signature']=[e['signature'][0]%2]
                e['cache_identity']['signature']=e['signature']
        self.assertEqual(method.event_contract(old,candidate,proof)['candidate_cache_identities'],14)
        next(e for e in candidate if e['kind']=='load')['offset']+=1
        with self.assertRaisesRegex(ValueError,'COMPUTATION_TENSOR_REQUEST'):method.event_contract(old,candidate,proof)

    def test_raw_final_signature_rejected(self):
        _,proof=method.transform(function(probe.BASELINE))
        with self.assertRaisesRegex(ValueError,'FINAL_MAC_SIGNATURE_IS_NOT_REPRESENTATIVE'):
            method.llvm_signature_provenance((probe.BASELINE/'prepare/R/module.ll').read_text(),proof)

    def test_unsupported_final_projection_rejected(self):
        proof={'changes':[{'scope':0,'form':{'kind':'modulo','origin':0,'period':3}}]}
        with self.assertRaisesRegex(ValueError,'LLVM_PROJECTION_SUPPORT_UNRESOLVED'):
            method.llvm_signature_provenance('',proof)

    def test_full_compile_chain_without_target_calls(self):
        path=os.environ.get('C3_IDENTITY_PREFLIGHT')
        if not path:self.skipTest('explicit zero-call artifact directory required')
        root=Path(path)/'actual_compile';root.mkdir(exist_ok=False)
        bindings=probe.read(probe.EVIDENCE/'freeze.json')['bindings'].copy()
        bindings.update({str(p.resolve()):probe.sha(p) for p in
                        [Path(method.__file__),Path(probe.__file__),Path(__file__)]})
        probe.write(root/'freeze.json',{'bindings':bindings})
        shutil.copyfile(probe.BASELINE/'hardware_archive.json',root/'hardware_archive.json')
        probe.prepare_worker(root)
        ready=probe.read(root/'prepare_complete.json')
        self.assertEqual(ready['status'],'IDENTITY_A_PREPARED')
        proof=probe.read(root/'prepare/R/late_projection.json')
        llvm=(root/'prepare/R/module.ll').read_text()
        self.assertEqual(len(method.llvm_signature_provenance(llvm,proof)['sites']),2)
        with self.assertRaisesRegex(ValueError,'FINAL_MAC_SIGNATURE_IS_NOT_REPRESENTATIVE'):
            method.llvm_signature_provenance(llvm.replace(' = and i32 ',' = or i32 '),proof)


if __name__=='__main__':unittest.main()
