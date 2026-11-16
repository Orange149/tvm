"""Safety-relevant offline graph fixtures; never execute an accelerator."""
import unittest
from copy import deepcopy

from c3_sync_release_audit import (Access, Graph, Obligations, address_mask, classify,
                                  access_model, expand_contract, kernel_access,
                                  later_weight_target, requirements)


def row(i, stage, flags=(0,0,0,0), description='GEMM'):
    return {'index': i, 'stage': stage, 'bits': flags, 'description': description,
            'detail': 'reset_out: 0'}


def obligations(graph, accesses):
    return (Obligations(requirements(accesses), graph.ancestors, True),
            Obligations(requirements(accesses, True), graph.ancestors, False))


def split_fixture(overwrite_input=False):
    rows = [row(0,1,description='LOAD INP'), row(1,1,(0,0,0,1),'LOAD WGT'),
            row(2,2,(1,0,1,0)), row(3,1,(0,1,0,0),'LOAD INP'),
            row(4,1,description='LOAD WGT')]
    accesses = [Access(writes={'INP':1}), Access(writes={'WGT':1}),
                Access(reads={'INP':1,'WGT':1}),
                Access(writes={'INP':1 if overwrite_input else 2}), Access(writes={'WGT':1})]
    graph = Graph(rows)
    token = next(t for t in graph.tokens if t['lane']==[2,1])
    return graph,token,accesses


def certificate_fixture(reset=False, store=False):
    rows=[row(0,2)]
    rows[0]['detail']='reset_out: %d\nouter loop - iter: 1, wgt: 0, inp: 0, acc: 0\ninner loop - iter: 1, wgt: 0, inp: 0, acc: 0' % reset
    contract={'args':[[0,int(reset),0,0,0,0,0,0]],'loops':[]}
    material=[dict(instruction=0,description='GEMM',kind='kernel',native=dict(begin=0,end=1))]
    if store:
        rows.append(row(1,3,description='STORE'))
        material.append(dict(instruction=1,description='STORE',kind='store',native=dict(sram=0,x=1,y=1)))
    cert=dict(submission_count=1,cache_definitions=dict(contracts={'k':contract},definitions={'k':[[0,0,0]]}),
              uop_values=dict(checks=[dict(values_match=True,compute_index=0,submission=1,fifo_index=0,
                                         uop_begin=0,uop_end=1,kernel_identity='k')]),material_map=material)
    return rows,cert


class ReleaseAuditTests(unittest.TestCase):
    def test_removing_scope_wait_loses_following_weight_protection(self):
        graph,token,accesses = split_fixture()
        live,physical = obligations(graph,accesses)
        result = classify(graph,live,physical,token)
        self.assertEqual(result['category'],'NECESSARY_PROTECTION')
        self.assertEqual(result['witness'],dict(source=2,target=4,domain='WGT',address=0,kind='WAR'))

    def test_move_preserves_weight_wait_and_releases_other_input_range(self):
        graph,token,accesses = split_fixture()
        live,physical = obligations(graph,accesses)
        self.assertEqual(later_weight_target(graph,token),4)
        result = classify(graph,live,physical,token,4)
        self.assertEqual(result['category'],'POSSIBLE_EARLIER_RELEASE')
        self.assertFalse(result['executable_or_combined_transform_certified'])
        changed = graph.hypothesis(token,4)
        self.assertFalse(changed[3]>>2&1)
        self.assertTrue(changed[4]>>2&1)

    def test_move_does_not_allow_overwriting_current_input(self):
        graph,token,accesses = split_fixture(True)
        live,physical = obligations(graph,accesses)
        result = classify(graph,live,physical,token,4)
        self.assertEqual(result['category'],'NECESSARY_PROTECTION')
        self.assertEqual(result['witness']['domain'],'INP')

    def test_each_hypothesis_starts_from_unchanged_original(self):
        graph,token,accesses = split_fixture()
        before=list(graph.ancestors)
        graph.hypothesis(token)
        graph.hypothesis(token,4)
        self.assertEqual(graph.ancestors,before)
        self.assertTrue(graph.ancestors[3]>>2&1)

    def test_reset_dead_read_collision_is_unresolved_not_proved_safe(self):
        graph=Graph([row(0,2,(0,0,1,0)),row(1,1,(0,1,0,0),'LOAD INP')])
        accesses=[Access(writes={'ACC':1,'OUT':1},dead_reads={'INP':1}),Access(writes={'INP':1})]
        live,physical=obligations(graph,accesses)
        result=classify(graph,live,physical,graph.tokens[0])
        self.assertEqual(result['category'],'HARDWARE_OBSERVABILITY_UNRESOLVED')

    def test_store_reads_out_and_reset_must_not_overwrite_it(self):
        graph=Graph([row(0,2,(0,0,0,1)),row(1,3,(1,0,1,0),'STORE'),row(2,2,(0,1,0,0))])
        accesses=[Access(writes={'OUT':1}),Access(reads={'OUT':1}),Access(writes={'ACC':1,'OUT':1})]
        live,physical=obligations(graph,accesses)
        token=next(t for t in graph.tokens if t['lane']==[3,2])
        result=classify(graph,live,physical,token)
        self.assertEqual(result['category'],'NECESSARY_PROTECTION')
        self.assertEqual(result['witness']['domain'],'OUT')

    def test_existing_dead_read_unordered_pair_is_not_a_baseline_error(self):
        graph=Graph([row(0,2),row(1,1,description='LOAD INP')])
        live,physical=obligations(graph,[Access(dead_reads={'INP':1}),Access(writes={'INP':1})])
        self.assertFalse(live.original_unordered)
        self.assertEqual(len(physical.original_unordered),1)

    def test_missing_original_live_order_fails_closed(self):
        graph=Graph([row(0,1,description='LOAD INP'),row(1,2)])
        with self.assertRaisesRegex(ValueError,'ORIGINAL_LIVE_MEMORY_ORDER_UNRESOLVED'):
            obligations(graph,[Access(writes={'INP':1}),Access(reads={'INP':1})])

    def test_moving_across_next_pop_is_rejected(self):
        graph=Graph([row(0,2,(0,0,1,0)),row(1,2,(0,0,1,0)),
                     row(2,1,(0,1,0,0),'LOAD INP'),row(3,1,(0,1,0,0),'LOAD WGT')])
        with self.assertRaisesRegex(ValueError,'MOVE_CROSSES_OR_COLLIDES'):
            graph.hypothesis(graph.tokens[0],3)
        self.assertIsNone(later_weight_target(graph,graph.tokens[0]))

    def test_reset_model_tracks_real_out_writes_and_dead_operand_reads(self):
        a=kernel_access({'args':[[0,1,2,0,0,0,0,0]],'loops':[[2,1,0,0]]})
        self.assertEqual(a.writes,{'ACC':12,'OUT':12})
        self.assertEqual(a.dead_reads,{'INP':1,'WGT':1,'ACC':12})
        self.assertFalse(a.reads)

    def test_alu_immediate_keeps_source_read_separate(self):
        a=kernel_access({'args':[[1,0,2,7,0,3,1,7]],'loops':[]})
        self.assertEqual(a.reads,{'ACC':4})
        self.assertEqual(a.dead_reads,{'ACC':128})
        self.assertEqual(a.writes['OUT'],4)

    def test_address_bounds_include_exclusive_end(self):
        self.assertEqual(address_mask([2047],'INP'),1<<2047)
        for address in (-1,2048):
            with self.assertRaises(ValueError):address_mask([address],'INP')

    def test_unmatched_tokens_rejected(self):
        with self.assertRaisesRegex(ValueError,'UNMATCHED_TOKEN'):
            Graph([row(0,2,(0,0,1,0))])

    def test_contract_must_match_base_uop_definition(self):
        rows,cert=certificate_fixture()
        cert['cache_definitions']['contracts']['k']['args'][0][3]=1
        with self.assertRaisesRegex(ValueError,'CONTRACT_DEFINITION_CONTENT_DIFFER'):
            access_model(rows,cert)

    def test_contract_must_match_actual_hardware_loops(self):
        rows,cert=certificate_fixture()
        cert['cache_definitions']['contracts']['k']['loops']=[[2,1,1,1]]
        with self.assertRaisesRegex(ValueError,'KERNEL_BINDING_HARDWARE_LOOP_CHANGED'):
            access_model(rows,cert)

    def test_actual_compute_fifo_index_must_match(self):
        rows,cert=certificate_fixture()
        cert['uop_values']['checks'][0]['fifo_index']=1
        with self.assertRaisesRegex(ValueError,'KERNEL_BINDING_REFERENCE_CHANGED'):
            access_model(rows,cert)

    def test_reset_temporary_out_must_not_be_observed(self):
        rows,cert=certificate_fixture(reset=True,store=True)
        with self.assertRaisesRegex(ValueError,'RESET_TEMPORARY_OUT_OBSERVED'):
            access_model(rows,cert)

    def test_step_limit_checked_before_building_large_offset_product(self):
        contract=dict(args=[[0,0,0,0,0,0,0,0]],loops=[[16000,0,0,0],[16000,0,0,0]])
        with self.assertRaisesRegex(ValueError,'AUDIT_EXPANDED_KERNEL_LIMIT'):
            list(expand_contract(contract))


if __name__=='__main__':
    unittest.main()
