"""Zero target-call checks for the bounded node34 baseline entry."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import c3_node34_tps_local as local


class Node34LocalTests(unittest.TestCase):
    def test_author_environment_identity_and_stale_namespace_rejected(self):
        import vta
        from c3_banerjee_local import author_namespace
        stale=author_namespace(local.AUTHOR,safe_uop_order=False)
        before=vta.get_env()
        with vta.Environment(dict(before.cfg_dict,TARGET='sim')):
            with self.assertRaisesRegex(ValueError,'ENVIRONMENT_MISMATCH'):
                local.check_schedule_environment(stale)
            fresh=author_namespace(local.AUTHOR,safe_uop_order=False)
            self.assertTrue(local.check_schedule_environment(fresh)['same_coproc_axis_identity'])
        self.assertIs(vta.get_env(),before)
        with self.assertRaisesRegex(ValueError,'ENVIRONMENT_MISMATCH'):
            local.check_schedule_environment(fresh)

    def test_real_relay_fusion_and_wrong_shift_rejected(self):
        source=(local.GRAPH/'relay_000.txt').read_text()
        extracted=local.extract_relay(source)
        self.assertIn('def @main(',extracted)
        self.assertIn('right_shift(%57, 6',extracted)
        self.assertNotIn('Primitive=1',extracted)
        import tvm
        parsed=tvm.relay.parse(extracted)
        inferred=tvm.relay.transform.InferType()(parsed)
        shapes=[[int(v) for v in p.checked_type.shape] for p in inferred['main'].params]
        self.assertEqual(shapes,local.SHAPES[:3])
        self.assertEqual([int(v) for v in inferred['main'].ret_type.shape],local.SHAPES[3])
        with self.assertRaisesRegex(ValueError,'NODE34_RELAY_CONTRACT'):
            local.extract_relay(source.replace('right_shift(%57, 6','right_shift(%57, 7'))
        with self.assertRaisesRegex(ValueError,'NOT_UNIQUE'):
            local.extract_relay(source+source)

    def test_frozen_configuration_and_budget(self):
        route=local.frozen_route()['identity']
        config=route['complete_config_entity']['entity']
        self.assertEqual([r[2][-1] for r in config[:5]],[1,7,7,1,16])
        self.assertEqual(config[-2:],[['oc_nthread','ot',2],['h_nthread','ot',1]])
        self.assertEqual(len(local.KINDS),7)
        self.assertEqual(len(local.SEEDS),7)
        self.assertEqual(route['workload'][2][1],[64,32,3,3,16,16])

    def test_padding_is_not_dram_payload(self):
        event={'kind':'load','domain':2,'pad':[1,1,1,1],'x':2,'y':2,'sram':0,
               'tensor':0,'offset':0,'stride':2}
        result=local.tensor_traffic([event,event])['INP']
        self.assertEqual(result['payload_bytes'],128)
        self.assertEqual(result['origin_count'],4)
        self.assertEqual(result['multiplicities'],[2])
        self.assertEqual(result['padding_vector_writes'],24)

    def test_failure_is_closed_without_retry(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)/'attempt'
            with patch.object(local,'start_body',side_effect=ValueError('PRECHECK_FAILURE')) as body:
                self.assertEqual(local.start(root),1)
                body.assert_called_once_with(root)
            self.assertEqual(local.read(root/'status.json')['status'],'STOPPED_CONTROLLER_GATE')
            self.assertEqual(local.read(root/'call_accounting.json')['completed_calls']['exact'],0)
            with self.assertRaises(FileExistsError):local.start(root)


if __name__=='__main__':unittest.main()
