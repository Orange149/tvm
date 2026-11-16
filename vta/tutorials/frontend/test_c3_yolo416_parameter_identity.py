from pathlib import Path
import tempfile
import unittest
import numpy as np
import tvm
from tvm import relay
from c3_tps_sync_release_local import HOLD
from c3_yolo416_parameter_identity import compare


class ParameterIdentityTests(unittest.TestCase):
    def test_real_TH_P_model_values_equal_despite_serialization_order(self):
        r=HOLD/'20261017_c3_yolo416_graph_prepare_run02'
        result=compare(r/'TH/params.bin',r/'P/params.bin')
        self.assertEqual(result['status'],'ALL_NAMED_MODEL_PARAMETERS_EXACTLY_EQUAL')
        self.assertFalse(result['serialized_bytes_equal'])
        self.assertTrue(result['tensors'])

    def test_changed_value_name_shape_or_dtype_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);a=r/'a.bin';b=r/'b.bin'
            a.write_bytes(relay.save_param_dict({'w':tvm.nd.array(np.array([1,2],dtype='int8'))}))
            for name,value in [('w',np.array([1,3],dtype='int8')),('other',np.array([1,2],dtype='int8')),
                               ('w',np.array([[1,2]],dtype='int8')),('w',np.array([1,2],dtype='int32'))]:
                b.write_bytes(relay.save_param_dict({name:tvm.nd.array(value)}))
                with self.assertRaises(ValueError):compare(a,b)

    def test_float_sign_bits_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);a=r/'a';b=r/'b'
            a.write_bytes(relay.save_param_dict({'f':tvm.nd.array(np.array([0.0],dtype='float32'))}))
            b.write_bytes(relay.save_param_dict({'f':tvm.nd.array(np.array([-0.0],dtype='float32'))}))
            with self.assertRaisesRegex(ValueError,'TRAINED_PARAMETER_VALUE_CHANGED'):compare(a,b)


if __name__=='__main__':unittest.main()
