"""Local checks for E2 strict comparison and endpoint accounting."""
import unittest
from qualify_vta_segment_counters import counter_differences
from audit_vta_endpoint_boundaries import edge


class CounterTests(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(counter_differences({"load":5,"diagnostic":10},{"load":5}),{})

    def test_missing_zero_is_not_accepted(self):
        self.assertEqual(counter_differences({},{"load":0}),{"load":{"expected":0,"observed":None}})

    def test_mismatch(self):
        self.assertTrue(counter_differences({"load":6},{"load":5}))


class EdgeTests(unittest.TestCase):
    def stage(self,sid,device,shapes):
        contract={"arity":len(shapes),"slots":[{"shape":s,"dtype":"float32"} for s in shapes]}
        return {"segment_id":sid,"device":device,"input_contract":contract,"output_contract":contract}

    def test_each_tensor_aligned_separately(self):
        row=edge(self.stage("vta","vta",[[1],[1]]),self.stage("cpu","cpu",[[1],[1]]))
        self.assertEqual(row["total_payload_bytes"],8)
        self.assertEqual(row["theoretical_k2_aligned_slot_bytes"],1024)

    def test_contract_mismatch_rejected(self):
        with self.assertRaisesRegex(AssertionError,"contract mismatch"):
            edge(self.stage("vta","vta",[[4]]),self.stage("cpu","cpu",[[8]]))


if __name__=="__main__":
    unittest.main()
