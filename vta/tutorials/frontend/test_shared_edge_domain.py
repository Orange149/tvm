"""Contract and Pareto regression checks; no board or model compilation."""
import random
import unittest

from audit_shared_edge_domain import numeric_sum, pareto_ids
from audit_vta_endpoint_boundaries import edge


class ParetoTests(unittest.TestCase):
    def test_against_brute_force_including_ties(self):
        rng = random.Random(260908)
        for size in range(50):
            rows = [{"topology_id":str(i), "m1_score_ms":rng.randrange(8),
                     "boundary_payload_bytes":rng.randrange(8)} for i in range(size)]
            expected = []
            for row in rows:
                a = (row["m1_score_ms"], row["boundary_payload_bytes"])
                dominated = False
                for other in rows:
                    b = (other["m1_score_ms"], other["boundary_payload_bytes"])
                    dominated |= b[0] <= a[0] and b[1] <= a[1] and b != a
                if not dominated:
                    expected.append(row["topology_id"])
            self.assertEqual(set(pareto_ids(rows)), set(expected))

    def test_repeated_primitive_invocations(self):
        self.assertEqual(numeric_sum([({"read_bytes":16},3),
                                      ({"read_bytes":4,"write_bytes":8},2)]),
                         {"read_bytes":56,"write_bytes":16})


class EdgeTests(unittest.TestCase):
    def pair(self, width):
        contract = {"slots":[{"shape":[1,width],"dtype":"float32"}]}
        return ({"segment_id":"cpu:00:00","device":"cpu","output_contract":contract},
                {"segment_id":"vta:01:02","device":"vta","input_contract":contract})

    def test_alignment_is_not_always_twice_payload(self):
        row=edge(*self.pair(3))
        self.assertEqual(row["total_payload_bytes"],12)
        self.assertEqual(row["theoretical_k2_aligned_slot_bytes"],512)

    def test_aligned_payload(self):
        row=edge(*self.pair(64))
        self.assertEqual(row["theoretical_k2_aligned_slot_bytes"],2*row["total_payload_bytes"])

    def test_mismatched_contract_rejected(self):
        producer,_=self.pair(3)
        _,consumer=self.pair(4)
        with self.assertRaisesRegex(AssertionError,"contract mismatch"):
            edge(producer,consumer)


if __name__ == "__main__":
    unittest.main()
