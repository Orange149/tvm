"""Local-only audit tests; synthetic observations never become real evidence."""
import copy
import unittest
from c3_schedule_evidence_audit import (
    SEEDS, aggregate, compare, identifiability, load_address_spans, source_coverage, weight_plan,
)


def calls():
    return [{"correct": True, "boot_id": "synthetic", "call_kind": "timing",
             "round": r, "seed": seed, "program": p, "latency_ms": t}
            for r in range(6) for p, t in (("base", 10.0), ("candidate", 9.0)) for seed in SEEDS]


def request(offset=0, x=2, y=2, stride=3, multiplicity=1, memory="wgt", element=256):
    g = {"direction": "load", "memory_name": memory, "element_bytes": element,
         "ddr_bounds_checked": True, "ddr_offset_elements": offset, "x_size": x, "y_size": y,
         "x_stride": stride, "sram_offset_elements": 0, "padding": [0, 0, 0, 0],
         "multiplicity": multiplicity}
    return {"request_groups": [g], "totals": {"load_buffer_2d_" + memory + "_bytes": x * y * element * multiplicity}}


class RelationsTests(unittest.TestCase):
    def test_residency_request_equation(self):
        for r, count in ((0, 112), (8, 64), (12, 40), (13, 34)):
            d = weight_plan(16, r, 1, 7, 18432, 262144)
            self.assertEqual(d["whole_group_loads"], count)
            self.assertEqual(d["request_bytes"], count * 18432)

    def test_two_slots_trade_capacity_not_request_count(self):
        one, two = [weight_plan(16, 12, k, 7, 18432, 262144) for k in (1, 2)]
        self.assertEqual(one["request_bytes"], two["request_bytes"])
        self.assertEqual(two["nominal_allocated_bytes"], 258048)
        self.assertEqual(two["resident_bytes"], 221184)
        self.assertEqual(two["max_resident_with_these_slots"], 12)
        self.assertTrue(two["not_an_overlap_or_time_prediction"])

    def test_too_many_resident_and_two_slots(self):
        with self.assertRaises(ValueError):
            weight_plan(16, 13, 2, 7, 18432, 262144)

    def test_requires_one_live_stream_slot(self):
        with self.assertRaises(ValueError):
            weight_plan(16, 16, 0, 7, 18432, 262144)

    def test_noninteger_invalid_geometry(self):
        with self.assertRaises(ValueError):
            weight_plan(16, 12.0, 2, 7, 18432, 262144)

    def test_dma_stride_preserves_holes(self):
        counts, d = source_coverage(request(), "wgt")
        self.assertEqual(set(counts), {0, 1, 3, 4})
        self.assertFalse(d["complete_contiguous_source"])
        self.assertEqual(d["unique_bytes"], 1024)

    def test_dma_duplicate_requests(self):
        counts, d = source_coverage(request(multiplicity=7, stride=2), "wgt")
        self.assertEqual(set(counts.values()), {7})
        self.assertEqual(d["repeat_histogram_elements"], {7: 4})
        self.assertEqual(d["requested_bytes"], 7168)
        self.assertTrue(d["not_physical_bus_traffic"])

    def test_no_unknown_address_as_pass(self):
        d = request()
        d["request_groups"][0]["ddr_bounds_checked"] = False
        with self.assertRaises(ValueError):
            source_coverage(d, "wgt")

    def test_bytes_and_unit_check(self):
        d = request()
        d["totals"]["load_buffer_2d_wgt_bytes"] = 4
        with self.assertRaises(ValueError):
            source_coverage(d, "wgt")

    def test_padding_generates_sram_not_ddr_bytes(self):
        d = request(memory="inp", element=16)
        d["request_groups"][0]["padding"] = [1, 1, 1, 1]
        _, source = source_coverage(d, "inp")
        self.assertEqual(source["requested_bytes"], 64)
        self.assertEqual(load_address_spans(d)["inp"], 256)

    def test_round_median_and_pairing(self):
        d = aggregate(calls(), ["base", "candidate"])
        self.assertEqual(d["median_ms"]["candidate"], 9)
        p = compare(d, "candidate", "base")
        self.assertAlmostEqual(p["paired_gain_pct"], 10)
        self.assertEqual(p["wins"], 6)

    def test_ties_are_not_wins(self):
        cc = calls()
        for c in cc:
            c["latency_ms"] = 10
        self.assertEqual(compare(aggregate(cc, ["base", "candidate"]), "candidate", "base")["wins"], 0)

    def test_no_incomplete_rounds(self):
        with self.assertRaises(ValueError):
            aggregate(calls()[:-1], ["base", "candidate"])

    def test_no_duplicate_seed_as_three_observations(self):
        cc = calls()
        cc[1]["seed"] = cc[0]["seed"]
        with self.assertRaises(ValueError):
            aggregate(cc, ["base", "candidate"])

    def test_no_boot_splicing(self):
        cc = calls()
        cc[-1]["boot_id"] = "other"
        with self.assertRaises(ValueError):
            aggregate(cc, ["base", "candidate"])

    def test_no_bad_or_missing_latency(self):
        for val in (None, 0, float("nan"), -1):
            cc = calls()
            cc[-1]["latency_ms"] = val
            with self.assertRaises(ValueError):
                aggregate(cc, ["base", "candidate"])

    def test_no_invalid_correctness(self):
        cc = calls()
        cc[0]["correct"] = False
        with self.assertRaises(ValueError):
            aggregate(cc, ["base", "candidate"])

    def test_selection_excluded_from_confirmation(self):
        cc = calls()
        cc += [{**c, "call_kind": "selection_timing", "latency_ms": 0.01} for c in calls()]
        self.assertEqual(aggregate(cc, ["base", "candidate"])["median_ms"]["base"], 10)

    def test_rank_is_not_validation(self):
        d = identifiability([[1, 1, 0], [1, 0, 1], [1, 2, 3]], 0, True)
        self.assertTrue(d["algebraically_identifiable"])
        self.assertFalse(d["fit_permitted"])

    def test_collinear_coefficients_rejected(self):
        d = identifiability([[1, 1, 1], [1, 2, 1], [1, 3, 1]], 2, True)
        self.assertEqual(d["rank"], 2)
        self.assertFalse(d["fit_permitted"])

    def test_missing_overlap_feature_blocks_five_cost_claim(self):
        self.assertFalse(identifiability([[1, 1], [1, 2]], 2, False)["fit_permitted"])

    def test_synthetic_data_never_mutated(self):
        cc = calls()
        before = copy.deepcopy(cc)
        aggregate(cc, ["base", "candidate"])
        self.assertEqual(cc, before)


if __name__ == "__main__":
    unittest.main()
