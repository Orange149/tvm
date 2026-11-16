"""Protocol/accounting tests; no TVM imports or target execution."""

from collections import Counter
from copy import deepcopy
import math
from statistics import median
import unittest

import c3_tps_aggregation_protocol as protocol


def identities():
    return {p: {"module_sha256": p * 64, "runtime": {"sha256": "common"}}
            for p in ("B", "N", "G")}


def passing_results(ledger, formal=None):
    rows = deepcopy(ledger["calls"])
    for row in rows:
        row.update(status="PASS", target_calls=1, elapsed_ms=99999.0)
        if row["phase"].startswith("diagnostic_"):
            row["elapsed_ms"] = None
        if row["phase"] == "formal":
            value = formal[row["program"]][row["round"] - 1] if formal else 10.0
            row["elapsed_ms"] = value + {
                0: -0.125, 20250901: 0.0, 20260910: 0.125}[row["seed"]]
    return rows


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.ledger = protocol.manifest("attempt01", identities())

    def test_180_calls_30_sessions_and_phase_budget(self):
        ledger = self.ledger
        self.assertEqual(ledger["total_target_calls"], 180)
        self.assertEqual(ledger["formal_samples"], 54)
        self.assertEqual(len(ledger["sessions"]), 30)
        self.assertEqual([r["call_index"] for r in ledger["calls"]], list(range(1, 181)))
        self.assertEqual(Counter(r["program"] for r in ledger["calls"]), {"B": 60, "N": 60, "G": 60})
        self.assertEqual(ledger["phase_counts"], {
            "qualification": 9, "weight_check": 9,
            "diagnostic_prefix": 27, "diagnostic_warm": 27,
            "warmup": 54, "formal": 54})
        self.assertEqual(Counter(s["program"] for s in ledger["sessions"]), {"B": 10, "N": 10, "G": 10})
        for session in ledger["sessions"]:
            rows = [r for r in ledger["calls"] if r["session_id"] == session["session_id"]]
            self.assertEqual(len(rows), 6)
            self.assertEqual([r["position"] for r in rows], list(range(1, 7)))
            self.assertEqual([r["call_index"] for r in rows],
                             list(range(session["first_call_index"], session["last_call_index"] + 1)))

    def test_all_six_orders_balance_positions_and_directed_neighbours(self):
        orders = self.ledger["round_orders"]
        self.assertEqual(orders, [list(x) for x in ("BNG", "NGB", "GBN", "GNB", "NBG", "BGN")])
        self.assertEqual(len({tuple(order) for order in orders}), 6)
        for program in "BNG":
            self.assertEqual([sum(order[i] == program for order in orders) for i in range(3)], [2, 2, 2])
        neighbours = Counter((row[i], row[i + 1]) for row in orders for i in range(2))
        self.assertEqual(set(neighbours.values()), {2})
        self.assertEqual(len(neighbours), 6)
        formal_sessions = [s for s in self.ledger["sessions"] if s["phase"] == "formal"]
        for r, order in enumerate(orders, 1):
            self.assertEqual([s["program"] for s in formal_sessions if s["round"] == r], order)

    def test_seed_once_per_round_and_matching_warm_positions(self):
        rotations = [[0, 20250901, 20260910], [20250901, 20260910, 0], [20260910, 0, 20250901]]
        for session in self.ledger["sessions"]:
            rows = [r for r in self.ledger["calls"] if r["session_id"] == session["session_id"]]
            if session["phase"] == "qualification":
                self.assertEqual([r["seed"] for r in rows], rotations[0] + [0, 0, 0])
                self.assertEqual([r["weight"] for r in rows], ["W0"] * 4 + ["W1", "W0"])
            else:
                seeds = rotations[session["rotation"]]
                self.assertEqual([r["seed"] for r in rows], seeds + seeds)
            if session["phase"] == "formal":
                self.assertEqual([r["phase"] for r in rows], ["warmup"] * 3 + ["formal"] * 3)
                self.assertEqual({r["collection"] for r in rows}, {"minimal"})
            else:
                self.assertEqual({r["collection"] for r in rows}, {"detailed"})
            self.assertEqual([r["timed"] for r in rows], [r["phase"] == "formal" for r in rows])

    def test_frozen_identity_is_copied_and_unsupported_pool_rejected(self):
        source = identities()
        ledger = protocol.manifest("identity", source)
        source["B"]["runtime"]["sha256"] = "changed"
        self.assertEqual(ledger["programs"]["B"]["runtime"]["sha256"], "common")
        ledger["calls"][0]["program_identity"]["runtime"]["sha256"] = "changed-row"
        self.assertEqual(ledger["programs"]["B"]["runtime"]["sha256"], "common")
        for bad in ({"B": {}}, {"B": {}, "N": {}, "G": {}, "Q1": {}}, {"B": [], "N": {}, "G": {}}):
            with self.subTest(pool=bad), self.assertRaises(ValueError):
                protocol.manifest("bad", bad)
        with self.assertRaises(ValueError):
            protocol.manifest("bad", {"B": {"x": math.nan}, "N": {}, "G": {}})


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.ledger = protocol.manifest("attempt01", identities())
        self.rows = passing_results(self.ledger)

    def test_complete_attempt_passes_but_missing_reordered_duplicate_do_not(self):
        self.assertEqual(protocol.validate_results(self.ledger, self.rows)["target_calls"], 180)
        variants = [self.rows[:-1], self.rows + [self.rows[-1]],
                    [self.rows[1], self.rows[0]] + self.rows[2:]]
        duplicate = deepcopy(self.rows)
        duplicate[1] = deepcopy(duplicate[0])
        variants.append(duplicate)
        for rows in variants:
            with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                protocol.validate_results(self.ledger, rows)

    def test_row_or_frozen_manifest_identity_change_is_rejected(self):
        for key, value in (("program_identity", {"module_sha256": "wrong"}),
                           ("seed", 1), ("session_id", "wrong"), ("collection", "minimal"),
                           ("position", 2), ("timed", 0)):
            rows = deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_results(self.ledger, rows)
        ledger = deepcopy(self.ledger)
        ledger["calls"][0]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "FROZEN_MANIFEST_CHANGED"):
            protocol.validate_results(ledger, self.rows)

    def test_failed_or_multiple_target_calls_are_rejected(self):
        for update in ({"status": "FAIL"}, {"target_calls": 2}, {"target_calls": True}):
            rows = deepcopy(self.rows)
            rows[0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                protocol.validate_results(self.ledger, rows)

    def test_nonfinite_nonpositive_and_missing_times_are_rejected(self):
        first_formal = next(i for i, r in enumerate(self.rows) if r["phase"] == "formal")
        for index in (0, first_formal):
            for value in (None, 0, -1, math.inf, -math.inf, math.nan, True, "1.0"):
                rows = deepcopy(self.rows)
                rows[index]["elapsed_ms"] = value
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    protocol.validate_results(self.ledger, rows)
        del self.rows[0]["elapsed_ms"]
        with self.assertRaises(ValueError):
            protocol.validate_results(self.ledger, self.rows)

    def test_diagnostic_null_is_allowed_but_invalid_time_is_not(self):
        diagnostic = next(i for i, r in enumerate(self.rows) if r["phase"] == "diagnostic_warm")
        self.assertIsNone(self.rows[diagnostic]["elapsed_ms"])
        protocol.validate_results(self.ledger, self.rows)
        self.rows[diagnostic]["elapsed_ms"] = -1
        with self.assertRaises(ValueError):
            protocol.validate_results(self.ledger, self.rows)

    def test_only_formal_samples_enter_paired_analysis(self):
        values = {"B": [10, 12, 14, 16, 18, 20],
                  "N": [11, 14, 15, 18, 19, 23],
                  "G": [9, 11, 13, 15, 17, 19]}
        report = protocol.analyse(self.ledger, passing_results(self.ledger, values))
        self.assertEqual(report["round_medians"], values)
        self.assertEqual(report["T"]["B"], 15)
        self.assertEqual(report["T"]["G"], 14)
        expected = [100 * (1 - g / b) for g, b in zip(values["G"], values["B"])]
        self.assertEqual(report["pairwise"]["G/B"]["round_gains_percent"], expected)
        self.assertEqual(report["pairwise"]["G/B"]["Gain_percent"], median(expected))
        self.assertNotEqual(median(expected), 100 * (1 - report["T"]["G"] / report["T"]["B"]))
        self.assertEqual(report["pairwise"]["G/B"]["absolute_ms"], 1)
        self.assertEqual(report["pairwise"]["G/B"]["wins"], 6)
        self.assertEqual(report["dispersion"]["B"]["MAD_ms"], 3)
        self.assertTrue(report["engineering_goal_met"])
        self.assertFalse(report["statistical_significance_claimed"])

    def test_four_wins_or_only_g_over_n_is_not_primary_success(self):
        cases = [
            {"B": [10] * 6, "N": [12] * 6, "G": [9, 9, 9, 9, 10, 11]},
            {"B": [10] * 6, "N": [12] * 6, "G": [10] * 6},
        ]
        for values in cases:
            report = protocol.analyse(self.ledger, passing_results(self.ledger, values))
            self.assertFalse(report["engineering_goal_met"])
            self.assertTrue(report["pairwise"]["G/N"]["engineering_goal_met"])
        self.assertEqual(protocol.analyse(self.ledger, passing_results(self.ledger, cases[0]))
                         ["pairwise"]["G/B"]["wins"], 4)


if __name__ == "__main__":
    unittest.main()
