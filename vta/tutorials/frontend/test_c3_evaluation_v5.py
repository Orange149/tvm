import unittest

import c3_evaluation_v5 as v5


def policies():
    result = {}
    for task in ("task_a", "task_b"):
        production = [task + "_inc"] + [task + "_p{}".format(index) for index in range(7)]
        legacy = [task + "_l{}".format(index) for index in range(6)]
        orders = {method: production[1:] for method in v5.METHODS}
        result[task] = {
            "incumbent_program_id": production[0],
            "production_program_pool": production,
            "legacy_controlled_program_pool": legacy,
            "orders": orders,
        }
    return result


class EvaluationV5Test(unittest.TestCase):
    def test_call_counts(self):
        self.assertEqual(v5.validate_call_counts(policies()), {
            "production_correctness": 48,
            "legacy_correctness": 36,
            "selection": 48,
            "production_reference": 288,
            "legacy_reference": 216,
        })

    def test_selection_has_one_seed_per_round_and_full_coverage(self):
        schedule = v5.selection_schedule(policies())
        for round_index, seed in enumerate(v5.SEEDS):
            rows = [row for row in schedule if row["round"] == round_index]
            self.assertEqual(len(rows), 16)
            self.assertEqual({row["seed"] for row in rows}, {seed})
            self.assertEqual(len({(row["execution_task_id"], row["program_id"])
                                  for row in rows}), 16)

    def test_exact_tie_prefers_incumbent(self):
        frozen = policies()
        rows = []
        for spec in v5.selection_schedule(frozen):
            rows.append({**spec, "latency_ms": 1.0})
        decisions = v5.select_winners(frozen, rows)
        for task, row in decisions.items():
            for method in v5.METHODS:
                for budget in v5.BUDGETS:
                    self.assertEqual(
                        row["methods"][method]["B{}".format(budget)]["search_winner"],
                        frozen[task]["incumbent_program_id"],
                    )

    def test_confirmation_self_comparison_is_not_duplicated(self):
        frozen = policies()
        winners = {task: row["incumbent_program_id"] for task, row in frozen.items()}
        schedule = v5.confirmation_schedule(frozen, winners)
        self.assertEqual(len(schedule), 2 * 6 * 3)
        self.assertEqual(len({row["ordinal"] for row in schedule}), len(schedule))

    def test_state_machine_rejects_skip_and_rewind(self):
        self.assertEqual(
            v5.transition_state("PRE_LATENCY_FROZEN", "SELECTION_RUNNING"),
            "SELECTION_RUNNING",
        )
        with self.assertRaises(ValueError):
            v5.transition_state("PRE_LATENCY_FROZEN", "WINNERS_SEALED")
        with self.assertRaises(ValueError):
            v5.transition_state("REFERENCE", "CONFIRMATION")


if __name__ == "__main__":
    unittest.main()
