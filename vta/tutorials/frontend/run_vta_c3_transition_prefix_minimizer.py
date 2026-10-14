#!/usr/bin/env python3
"""Find the smallest frozen execution prefix that corrupts a mode-10 successor.

This wrapper reuses the independently reloading, correctness-only protocol in
``run_vta_c3_module_transition_diagnostic``.  It changes only the pre-frozen
case matrix and records itself as the source of the resulting artifact.
"""

from __future__ import annotations

import run_vta_c3_module_transition_diagnostic as diagnostic


diagnostic.SCHEMA = "c3_transition_prefix_minimizer_v1"


def repeat(candidate_id, seeds=diagnostic.SEEDS):
    return [(candidate_id, seed) for seed in seeds]


def cases(_all_ids):
    a1, a2, a3, target = diagnostic.A1, diagnostic.A2, diagnostic.A3, diagnostic.TARGET
    preload = [a1, a2, a3, target]
    return [
        {
            "case_id": "target_control",
            "preload": preload,
            "calls": [(target, 0)],
            "question": "Fresh-session target control.",
        },
        {
            "case_id": "a1_once_then_target",
            "preload": preload,
            "calls": [(a1, 0), (target, 0)],
            "question": "Can one A1 execution trigger the failure?",
        },
        {
            "case_id": "a1_three_then_target",
            "preload": preload,
            "calls": repeat(a1) + [(target, 0)],
            "question": "Can three A1 executions trigger the failure?",
        },
        {
            "case_id": "a2_once_then_target",
            "preload": preload,
            "calls": [(a2, 0), (target, 0)],
            "question": "Can one A2 execution trigger the failure?",
        },
        {
            "case_id": "a2_three_then_target",
            "preload": preload,
            "calls": repeat(a2) + [(target, 0)],
            "question": "Can three A2 executions trigger the failure?",
        },
        {
            "case_id": "a3_once_then_target",
            "preload": preload,
            "calls": [(a3, 0), (target, 0)],
            "question": "Can one A3 execution trigger the failure?",
        },
        {
            "case_id": "a3_three_then_target",
            "preload": preload,
            "calls": repeat(a3) + [(target, 0)],
            "question": "Can three A3 executions trigger the failure?",
        },
        {
            "case_id": "a1_a2_three_then_target",
            "preload": preload,
            "calls": repeat(a1) + repeat(a2) + [(target, 0)],
            "question": "Does the A1+A2 prefix trigger without A3?",
        },
        {
            "case_id": "a1_a3_three_then_target",
            "preload": preload,
            "calls": repeat(a1) + repeat(a3) + [(target, 0)],
            "question": "Does the A1+A3 prefix trigger without A2?",
        },
        {
            "case_id": "a2_a3_three_then_target",
            "preload": preload,
            "calls": repeat(a2) + repeat(a3) + [(target, 0)],
            "question": "Does the A2+A3 prefix trigger without A1?",
        },
        {
            "case_id": "one_each_then_target",
            "preload": preload,
            "calls": [(a1, 0), (a2, 0), (a3, 0), (target, 0)],
            "question": "Is one execution of each predecessor sufficient?",
        },
        {
            "case_id": "formal_prefix_reproduction",
            "preload": preload,
            "calls": repeat(a1) + repeat(a2) + repeat(a3) + [(target, 0)],
            "question": "Reproduce the exact formal prefix as a positive failure control.",
        },
    ]


def classify(results):
    return {
        "target_correct_by_case": {
            row["case_id"]: row["target_outputs_correct"] for row in results
        },
        "interpretation_rule": (
            "a target failure identifies a sufficient predecessor multiset/order; "
            "a pass only rejects that exact frozen prefix"
        ),
    }


if __name__ == "__main__":
    diagnostic.frozen_cases = cases
    diagnostic.classify = classify
    # Make the reused artifact sealer and command recorder bind this wrapper,
    # not the imported protocol implementation.
    diagnostic.__file__ = __file__
    diagnostic.run(diagnostic.parse_args())
