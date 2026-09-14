#!/usr/bin/env python3

from audit_vta_fused_program_reranker_holdout import ratio_is_exact_double


def test_ratio_is_exact_double_accepts_signed_and_zero_fields():
    assert ratio_is_exact_double(
        {"a": -20, "b": 6, "c": 0}, {"a": -10, "b": 3, "c": 0}
    )
    assert not ratio_is_exact_double({"a": -19}, {"a": -10})
