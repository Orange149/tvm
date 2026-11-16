import pytest

from c3_completion_policy import (
    DEPENDENCY_TOKENS,
    SCOPED_FORCE_SERIAL,
    TRANSITION_SAFE_FORCE_SERIAL,
    completion_policy_for_candidate,
    policy_set_compatibility,
)


def test_registered_modes_resolve_without_numeric_mode_dependency():
    assert completion_policy_for_candidate({"candidate_mode": "original"})["name"] == DEPENDENCY_TOKENS
    assert completion_policy_for_candidate(
        {"residence_mode": "input_weight_resident_composable"}
    )["name"] == SCOPED_FORCE_SERIAL
    transition_safe = completion_policy_for_candidate(
        {"residence_mode": "input_weight_resident_transition_safe"}
    )
    assert transition_safe["name"] == TRANSITION_SAFE_FORCE_SERIAL
    assert transition_safe["entry_drains_before_debug"] == 1
    assert transition_safe["production_eligible"] is True


def test_explicit_policy_must_match_registered_mode():
    with pytest.raises(ValueError, match="mode/policy mismatch"):
        completion_policy_for_candidate({
            "candidate_mode": "original", "completion_policy": SCOPED_FORCE_SERIAL,
        })


def test_diagnostic_and_legacy_policies_are_not_production_compatible():
    result = policy_set_compatibility(["global_force_serial_diagnostic_v1"])
    assert result["compatible"] is False
    assert result["reasons"] == [
        "policy_not_production_eligible:global_force_serial_diagnostic_v1"
    ]


def test_predecessor_sensitive_mode_ten_is_not_production_compatible():
    result = policy_set_compatibility([SCOPED_FORCE_SERIAL])
    assert result["compatible"] is False
    assert result["reasons"] == [
        "policy_not_production_eligible:{}".format(SCOPED_FORCE_SERIAL)
    ]
