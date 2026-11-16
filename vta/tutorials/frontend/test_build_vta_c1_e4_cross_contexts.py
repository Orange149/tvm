"""Unit tests for the fail-closed E4 paired-context builder."""

import pytest
from tvm import relay

import build_vta_c1_e4_cross_contexts as e4build
import freeze_vta_c1_e4_fixed_tiles as freeze


def frozen_row(workload_id="Y10"):
    payload = freeze.build_freeze(
        freeze.DEFAULT_READINESS,
        {
            key: freeze.GROUPED / value
            for key, value in freeze.DEFAULT_QUALIFICATION_DIRS.items()
        },
    )
    return next(row for row in payload["rows"] if row["workload_id"] == workload_id)


def test_relay_contexts_have_same_conv_workload_and_distinct_inputs():
    row = frozen_row()
    plain = e4build.relay_module(row["workload"], False)
    residual = e4build.relay_module(row["workload"], True)

    assert len(plain["main"].params) == 2
    assert len(residual["main"].params) == 3
    assert isinstance(plain["main"].body, relay.Call)
    assert isinstance(residual["main"].body, relay.Call)


def test_dispatch_identity_accepts_only_exact_frozen_route():
    row = frozen_row()
    expected = e4build.route_from_frozen(row)["identity"]
    audit = {
        "schedule_hits": [
            {
                "candidate_id": row["candidate_id"],
                "complete_config_entity": expected["complete_config_entity"],
                "public_mode": expected["public_mode"],
                "implementation_mode": expected["implementation_mode"],
                "workload": expected["workload"],
            }
        ]
    }

    assert e4build.verify_dispatch_identity(audit, row)["exact"]

    audit["schedule_hits"][0]["implementation_mode"] = 1
    with pytest.raises(RuntimeError, match="implementation_mode"):
        e4build.verify_dispatch_identity(audit, row)


def test_dispatch_identity_rejects_missing_or_duplicate_hits():
    row = frozen_row()
    with pytest.raises(RuntimeError, match="exactly one"):
        e4build.verify_dispatch_identity({"schedule_hits": []}, row)
    with pytest.raises(RuntimeError, match="exactly one"):
        e4build.verify_dispatch_identity({"schedule_hits": [{}, {}]}, row)
