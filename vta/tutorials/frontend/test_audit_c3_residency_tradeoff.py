"""Offline auditor serialization regression; never dispatches device work."""

from audit_c3_residency_tradeoff import json_equivalent


def test_json_tuple_list_roundtrip_preserves_every_address():
    parsed = {"submissions": [{"uop_load_ranges": [(0, 224), (224, 448)]}]}
    recorded = {"submissions": [{"uop_load_ranges": [[0, 224], [224, 448]]}]}
    assert json_equivalent(parsed, recorded)
    recorded["submissions"][0]["uop_load_ranges"][1][1] = 449
    assert not json_equivalent(parsed, recorded)


def test_missing_field_is_not_ignored():
    assert not json_equivalent({"mac": 112, "reset": 112}, {"mac": 112})
