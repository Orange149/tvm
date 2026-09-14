import json
from pathlib import Path

import run_vta_p7r121_tophub_adapter_diagnostic as p7r121


def parent_contract():
    return json.loads(Path(p7r121.DEFAULT_PARENT).read_text(encoding="utf-8"))


def test_two_distinct_frozen_variants():
    specs = p7r121.variant_specs(parent_contract())
    assert [row["variant"] for row in specs] == [
        "source_tophub",
        "residency_mode0_adapter",
    ]
    assert len({row["variant_id"] for row in specs}) == 2
    assert specs[0]["config_index"] == 526
    assert specs[1]["config_index"] == 650
    assert specs[0]["semantic_knobs"] == specs[1]["semantic_knobs"]


def test_outcome_is_fail_closed():
    def row(name, values):
        return {"variant": name, "seeds": [{"correct": value} for value in values]}

    assert p7r121.outcome(
        [row("source_tophub", [True] * 3), row("residency_mode0_adapter", [False] * 3)]
    ).endswith("v3_source_canary_required")
    assert p7r121.outcome(
        [row("source_tophub", [False] * 3), row("residency_mode0_adapter", [False] * 3)]
    ) == "both_fail_p7r120_remains_blocked"


def test_parent_and_sources_are_still_frozen():
    contract, ledger_hash = p7r121.verify_parent(p7r121.DEFAULT_PARENT)
    assert contract["sealed_reference"]["included_in_search_candidates"] is False
    assert len(ledger_hash) == 64
