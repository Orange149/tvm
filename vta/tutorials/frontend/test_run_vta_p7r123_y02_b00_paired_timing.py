import run_vta_p7r123_y02_b00_paired_timing as p7r123


def frozen_inputs():
    contract = p7r123.load_json(p7r123.DEFAULT_P7R122 / "contract.json")
    rows = p7r123.load_jsonl(p7r123.DEFAULT_P7R122 / "candidate_correctness.jsonl")
    return contract, rows


def test_only_two_correct_b00_identities_selected():
    entries = p7r123.select_entries(*frozen_inputs())
    assert [row["public_mode"] for row in entries] == ["original", "weight_resident_barrier"]
    assert {row["family_id"] for row in entries} == {"Y02B00"}


def test_seven_rounds_are_maximally_position_balanced():
    entries = p7r123.select_entries(*frozen_inputs())
    orders = p7r123.timing_orders(entries)
    assert p7r123.validate_orders(orders, entries)
    assert orders[0] == orders[2] == orders[4] == orders[6]
    assert orders[1] == orders[3] == orders[5]


def test_failed_p7r122_identities_are_excluded():
    entries = p7r123.select_entries(*frozen_inputs())
    ids = {row["candidate_id"] for row in entries}
    failed = {
        row["candidate_id"]
        for row in frozen_inputs()[1]
        if row["status"] != "passed"
    }
    assert ids.isdisjoint(failed)
