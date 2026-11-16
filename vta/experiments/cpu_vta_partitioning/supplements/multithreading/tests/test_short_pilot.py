"""Exercise the smaller protocol through the real receipt and recovery paths."""
from pathlib import Path

import pytest

import short_pilot as pilot
from cpu_vta_partitioning.common import read
from cpu_vta_partitioning.transport import TransportError
from test_campaign import environment, FakeBoard


@pytest.fixture
def short_environment(environment, monkeypatch):
    old_root, bundle, board = environment
    monkeypatch.setattr(pilot, "validate", lambda p: read(Path(p) / "manifest.json"))
    monkeypatch.setattr(pilot, "parent_identity", lambda: "test-parent")
    return old_root.parent / "short-pilot", bundle, board


def test_short_pilot_complete_resume_no_duplicate_and_small_inputs(short_environment):
    root, bundle, board = short_environment
    result = pilot.run(root, bundle, {}, board)
    assert result["component_signatures"] == 31
    assert result["mapping_configurations"] == 225
    assert result["input_images"] == 2
    assert not result["formal_significance"]
    assert read(root / "state.json")["status"] == "COMPLETE"
    assert all(v["same_configuration"] for v in result["existing_vs"].values())
    assert result["separate_vs_overlap"]["throughput_gain"] == 0
    calls = len(board.launches)
    assert pilot.run(root, bundle, {}, board) == result
    assert len(board.launches) == calls
    documents = [read(p)["task"] for p in (root / "accepted" / board.boot).glob("*.json")]
    assert {tuple(d["input_indices"]) for d in documents} == {(0,), (16,)}
    assert max(sum(d[k] for k in ("warmup", "scored", "tail")) for d in documents) == 60
    assert sum(d["kind"] == "calibration" for d in documents) == 31


def test_short_pilot_disconnect_recovers_without_duplicate_and_rejects_boot_change(short_environment):
    root, bundle, board = short_environment
    board.disconnect = True
    with pytest.raises(TransportError):
        pilot.run(root, bundle, {}, board)
    assert len(board.launches) == 1
    pilot.run(root, bundle, {}, board)
    assert len(board.launches) == len(list((root / "accepted" / board.boot).glob("*.json")))
    other = root.parent / "interrupted-pilot"
    # The fixture deduplicates by boot/task, while the real transport also keys
    # jobs by run directory. Use a fresh fake for this distinct batch.
    board = FakeBoard()
    board.disconnect = True
    with pytest.raises(TransportError):
        pilot.run(other, bundle, {}, board)
    board.boot = "00000000-0000-0000-0000-000000000099"
    with pytest.raises(ValueError, match="boot changed"):
        pilot.run(other, bundle, {}, board)
    assert (other / "accepted").exists() is False
    assert len(list((other / "raw").glob("*/*/collected/result.jsonl"))) == 1


def test_short_pilot_report_rejects_changed_raw(short_environment):
    root, bundle, board = short_environment
    pilot.run(root, bundle, {}, board)
    receipt = next((root / "accepted" / board.boot).glob("*.json"))
    raw = next(root / n for n in read(receipt)["files"] if n.endswith("result.jsonl"))
    raw.write_text(raw.read_text() + "\n")
    with pytest.raises(ValueError, match="missing or changed"):
        pilot.analyze(root)


def test_short_pilot_calibration_import_does_not_repeat_components(short_environment):
    source, bundle, board = short_environment
    pilot.run(source, bundle, {}, board)
    target = source.parent / "imported-pilot"
    other = FakeBoard()
    result = pilot.run(target, bundle, {}, other, calibration_source=source)
    assert result["calibration_import"]["run_dir"] == str(source.resolve())
    assert not (target / "records/calibration").exists()
    tasks = [read(p)["task"] for p in (target / "accepted" / other.boot).glob("*.json")]
    assert not any(t["kind"] in ("calibration", "qualify_component") for t in tasks)
    assert pilot.analyze(target) == result
