"""Controller failure/serialization checks; never contact a board."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from run_c3_input_early_release import ORDER, compare_outputs, run_logged, write


def fixture(tmp_path):
    reference_dir, actual_dir = tmp_path / "reference", tmp_path / "actual"
    reference_dir.mkdir()
    actual_dir.mkdir()
    blob = bytes(range(100))
    (reference_dir / "outputs.bin").write_bytes(blob)
    (actual_dir / "outputs.bin").write_bytes(blob)
    frames = [{"frame_id": index, "input_index": index, "complete": True,
               "output_offset": index, "output_bytes": 1} for index in range(100)]
    reference = {"local_directory": str(reference_dir), "frames": copy.deepcopy(frames)}
    actual = {"local_directory": str(actual_dir), "frames": frames, "attempt": "fixture"}
    return reference, actual


def test_complete_snapshot_check_is_postprocessing(tmp_path):
    reference, actual = fixture(tmp_path)
    compare_outputs(actual, reference)
    assert actual["outputs_verified"] is True
    assert json.loads((tmp_path / "actual/output_check.json").read_text())["frames"] == 100


@pytest.mark.parametrize("mutate", [
    lambda ref, got: ref["frames"][1].update(input_index=0),
    lambda ref, got: ref["frames"][1].update(output_offset=0),
    lambda ref, got: ref["frames"].pop(),
    lambda ref, got: got["frames"][1].update(output_offset=0),
    lambda ref, got: got["frames"][1].update(complete=False),
    lambda ref, got: got["frames"][1].update(output_bytes=0),
])
def test_missing_or_aliased_snapshot_is_not_valid_output(tmp_path, mutate):
    reference, actual = fixture(tmp_path)
    mutate(reference, actual)
    with pytest.raises(ValueError):
        compare_outputs(actual, reference)
    assert actual.get("outputs_verified") is not True


def test_mismatch_records_counterexample_not_zero_time(tmp_path):
    reference, actual = fixture(tmp_path)
    (tmp_path / "actual/outputs.bin").write_bytes(bytes(100))
    with pytest.raises(ValueError):
        compare_outputs(actual, reference)
    failure = json.loads((tmp_path / "actual/first_difference.json").read_text())
    assert failure["frame_id"] == 1
    assert "latency_ms" not in failure


def test_failed_process_has_durable_log_and_no_retry(tmp_path):
    with pytest.raises(RuntimeError):
        run_logged([sys.executable, "-c", "raise RuntimeError('fixture')"], tmp_path, "failure")
    record = json.loads((tmp_path / "failure.json").read_text())
    assert record["returncode"] != 0


def test_timeout_is_unknown_completion_not_a_zero_sample(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        run_logged([sys.executable, "-c", "import time; time.sleep(1)"], tmp_path, "timeout", timeout=.02)
    record = json.loads((tmp_path / "timeout.json").read_text())
    assert record["returncode"] is None
    assert record["target_completion_unknown"] is True
    assert record["automatic_retry"] is False


def test_atomic_json_is_complete_and_stops_on_leftover_pending(tmp_path):
    path = tmp_path / "state.json"
    write(path, {"phase": "PREPARED"})
    assert json.loads(path.read_text()) == {"phase": "PREPARED"}
    (tmp_path / "state.json.pending").write_text("uncommitted")
    with pytest.raises(FileExistsError):
        write(path, {"phase": "COMPLETE"})
    assert json.loads(path.read_text())["phase"] == "PREPARED"


def test_formal_order_each_variant_occupies_each_position_once():
    assert len(ORDER) == 4
    for position in range(4):
        assert {row[position] for row in ORDER} == {"L1", "L2", "C1", "E1"}
