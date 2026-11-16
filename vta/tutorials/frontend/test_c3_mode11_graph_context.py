"""Regression tests for the C3 mode11 graph-context v6.1 diagnostic."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from c3_mode11_graph_context import (  # noqa: E402
    StrictDebugCursor,
    arrays_equal,
    classify_replay,
    core_reference,
    first_difference,
    frozen_graph_contract,
    install_debug_api_traps,
    ordinary_prefix_tap_graph,
    successor62_reference,
)


GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
GRAPH = (
    GROUPED / "20260926_c3_yolo512_evaluation_v5_graph_build_run01/incumbent/graph.json"
)


class FakeDebug:
    def __init__(self):
        self.executed = []
        self.read = []

    def _execute_node(self, node):
        self.executed.append(node)

    def _get_node_output(self, node, output):
        self.read.append((node, output))
        return np.array([node, output], dtype="int32")

    def run(self):
        raise AssertionError("original run must be replaced")


def test_graph_entry_mapping_and_sid12_lifetime():
    contract = frozen_graph_contract(json.loads(GRAPH.read_text(encoding="utf-8")))
    assert contract["abi_order"] == ["X", "p27", "p28", "Y"]
    assert {name: row["entry_id"] for name, row in contract["tensors"].items()} == {
        "X": 58, "p27": 59, "p28": 60, "Y": 61, "Y62": 62,
    }
    assert contract["tensors"]["X"]["storage_id"] == 12
    assert contract["tensors"]["Y62"]["storage_id"] == 12
    assert contract["storage_lifetime"]["assert_X_unchanged_after_node62"] is False
    assert contract["storage_lifetime"]["post_node62_target_output"] == (
        "UNOBSERVABLE_WITH_CURRENT_RUNTIME"
    )


def test_ordinary_prefix_tap_preserves_prefix_and_storage_plan():
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    tapped = ordinary_prefix_tap_graph(graph)
    assert tapped["nodes"][:62] == graph["nodes"][:62]
    assert tapped["nodes"][62]["op"] == "null"
    assert tapped["heads"] == [[58, 0, 0], [61, 0, 0]]
    assert tapped["attrs"] == graph["attrs"]
    assert tapped["node_row_ptr"] == graph["node_row_ptr"]
    assert tapped["arg_nodes"] == graph["arg_nodes"]


def test_successor62_reference_contract():
    y = np.zeros((1, 16, 16, 16, 1, 16), dtype="int8")
    output = successor62_reference(y)
    assert output.shape == (1, 255, 16, 16)
    assert output.dtype == np.float32
    # Frozen node62 keeps two raw fields per anchor and applies sigmoid elsewhere.
    reshaped = output.reshape(1, 3, 85, 16, 16)
    np.testing.assert_array_equal(reshaped[:, :, 2:4], 0.0)
    np.testing.assert_array_equal(reshaped[:, :, :2], 0.5)
    np.testing.assert_array_equal(reshaped[:, :, 4:], 0.5)


def test_monotonic_debug_cursor_and_last_node_read():
    fake = FakeDebug()
    cursor = StrictDebugCursor(fake)
    cursor.execute_node(58)
    np.testing.assert_array_equal(cursor.get_node_output(58), [58, 0])
    cursor.execute_node(61)
    with pytest.raises(ValueError, match="last executed"):
        cursor.get_node_output(58)
    with pytest.raises(ValueError, match="strictly increasing"):
        cursor.execute_node(61)
    assert fake.executed == [58, 61]


def test_debug_cursor_executes_every_intervening_prefix_node():
    fake = FakeDebug()
    cursor = StrictDebugCursor(fake)
    assert cursor.execute_through(3) == [0, 1, 2, 3]
    np.testing.assert_array_equal(cursor.get_node_output(3), [3, 0])
    assert cursor.execute_through(5) == [4, 5]
    assert fake.executed == [0, 1, 2, 3, 4, 5]


def test_forbidden_debug_apis_are_trapped():
    fake = FakeDebug()
    installed = install_debug_api_traps(fake)
    assert set(installed) == set(StrictDebugCursor.BANNED)
    for name in installed:
        with pytest.raises(RuntimeError, match="forbidden debug API"):
            getattr(fake, name)()


def test_first_difference_and_metadata_are_explicit():
    left = np.array([[1, 2]], dtype="int8")
    right = np.array([[1, 3]], dtype="int8")
    diff = first_difference(left, right)
    assert diff["flat_index"] == 1
    assert diff["index"] == [0, 1]
    assert diff["expected"] == 2 and diff["actual"] == 3
    assert arrays_equal(left, left.copy())
    assert first_difference(left, left.astype("int16"))["kind"] == "metadata"


def test_core_reference_is_three_argument_contract_only():
    x = np.zeros((1, 32, 16, 16, 1, 16), dtype="int8")
    weight = np.zeros((16, 32, 1, 1, 16, 16), dtype="int8")
    weight[0, 0, 0, 0, 0, 0] = 1
    x[0, 0, 0, 0, 0, 0] = -128
    output = core_reference(x, weight)
    assert output.shape == (1, 16, 16, 16, 1, 16)
    assert output.dtype == np.int8
    # Arithmetic -128 >> 8 is -1, then the A-only [0,127] clip produces zero.
    assert output[0, 0, 0, 0, 0, 0] == 0


def test_invalid_debug_prefix_precedes_replay_interpretation():
    matrix = {
        "R": {"status": "passed"},
        "S": {"status": "DEBUG_PREFIX_REFERENCE_INVALID"},
    }
    earliest, flags = classify_replay(matrix)
    assert earliest == "DEBUG_PREFIX_REFERENCE_INVALID"
    assert "DEBUG_PATH_CHANGED_REPRODUCTION" in flags
    assert "REFERENCE_REPLAY_NOT_RUN" in flags


def test_incumbent_B_gate_precedes_mode11_interpretation():
    matrix = {
        "R": {"status": "passed"},
        "B": {
            "incumbent_matches_reference": False,
            "mode11_matches_reference": False,
        },
    }
    earliest, _ = classify_replay(matrix)
    assert earliest == "REFERENCE_REPLAY_CONTRACT_INVALID"


def test_stable_B_divergence_and_p28_flags_are_nonexclusive():
    base = {
        "R": {"status": "passed"},
        "B": {
            "incumbent_matches_reference": True,
            "mode11_matches_reference": False,
            "mode11_repeat_matches_reference": False,
            "mode11_zero_p28_matches_incumbent": True,
        },
    }
    earliest, flags = classify_replay(base)
    assert earliest == "FUSED_FUNCTION_INDEPENDENT_BUFFER_DIVERGENCE"
    assert "P28_SENSITIVE" in flags
    base["B"]["mode11_zero_p28_matches_incumbent"] = False
    _, flags = classify_replay(base)
    assert "P28_ZERO_STILL_FAILS" in flags


def test_tap_runner_has_no_debug_or_timer_path():
    source = (HERE / "run_vta_c3_mode11_ordinary_prefix_tap.py").read_text(
        encoding="utf-8"
    )
    assert "from tvm.contrib.debugger" not in source
    assert "debug_executor.create(" not in source
    assert "time_evaluator(" not in source
    assert "debug_get_output(" not in source
    assert "run_individual(" not in source
    assert '"candidate_reselection": False' in source
    assert '"latency_values_collected": False' in source


def test_runner_has_no_timer_or_reselection_path():
    source = (HERE / "run_vta_c3_mode11_graph_context_diagnostic.py").read_text(
        encoding="utf-8"
    )
    assert "time_evaluator(" not in source
    assert "debug_get_output(" not in source
    assert "run_individual(" not in source
    assert '"candidate_reselected": False' in source
    assert '"latency_values_collected": False' in source

