"""No-SSH/FPGA tests of the isolated board adapter and frozen call boundaries."""

import json
from collections import Counter

import numpy as np
import pytest

from c3_five_tradeoff_board import Runner, PROGRAMS, ORDERS, NEW, expectations, verify_profile
from test_c3_residency_tradeoff_board import Buffer


def test_native_call_preserves_new_program_label_and_profile_boundary(tmp_path):
    runner = Runner.__new__(Runner)
    runner.out, runner.boot, runner.rows = tmp_path, "one-boot", []
    (tmp_path / "programs").mkdir()
    runner.x, runner.weight, runner.add, runner.y = {0: np.array([1])}, np.array([2]), np.array([3]), {0: np.array([4])}
    runner.totals = {NEW: {"load_buffer_2d_wgt_calls": 40}}
    trace, calls = [], []
    buffers = tuple(Buffer(np.array([v]), trace) for v in (1, 2, 3, -113))
    def target(*args):
        calls.append(args)
        args[-1].array = np.array([4])
        trace.append("target")
    profile = {"load_buffer_2d_wgt_calls": 40, "push_gemm_op_calls": 224,
               "driver_run_calls": 1, "mem_copy_to_host_calls": 0}
    funcs = {"vta.runtime.profiler_clear": lambda: trace.append("clear"),
             "vta.runtime.profiler_status": lambda: (trace.append("snapshot") or json.dumps(profile))}
    from c3_residency_tradeoff_board import SYMBOL
    row = runner.call({SYMBOL: target}, funcs, NEW, buffers, 0, "correctness", "mock")
    assert len(calls) == 1 and row["program"] == NEW
    assert trace.index("target") < trace.index("snapshot") < trace.index("read")
    assert (tmp_path / "programs" / (NEW + ".jsonl")).is_file()
    dispatch = [json.loads(x) for x in (tmp_path / "dispatch.jsonl").read_text().splitlines()]
    assert dispatch[0]["program"] == NEW
    assert row["latency_ms"] is None


def test_count_alias_does_not_allow_wrong_gemm_commands():
    assert expectations(NEW)["wgt_loads"] == 40
    good = {"reset": 112, "mac": 112, "wgt_loads": 40}
    profile = {"push_gemm_op_calls": 224, "driver_run_calls": 1}
    assert verify_profile(NEW, profile, good) == {"mac": 112, "wgt_loads": 40}
    with pytest.raises(ValueError, match="RESET/MAC"):
        verify_profile(NEW, profile, {**good, "mac": 896})


def test_four_program_approximate_balance_and_pair_order():
    assert len(ORDERS) == 6
    assert all(set(row) == set(PROGRAMS) for row in ORDERS)
    assert sum(row.index(NEW) < row.index("R13-G") for row in ORDERS) == 3
    assert sum(row.index(NEW) < row.index("Input-7x7") for row in ORDERS) == 3
    positions = Counter((name, i) for row in ORDERS for i, name in enumerate(row))
    assert set(positions.values()) == {1, 2}


def test_no_board_for_helper_binary_certificate():
    from c3_residency_tradeoff_board import helper_certificate
    data = helper_certificate()
    assert all(not row["defines_VTA_runtime"] for row in data.values())


def test_native_call_rejects_mismatched_source_requests(tmp_path):
    from c3_residency_tradeoff_board import validate_request_profile
    with pytest.raises(ValueError, match="request mismatch"):
        validate_request_profile({"load_buffer_2d_wgt_calls": 34, "mem_copy_to_host_calls": 0},
                                 {"load_buffer_2d_wgt_calls": 40})
