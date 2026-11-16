"""No-FPGA tests of call ordering, calibration quarantine and old-anchor safety."""

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest
import tvm

import c3_residency_tradeoff as p
import c3_residency_tradeoff_board as b
import c3_residency_tradeoff_analysis as analysis
from test_c3_residency_tradeoff import rows


class Buffer:
    def __init__(self, array, log):
        self.array, self.log = np.array(array, copy=True), log
        self.handle = ctypes.c_void_p(id(self))

    def copyfrom(self, array):
        self.array = np.array(array, copy=True)
        self.log.append("prepare")

    def numpy(self):
        self.log.append("read")
        return self.array


def test_profile_snapshot_precedes_any_read_and_forwards_once(tmp_path):
    runner = b.Runner.__new__(b.Runner)
    runner.out, runner.boot, runner.rows = tmp_path, "one-boot", []
    (tmp_path / "programs").mkdir()
    runner.x, runner.weight, runner.add, runner.y = {0: np.array([1])}, np.array([2]), np.array([3]), {0: np.array([4])}
    runner.totals = {"R13-G": {"load_buffer_2d_wgt_calls": 34}}
    trace = []
    arrays = tuple(Buffer(np.array([x]), trace) for x in (1, 2, 3, -113))
    profile = {"push_gemm_op_calls": 224, "driver_run_calls": 1,
               "load_buffer_2d_wgt_calls": 34, "mem_copy_to_host_calls": 0}
    funcs = {"vta.runtime.profiler_clear": lambda: trace.append("clear"),
             "vta.runtime.profiler_status": lambda: (trace.append("snapshot") or json.dumps(profile))}
    invocations = []

    def target(*received):
        invocations.append(received)
        received[-1].array = np.array([4])
        trace.append("target")

    row = runner.call({b.SYMBOL: target}, funcs, "R13-G", arrays, 0, "correctness", "mock")
    assert len(invocations) == 1
    assert all(x is y for x, y in zip(invocations[0], arrays))
    assert trace.index("target") < trace.index("snapshot") < trace.index("read")
    assert row["correct"] and row["arguments_unchanged"]
    assert row["latency_ms"] is None


def test_sentinel_differs_from_every_expected_element():
    values = np.array([-113, 0, 112, 127], dtype="int8")
    assert np.all(b.sentinel(values) != values)


def test_statics_use_actual_post_fold_addresses():
    from c3_residency_tradeoff_evidence import check_runtime
    assert check_runtime()["capacity_elements"]["WGT"] == 1024
    root = b.HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20261012_c3_residency_instruction_tradeoff_run01"
    for name in ("R0-G", "R8-G", "R12-G", "compat_R13-G"):
        cert = p.read(root / "local" / name / "comparison_certificate.json")
        assert cert["status"] == "PASSED"
        assert len(cert["queue_certificate"]["cases"]) == 112
        assert cert["max_kernel_uops"] == 224
    assert p.read(root / "organization_certificate.json")["transform_structurally_reproduces_new_body"]


def test_comparison_rejects_substantive_sram_difference():
    from c3_residency_tradeoff_evidence import function, ANCHORS
    old = function(ANCHORS["R13-G"])
    from tvm import tir
    changed = {"done": False}

    def post(node):
        if (not changed["done"] and isinstance(node, tir.Call)
                and node.op.name == "tir.vta.uop_push" and int(node.args[1]) == 0
                and int(node.args[0]) == 0):
            args = list(node.args)
            args[4] = args[4] + 1
            changed["done"] = True
            return tir.Call(node.dtype, node.op, args, node.span)
    new = old.with_body(tir.stmt_functor.ir_transform(old.body, None, post))
    assert changed["done"]
    assert not tvm.ir.structural_equal(old, new, map_free_vars=True)


def test_model_seal_does_not_open_validation_journals(tmp_path, monkeypatch):
    (tmp_path / "board/programs").mkdir(parents=True)
    p.write_new(tmp_path / "board/summary.json", {"status": "FORMAL_ACQUISITION_COMPLETE",
                "timed_calls": 108, "warmup_calls": 108, "boot_id": "boot-one"})
    synthetic = rows({"F13-B": 7, "R13-G": 5, "R0-G": 8, "R8-G": 6, "R12-G": 5.2, "Input-7x7": 5.1})
    for name in p.CALIBRATION:
        data = [r for r in synthetic if r["program"] == name]
        for r in data:
            r["model_features"] = {"wgt_loads": p.expectations(name)["wgt_loads"],
                                   "mac": p.expectations(name)["mac"]}
        (tmp_path / "board/programs" / (name + ".jsonl")).write_text(
            "".join(json.dumps(r) + "\n" for r in data))
    accessed = []
    original = analysis.journal

    def guarded(path):
        accessed.append(Path(path).stem)
        assert Path(path).stem in p.CALIBRATION
        return original(path)
    monkeypatch.setattr(analysis, "journal", guarded)
    model = analysis.calibration_seal(tmp_path)
    assert model["status"] == "MODEL_SEALED"
    assert set(accessed) == set(p.CALIBRATION)
    assert (tmp_path / "analysis/model.json").is_file()
    assert (tmp_path / "analysis/seal_receipt.json").is_file()


def test_helper_binary_has_no_second_runtime():
    certificate = b.helper_certificate()
    assert not certificate["timer"]["defines_VTA_runtime"]
    assert not certificate["diagnostic"]["defines_VTA_runtime"]


def test_unknown_or_contaminated_profile_does_not_pass():
    with pytest.raises(ValueError):
        b.validate_request_profile({"mem_copy_to_host_calls": 1}, {})
    with pytest.raises(ValueError):
        b.validate_request_profile({"mem_copy_to_host_calls": 0, "template_replay_hits": 1}, {})
