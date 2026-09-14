"""Tests for exact Relay-side VTA residency routing."""

import json
from pathlib import Path

import pytest

from tvm.autotvm.task.space import ConfigEntity

from vta.top.residency_dispatch import ExplicitResidencyDispatch, resolve_schedule_route


ROOT = Path(__file__).resolve().parent
R50B = ROOT / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
R50B /= "20260912_p7r220_r50b_full_local_oracle_completion_run01/candidates_v2.jsonl"


def route():
    return next(
        row for row in map(json.loads, R50B.read_text(encoding="utf-8").splitlines())
        if row["candidate_id"].startswith("6b1b47da86ed")
    )


def config(payload=None):
    payload = payload or route()["identity"]["complete_config_entity"]
    return ConfigEntity.from_json_dict(dict(payload, index=-1))


class FakeOp:
    def __init__(self, attrs=None, inputs=None):
        self.attrs = attrs or {}
        self.input_tensors = inputs or []


class FakeTensor:
    def __init__(self, op):
        self.op = op


def output_for(workload):
    conv = FakeTensor(FakeOp({"workload": workload}))
    return FakeTensor(FakeOp(inputs=[conv]))


def test_default_is_original():
    assert resolve_schedule_route(config(), [output_for(route()["identity"]["workload"])]) == "original"


def test_exact_route_binds_config_and_schedule():
    raw = route()
    with ExplicitResidencyDispatch([raw]) as dispatch:
        selected = dispatch.query(None, tuple(raw["identity"]["workload"]))
        mode = resolve_schedule_route(selected, [output_for(raw["identity"]["workload"])])
    assert mode == 4
    assert dispatch.summary()["all_routes_scheduled"]
    assert len(dispatch.config_hits) == len(dispatch.schedule_hits) == 1


def test_matched_workload_rejects_wrong_config():
    raw = route()
    wrong = json.loads(json.dumps(raw["identity"]["complete_config_entity"]))
    wrong["entity"][2][2][-1] = 7
    with ExplicitResidencyDispatch([raw]):
        with pytest.raises(RuntimeError, match="ConfigEntity differed"):
            resolve_schedule_route(config(wrong), [output_for(raw["identity"]["workload"])])


def test_invalid_public_mode_is_rejected():
    raw = json.loads(json.dumps(route()))
    raw["identity"]["public_mode"] = "paper_inspired_hybrid"
    with pytest.raises(ValueError, match="unsupported production residency mode"):
        ExplicitResidencyDispatch([raw])


def test_combined_cheng_route_binds_mode_five():
    raw = json.loads(json.dumps(route()))
    raw["public_mode"] = "input_weight_resident_barrier"
    raw["implementation_mode"] = 5
    raw["identity"]["public_mode"] = "input_weight_resident_barrier"
    raw["identity"]["implementation_mode"] = 5
    with ExplicitResidencyDispatch([raw]) as dispatch:
        selected = dispatch.query(None, tuple(raw["identity"]["workload"]))
        mode = resolve_schedule_route(
            selected, [output_for(raw["identity"]["workload"])]
        )
    assert mode == 5
    assert dispatch.summary()["all_routes_scheduled"]
