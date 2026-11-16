from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_output_control_profile as c


def fixture(tmp_path, monkeypatch):
    (tmp_path / "zero.bin").write_bytes(bytes(4))
    monkeypatch.setattr(c, "BUNDLE", tmp_path)
    refs = [dict(capsule_path="zero.bin", shape=[1], dtype="float32")]
    monkeypatch.setattr(c, "read", lambda path: dict(references=dict(resnet18=refs)))
    value = dict(kind="isolated_output_serialization", DNN_inferences=0, FPGA_calls=0,
        warmup=2, scored=8, ring_count=1, stage_count=7, output_bytes=4, affinity="0,1,2,3",
        raw_outputs=[dict(index=0, shape=[1], dtype="float32", nbytes=4, fnv1a64="315446a086a23133")],
        samples=[dict(CPU_ns=100000, wall_ns=100000, JSON_bytes=200) for _ in range(8)])
    return value, dict(network="resnet18", ring_count=1)


def test_wrong_output_execution_identity_and_missing_samples_rejected(tmp_path, monkeypatch):
    value, spec = fixture(tmp_path, monkeypatch)
    assert c.validate_output(value, spec) == value
    for mutate in (
        lambda v: v.update(DNN_inferences=1),
        lambda v: v.update(FPGA_calls=1),
        lambda v: v.update(ring_count=4),
        lambda v: v.update(affinity="0"),
        lambda v: v["raw_outputs"][0].update(fnv1a64="0000000000000000"),
        lambda v: v["samples"].pop(),
    ):
        bad = deepcopy(value)
        mutate(bad)
        with pytest.raises(ValueError):
            c.validate_output(bad, spec)


def test_qemu_clock_exception_never_applies_to_native_samples(tmp_path, monkeypatch):
    value, spec = fixture(tmp_path, monkeypatch)
    value["samples"][0]["CPU_ns"] = 130000
    with pytest.raises(ValueError):
        c.validate_output(value, spec)
    c.validate_output(value, spec, native=False)
