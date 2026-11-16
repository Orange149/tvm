"""Small offline boundary checks; never build/copy the real payload or use SSH."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from cpu_policies.pilot_common import ROOT, identity, load_legacy, protocol
from cpu_policies import pilot_prepare


def test_protocol_is_single_input_single_boot_and_has_no_performance_gates():
    value = protocol()
    assert value["calibration"] == {"indices": [0], "warmup": 2, "scored": 8, "tail": 0}
    assert value["performance"] == {"indices": [0], "warmup": 8, "scored": 16, "tail": 4}
    assert value["qualification_indices"] == [0]
    assert value["rounds"] == 1 and value["independent_confirmation_boots"] == 0
    assert value["performance_rejection_threshold"] is None
    assert not value["anchors"] and not value["automatic_window_extension"]
    assert value["resource_policies"]["shared4"]["cpu_cores"] == [0, 1, 2, 3]
    assert value["resource_policies"]["reserved3"] == {
        "cpu_cores": [0, 1, 2], "host_mask": [3], "control_mask": [3], "reserved_host": True}


@pytest.mark.parametrize("arguments", [
    ["board", "run", "--bundle", "/unused"],
    ["board", "resume", "--run-dir", "/unused"],
])
def test_board_cli_without_execute_flag_rejects_before_access(arguments, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("pilot_cli_under_test", ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["run.py", *arguments])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "requires --execute-board" in capsys.readouterr().err


def test_copy_rejects_changed_source_and_preserves_original_bytes(tmp_path):
    load_legacy()
    from cpu_vta_partitioning.common import sha256
    source = tmp_path / "original"
    source.write_bytes(b"frozen-original")
    expected = sha256(source)
    target = tmp_path / "bundle" / "copied"
    pilot_prepare._copy(source, target, expected)
    assert source.read_bytes() == target.read_bytes() == b"frozen-original"
    target.write_bytes(b"changed-copy")
    assert sha256(source) == expected
    with pytest.raises(ValueError, match="hash differs"):
        pilot_prepare._copy(source, tmp_path / "rejected", "0" * 64)
    assert not (tmp_path / "rejected").exists()


def tiny_bundle(tmp_path, monkeypatch):
    load_legacy()
    from cpu_vta_partitioning.common import digest, inventory, sha256
    from cpu_vta_partitioning.prepare import payload
    monkeypatch.setattr(pilot_prepare, "load_legacy", lambda value: Path(value))
    monkeypatch.setattr(payload, "execution_files", lambda value: {})
    bundle = tmp_path / "bundle"
    (bundle / "plans").mkdir(parents=True)
    (bundle / "runner").mkdir()
    (bundle / "inputs").mkdir()
    (bundle / "references").mkdir()
    runner = bundle / "runner" / "pinned"
    runner.write_bytes(b"test-fixture-not-an-executable")
    input_file = bundle / "inputs/000.bin"
    reference_file = bundle / "references/000.npy"
    input_file.write_bytes(b"fixture-input")
    reference_file.write_bytes(b"fixture-reference-not-loaded-by-validator")

    def put(name, value):
        (bundle / name).write_text(json.dumps(value))

    put("plans/pilot_protocol.json", protocol())
    put("plans/programs.json", {**{f"cpu{i}": {"device": "cpu"} for i in range(85)},
                               **{f"vta{i}": {"device": "vta"} for i in range(87)}})
    put("plans/executors.json", {"policy": {"path": "runner/pinned", "sha256": sha256(runner)}})
    put("plans/inputs.json", {"available_indices": [0], "records": [{"index": 0,
        "input": "inputs/000.bin", "frontiers": {"0": {
            "raw": "inputs/000.bin", "npy": "references/000.npy"}}}],
        "files": {str(p.relative_to(bundle)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                  for p in (input_file, reference_file)}})
    put("plans/board_payload.json", {"files": {}})
    put("OFFLINE_READINESS.json", {"status": "PREPARED_OFFLINE", "tests": {"passed": True}})
    manifest = {"experiment": protocol()["experiment"], "controller_identity": identity(),
                "files": inventory(bundle)}
    manifest["bundle_id"] = digest(manifest)[:24]
    put("manifest.json", manifest)
    return bundle, manifest


def test_small_reference_closure_remains_valid_after_relocation(tmp_path, monkeypatch):
    bundle, manifest = tiny_bundle(tmp_path, monkeypatch)
    assert pilot_prepare.validate_bundle(bundle) == manifest
    copied = tmp_path / "relocated"
    shutil.copytree(bundle, copied)
    assert pilot_prepare.validate_bundle(copied) == manifest


def test_invalid_manifest_identity_is_rejected_offline(tmp_path, monkeypatch):
    bundle, manifest = tiny_bundle(tmp_path, monkeypatch)
    manifest["bundle_id"] = "0" * 24
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest identity"):
        pilot_prepare.validate_bundle(bundle)


def test_changed_frozen_asset_is_rejected_offline(tmp_path, monkeypatch):
    bundle, _ = tiny_bundle(tmp_path, monkeypatch)
    (bundle / "runner/pinned").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="missing or changed artifact"):
        pilot_prepare.validate_bundle(bundle)


def test_inconsistent_single_input_index_is_rejected(tmp_path, monkeypatch):
    bundle, manifest = tiny_bundle(tmp_path, monkeypatch)
    from cpu_vta_partitioning.common import digest, inventory
    (bundle / "plans/inputs.json").write_text(json.dumps({
        "available_indices": [0], "records": [{"index": 16}]}))
    manifest["files"] = inventory(bundle, exclude=("manifest.json",))
    manifest.pop("bundle_id")
    manifest["bundle_id"] = digest(manifest)[:24]
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="input|reference"):
        pilot_prepare.validate_bundle(bundle)


def test_unfrozen_input_reference_is_rejected_even_with_valid_manifest(tmp_path, monkeypatch):
    bundle, manifest = tiny_bundle(tmp_path, monkeypatch)
    from cpu_vta_partitioning.common import digest
    # A listed input reference must itself be frozen, not merely present.
    manifest["files"].pop("references/000.npy")
    manifest.pop("bundle_id")
    manifest["bundle_id"] = digest(manifest)[:24]
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reference closure"):
        pilot_prepare.validate_bundle(bundle)
