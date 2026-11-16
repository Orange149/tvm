from types import SimpleNamespace
import pytest
import run_vta_c3_reuse_fixed_driver as adapter


def test_board_requires_explicit_execute_before_contact():
    with pytest.raises(ValueError, match="execute-board"):
        adapter.run_board(SimpleNamespace(execute_board=False))


def test_observer_cannot_be_promoted_to_fixed_runtime(tmp_path, monkeypatch):
    original = dict(parent_manifest_sha256="pool", source_bindings={})
    runtime = dict(parent_manifest_sha256="pool", source_bindings={},
                   handshake_mode="valid_observer", completion_decision_changed=False)
    qualified = dict(phase="handshake_qualification_analysis", overlay_manifest_sha256="ledger")
    records = iter((original, runtime, qualified))
    monkeypatch.setattr(adapter, "verify", lambda p: next(records))
    monkeypatch.setattr(adapter, "read", lambda p: dict(status="HANDSHAKE_MICRO_QUALIFICATION_PASS",
                        completion_decision_changed=True, immediate_wrong=0, changed_after_return=0,
                        driver_submissions_verified=90, complete_operator_calls=42))
    monkeypatch.setattr(adapter, "file_hash", lambda p: "ledger")
    output = tmp_path / "should_not_exist"
    args = SimpleNamespace(contract=str(tmp_path / "base/contract.json"),
                           runtime_contract=str(tmp_path / "runtime/contract.json"),
                           qualification_dir=str(tmp_path / "qualification"), output_dir=str(output))
    with pytest.raises(ValueError, match="observer"):
        adapter.prepare(args)
    assert not output.exists()
