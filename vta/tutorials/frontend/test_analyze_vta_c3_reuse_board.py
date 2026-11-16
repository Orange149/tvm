from types import SimpleNamespace
import pytest

import analyze_vta_c3_reuse_board as analysis
from run_vta_c3_reuse_supplement import MODES


def tables():
    return [dict(family_id="M1", workload_id="M1", mode=m, candidate_id=m,
                 control=False, inp_bytes=100 if m == "original" else 50,
                 wgt_bytes=100 if m == "original" else 50) for m in MODES]


def test_one_boot_never_claims_stability():
    values = {m: [float(4-i)]*8 for i, m in enumerate(MODES)}
    row = analysis.family_results(tables(), values)[0]
    assert row["four_fpga_measured"] and row["effective_dual_reuse"]
    assert row["combination_fastest"]
    assert row["stable_cross_boot_claim"] is False


def test_missing_mode_not_replaced():
    values = {m: [1.]*8 for m in MODES if m != "input_weight_resident_barrier"}
    row = analysis.family_results(tables(), values)[0]
    assert not row["four_fpga_measured"]
    assert "median_ms" not in row


def test_incomplete_rounds_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        analysis.family_results(tables(), {m: [1.]*7 for m in MODES})


def test_invalid_session_cannot_supply_timing(tmp_path, monkeypatch):
    (tmp_path / "invalid_session.json").write_text("{}")
    monkeypatch.setattr(analysis, "verify", lambda p: {})
    with pytest.raises(ValueError, match="invalid session"):
        analysis.run(SimpleNamespace(board_dir=str(tmp_path), output_dir=str(tmp_path / "not_created")))
    assert not (tmp_path / "not_created").exists()
