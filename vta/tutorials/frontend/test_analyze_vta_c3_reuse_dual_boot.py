import copy
from pathlib import Path
from types import SimpleNamespace
import pytest

import analyze_vta_c3_reuse_dual_boot as a


def observations(values=None):
    return [dict(round=n,seed=a.SEEDS[n%3],correct=True,mismatch_count=0,
                 actual_sha256="a",expected_sha256="a",latency_ms=(values or [1.]*8)[n]) for n in range(8)]


def test_round_alignment_uses_explicit_round_not_append_order():
    rows=observations(list(range(1,9)))
    assert a.round_values(list(reversed(rows))) == list(range(1,9))


@pytest.mark.parametrize("damage",["missing","duplicate","seed","wrong","nan","negative"])
def test_incomplete_or_wrong_rounds_reject(damage):
    rows=observations()
    if damage=="missing": rows.pop()
    if damage=="duplicate": rows[7]["round"]=6
    if damage=="seed": rows[0]["seed"]=999
    if damage=="wrong": rows[0]["correct"]=False
    if damage=="nan": rows[0]["latency_ms"]=float("nan")
    if damage=="negative": rows[0]["latency_ms"]=-1
    with pytest.raises(ValueError): a.round_values(rows)


def test_stability_requires_two_boots_improvement_and_round_wins():
    s=dict(improvement_pct=2.,wins=6)
    assert a.stable([s,s])
    assert not a.stable([s])
    assert not a.stable([s,dict(improvement_pct=1.99,wins=8)])
    assert not a.stable([s,dict(improvement_pct=20,wins=5)])


def test_paired_median_is_not_ratio_of_medians():
    p=a.paired([1.]*4+[100.]*4,[.5]*8)
    assert p["median_paired_improvement_pct"] != p["improvement_pct"]
    assert p["wins"]==8


def test_alignment_is_per_command_queue_not_sum_then_align():
    assert a.aligned_capacity(1)+a.aligned_capacity(1)==8192
    assert a.aligned_capacity(4097)==8192
    with pytest.raises(ValueError): a.aligned_capacity(None)


def test_negative_additive_interaction_does_not_imply_combination_fastest():
    o,i,w,iw=10,15,5,6
    assert iw-i-w+o < 0
    assert iw > w


def test_invalid_session_rejected_before_binding_or_timing(tmp_path,monkeypatch):
    (tmp_path/"invalid_session.json").write_text("{}")
    monkeypatch.setattr(a,"verify",lambda p:{})
    with pytest.raises(ValueError,match="invalid session"):
        a.validate_board(tmp_path,[],[],{})


def test_completed_real_boards_pass_full_dispatch_dma_and_order_audit():
    root=Path(__file__).parent/"report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
    if not all((root/n/"artifact_hashes.json").exists() for n in a.BOARDS):
        pytest.skip("local frozen evidence not present")
    frozen=a.verify(root/a.FREEZE)
    candidates=a.read(root/a.FREEZE/"candidates.json")
    controls=a.read(root/a.FREEZE/"controls.json")
    for n in a.BOARDS:
        _,summary,rows=a.validate_board(root/n,candidates,controls,frozen)
        assert len(rows)==659 and summary["correct_identities"]==48
    poisoned=copy.deepcopy(frozen)
    poisoned["frozen_mode_orders"][0].reverse()
    with pytest.raises(ValueError,match="Latin"):
        a.validate_board(root/a.BOARDS[0],candidates,controls,poisoned)
