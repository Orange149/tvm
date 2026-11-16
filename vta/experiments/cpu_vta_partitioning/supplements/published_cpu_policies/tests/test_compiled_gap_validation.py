from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_compiled_gap_validation as p
from cpu_policies.open_resources import domain


def item(n, mask, score):
    rows=[domain.stage("cpu",0,3,n,mask),domain.stage("vta",3,13,1,[0,1,2,3]),domain.stage("cpu",13,14,1,[0])]
    c=domain.candidate(rows,"shared4","off")
    return dict(configuration=c,static_score_ns=score)


def test_selection_uses_thread_strata_and_metadata_not_pipeline_results():
    a=item(3,[0,1,2],3);b=item(3,[1,2,3],2);c=item(4,[0,1,2,3],4)
    pool={n:{str(i):deepcopy(x) for i,x in enumerate((a,b,c))} for n in ("resnet18","yolov3_tiny416")}
    selected=p.choose(pool,{p.fingerprint(b["configuration"])})
    assert all([v["configuration"]["stages"][0]["threads"] for v in rows]==[3,4] for rows in selected.values())
    assert all(rows[0]["configuration"]==a["configuration"] for rows in selected.values())
    with pytest.raises(ValueError):p.choose(pool,{p.fingerprint(c["configuration"])})
    d=item(2,[0,1],8)
    for rows in pool.values():rows["two"]=d
    fallback=p.choose(pool,{p.fingerprint(c["configuration"])})
    assert all([v["configuration"]["stages"][0]["threads"] for v in rows]==[2,3] for rows in fallback.values())


def test_board_access_still_requires_explicit_execution_before_any_network(monkeypatch):
    monkeypatch.setattr(p,"read",lambda path:dict(expected_boot_id=p.BOOT))
    with pytest.raises(ValueError,match="execute-board"):
        p.board()
