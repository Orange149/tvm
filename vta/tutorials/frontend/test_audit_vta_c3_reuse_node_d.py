from pathlib import Path
import pytest

import audit_vta_c3_reuse_node_d as a


def row(workload,cid,byte,latency,calls=1):
    return dict(workload_id=workload,candidate_id=cid,family_id=cid,mode=a.IW,control=False,
                logical_data_DMA_bytes=byte,logical_data_DMA_calls=calls,
                C1_median_ms=latency,C2_median_ms=latency,command_backing_bytes=8192)


def test_least_bytes_can_miss_oracle_and_is_only_retrospective():
    rows=[r for m in ("M1","M2","M3") for r in (row(m,"small",10,3),row(m,"fast",20,1))]
    result=a.rank_diagnostics(rows)
    assert len(result)==6
    assert all(r["regret_pct"]==200 and r["first_pool_2pct_rank"]==2 for r in result)
    assert all("retrospective" in r["scope"] for r in result)


def test_pareto_does_not_trade_different_domains_by_sum():
    small=row("M1","small",10,3)
    fast=row("M1","fast",20,1)
    assert not a.dominates(small,fast,1) and not a.dominates(fast,small,1)
    worse=row("M1","worse",20,4)
    assert a.dominates(small,worse,1)


def test_real_M3_counterexample_is_not_suppressed():
    root=Path(__file__).parent/"report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
    path=root/"20260915_c3_reuse_node_d_dual_boot_run01/four_mode_space_dma_latency.csv"
    if not path.exists():pytest.skip("local frozen evidence absent")
    result=a.rank_diagnostics(a.numeric_rows(path))
    m3=[r for r in result if r["geometry"]=="M3"]
    assert all(not r["within_2pct"] and r["regret_pct"]>200 for r in m3)
    assert all(r["selected_family"]=="M3D0056" for r in m3)
