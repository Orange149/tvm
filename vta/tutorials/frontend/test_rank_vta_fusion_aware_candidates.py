"""Tests for deterministic fusion-aware candidate ordering."""

from rank_vta_fusion_aware_candidates import fusion_order


def candidate(candidate_id, score, boundary):
    return {
        "candidate_id": candidate_id,
        "topology_id": candidate_id,
        "path": [["vta:01:01", 1]],
        "m1_score_ms": score,
        "m1_components": {"single_vta_plus_mutex_boundary_ms": score},
        "boundary_bytes": boundary,
    }


def estimate(status="exact_reuse", fragmented=0):
    return {
        "status": status,
        "structured_vector": {
            "cpu_read_bytes": 0,
            "cpu_write_bytes": 0,
            "small_load_calls": fragmented,
            "strided_load_calls": 0,
            "strided_store_calls": 0,
            "convert_calls": 0,
        },
    }


def test_fusion_order_keeps_m1_incumbent_first_and_does_not_drop_rows():
    rows = [candidate("incumbent", 1.0, 100), candidate("dominated", 1.01, 200), candidate("front", 1.01, 50)]
    order = fusion_order(rows, {"vta:01:01": estimate()}, band_fraction=0.02)
    assert order[0]["candidate_id"] == "incumbent"
    assert [row["candidate_id"] for row in order] == ["incumbent", "front", "dominated"]


def test_non_exact_estimate_abstains_inside_band():
    rows = [candidate("incumbent", 1.0, 100), candidate("unknown", 1.01, 50)]
    order = fusion_order(rows, {"vta:01:01": estimate("new_fusion_class")})
    assert [row["candidate_id"] for row in order] == ["incumbent", "unknown"]
    assert not order[1]["transfer_exact_reuse"]


def test_exact_ties_use_topology_then_candidate_id():
    rows = [
        candidate("incumbent", 1.0, 100),
        candidate("z_variant", 1.01, 50),
        candidate("a_variant", 1.01, 50),
    ]
    rows[1]["topology_id"] = "a_topology"
    rows[2]["topology_id"] = "z_topology"
    order = fusion_order(rows, {"vta:01:01": estimate()})
    assert [row["candidate_id"] for row in order] == ["incumbent", "z_variant", "a_variant"]
