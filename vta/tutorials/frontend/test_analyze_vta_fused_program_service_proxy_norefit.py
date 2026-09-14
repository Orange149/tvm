from analyze_vta_fused_program_service_proxy_norefit import (
    pairwise_order_accuracy,
    service_score,
    total_dma,
    verify_artifacts_compatible,
)
import hashlib
import json


def traffic(load_bytes, load_calls, store_bytes=10, store_calls=1):
    return {
        "load_buffer_2d_bytes": load_bytes,
        "load_buffer_2d_calls": load_calls,
        "store_buffer_2d_bytes": store_bytes,
        "store_buffer_2d_calls": store_calls,
    }


def test_total_dma_and_service_score_include_load_and_store():
    row = traffic(100, 3, 20, 2)
    assert total_dma(row) == {"bytes": 120, "calls": 5}
    assert service_score(row, 64, 2, 128) == 120 + 64 * 5 + 128 * 2


def test_pairwise_order_accuracy_reports_the_inversion():
    rows = [
        {"program_id": "a", "metric": 1, "latency_ms": 2},
        {"program_id": "b", "metric": 2, "latency_ms": 1},
        {"program_id": "c", "metric": 3, "latency_ms": 3},
    ]
    result = pairwise_order_accuracy(rows, "metric")
    assert result["correct"] == 2
    assert result["total"] == 3
    assert len(result["discordant_pairs"]) == 1


def test_legacy_files_hash_manifest_is_verified(tmp_path):
    payload = tmp_path / "payload.txt"
    payload.write_text("frozen\n", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "artifact_hashes.json").write_text(
        json.dumps({"schema": "legacy", "files": {"payload.txt": digest}}),
        encoding="utf-8",
    )
    assert verify_artifacts_compatible(tmp_path) == {"payload.txt": digest}
