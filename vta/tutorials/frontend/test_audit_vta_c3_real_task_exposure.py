"""Tests for exact C3 real-task exposure binding."""

import audit_vta_c3_real_task_exposure as audit


def task(model, signature, layers):
    return {"model": model, "geometry_signature": signature, "source_layers": layers}


def test_known_aliases_require_model_geometry_and_layer():
    y00 = task("yolov3_tiny_416_darknet", audit.ALIASES["Y00"]["signature"], ["conv2"])
    assert audit.alias_matches(y00, "Y00")
    wrong_layer = dict(y00, source_layers=["conv4"])
    assert not audit.alias_matches(wrong_layer, "Y00")
    wrong_model = dict(y00, model="relay_testing_resnet18")
    assert not audit.alias_matches(wrong_model, "Y00")


def test_resnet_alias_is_geometry_bound_not_call_name_bound():
    h1 = task("relay_testing_resnet18", audit.ALIASES["R18-H1"]["signature"], ["packed_conv_call_01"])
    assert audit.alias_matches(h1, "R18-H1")
    assert not audit.alias_matches(h1, "R18-H2")


def test_render_keeps_unseen_and_exposed_separate():
    rows = [
        {"model": "a", "exposure_status": "latency_exposed_exact_alias_binding", "strict_latency_unseen_eligible": False},
        {"model": "a", "exposure_status": "strict_latency_unseen_candidate_after_known_evidence_audit", "strict_latency_unseen_eligible": True},
        {"model": "b", "exposure_status": "latency_or_board_geometry_exposure_unresolved", "strict_latency_unseen_eligible": False},
    ]
    counts, by_model, text = audit.render(rows)
    assert counts == {"dispatch_scopes": 3, "latency_exposed": 1, "unresolved": 1, "strict_unseen_candidates": 1}
    assert by_model["a"]["strict_unseen_candidates"] == 1
    assert "No global optimum" in text
