from build_vta_c3_yolo608_graph_deployments import route_payload


def test_route_payload_preserves_exact_workload_config_and_program_identity():
    pool = {"workload_id": "W", "execution_task_id": "T",
            "workload": ["conv2d_packed.vta", "data", "weight"]}
    candidate = {"candidate_id": "C", "residence_mode": "input_weight_resident_barrier",
                 "implementation_mode": 5, "complete_config_entity": {"entity": [["x", "ot", 1]]}}
    route = route_payload(pool, candidate, "P")
    assert route["candidate_id"] == "P"
    assert route["representative_candidate_id"] == "C"
    assert route["identity"]["workload"] == pool["workload"]
    assert route["identity"]["implementation_mode"] == 5
    assert route["identity"]["complete_config_entity"] == candidate["complete_config_entity"]
