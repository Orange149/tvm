import json

import audit_vta_c1_e4_natural_placement_reuse as audit


def frozen_row(index):
    workload = ["conv", ["TENSOR", [1, index + 1], "int8"]]
    config = {"entity": [["tile", "sp", [-1, index + 1]]], "code_hash": None}
    return {
        "workload_id": "W{:02d}".format(index),
        "candidate_id": "candidate-{}".format(index),
        "workload": workload,
        "config": config,
        "tile_identity_sha256": "tile-{}".format(index),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_requires_exact_route_and_normalized_primitive(tmp_path):
    rows = [frozen_row(index) for index in range(12)]
    fixed = tmp_path / "fixed.json"
    write_json(fixed, {"rows": rows, "artifact_sha256": "freeze"})

    exact_builds = []
    for index in range(2):
        path = tmp_path / "fullgraph_exact_{}".format(index) / "build.json"
        write_json(
            path,
            {
                "dispatch_audit": {
                    "schedule_hits": [
                        {
                            "candidate_id": rows[0]["candidate_id"],
                            "workload": rows[0]["workload"],
                            "complete_config_entity": rows[0]["config"],
                            "public_mode": "original",
                            "implementation_mode": 0,
                        }
                    ]
                },
                "primitive_inventory": [
                    {"signature": "primitive", "normalized": {"op": "conv"}}
                ],
            },
        )
        exact_builds.append(path)

    mode_mismatch = tmp_path / "fullgraph_mode_mismatch" / "build.json"
    write_json(
        mode_mismatch,
        {
            "dispatch_audit": {
                "schedule_hits": [
                    {
                        "candidate_id": "different-candidate",
                        "workload": rows[1]["workload"],
                        "complete_config_entity": rows[1]["config"],
                        "public_mode": "input_stationary",
                        "implementation_mode": 1,
                    }
                ]
            },
            "primitive_inventory": [
                {"signature": "primitive", "normalized": {"op": "conv"}}
            ],
        },
    )

    result = audit.build_audit(fixed, [*exact_builds, mode_mismatch])

    assert result["summary"]["exact_reusable_natural_context_count"] == 2
    assert result["summary"]["natural_cross_placement_ready_workload_count"] == 1
    assert result["rows"][0]["natural_cross_placement_ready"]
    assert result["rows"][1]["matches"][0]["status"] == "tile_only_schedule_mode_mismatch"
    assert not result["summary"]["all_twelve_ready"]


def test_discovery_only_selects_fullgraph_builds(tmp_path):
    full = tmp_path / "candidate_fullgraph_run" / "build.json"
    other = tmp_path / "operator_run" / "build.json"
    write_json(full, {})
    write_json(other, {})

    assert audit.discover_builds([tmp_path]) == [full.resolve()]

