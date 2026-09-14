import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import plan_vta_incumbent_protected_routes as planner


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proposal(candidate_id, baseline, delta, wins, evidence_path, evidence_hash):
    return {
        "route": {"candidate_id": candidate_id, "workload_id": candidate_id},
        "baseline_route_candidate_ids": list(baseline),
        "proposed_route_candidate_ids": list(baseline) + [candidate_id],
        "evaluation": {
            "graph_outputs_equal": True,
            "live_memory_fit": True,
            "paired_rounds": 7,
            "proposed_paired_wins": wins,
            "median_paired_delta_ms": delta,
        },
        "evidence": [{
            "path": str(evidence_path),
            "sha256": evidence_hash,
            "assertions": [{"path": "ok", "equals": True}],
        }],
    }


def contract(tmp_path, proposals):
    path = tmp_path / "contract.json"
    dump(path, {
        "schema": "c3_incumbent_protected_route_contract_v1",
        "status": "frozen_before_route_composition",
        "workspace_root": str(tmp_path),
        "policy": {
            "algorithm": "greedy_sequential_incumbent_protection",
            "minimum_paired_rounds": 5,
            "minimum_win_fraction": 0.5,
            "minimum_improvement_ms": 0.0,
        },
        "protected_incumbent": {"route_candidate_ids": []},
        "proposals": proposals,
    })
    return path


def test_accept_then_reject_preserves_incumbent(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence_hash = dump(evidence, {"ok": True})
    proposals = [
        proposal("good", [], -1.0, 7, evidence, evidence_hash),
        proposal("bad", ["good"], 2.0, 0, evidence, evidence_hash),
    ]
    contract_path = contract(tmp_path, proposals)
    output = tmp_path / "out"
    planner.run(SimpleNamespace(contract=str(contract_path), output_dir=str(output)))
    result = json.loads((output / "selected_routes.json").read_text(encoding="utf-8"))
    assert result["selected_route_candidate_ids"] == ["good"]
    assert [row["accepted"] for row in result["decisions"]] == [True, False]


def test_rejects_stale_baseline(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence_hash = dump(evidence, {"ok": True})
    proposals = [proposal("bad", ["not-current"], -1.0, 7, evidence, evidence_hash)]
    with pytest.raises(RuntimeError, match="current incumbent"):
        planner.run(SimpleNamespace(
            contract=str(contract(tmp_path, proposals)), output_dir=str(tmp_path / "out")
        ))


def test_rejects_mutated_evidence(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence_hash = dump(evidence, {"ok": True})
    proposals = [proposal("good", [], -1.0, 7, evidence, evidence_hash)]
    contract_path = contract(tmp_path, proposals)
    dump(evidence, {"ok": False})
    with pytest.raises(RuntimeError, match="evidence hash mismatch"):
        planner.run(SimpleNamespace(
            contract=str(contract_path), output_dir=str(tmp_path / "out")
        ))
