import json

import run_vta_p7r122_y02_health_gated_correctness as p7r122


def test_knob_audit_separates_p7r121_thread_stratum():
    parent = p7r122.load_json(p7r122.DEFAULT_P7R120)
    rows = parent["candidate_pool"]["candidates"]
    audit = p7r122.audit_knobs(rows, {"status": "both_fail_p7r120_remains_blocked"})
    assert audit["six_candidate_common"] == {"oc_nthread": 1, "h_nthread": 1}
    assert audit["p7r121_failed_tophub"]["oc_nthread"] == 2


def test_w05_health_canary_is_historically_three_seed_correct():
    assert p7r122.prior_w05_pass(p7r122.DEFAULT_W05_EVIDENCE) == {
        "records": 2,
        "seed_checks": 6,
        "binary_sha256": ["c6eb7e376243053efd91f7363aa9f5f7dcc39c494b914f587f2ce36a4fb34ef3"],
        "tir_sha256": ["0550a74a5a1446e8a31c9ba407778562b419ab3e937db504da923bb68be31060"],
    }


def test_health_spec_is_original_config575():
    manifest = p7r122.load_json(p7r122.DEFAULT_W05_MANIFEST)
    workload = p7r122.load_json(p7r122.DEFAULT_W05_CONTRACT)["workload"]
    spec = p7r122.w05_health_spec(manifest, workload)
    assert spec["debug"]["config_index"] == 575
    assert spec["implementation_mode"] == 0
    assert spec["role"] == "health_gate_only"
