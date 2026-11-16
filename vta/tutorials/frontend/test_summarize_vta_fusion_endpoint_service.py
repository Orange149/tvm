import copy
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("summarize_vta_fusion_endpoint_service.py")
SPEC = importlib.util.spec_from_file_location("endpoint_summary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
REAL_BOOT = Path(__file__).parent / (
    "report_out/stage_tile_cotuning/compile_context_audit/"
    "c1_fusion_endpoint_service_boot_e274_run2"
)


def test_real_boot_passes_complete_single_boot_validation():
    summary, contract, samples = MODULE.load_boot(REAL_BOOT)
    assert summary["samples"] == 81
    assert contract["repeat"] == 3
    assert len(samples) == 81
    assert [row["sequence"] for row in samples] == list(range(81))


def test_three_boot_summary_rejects_duplicate_physical_boots():
    with pytest.raises(ValueError, match="three distinct"):
        MODULE.summarize_runs([REAL_BOOT, REAL_BOOT, REAL_BOOT])


def test_contract_identity_excludes_only_boot_specific_snapshot():
    _summary, contract, _samples = MODULE.load_boot(REAL_BOOT)
    changed_boot = copy.deepcopy(contract)
    changed_boot["boot_id"] = "another-boot"
    changed_boot["board_before"] = {"boot": "another-boot"}
    assert MODULE.contract_identity(changed_boot) == MODULE.contract_identity(contract)

    changed_protocol = copy.deepcopy(contract)
    changed_protocol["warmup"] += 1
    assert MODULE.contract_identity(changed_protocol) != MODULE.contract_identity(contract)
