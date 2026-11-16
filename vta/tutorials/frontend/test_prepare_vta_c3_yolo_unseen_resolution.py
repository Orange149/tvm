import json
from pathlib import Path

from prepare_vta_c3_yolo_unseen_resolution import (
    derive_cfg,
    graph_memory_contract,
    resolution_admission,
)


def test_derive_cfg_changes_only_net_width_and_height(tmp_path):
    source = tmp_path / "base.cfg"
    source.write_text("[net]\nwidth=416\nheight=416\nchannels=3\n[convolutional]\nwidth=9\n")
    derived = derive_cfg(source, 608)
    assert derived == "[net]\nwidth=608\nheight=608\nchannels=3\n[convolutional]\nwidth=9\n"


def test_graph_memory_contract_uses_max_alias_size_and_command_backing():
    graph = {
        "attrs": {
            "shape": ["list_shape", [[1, 16], [1, 32], [1, 8]]],
            "dltype": ["list_str", ["int8", "int8", "int32"]],
            "storage_id": ["list_int", [0, 0, 1]],
            "device_index": ["list_int", [12, 12, 1]],
        }
    }
    result = graph_memory_contract(json.dumps(graph))
    assert result["graph_pool_aligned_bytes"] == 512
    assert result["static_contract_bytes"] == 512 + 2 * 32 * 1024 * 1024


def test_latency_exposed_resolutions_are_excluded_and_512_remains_eligible():
    exposed = resolution_admission(608)
    assert exposed["eligible"] is False
    assert exposed["exposure"]["evidence"] == (
        "20260924_c3_yolo608_composable_confirmation_boot1_run01"
    )
    exposed_576 = resolution_admission(576)
    assert exposed_576["eligible"] is False
    assert exposed_576["exposure"]["evidence"] == (
        "20260924_c3_yolo576_transition_safe_timing_boot1_run01"
    )
    exposed_544 = resolution_admission(544)
    assert exposed_544["eligible"] is False
    assert exposed_544["exposure"]["evidence"] == (
        "20260924_c3_yolo544_selector_v4_timing_boot1_run01"
    )
    assert resolution_admission(512)["eligible"] is True
