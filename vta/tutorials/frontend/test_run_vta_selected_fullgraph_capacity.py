"""Unit tests for selected full-graph command-capacity helpers."""

from run_vta_selected_fullgraph_capacity import align_up, canonical_sha256, environment


def test_page_alignment_and_manifest_environment():
    assert align_up(1065280) == 1069056
    assert align_up(1320) == 4096
    env = environment({"insn_bytes": 1069056, "uop_bytes": 4096}, "manifest")
    assert env["VTA_INSN_BUFFER_BYTES"] == 1069056
    assert env["VTA_UOP_BUFFER_BYTES"] == 4096
    assert env["VTA_REPLAY_POLICY"] == "disabled"
    assert env["VTA_COMMAND_MANIFEST_ID"] == "manifest"


def test_identity_hash_is_canonical():
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})
