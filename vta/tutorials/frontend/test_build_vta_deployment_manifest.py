"""Tests for manifest-driven VTA command-memory deployment."""

import copy

import pytest

from build_vta_deployment_manifest import build_manifest, validate_manifest
from test_qualify_vta_command_resource_certificate import evidence, provisional
from qualify_vta_command_resource_certificate import qualify_certificate


def qualified():
    return qualify_certificate(provisional(), *evidence())


def test_manifest_derives_concurrent_capacity_from_qualified_certificate():
    certificate = qualified()
    manifest = build_manifest(certificate, "a", queue_instances=3)
    assert manifest["command_memory"]["per_instance"]["total_capacity_bytes"] == 8192
    assert manifest["command_memory"]["deployment_total_capacity_bytes"] == 24576
    assert manifest["command_memory"]["queue_instances"] == 3
    assert manifest["primary_candidate_id"] == "a"
    assert validate_manifest(manifest, certificate)


def test_provisional_certificate_builds_a_pending_contract():
    manifest = build_manifest(provisional(), "a")
    assert manifest["status"] == "qualification_pending"
    assert "board_qualification_key" not in manifest
    assert manifest["launcher_environment"]["VTA_COMMAND_MANIFEST_ID"] == manifest["manifest_id"]


def test_pending_and_qualified_manifest_keep_the_same_preexecution_id():
    pending = build_manifest(provisional(), "a")
    ready = build_manifest(qualified(), "a")
    assert pending["manifest_id"] == ready["manifest_id"]
    assert "qualification_attestation" not in pending
    assert ready["qualification_attestation"]["binary_sha256_by_candidate_id"]


def test_unknown_primary_is_rejected():
    with pytest.raises(ValueError, match="primary"):
        build_manifest(qualified(), "missing")


def test_manifest_tampering_is_detected():
    certificate = qualified()
    manifest = build_manifest(certificate, "a")
    tampered = copy.deepcopy(manifest)
    tampered["command_memory"]["queue_instances"] = 2
    with pytest.raises(ValueError, match="manifest ID"):
        validate_manifest(tampered, certificate)
