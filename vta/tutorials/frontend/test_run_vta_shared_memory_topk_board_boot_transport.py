"""Tests for the connect-only retry boundary."""

import subprocess

import run_vta_shared_memory_topk_board_boot_transport as transport


def completed(returncode, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess(["ssh"], returncode, stdout, stderr)


def test_only_unstarted_ssh_connection_failures_are_retryable():
    assert transport.connect_only_failure(
        completed(255, stderr=b"ssh: connect to host x: No route to host\n")
    )
    assert transport.connect_only_failure(completed(255), "echo BOOT=$(cat boot_id)")
    assert not transport.connect_only_failure(completed(255), "mkdir /tmp/x")
    assert not transport.connect_only_failure(
        completed(255, stdout=b"remote command began\n", stderr=b"Connection reset by peer")
    )
    assert not transport.connect_only_failure(completed(1, stderr=b"No route to host"))
    assert not transport.connect_only_failure(completed(255, stderr=b"permission denied"))
