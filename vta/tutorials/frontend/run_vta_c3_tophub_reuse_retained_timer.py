#!/usr/bin/env python3
"""Non-mutating recovery adapter: keep all RPC loaded-module handles alive.

The original comparison runner and its sealed local preparation stay read-only.
In particular a registry callback cannot outlive its timer-helper DSO handle.
"""
from pathlib import Path

import run_vta_c3_tophub_reuse_comparison as experiment
import run_vta_p7r120_yolo_rpc_board as transport

LOADED_MODULES = []
ORIGINAL_LOAD = transport.load_ephemeral
ORIGINAL_VERIFY = experiment.s.verify


def retained_load(remote, binary):
    module = ORIGINAL_LOAD(remote, binary)
    LOADED_MODULES.append(module)
    return module


def bound_verify(directory):
    contract = ORIGINAL_VERIFY(directory)
    if contract.get("schema") == "c3_tophub_four_mode_v1":
        contract["source_bindings"] = {**contract["source_bindings"],
            str(Path(__file__).resolve()): experiment.s.file_hash(__file__)}
        contract["timer_module_retained_for_whole_session"] = True
        contract["recovery_scope"] = "rerun whole board session; no previous timing reused"
    return contract


if __name__ == "__main__":
    transport.load_ephemeral = retained_load
    experiment.s.verify = bound_verify
    experiment.main()
