#!/usr/bin/env python3
"""Audit actual local prerequisites without loading a model or resetting a board."""

import argparse
from pathlib import Path
import subprocess

from thesis_c1_memory_ablation import INPUT_SCHEMA, file_hash, freeze, read_json, seal, verify_ref, verify_seal, write_new
from thesis_c1_memory_evidence import inputs_manifest


FRONTEND = Path(__file__).resolve().parent
BASE = FRONTEND / "report_out/stage_tile_cotuning"


def check(inventory=None, probe_board=False, host="root@192.168.1.247", input_manifest=None, runner_build=None):
    root = FRONTEND.parents[2]
    canonical = BASE / "thesis_supplement_experiments/results/01/20260916T011355Z_completion_0670cfb1/raw/canonical_split_preparation_02/canonical_qir.json"
    weights = Path.home() / ".mxnet/models/resnet18_v1-a0666292.params"
    files = {}
    for name, path in (("canonical_qir", canonical), ("pretrained_weights", weights),
                       ("bitstream", root / "apps/vta_rpc/firmware/vta_hpc.bit"),
                       ("host_runtime", root / "build/libtvm_runtime.so")):
        files[name] = {"path": str(path), "exists": path.is_file(),
                       "sha256": file_hash(path) if path.is_file() else None}
    old = BASE / "thesis_supplement_experiments/results/06/20260916T124859Z_single_image_audit_bfec496d/raw/c1_frozen/fair_support_contract.json"
    historical = None
    if old.exists():
        data = read_json(old)
        support = data.get("support", [])
        historical = {"path": str(old), "sha256": file_hash(old),
                      "configurations": len(support), "island_values": sorted({r["vta_islands"] for r in support}),
                      "slot_values": sorted({r["additional_slot_bytes"] for r in support}),
                      "unknown_dma_rows": sum(r.get("vta_dma_calls") is None for r in support),
                      "reorder_band_size": len(data.get("semantic_audit", {}).get("band_candidate_ids", [])),
                      "usable_as_new_paired_inventory": False}
    blockers, prepared = [], {}
    if input_manifest:
        try:
            records = inputs_manifest({"artifacts": {"inputs": {"path": str(input_manifest.resolve()),
                                             "sha256": file_hash(input_manifest)}, "canonical_qir": files["canonical_qir"]}})
            prepared["inputs"] = {"path": str(input_manifest.resolve()), "sha256": file_hash(input_manifest),
                                  "real_inputs": 64, "diagnostic_inputs": 3,
                                  "frontier_tensors_per_input": len(records[0]["frontiers"]),
                                  "status": "host_reference_reaudited_not_board_qualified"}
        except (ValueError, KeyError, OSError) as error:
            blockers.append("Input reference preparation invalid: " + str(error))
    if runner_build:
        try:
            build = read_json(runner_build)
            verify_seal(build)
            if build.get("schema") != "c1_memory_runner_build_v1" or build.get("status") != "built_not_board_qualified":
                raise ValueError("unexpected native runner build manifest")
            for name in ("runner", "runtime", "driver"):
                verify_ref(build[name])
            for name, expected in build["sources"].items():
                if file_hash(root / "vta/apps/native_deploy" / name) != expected:
                    raise ValueError("native source differs from built binary: " + name)
            prepared["runner"] = {"manifest_path": str(runner_build.resolve()), "manifest_sha256": file_hash(runner_build),
                                  **build["runner"], "status": build["status"]}
        except (ValueError, KeyError, OSError) as error:
            blockers.append("Runner build invalid: " + str(error))
    if inventory is None:
        blockers.append("No current-graph inventory in c1_memory_ablation_inventory_v1 was supplied: 3/3/2 island quotas, M1 thread optima, final TIR DMA and paired shared plans are not frozen.")
        if "inputs" not in prepared:
            blockers.append("No validated 64-real-plus-3-diagnostic input/frontier reference manifest was supplied.")
        if "runner" not in prepared:
            blockers.append("The modified native runner needs a verified AArch64 build; an old single-island certificate cannot qualify this runner.")
        blockers.append("Multi-worker completion qualification remains unexecuted; no old single-island certificate is promoted.")
    else:
        try:
            payload = read_json(inventory)
            if payload.get("schema") != INPUT_SCHEMA:
                raise ValueError("inventory schema is not " + INPUT_SCHEMA)
            contract = freeze(payload, inventory.resolve().parent)
            inputs_manifest(contract)
        except (ValueError, KeyError, OSError) as error:
            blockers.append(str(error))
    board = {"contacted": False, "mutated": False, "boot_id": None}
    if probe_board:
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes",
                "-o", "HostKeyAlgorithms=+ssh-rsa", "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa", host,
                "cat /proc/sys/kernel/random/boot_id"]
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=12, check=False)
            board.update(contacted=True, command=args, returncode=result.returncode,
                         stdout=result.stdout, stderr=result.stderr,
                         boot_id=result.stdout.strip() if result.returncode == 0 else None)
            if result.returncode:
                blockers.append("Read-only SSH boot probe failed; no device state was changed.")
        except (OSError, subprocess.TimeoutExpired) as error:
            board.update(contacted=True, error=str(error))
            blockers.append("Read-only SSH boot probe did not complete.")
    return seal({"schema": "c1_memory_ablation_readiness_v1", "status": "prerequisites_missing" if blockers else "local_inputs_ready_pending_board_qualification",
                 "local_sources": files, "prepared_artifacts": prepared, "historical_single_image_support": historical, "board": board,
                 "blockers": blockers, "experiment_completed": False, "new_model_calls": 0,
                 "new_performance_windows": 0, "new_independent_boots_collected": 0,
                 "qualification_scope": "local readiness only; no hardware or ranking result"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inputs-manifest", type=Path)
    parser.add_argument("--runner-build", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-board", action="store_true")
    parser.add_argument("--host", default="root@192.168.1.247")
    args = parser.parse_args()
    result = check(args.inventory, args.probe_board, args.host, args.inputs_manifest, args.runner_build)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_new(args.output_dir / "readiness.json", result)
    lines = ["# C1 内存感知消融：实现与实际前置检查", "", "状态：" + result["status"], "",
             "本次仅核验本地材料和可选只读 SSH 身份；新增模型调用、性能窗口与独立实验 boot 均为 0。", "",
             "## 已准备材料", ""]
    lines += ["- {}：{}。".format(key, value["status"]) for key, value in result["prepared_artifacts"].items()]
    lines += ["", "## 缺失项", ""]
    lines += ["- " + value for value in result["blockers"]]
    lines += ["", "旧单图支持集只用于缺口核查，没有导入其性能标签，也没有将合成测试算作实验。", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(result["status"])
    for blocker in result["blockers"]:
        print("- " + blocker)


if __name__ == "__main__":
    main()
