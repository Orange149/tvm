#!/usr/bin/env python3
"""Local, experiment-only node24 partial-residency qualification.

The first stage (``probe``) captures the exact post-StorageRewrite Input TIR.
It does not modify the production schedule or contact the FPGA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tvm
from tvm import autotvm, relay, tir
from tvm.contrib import graph_executor
import vta
from vta.testing import simulator  # Registers ext_dev FSim device API.

from vta.top.residency_dispatch import ExplicitResidencyDispatch
from vta.top.vta_node24_partial import (
    Node24GroupGEMM,
    Node24GroupWeightDMA,
    Node24SpatialBoundaryBridge,
    Node24WeightSlots,
    SplitNode24AltOutputDMA,
    SplitNode24InputDMA,
)
from c3_reuse_semantics import extract_requests
from c3_thread_sync_audit import final_tir_audit


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
G2 = GROUPED / "20261008_c3_yolo224_incremental_prepare_run03/G2"
TASKS = GROUPED / "20260926_c3_yolo224_static_prepare_run01/resolution_224/result.json"
TASK_ID = "278686c6edcaf8fd3eaf3a074ebb32a2c8d65cdd0cc4cece0cd49f85b7ad9822"
SYMBOL = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast"
TILE = (1, 2, 14, 1, 1)
SEEDS = (0, 20250901, 20260910)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def frozen_fusion(group_count=16):
    if group_count not in (14, 15, 16):
        raise ValueError("boundary group count must be 14, 15, or 16")
    text = (G2 / "relay_000.txt").read_text()
    anchor = "%143 = fn ("
    if text.count(anchor) != 1:
        raise ValueError("frozen node24 Relay block not unique")
    block = "fn (" + text.split(anchor, 1)[1].split("} /* ty=fn", 1)[0] + "}"
    if "Tensor[(16, 8, 3, 3, 16, 16), int8]" not in block or "right_shift(%78, 7" not in block:
        raise ValueError("frozen four-parameter fusion contract drifted")
    if group_count != 16:
        block = block.replace("Tensor[(16, 8, 3, 3, 16, 16), int8]",
                              "Tensor[({}, 8, 3, 3, 16, 16), int8]".format(group_count))
        block = block.replace("Tensor[(16, 1, 1, 1, 16), int32]",
                              "Tensor[({}, 1, 1, 1, 16), int32]".format(group_count))
        block = block.replace("Tensor[(1, 16, 14, 14, 1, 16),",
                              "Tensor[(1, {}, 14, 14, 1, 16),".format(group_count))
        block = block.replace("channels=256", "channels={}".format(group_count * 16))
        if "channels=256" in block or "Tensor[(1, 16, 14, 14" in block:
            raise ValueError("boundary Relay group transformation incomplete")
    block = block.replace(", Primitive=1", "").replace("fn (", "def @main(", 1)
    return relay.parse('#[version = "0.0.5"]\n' + block)


def route(mode="input_stationary", tile=TILE, group_count=16):
    task_rows = json.loads(TASKS.read_text())["tasks"]
    task = next(row for row in task_rows if row["task_id"] == TASK_ID)
    entity = [[name, "sp", [-1, value]] for name, value in zip(
        ("tile_b", "tile_h", "tile_w", "tile_ci", "tile_co"), tile
    )]
    entity += [["oc_nthread", "ot", 1], ["h_nthread", "ot", 1]]
    workload = json.loads(json.dumps(task["workload"]))
    if group_count != 16:
        workload[2][1][0] = group_count
    return {
        "candidate_id": "node24-{}-b{}h{}w{}ci{}co{}-o1".format(mode, *tile),
        "execution_task_id": TASK_ID if group_count == 16 else
        "boundary-{}-groups-derived-from-{}".format(group_count, TASK_ID),
        "identity": {
            "workload": workload,
            "complete_config_entity": {"code_hash": None, "entity": entity},
            "public_mode": mode,
            "implementation_mode": {
                "node24_partial_input_experimental": 12,
                "input_stationary": 1,
                "input_weight_resident_transition_safe": 11,
                "weight_resident_barrier": 4,
            }[mode],
        },
    }


@tvm.instrument.pass_instrument
class TargetTIRArchive:
    def __init__(self):
        self.snapshots = []

    def run_before_pass(self, mod, info):
        if str(info.name) == "tir.InjectCopyIntrin":
            self._capture(mod, "before_dma")

    def run_after_pass(self, mod, info):
        if str(info.name) not in (
            "tir.vta.EarlyRewrite", "tir.vta.Node24WeightSlots",
            "tir.vta.Node24GroupWeightDMA",
            "tir.vta.Node24GroupGEMM",
            "tir.vta.CPUAccessRewrite",
        ):
            return
        self._capture(mod, str(info.name))

    def _capture(self, mod, label):
        functions = []
        for function in mod.functions.values():
            if not isinstance(function, tir.PrimFunc) or function.attrs is None:
                continue
            if "global_symbol" in function.attrs and str(function.attrs["global_symbol"]) == SYMBOL:
                functions.append(function)
        if len(functions) != 1:
            return
        function = functions[0]
        self.snapshots.append({
            "pass": label,
            "script": function.script(),
            "json": tvm.ir.save_json(function),
        })


def _arrays(seed, group_count=16):
    rng = np.random.default_rng(seed)
    return (
        rng.integers(-32, 33, size=(1, 8, 14, 14, 1, 16), dtype=np.int8),
        rng.integers(-16, 17, size=(group_count, 8, 3, 3, 16, 16), dtype=np.int8),
        rng.integers(-2048, 2049, size=(group_count, 1, 1, 1, 16), dtype=np.int32),
    )


def _reference(x, w, add):
    inp = x[0, :, :, :, 0, :].transpose(1, 2, 0, 3).reshape(14, 14, 128).astype(np.int32)
    kernel = w.transpose(2, 3, 1, 5, 0, 4).reshape(3, 3, 128, w.shape[0] * 16).astype(np.int32)
    padded = np.pad(inp, ((1, 1), (1, 1), (0, 0)))
    channels = w.shape[0] * 16
    accum = np.zeros((14 * 14, channels), dtype=np.int32)
    for ky in range(3):
        for kx in range(3):
            accum += padded[ky:ky + 14, kx:kx + 14, :].reshape(196, 128) @ kernel[ky, kx]
    accum += add[:, 0, 0, 0, :].reshape(1, channels)
    scalar = np.clip(accum >> 7, -127, 127).astype(np.int8).reshape(14, 14, w.shape[0], 16)
    return scalar.transpose(2, 0, 1, 3)[None, :, :, :, None, :]


def probe(output, mode, resident_groups=None, keep_resident=False, fsim=False,
          board_compile=False, tile=TILE, group_count=16, batch_group_load=False,
          group_gemm=False):
    if batch_group_load and (mode != "node24_partial_input_experimental" or
                             resident_groups != 13 or group_count != 16 or tile != TILE):
        raise ValueError("batched group DMA requires frozen node24 A13/F13 geometry")
    if group_gemm and not (batch_group_load and keep_resident):
        raise ValueError("group GEMM merge requires frozen F13-B")
    program = frozen_fusion(group_count)
    candidate = route(mode, tile, group_count)
    if resident_groups is not None:
        candidate["candidate_id"] += "-{}{}".format(
            "F" if keep_resident else "A", resident_groups)
    if batch_group_load:
        candidate["candidate_id"] += "-B"
    if group_gemm:
        candidate["candidate_id"] += "-G"
    cfg = dict(vta.get_env().cfg_dict)
    if not board_compile:
        cfg["TARGET"] = "sim"
    with vta.Environment(cfg):
        env = vta.get_env()
        target = tvm.target.Target(env.target, host=env.target_host)
        archive = TargetTIRArchive()
        relay.backend.te_compiler.get().clear()
        try:
            with autotvm.tophub.context(target), ExplicitResidencyDispatch([candidate]) as dispatch:
                build = vta.build_config()
                lower_passes = list(build.config["tir.add_lower_pass"])
                if mode == "input_stationary" and tile == (1, 7, 7, 4, 2):
                    lower_passes.insert(1, (1, SplitNode24InputDMA()))
                    lower_passes.insert(2, (1, SplitNode24AltOutputDMA()))
                if mode == "node24_partial_input_experimental":
                    lower_passes.insert(1, (1, SplitNode24InputDMA()))
                    if resident_groups is not None:
                        lower_passes.append((2, Node24WeightSlots(
                            resident_groups, keep_resident, group_count=group_count)))
                    if batch_group_load:
                        lower_passes.append((2, Node24GroupWeightDMA()))
                    lower_passes.append((2, Node24SpatialBoundaryBridge()))
                    if group_gemm:
                        before_cpu = next(i for i, (_, p) in enumerate(lower_passes)
                                          if "CPUAccessRewrite" in str(p))
                        lower_passes.insert(before_cpu, (3, Node24GroupGEMM()))
                with tvm.transform.PassContext(
                    opt_level=3,
                    disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
                    instruments=[archive],
                    config={"tir.add_lower_pass": lower_passes},
                ):
                    graph, lib, params = relay.build(program, target=target)
        finally:
            for snapshot in archive.snapshots:
                stem = (
                    "node24_pre_dma" if snapshot["pass"] == "before_dma" else
                    "node24_early" if snapshot["pass"].endswith("EarlyRewrite") else
                    "node24_slots" if snapshot["pass"].endswith("Node24WeightSlots") else
                    "node24_batched" if snapshot["pass"].endswith("Node24GroupWeightDMA") else
                    "node24_gemm" if snapshot["pass"].endswith("Node24GroupGEMM") else
                    "node24_final"
                )
                (output / (stem + ".tir")).write_text(snapshot["script"])
                (output / (stem + ".tir.json")).write_text(snapshot["json"])
        if not dispatch.summary()["all_routes_scheduled"] or params:
            raise ValueError("node24 exact Input route was not scheduled")
        expected_snapshots = (6 if group_gemm else 5 if batch_group_load else
                              4 if resident_groups is not None else 3)
        if len(archive.snapshots) != expected_snapshots:
            raise ValueError("missing a pre-DMA, early, slots, or final node24 TIR snapshot")
        (output / "graph.json").write_text(graph)
        result = {
            "status": "NODE24_TIR_PROBED_NOT_QUALIFIED",
            "task_id": TASK_ID,
            "tile": tile,
            "group_count": group_count,
            "resident_groups": resident_groups,
            "keep_resident": keep_resident,
            "batch_group_load": batch_group_load,
            "group_gemm": group_gemm,
            "route": candidate,
            "dispatch": dispatch.summary(),
            "fusion_relay_sha256": sha(tvm.ir.save_json(program).encode()),
            "snapshots": [
                {"pass": row["pass"], "sha256": sha(row["json"].encode())}
                for row in archive.snapshots
            ],
            "board_contacted": False,
            "target": env.TARGET,
        }
        if board_compile:
            from build_vta_yolov3_tiny_relay_residency_dispatch_pair import cross_compiler
            lib.export_library(str(output / "graphlib.so"), fcompile=cross_compiler())
            result["graphlib_sha256"] = sha((output / "graphlib.so").read_bytes())
        if resident_groups is not None or mode in (
                "input_stationary", "input_weight_resident_transition_safe"):
            final = next(row for row in archive.snapshots
                         if row["pass"].endswith("CPUAccessRewrite"))
            final_function = tvm.ir.load_json(final["json"])
            final_module = tvm.IRModule({"main": final_function})
            requests = extract_requests(final_module, env)
            address = final_tir_audit(final_module, requests)
            (output / "request_semantics.json").write_text(
                json.dumps(requests, indent=2, sort_keys=True) + "\n")
            (output / "address_audit.json").write_text(
                json.dumps(address, indent=2, sort_keys=True) + "\n")
            result["request_totals"] = requests["totals"]
            result["address_status"] = address["status"]
            result["max_kernel_uops"] = address["max_kernel_uops"]
        if fsim:
            runner = graph_executor.create(graph, lib, tvm.device("ext_dev", 0))
            rows = []
            for seed in (*SEEDS, 0):
                x, w, add = _arrays(seed, group_count)
                before = [sha(value.tobytes()) for value in (x, w, add)]
                runner.set_input("p015", x)
                runner.set_input("p17", w)
                runner.set_input("p26", add)
                runner.run()
                actual = runner.get_output(0).numpy()
                expected = _reference(x, w, add)
                mismatch = np.argwhere(actual != expected)
                row = {
                    "seed": seed,
                    "correct": len(mismatch) == 0,
                    "actual_sha256": sha(actual.tobytes()),
                    "expected_sha256": sha(expected.tobytes()),
                    "arguments_unchanged": before == [sha(value.tobytes()) for value in (x, w, add)],
                    "first_difference": mismatch[0].tolist() if len(mismatch) else None,
                }
                rows.append(row)
            if rows[0]["expected_sha256"] == rows[1]["expected_sha256"]:
                raise ValueError("A/B test weights did not distinguish expected outputs")
            if rows[0]["actual_sha256"] != rows[-1]["actual_sha256"]:
                raise ValueError("A-B-A did not reproduce A on same runner")
            result["fsim"] = rows
            result["status"] = "LOCAL_FSIM_PASSED_BOARD_NOT_QUALIFIED" if all(
                row["correct"] and row["arguments_unchanged"] for row in rows
            ) else "LOCAL_FSIM_FAILED"
        save_json(output / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "fsim", "board_prepare"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("input_stationary", "weight_resident_barrier",
                                           "input_weight_resident_transition_safe",
                                           "node24_partial_input_experimental"),
                        default="input_stationary")
    parser.add_argument("--tile-h", type=int, default=2)
    parser.add_argument("--tile-w", type=int, default=14)
    parser.add_argument("--tile-ci", type=int, default=1)
    parser.add_argument("--tile-co", type=int, default=1)
    parser.add_argument("--resident-groups", type=int, choices=(12, 13))
    parser.add_argument("--group-count", type=int, choices=(14, 15, 16), default=16)
    parser.add_argument("--keep-resident", action="store_true")
    parser.add_argument("--batch-group-load", action="store_true")
    parser.add_argument("--group-gemm", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.resident_groups is not None and args.mode != "node24_partial_input_experimental":
        parser.error("weight slots require the grouped experimental mode")
    if args.keep_resident and args.resident_groups is None:
        parser.error("--keep-resident requires --resident-groups")
    tile = (1, args.tile_h, args.tile_w, args.tile_ci, args.tile_co)
    if args.resident_groups is not None and tile != TILE:
        parser.error("node24 A/F variants must use the frozen b1h2w14ci1co1 Tile")
    if args.batch_group_load and (args.mode != "node24_partial_input_experimental" or
                                   args.resident_groups != 13 or args.group_count != 16):
        parser.error("--batch-group-load requires frozen A13/F13 geometry")
    if args.group_gemm and not (args.batch_group_load and args.keep_resident):
        parser.error("--group-gemm requires frozen F13-B")
    result = probe(args.output_dir, args.mode, args.resident_groups, args.keep_resident,
                   fsim=args.stage == "fsim", board_compile=args.stage == "board_prepare",
                   tile=tile, group_count=args.group_count,
                   batch_group_load=args.batch_group_load,
                   group_gemm=args.group_gemm)
    print(json.dumps({"status": result["status"], "snapshots": result["snapshots"]}, indent=2))


if __name__ == "__main__":
    main()
