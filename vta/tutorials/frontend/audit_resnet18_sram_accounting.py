#!/usr/bin/env python3
"""Audit archived ResNet18 SRAM evidence without compiling or contacting a board.

Re-read hash-bound post-StorageRewrite TIR, not layer sizes or mode names.
The output describes static allocation spans, never time-averaged occupancy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
QUALIFICATION = C3 / "07_grouped_holdout/20260915_c3_tophub_h1_prepare_run02"
BOARD = C3 / "07_grouped_holdout/20260915_c3_tophub_h1_board_run02"
LEGACY = HERE / "report_out/native_stage_pipeline_searches/batch004/summary.json"
SCOPES = {"inp": "local.inp_buffer", "wgt": "local.wgt_buffer", "acc": "local.acc_buffer"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def storage_allocations(module):
    import tvm
    allocations = []

    def visit(node):
        if not isinstance(node, tvm.tir.Allocate):
            return
        if any(not isinstance(x, tvm.tir.IntImm) for x in node.extents):
            raise ValueError("dynamic allocation cannot be assigned a static utilization")
        dtype = tvm.DataType(str(node.dtype))
        size = dtype.bits * dtype.lanes
        extents = [int(x) for x in node.extents]
        for extent in extents:
            if extent <= 0:
                raise ValueError("nonpositive allocation extent")
            size *= extent
        if size % 8:
            raise ValueError("non-byte-aligned allocation")
        allocations.append({"buffer": str(node.buffer_var),
                            "scope": str(node.buffer_var.type_annotation.storage_scope),
                            "bytes": size // 8, "extents": extents})

    for function in module.functions.values():
        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    return allocations


def domain_metrics(allocations, hardware):
    result = {}
    for domain, scope in SCOPES.items():
        entries = [a for a in allocations if a["scope"] == scope]
        capacity = hardware[domain + "_buff_size"]
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("invalid byte capacity")
        span = sum(a["bytes"] for a in entries) if entries else None
        # Multiple arenas require placement/liveness reasoning. Do not call their
        # sum a simultaneous peak, and never turn missing allocations into zero.
        exact = len(entries) == 1
        result[domain + "_capacity_bytes"] = capacity
        result[domain + "_arena_span_bytes"] = span
        result[domain + "_arena_capacity_pct"] = 100 * span / capacity if exact else None
        result[domain + "_status"] = (
            "single_compiled_arena" if exact else "span_sum_upper_bound" if entries else "unknown"
        )
    return result


def reconstruct_legacy(best):
    # AXU5EVB BATCH=1, BLOCK_IN=BLOCK_OUT=16, i8/w8/a32.
    capacities = {"inp": 32768, "wgt": 262144, "acc": 131072, "out": 32768}
    widths = {"inp": 16, "wgt": 256, "acc": 64, "out": 16}
    rows = []
    for domain, capacity in capacities.items():
        old_pct = best["onchip_" + domain + "_util_pct"]
        wrong_capacity = capacity * widths[domain]
        numerator = old_pct / 100 * wrong_capacity
        if abs(numerator - round(numerator)) > 1e-6:
            raise ValueError("historical byte reconstruction is not integral")
        rows.append({"domain": domain, "capacity_bytes": capacity,
                     "legacy_capacity_bytes": wrong_capacity,
                     "legacy_pct": old_pct, "inferred_whole_unit_bytes": round(numerator),
                     "whole_unit_capacity_ratio_pct": 100 * numerator / capacity,
                     "compiled_sram_utilization_pct": None})
    return rows


def audit(qualification, board, legacy):
    import tvm
    import vta
    # Register VTA-specific TIR operators before deserializing archived modules.
    _ = vta.get_env().dev
    inputs = {}
    manifests = {}

    def verified(directory, relative):
        manifest_path = directory / "artifact_hashes.json"
        if directory not in manifests:
            manifests[directory] = json.loads(manifest_path.read_text())["artifacts"]
            inputs[str(manifest_path)] = sha(manifest_path)
        path = directory / relative
        digest = sha(path)
        if manifests[directory].get(relative) != digest:
            raise ValueError("artifact hash mismatch: " + str(path))
        inputs[str(path)] = digest
        return path

    contract = json.loads(verified(qualification, "contract.json").read_text())
    rows = [json.loads(s) for s in verified(qualification, "qualification.jsonl").read_text().splitlines()]
    board_summary = json.loads(verified(board, "summary.json").read_text())
    if not board_summary["timing_valid"] or not board_summary["all_timing_calls_correct"]:
        raise ValueError("archived board session is not qualified")
    timed = {row["candidate_id"] for row in board_summary["comparisons"]}
    timed.add(contract["baseline_candidate_id"])
    output = []
    for candidate in rows:
        local = candidate["local_qualification"]
        cid = candidate["candidate_id"]
        row = {"candidate_id": cid, "family_id": candidate["family_id"],
               "workload_id": candidate["workload_id"], "mode": candidate["residence_mode"],
               "lower_status": local["status"], "board_timed": cid in timed,
               "whole_graph_peak_pct": None, "dynamic_occupancy_pct": None,
               "uop_sram_peak_bytes": None}
        if local["status"] == "passed":
            if local["storage_arenas"]["capture"] != "after StorageRewrite, before LowerDeviceStorageAccessInfo":
                raise ValueError("unexpected storage capture phase")
            path = verified(qualification, "tir/" + cid + ".storage.json")
            if sha(path) != local["storage_tir_sha256"]:
                raise ValueError("storage TIR identity mismatch")
            allocations = storage_allocations(tvm.ir.load_json(path.read_text()))
            archived = local["storage_arenas"]["allocations"]
            if sorted(allocations, key=str) != sorted(archived, key=str):
                raise ValueError("recounted allocation differs from archived record")
            lowered = verified(qualification, "tir/" + cid + ".json")
            if sha(lowered) != local["tir_sha256"]:
                raise ValueError("lowered TIR identity mismatch")
            cross = candidate["cross_qualification"]
            if cross["relowered_tir_sha256"] != sha(lowered):
                raise ValueError("cross compilation used a different program")
            binary = verified(qualification, "binaries/" + cid + ".so")
            if sha(binary) != cross["binary_sha256"]:
                raise ValueError("binary identity mismatch")
            row.update(domain_metrics(allocations, candidate["hardware_fingerprint"]))
            row.update(storage_tir_sha256=sha(path), lowered_tir_sha256=sha(lowered),
                       binary_sha256=sha(binary))
        output.append(row)
    if {r["candidate_id"] for r in output if r["lower_status"] == "passed"} != timed:
        raise ValueError("board/local candidate sets differ")
    inputs[str(legacy)] = sha(legacy)
    old = json.loads(legacy.read_text())["best"]
    correction = reconstruct_legacy(old)
    if any(sha(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError("source changed during read-only audit")
    return {"schema": "resnet18_sram_accounting_audit_v1", "board_contacted": False,
            "scope": "archived R18-H1 operator candidates; not the May full-graph programs",
            "whole_graph_sram_peak_pct": None, "dynamic_sram_occupancy_pct": None,
            "baseline_candidate_id": contract["baseline_candidate_id"],
            "legacy_candidate_id": old["candidate_id"], "legacy_reconstruction": correction,
            "candidates": output, "input_sha256": inputs,
            "input_files_unchanged": True, "auditor_sha256": sha(Path(__file__))}


def write_outputs(directory, result):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    for filename, rows in (("candidate_arenas.csv", result["candidates"]),
                           ("legacy_reconstruction.csv", result["legacy_reconstruction"])):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (directory / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
    passed = [r for r in result["candidates"] if r["lower_status"] == "passed"]
    lines = ["# ResNet18 SRAM 统计纠错与历史 arena 复核", "",
             "38.28125% 撤回作为 SRAM 利用率：旧代码把 byte 容量再次乘 vector 字节数，且分子是整计算单元张量估算。",
             "输入示例：200704 / (32768 × 16) × 100 = 38.28125%。改为 200704 / 32768 = 6.125 倍，只说明整张量需要分块。",
             "同样撤回据此得出的‘整网 SRAM 不是约束、没有 spill’判断；真实时间记录不被改写。", "",
             "以下重读封存的 StorageRewrite 后 TIR，逐 Allocate 计算 extents × dtype 字节数，核对历史记录、TIR 和二进制哈希。",
             "这是 64→64、56×56、3×3 的 R18-H1 孤立算子历史配置；不是 5 月整网的重新编译，也不是逐周期有效占用。",
             "每个已报告存储域仅有一个 arena，包含虚拟线程变换后的分配；多个 arena 的和只能作为上界。", "",
             "| family | mode | INP B / % | WGT B / % | ACC B / % |",
             "|---|---|---:|---:|---:|"]
    for row in passed:
        values = [f'{row[d + "_arena_span_bytes"]} / {row[d + "_arena_capacity_pct"]:.4f}'
                  if row[d + "_arena_capacity_pct"] is not None else "unknown"
                  for d in SCOPES]
        lines.append(f'| {row["family_id"]} | {row["mode"]} | ' + " | ".join(values) + " |")
    lines += ["", f"共 {len(result['candidates'])} 个请求身份，{len(passed)} 个成功分配并有历史板端计时；失败身份留在 CSV，缺失占用为空而非零。",
              "INP/WGT/ACC 容量分别为 32768/262144/131072 B。各域独立，不能相加成一个百分比。",
              "OUT 与 UOP 未在这些 arena 中获得完整分配证据，保持 unknown；DDR 命令缓冲峰值不能替代片上 UOP 峰值。",
              "ResNet18 整网静态峰值与动态平均有效占用仍为 unknown；不能把本表最大值外推为整网数值。",
              "所有读取的历史输入哈希在 audit.json 中保留，并在结束前复核不变。", ""]
    (directory / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-dir", type=Path, default=QUALIFICATION)
    parser.add_argument("--board-dir", type=Path, default=BOARD)
    parser.add_argument("--legacy-summary", type=Path, default=LEGACY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.qualification_dir, args.board_dir, args.legacy_summary)
    write_outputs(args.output, result)
    print(json.dumps({"output": str(args.output), "candidates": len(result["candidates"]),
                      "verified_inputs": len(result["input_sha256"]), "board_contacted": False}))


if __name__ == "__main__":
    main()
