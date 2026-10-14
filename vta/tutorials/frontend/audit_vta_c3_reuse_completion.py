#!/usr/bin/env python3
"""Verify the failed C1 and delayed-read evidence, without performance promotion."""
import argparse
from collections import Counter
import csv
from pathlib import Path

from run_vta_c3_reuse_supplement import read, write, verify, seal, file_hash


def rows(path):
    import json
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check_dispatch_accounting(results, timeline):
    begins = [r for r in timeline if r["event"] == "dispatch_begin"]
    if len(begins) != len(results):
        raise ValueError("missing dispatch/result or unfinished call")
    for dispatch, result in zip(begins, results):
        if (dispatch["candidate_id"], dispatch["seed"], dispatch["phase"]) != (result["candidate_id"], result["seed"], result["phase"]):
            raise ValueError("dispatch identity/order differs from result")
        if result["complete_operator_calls"] != 1 or result["hidden_warmup_calls"] != 0:
            raise ValueError("unaccounted calls or hidden warmup")
    return len(results)


def check_logical_dma(result, candidate):
    expected = candidate["local_qualification"]["semantics"]["totals"]
    profile = result["runtime_profile_complete"]
    for key, value in expected.items():
        if profile.get(key) != value:
            raise ValueError("static/runtime DMA mismatch: " + key)
    # Each operator can contain several accelerator submissions; compare count,
    # never divide the byte or driver-time totals by this count.
    submissions = candidate["fsim_qualification"]["command_features"]["values"]["submissions"]
    if profile["driver_run_calls"] != submissions:
        raise ValueError("FSim/FPGA submission count mismatch")


def delayed_evidence(results):
    wrong = [r for r in results if not r["correct"]]
    for r in results:
        if r["changed_after_return"] != (r["actual_sha256"] != r["delayed_sha256"]):
            raise ValueError("delayed hash change flag inconsistent")
        if r["delayed_correct"] != (r["delayed_mismatch_count"] == 0):
            raise ValueError("delayed correctness flag inconsistent")
        if r["delayed_correct"] and r["delayed_sha256"] != r["expected_sha256"]:
            raise ValueError("delayed correct output hash inconsistent")
    return {"immediate_wrong": len(wrong),
            "became_correct_without_second_execution": sum(r["delayed_correct"] for r in wrong),
            "changed_after_return": sum(r["changed_after_return"] for r in results)}


def run(args):
    failed, diagnosis = Path(args.failed_dir).resolve(), Path(args.diagnosis_dir).resolve()
    failure_contract, diagnostic_contract = verify(failed), verify(diagnosis)
    invalid = read(failed / "invalid_session.json")
    if invalid.get("all_timing_invalid") is not True:
        raise ValueError("requires explicitly invalidated timing session")
    if (diagnosis / "invalid_session.json").exists():
        raise ValueError("diagnostic session did not close")
    parent = Path(failure_contract["parent_directory"])
    verify(parent)
    parent_hash = file_hash(parent / "artifact_hashes.json")
    if any(c["parent_manifest_sha256"] != parent_hash for c in (failure_contract, diagnostic_contract)):
        raise ValueError("evidence does not share exact frozen pool")
    candidates = read(parent / "candidates.json") + read(parent / "controls.json")
    lookup = {r["candidate_id"]: r for r in candidates}
    lookup[failure_contract["health"]["candidate_id"]] = failure_contract["health"]
    failure_rows, diagnostic_rows = rows(failed / "results.jsonl"), rows(diagnosis / "results.jsonl")
    failure_count = check_dispatch_accounting(failure_rows, rows(failed / "timeline.jsonl"))
    diagnosis_count = check_dispatch_accounting(diagnostic_rows, rows(diagnosis / "timeline.jsonl"))
    for result in failure_rows + diagnostic_rows:
        check_logical_dma(result, lookup[result["candidate_id"]])
    delayed = delayed_evidence(diagnostic_rows)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "contract.json", {"phase": "completion_audit", "source_sha256": file_hash(__file__),
          "input_manifests": {str(d): file_hash(d / "artifact_hashes.json") for d in (failed, diagnosis, parent)},
          "formal_latency_labels_emitted": False, "board_contacted": False})
    costs = []
    for name, records in (("invalid_C1", failure_rows), ("diagnosis", diagnostic_rows)):
        for phase in dict.fromkeys(r["phase"] for r in records):
            selected = [r for r in records if r["phase"] == phase]
            costs.append({"session": name, "phase": phase, "complete_operator_calls": len(selected),
                          "wrong_outputs": sum(not r["correct"] for r in selected),
                          "calls_host_wall_seconds": sum(r["host_wall_seconds"] for r in selected)
                              if all("host_wall_seconds" in r for r in selected) else None,
                          "rpc_execution_wall_seconds": sum(r["rpc_call_wall_seconds"] for r in selected),
                          "logical_load_bytes": sum(r["runtime_profile_complete"]["load_buffer_2d_bytes"] for r in selected),
                          "logical_store_bytes": sum(r["runtime_profile_complete"]["store_buffer_2d_bytes"] for r in selected),
                          "logical_dma_calls": sum(r["runtime_profile_complete"]["load_buffer_2d_calls"] + r["runtime_profile_complete"]["store_buffer_2d_calls"] for r in selected),
                          "FPGA_submissions": sum(r["runtime_profile_complete"]["driver_run_calls"] for r in selected)})
    sequences = []
    for candidate in candidates:
        command = candidate["fsim_qualification"]["command_features"]["values"]
        sequence = command["submit_sequence"]
        if len(sequence) != command["submissions"]:
            raise ValueError("incomplete local command sequence")
        for submit in sequence:
            sequences.append({"family_id": candidate["family_id"], "mode": candidate["residence_mode"],
                              "control": candidate.get("experimental_reload_control", False),
                              "candidate_id": candidate["candidate_id"],
                              "submit": submit["submit_ordinal_within_inference"],
                              "insn_bytes": submit["insn_bytes"], "uop_bytes": submit["uop_bytes"],
                              "logical_load_bytes": submit["load_bytes"], "logical_store_bytes": submit["store_bytes"],
                              "control_only_by_DMA": submit["load_bytes"] == submit["store_bytes"] == 0,
                              "evidence_source": "frozen local FSim queue diagnostics; not per-submit FPGA telemetry"})
    diagnostic_table = [{"family_id": r["family_id"], "mode": r["residence_mode"], "seed": r["seed"],
                         "phase": r["phase"], "position": r.get("sequence_position", "health"),
                         "immediate_correct": r["correct"], "delayed_correct": r["delayed_correct"],
                         "immediate_mismatches": r["mismatch_count"], "delayed_mismatches": r["delayed_mismatch_count"],
                         "driver_run_total_us_per_complete_operator": r["runtime_profile_complete"]["driver_run_total_us"],
                         "FPGA_submissions_per_complete_operator": r["runtime_profile_complete"]["driver_run_calls"],
                         "changed_after_return": r["changed_after_return"]} for r in diagnostic_rows]
    for name, records in (("phase_costs.csv", costs), ("local_submit_sequences.csv", sequences),
                          ("delayed_read_evidence.csv", diagnostic_table)):
        with (output / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    report = {"status": "COMPLETION_BOUNDARY_AUDIT_COMPLETE_NO_NEW_PERFORMANCE",
              "invalid_C1_operator_calls": failure_count, "diagnosis_operator_calls": diagnosis_count,
              "all_calls_static_DMA_and_submission_verified": failure_count + diagnosis_count,
              "invalid_session_promoted_latency_labels": 0, "delayed_evidence": delayed,
              "failure_phase_counts": dict(Counter(r["phase"] for r in failure_rows)),
              "phase_costs": costs,
              "C1_W0_to_failure_seconds": invalid["W0_to_failure_seconds"],
              "C1_T0_to_failure_seconds": invalid["T0_to_failure_seconds"],
              "diagnosis_wall_seconds": read(diagnosis / "summary.json")["wall_seconds"],
              "MMIO_root_cause_confirmed": False, "bitstream_generated_register_map_available": False}
    write(output / "summary.json", report)
    relevant = [r for r in sequences if r["family_id"].startswith("M2") and r["mode"] == "input_weight_resident_barrier" and not r["control"]]
    lines = ["# 完成边界独立审计", "", "全部460次完整算子的dispatch、逻辑DMA和submission与冻结身份核对；无性能标签晋升。", "",
             "## IW提交结构必须按真实队列解释", "",
             "| family | local submit | insn B | LOAD B | STORE B |", "|---|---:|---:|---:|---:|"]
    for r in relevant:
        lines.append("| {} | {} | {} | {} | {} |".format(r["family_id"], r["submit"], r["insn_bytes"], r["logical_load_bytes"], r["logical_store_bytes"]))
    lines += ["", "本地三组M2 IW的第一个提交含全部LOAD/STORE，第二个提交是64 B且无数据DMA的控制尾段。",
              "不能把这两次提交写成‘先独立权重预加载，再计算’。FPGA profile只给完整算子累计数，",
              "尚不能定位哪次提交误判完成，FINISH计数也来自源码推导，不是FPGA逐提交解码。", "",
              "直接证据是7/7立即错误后来变正确；stale-done仍只是待核验假设。",
              "当前只有pkg_config的WR=0x10/RD=0x18源码配置，无该bitstream生成的done/valid寄存器定义；",
              "不能把RD+4的valid位或WR寄存器清零当成已验证修复。下一步必须先核验完成协议。"]
    (output / "AUDIT.md").write_text("\n".join(lines) + "\n")
    (output / "completion_read_boundary.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="220" viewBox="0 0 1120 220">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<style>text{font-family:monospace;font-size:14px} .title{font-size:17px}</style>'
        '<text x="20" y="30" class="title">Observed shared-output consumption boundary (one boot, 7 calls)</text>'
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8" fill="#555"/></marker></defs>'
        '<rect x="20" y="60" width="240" height="80" fill="#edf3fa" stroke="#555"/>'
        '<text x="35" y="90">Runtime returns</text><text x="35" y="116">driver total ~137 us</text>'
        '<path d="M260,100 L300,100" stroke="#555" marker-end="url(#arrow)"/>'
        '<rect x="310" y="60" width="240" height="80" fill="#fff0ef" stroke="#555"/>'
        '<text x="325" y="90">Immediate CPU output read</text><text x="325" y="116">Incorrect (7/7)</text>'
        '<path d="M550,100 L590,100" stroke="#555" marker-end="url(#arrow)"/>'
        '<rect x="600" y="60" width="240" height="80" fill="#edf7ed" stroke="#555"/>'
        '<text x="615" y="90">Delayed CPU output read</text><text x="615" y="116">Correct (7/7)</text>'
        '<text x="860" y="90">No second execution</text><text x="860" y="116">Output hash changed</text>'
        '<text x="20" y="170">Order only, not a calibrated timeline: actual STORE completion time is unobserved.</text>'
        '<text x="20" y="194">10 ms delayed read is diagnostic, NOT a validated synchronization fix or latency label.</text></svg>\n')
    for directory in (failed, diagnosis, parent):
        verify(directory)
    seal(output)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-dir", required=True)
    parser.add_argument("--diagnosis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
