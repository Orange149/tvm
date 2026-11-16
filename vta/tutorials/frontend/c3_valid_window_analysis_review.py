#!/usr/bin/env python3
"""Independent local review of the fixed-window analyzer; never contacts a board.

Fixtures are deliberately synthetic ledgers in a temporary directory. They are
not experimental results and cannot be used as FPGA qualification evidence.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import c3_valid_window_analysis as analysis
import c3_valid_window_board as board


DOCUMENT = (Path(__file__).resolve().parent / "report_out/stage_tile_cotuning/"
            "c3_dma_residency_autotune/00_governance/"
            "C3_VALID_WINDOW_MICROKERNEL_AGGREGATION_PLAN_V2.md")
BOOT = "SYNTHETIC-LOCAL-REVIEW-NOT-A-BOARD-BOOT"
BITSTREAM = "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6"
DEFAULT_LATENCIES = {
    "Q1": [10, 20, 30, 40, 50, 60],
    "Q2": [10, 19, 28, 37, 48, 58],
    "Q4": [8, 16, 24, 32, 40, 48],
    "Q8": [9, 18, 27, 36, 45, 54],
    "TPS": [7, 14, 21, 28, 35, 42],
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def write_lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


def fixture(root, latencies=None, aliases=None):
    latency = latencies or DEFAULT_LATENCIES
    protocol = board.plan(aliases)
    protocol.update(required_boot_id=BOOT, source_sha256=board.execution_sources(),
                    document=str(DOCUMENT), document_sha256=analysis.sha(DOCUMENT),
                    runtime_sha256={"libvta.so": "SYNTHETIC-VTA", "libtvm_runtime.so": "SYNTHETIC-TVM"})
    write(root / "board_protocol.json", protocol)
    write(root / "board_prepared.json", {"synthetic_review_fixture": True})
    for phase, status in (
            ("qualification", "FIXED_POOL_FPGA_QUALIFIED"),
            ("diagnostic", "ALL_ROUND_PREFIX_AND_UOP_DIAGNOSTICS_QUALIFIED"),
            ("board", "FORMAL_FIXED_POOL_ACQUISITION_COMPLETE")):
        write(root / phase / "summary.json", {
            "schema": board.SCHEMA, "status": status, "boot_id": BOOT,
            "all_correct": True, "prepared_sha256": analysis.sha(root / "board_prepared.json"),
            "protocol_sha256": analysis.sha(root / "board_protocol.json"),
        })
    calls, sessions, dispatch, helpers = [], [], [], []
    for round_index, order in enumerate(protocol["orders"]):
        for program in order:
            session_id = "synthetic-board:%03d" % len(sessions)
            sessions.append({
                "session_id": session_id, "program": program, "status": "PASS",
                "environment": {"boot_id": BOOT, "masks": ["3"]},
                "runtime_attestation": deepcopy(protocol["runtime_sha256"]),
                "start_state": {"udmabuf_size": 201326592},
                "bitstream_load_record": {"sha256": BITSTREAM},
                "inventory": {"replay_initial": {"mode": "disabled"}},
            })
            for call_kind in ("warmup", "timing"):
                for position, seed in enumerate(protocol["seed_orders"][round_index]):
                    iid = "synthetic-board:%06d" % len(calls)
                    row = {
                        "invocation_id": iid, "call_kind": call_kind,
                        "program": program, "seed": seed, "round": round_index + 1,
                        "seed_position": position, "session_id": session_id,
                        "buffer_handles": [session_id + ":X", session_id + ":W", session_id + ":ADD", session_id + ":Y"],
                        "status": "PASS", "boot_id": BOOT, "correct": True,
                        "arguments_unchanged": True, "complete_operator_calls": 1,
                        "hidden_warmup_calls": 0, "diagnostic_signature_matched": True,
                        # Extreme three-seed values ensure the seed median is used.
                        "latency_ms": (latency[program][round_index] * (0.5, 1.0, 2.0)[position]
                                       if call_kind == "timing" else None),
                    }
                    calls.append(row)
                    dispatch.extend({"invocation_id": iid, "event": event,
                                     "program": program, "call_kind": call_kind}
                                    for event in ("begin", "end"))
                    if call_kind == "timing":
                        helpers.append({"invocation_id": iid, "count": position + 1})
    write_lines(root / "board/calls.jsonl", calls)
    write_lines(root / "board/sessions.jsonl", sessions)
    write_lines(root / "board/dispatch.jsonl", dispatch)
    write_lines(root / "board/helper_counts.jsonl", helpers)
    # A deliberately huge timed qualification call is separate from formal data.
    write_lines(root / "qualification/calls.jsonl", [{
        "call_kind": "correctness", "first_cache_build": True, "latency_ms": 999999.0,
        "excluded_from_formal_samples": True,
    }])
    return protocol


def run(root):
    with redirect_stdout(io.StringIO()):
        return analysis.analyze(root)


def mutate_lines(root, relative, edit):
    path = root / relative
    rows = analysis.lines(path)
    edit(rows)
    write_lines(path, rows)


def review():
    passed = []
    with TemporaryDirectory(prefix="c3-analysis-review-") as tmp:
        base = Path(tmp)
        root = base / "formula"
        fixture(root)
        result = run(root)
        assert result["round_medians_ms"] == DEFAULT_LATENCIES
        assert result["T_ms"] == {name: median(values) for name, values in DEFAULT_LATENCIES.items()}
        primary = result["primary_Q8_versus_Q1"]
        assert all(abs(value - 10) < 1e-12 for value in primary["paired_improvements_pct"])
        assert primary["wins"] == 6 and primary["ties"] == 0
        assert primary["paired_ms_saved"] == [1, 2, 3, 4, 5, 6]
        assert primary["median_paired_ms_saved"] == 3.5
        assert result["round_range_ms"]["Q1"] == [10, 60]
        assert result["research_replication_engineering_gate"]
        assert not result["practical_graph_engineering_gate"]
        assert result["Q_pool_minimum_T_program"] == "Q4"
        assert result["formal_timings"] == result["formal_warmups"] == 90
        assert not result["statistical_significance_claimed"] and not result["graph_performance_measured"]
        passed.append("three-seed median; six-round T/G/ms/range; separate research/practical gates; qualification excluded")

        root = base / "paired-ratio"
        latency = deepcopy(DEFAULT_LATENCIES)
        latency["Q1"] = [1, 2, 3, 100, 100, 100]
        latency["Q8"] = [0.8, 1.8, 2.8, 50, 90, 99]
        fixture(root, latency)
        result = run(root)
        expected_g = median(100 * (1 - c / b) for c, b in zip(latency["Q8"], latency["Q1"]))
        assert result["primary_Q8_versus_Q1"]["G_pct"] == expected_g
        assert abs(expected_g - 100 * (1 - result["T_ms"]["Q8"] / result["T_ms"]["Q1"])) > 20
        passed.append("G is median of six paired percentages, not percentage of T medians")

        for wins in (4, 5):
            root = base / ("strict-wins-%d" % wins)
            latency = {name: [100.0] * 6 for name in board.LABELS}
            latency["Q8"] = [90.0] * wins + [100.0] * (6 - wins)
            fixture(root, latency)
            result = run(root)
            pair = result["primary_Q8_versus_Q1"]
            assert pair["wins"] == wins and pair["ties"] == 6 - wins
            assert result["research_replication_engineering_gate"] == (wins >= 5)
        passed.append("strict wins and exact ties; 4/6 fails and 5/6 passes engineering gate")

        root = base / "practical-pass"
        latency = {name: [value] * 6 for name, value in
                   (("Q1", 10.0), ("Q2", 9.7), ("Q4", 8.0), ("Q8", 8.5), ("TPS", 8.1))}
        fixture(root, latency)
        result = run(root)
        assert result["Q_pool_minimum_T_program"] == "Q4"
        assert result["practical_graph_engineering_gate"]
        assert result["primary_Q8_versus_Q1"]["candidate"] == "Q8"
        passed.append("Q4 practical candidate does not replace predeclared Q8 research comparison")

        root = base / "minimum-t-tie"
        fixture(root, {name: [10.0] * 6 for name in board.LABELS})
        result = run(root)
        assert result["Q_pool_minimum_T_program"] == "Q1"
        assert not result["research_replication_engineering_gate"]
        passed.append("exact minimum-T tie prefers Q1")

        root = base / "dedup"
        fixture(root, aliases={"Q2": "Q1", "Q4": "Q8"})
        result = run(root)
        assert set(result["T_ms"]) == {"Q1", "Q8", "TPS"}
        assert result["formal_timings"] == result["formal_warmups"] == 54
        passed.append("fixed-order deduplicated pool keeps canonical labels and reduced budget")

        negatives = {
            "missing_timing": lambda root: mutate_lines(root, "board/calls.jsonl", lambda rows: rows.pop()),
            "duplicate_invocation": lambda root: mutate_lines(root, "board/calls.jsonl", lambda rows: rows[1].update(invocation_id=rows[0]["invocation_id"])),
            "nonfinite_latency": lambda root: mutate_lines(root, "board/calls.jsonl", lambda rows: rows[3].update(latency_ms=float("nan"))),
            "unexpected_target_call": lambda root: mutate_lines(root, "board/calls.jsonl", lambda rows: rows[3].update(complete_operator_calls=2)),
            "failed_output": lambda root: mutate_lines(root, "board/calls.jsonl", lambda rows: rows[3].update(correct=False)),
            "changed_arguments": lambda root: mutate_lines(root, "board/calls.jsonl", lambda rows: rows[3].update(arguments_unchanged=False)),
            "unmatched_diagnostic": lambda root: mutate_lines(root, "board/calls.jsonl", lambda rows: rows[3].update(diagnostic_signature_matched=False)),
            "missing_dispatch_end": lambda root: mutate_lines(root, "board/dispatch.jsonl", lambda rows: rows.pop()),
            "wrong_helper_count": lambda root: mutate_lines(root, "board/helper_counts.jsonl", lambda rows: rows[0].update(count=2)),
            "wrong_cpu": lambda root: mutate_lines(root, "board/sessions.jsonl", lambda rows: rows[0]["environment"].update(masks=["0-3"])),
            "wrong_runtime": lambda root: mutate_lines(root, "board/sessions.jsonl", lambda rows: rows[0]["runtime_attestation"].update({"libvta.so": "OLD-RUNTIME"})),
            "wrong_seed_order": lambda root: mutate_lines(root, "board/calls.jsonl", lambda rows: rows[3].update(seed=20260910)),
            "changed_parameter_handle": lambda root: mutate_lines(root, "board/calls.jsonl", lambda rows: rows[3].update(buffer_handles=["CHANGED"])),
            "invalid_attempt": lambda root: write(root / "board/invalid_attempt.json", {"synthetic": True}),
            "missing_phase_summary": lambda root: (root / "qualification/summary.json").unlink(),
        }
        for name, mutation in negatives.items():
            root = base / ("negative-" + name)
            fixture(root)
            mutation(root)
            try:
                run(root)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError("invalid synthetic ledger accepted: " + name)
            assert not (root / "analysis.json").exists()
            passed.append("reject " + name)
    return {"status": "INDEPENDENT_LOCAL_SYNTHETIC_REVIEW_PASSED", "board_contacted": False,
            "synthetic_data_is_experimental_evidence": False, "checks": passed}


if __name__ == "__main__":
    print(json.dumps(review(), indent=2, sort_keys=True))
