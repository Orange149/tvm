#!/usr/bin/env python3
"""Generate deterministic validation artifacts for the RAMPS event graph."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import resource_aware_maxplus as ramps


DEFAULT_OUTPUT_DIR = (
    "vta/tutorials/frontend/report_out/resource_aware_maxplus/"
    "stage3_event_graph_validation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def record(name: str, stages: List[ramps.StageService], queue_depth: int = 2, **kwargs):
    return ramps.PipelineRecord(
        model="synthetic",
        candidate_id=name,
        source_path="deterministic_validation",
        group_id="stage3",
        measured_cycle_ms=None,
        queue_depth=queue_depth,
        stages=stages,
        effective_segment_count=len(stages),
        **kwargs,
    )


def cpu(name: str, service_ms: float, threads: int = 1) -> ramps.StageService:
    return ramps.StageService(
        name,
        "cpu",
        threads=threads,
        compute_ms=service_ms,
        core_demand_ms=service_ms,
        core_demand_source="synthetic_process_cpu_time",
    )


def validate_case(
    item: ramps.PipelineRecord,
    expected_ms: float,
    expected_resource: str,
    required_critical_resource: str = "",
) -> Dict[str, Any]:
    payload = ramps.timed_event_graph_payload(
        item, require_measured_core_demand=True
    )
    analytical = float(payload["analytical_cycle_ms"])
    spectral = float(payload["karp_spectral_radius_ms"])
    critical_resources = {
        str(cycle["resource"]) for cycle in payload.get("critical_cycles", [])
    }
    passed = (
        math.isclose(analytical, expected_ms, rel_tol=1.0e-12, abs_tol=1.0e-12)
        and math.isclose(analytical, spectral, rel_tol=1.0e-12, abs_tol=1.0e-12)
        and payload["bottleneck_cycle"]["resource"] == expected_resource
        and (
            not required_critical_resource
            or required_critical_resource in critical_resources
        )
    )
    return {
        "name": item.candidate_id,
        "expected_cycle_ms": expected_ms,
        "expected_resource": expected_resource,
        "analytical_cycle_ms": analytical,
        "karp_spectral_radius_ms": spectral,
        "spectral_validation_abs_error_ms": abs(analytical - spectral),
        "actual_resource": payload["bottleneck_cycle"]["resource"],
        "critical_resources": sorted(critical_resources),
        "passed": passed,
        "artifact": payload,
    }


def build_validation() -> Dict[str, Any]:
    cases = [
        validate_case(
            record(
                "balanced_cpu_vta_cpu",
                [cpu("cpu0", 10.0), ramps.StageService("vta1", "vta", compute_ms=10.0), cpu("cpu2", 10.0)],
                queue_depth=1,
            ),
            10.0,
            "stage_worker",
        ),
        validate_case(
            record(
                "two_vta_islands_mutex",
                [
                    cpu("cpu0", 8.0),
                    ramps.StageService("vta1", "vta", compute_ms=7.0),
                    cpu("cpu2", 8.0),
                    ramps.StageService("vta3", "vta", compute_ms=9.0),
                    cpu("cpu4", 8.0),
                ],
            ),
            16.0,
            "vta_mutex",
        ),
        validate_case(
            record(
                "independent_cpu_vta",
                [cpu("cpu0", 9.0), ramps.StageService("vta1", "vta", compute_ms=11.0)],
            ),
            11.0,
            "stage_worker",
        ),
        validate_case(
            record(
                "boundary_communication_bottleneck",
                [cpu("cpu0", 5.0), ramps.StageService("vta1", "vta", compute_ms=5.0)],
                boundary_ms=14.0,
            ),
            14.0,
            "bridge",
        ),
        validate_case(
            record(
                "ps_pl_load_co_critical",
                [
                    ramps.StageService(
                        "vta0",
                        "vta",
                        compute_ms=1.0,
                        load_ms=12.0,
                        overlap_factor=0.0,
                    )
                ],
            ),
            12.0,
            "stage_worker",
            required_critical_resource="ps_pl_load",
        ),
    ]

    fifo_base = [cpu("cpu0", 20.0), ramps.StageService("vta1", "vta", compute_ms=10.0), cpu("cpu2", 5.0)]
    shallow = ramps.timed_event_graph_payload(
        record("fifo_q1", fifo_base, queue_depth=1),
        require_measured_core_demand=True,
    )
    deep = ramps.timed_event_graph_payload(
        record("fifo_q4", fifo_base, queue_depth=4),
        require_measured_core_demand=True,
    )
    fifo_channels = shallow["marked_event_graph"]["channels"]
    fifo_check = {
        "name": "bounded_fifo_backpressure",
        "q1_cycle_ms": shallow["analytical_cycle_ms"],
        "q4_cycle_ms": deep["analytical_cycle_ms"],
        "q1_free_slot_tokens": [
            item["initial_tokens"]
            for item in fifo_channels
            if item["kind"] == "fifo_free_slots"
        ],
        "passed": (
            all(
                item["initial_tokens"] == 1
                for item in fifo_channels
                if item["kind"] == "fifo_free_slots"
            )
            and float(deep["analytical_cycle_ms"])
            <= float(shallow["analytical_cycle_ms"])
            and math.isclose(float(shallow["analytical_cycle_ms"]), 20.0)
        ),
        "interpretation": (
            "Reverse free-slot tokens encode producer backpressure. In this "
            "deterministic linear case, queue depth changes buffering but not "
            "the 20 ms physical bottleneck."
        ),
    }
    results = [
        {key: value for key, value in item.items() if key != "artifact"}
        for item in cases
    ] + [fifo_check]
    return {
        "schema_version": 1,
        "publication_mode": True,
        "all_passed": all(item["passed"] for item in results),
        "results": results,
        "event_graphs": {item["name"]: item["artifact"] for item in cases},
    }


def write_report(output_dir: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# RAMPS Stage 3 Event-Graph Validation",
        "",
        "This report validates deterministic mechanisms before board calibration. It does not",
        "constitute hardware accuracy evidence.",
        "",
        "| Case | Expected | Analytical | Karp | Critical resources | Pass |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in payload["results"]:
        if "expected_cycle_ms" not in item:
            continue
        lines.append(
            "| `{}` | {:.3f} | {:.3f} | {:.3f} | `{}` | {} |".format(
                item["name"],
                item["expected_cycle_ms"],
                item["analytical_cycle_ms"],
                item["karp_spectral_radius_ms"],
                ", ".join(item["critical_resources"]),
                "yes" if item["passed"] else "no",
            )
        )
    fifo = payload["results"][-1]
    lines.extend(
        [
            "",
            "## FIFO Interpretation",
            "",
            fifo["interpretation"],
            "",
            "The marked graph contains forward zero-token data dependencies and reverse",
            "free-slot channels carrying `queue_depth` tokens. Queue depth is frozen in the",
            "current paper instance; it is not claimed as a searched optimization variable.",
            "",
            "## Evidence Boundary",
            "",
            "These synthetic checks establish implementation semantics and analytical/Karp",
            "consistency. CPU contention magnitudes, VTA overlap, DMA bandwidth and boundary",
            "service accuracy still require the no-fallback hardware calibration in stage 4.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_validation()
    (output_dir / "validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(output_dir, payload)
    if not payload["all_passed"]:
        raise SystemExit("RAMPS event-graph validation failed")
    print("RAMPS event-graph validation passed: {}".format(output_dir))


if __name__ == "__main__":
    main()
