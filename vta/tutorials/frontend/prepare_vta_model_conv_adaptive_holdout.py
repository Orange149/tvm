#!/usr/bin/env python3
"""Freeze an explicit model convolution before target qualification or labels."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import vta

import prepare_vta_p7r115_yolo_confirmation as base
from vta_geometry_catalog import audit_sources, assert_unreserved


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESNET_SOURCE = REPO / "python" / "tvm" / "relay" / "testing" / "resnet.py"


def extract_resnet50_conv(layer):
    """Return the exact Relay geometry for a named ResNet50-v2 convolution."""
    import tvm
    from tvm import relay
    from tvm.relay.testing import resnet

    mod, _ = resnet.get_workload(
        num_layers=50, batch_size=1, image_shape=(3, 224, 224), dtype="float32"
    )
    mod = relay.transform.InferType()(mod)
    wanted = layer + "_weight"
    matches = []

    class Visitor(relay.ExprVisitor):
        def visit_call(self, call):
            super().visit_call(call)
            if not (isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.conv2d"):
                return
            if not (len(call.args) > 1 and isinstance(call.args[1], relay.Var)):
                return
            if call.args[1].name_hint != wanted:
                return
            shape = lambda typ: [int(value) for value in typ.shape]
            data_shape = shape(call.args[0].checked_type)
            weight_shape = shape(call.args[1].checked_type)
            output_shape = shape(call.checked_type)
            strides = [int(value) for value in call.attrs.strides]
            padding = [int(value) for value in call.attrs.padding]
            matches.append({
                "weight_parameter": wanted,
                "input_shape_nchw": data_shape,
                "weight_shape_oihw": weight_shape,
                "output_shape_nchw": output_shape,
                "strides": strides,
                "padding": padding,
                "ci": data_shape[1],
                "co": weight_shape[0],
                "height": data_shape[2],
                "width": data_shape[3],
                "kernel": weight_shape[2],
                "stride": strides[0],
                "symmetric_padding": padding[0],
            })

    Visitor().visit(mod["main"])
    if len(matches) != 1:
        raise ValueError("expected exactly one Relay convolution named {} but found {}".format(
            wanted, len(matches)
        ))
    return matches[0]


def extract_yolov3_tiny_conv(layer, cfg, weights, darknet_lib):
    """Return one Darknet-imported YOLOv3-tiny convolution geometry."""
    import tvm
    from tvm import relay
    from tvm.relay.testing.darknet import __darknetffi__

    assets = [Path(cfg), Path(weights), Path(darknet_lib)]
    for asset in assets:
        if not asset.is_file():
            raise FileNotFoundError(asset)
    net = __darknetffi__.dlopen(str(darknet_lib)).load_network(
        str(cfg).encode("utf-8"), str(weights).encode("utf-8"), 0
    )
    mod, _ = relay.frontend.from_darknet(
        net, dtype="float32", shape=(1, int(net.c), int(net.h), int(net.w))
    )
    mod = relay.transform.InferType()(mod)
    match = re.fullmatch(r"conv(\d+)", layer)
    if not match:
        raise ValueError("YOLO layer must use the conv<N> source name")
    wanted = "LAYERTYPE_CONVOLUTIONAL{}_weight".format(match.group(1))
    matches = []

    class Visitor(relay.ExprVisitor):
        def visit_call(self, call):
            super().visit_call(call)
            if not (isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.conv2d"):
                return
            if not (len(call.args) > 1 and isinstance(call.args[1], relay.Var)):
                return
            if call.args[1].name_hint != wanted:
                return
            shape = lambda typ: [int(value) for value in typ.shape]
            data_shape = shape(call.args[0].checked_type)
            weight_shape = shape(call.args[1].checked_type)
            output_shape = shape(call.checked_type)
            strides = [int(value) for value in call.attrs.strides]
            padding = [int(value) for value in call.attrs.padding]
            matches.append({
                "weight_parameter": wanted,
                "input_shape_nchw": data_shape,
                "weight_shape_oihw": weight_shape,
                "output_shape_nchw": output_shape,
                "strides": strides,
                "padding": padding,
                "ci": data_shape[1],
                "co": weight_shape[0],
                "height": data_shape[2],
                "width": data_shape[3],
                "kernel": weight_shape[2],
                "stride": strides[0],
                "symmetric_padding": padding[0],
                "network_input_nchw": [1, int(net.c), int(net.h), int(net.w)],
            })

    Visitor().visit(mod["main"])
    if len(matches) != 1:
        raise ValueError("expected exactly one Relay convolution named {} but found {}".format(
            wanted, len(matches)
        ))
    return matches[0]


def verify_declared_model_geometry(args):
    """Fail closed when an exact ResNet layer name disagrees with Relay InferType."""
    if args.model == "resnet50_v2" and re.fullmatch(
        r"stage\d+_unit\d+_conv[123]", args.layer
    ):
        actual = extract_resnet50_conv(args.layer)
    elif args.model.startswith("yolov3_tiny") and re.fullmatch(r"conv\d+", args.layer):
        if not args.darknet_weights or not args.darknet_lib:
            raise ValueError("YOLO source verification requires --darknet-weights and --darknet-lib")
        actual = extract_yolov3_tiny_conv(
            args.layer, args.model_source, args.darknet_weights, args.darknet_lib
        )
    else:
        return None
    declared = {
        "ci": args.ci,
        "co": args.co,
        "height": args.height,
        "width": args.width,
        "kernel": args.kernel,
        "stride": args.stride,
        "symmetric_padding": args.kernel // 2 if args.padding is None else args.padding,
    }
    mismatches = {
        key: {"declared": declared[key], "model": actual[key]}
        for key in declared if declared[key] != actual[key]
    }
    if mismatches:
        raise ValueError("declared convolution disagrees with Relay model: " + json.dumps(
            mismatches, sort_keys=True
        ))
    return actual


def workload(args, env):
    if args.ci % env.BLOCK_IN or args.co % env.BLOCK_OUT:
        raise ValueError("channels must be exactly representable in packed VTA layout")
    pad = args.kernel // 2 if args.padding is None else args.padding
    return [
        "conv2d_packed.vta",
        ["TENSOR", [env.BATCH, args.ci // env.BLOCK_IN, args.height, args.width,
                    1, env.BLOCK_IN], "int8"],
        ["TENSOR", [args.co // env.BLOCK_OUT, args.ci // env.BLOCK_IN,
                    args.kernel, args.kernel, env.BLOCK_OUT, env.BLOCK_IN], "int8"],
        [args.stride, args.stride], [pad, pad, pad, pad], [1, 1],
        "NCHW1n16c", "int32",
    ]


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable holdout " + str(output))
    if args.families != 8:
        raise ValueError("protocol requires exactly eight same-tile families")
    model_source_geometry = verify_declared_model_geometry(args)
    env = vta.get_env()
    target_workload = workload(args, env)
    signature = base.workload_signature(target_workload)
    exposure = audit_sources(base.C3, base.EXPOSURE_SOURCES.values())
    replay_reference = None
    if args.cost_replay_of:
        replay_root = Path(args.cost_replay_of).resolve()
        replay_reference = json.loads((replay_root / "contract.json").read_text())
        expected = {
            "workload_id": args.workload_id,
            "source_model": args.model,
            "source_layer": args.layer,
            "explicit_conv": {
                "ci": args.ci,
                "co": args.co,
                "height": args.height,
                "width": args.width,
                "kernel": args.kernel,
                "stride": args.stride,
                "padding": args.kernel // 2 if args.padding is None else args.padding,
            },
            "workload": target_workload,
            "selection_seed": args.selection_seed,
            "families": args.families,
        }
        observed = {
            "workload_id": replay_reference.get("workload_id"),
            "source_model": replay_reference.get("source_model"),
            "source_layer": replay_reference.get("source_layer"),
            "explicit_conv": replay_reference.get("explicit_conv"),
            "workload": replay_reference.get("workload"),
            "selection_seed": replay_reference.get("selection", {}).get("seed"),
            "families": replay_reference.get("selection", {}).get("families"),
        }
        if observed != expected:
            raise ValueError(
                "cost replay arguments differ from frozen reference: "
                + json.dumps({"expected": expected, "observed": observed}, sort_keys=True)
            )
    else:
        assert_unreserved(signature, exposure)

    model_source = Path(args.model_source).resolve()
    if not model_source.is_file():
        raise FileNotFoundError(model_source)
    source_hashes = {str(path.relative_to(REPO)): base.sha256_file(path)
                     for path in base.SCHEDULE_SOURCES}
    schedule_version = base.canonical_sha256(source_hashes)
    fingerprint = base.hardware_fingerprint(env)
    tasks = base.create_tasks(target_workload, env)
    old_seed = base.SELECTION_SEED
    try:
        base.SELECTION_SEED = args.selection_seed
        domain, selected, mode_domains = base.enumerate_shared_domain(
            args.workload_id, target_workload, tasks, env, args.families
        )
    finally:
        base.SELECTION_SEED = old_seed

    candidates = []
    for ordinal, row in enumerate(selected):
        family_id = "{}F{:02d}".format(args.workload_id, ordinal)
        for mode, number in base.MODE_NUMBERS.items():
            identity = base.candidate_identity_record(
                fingerprint, base.TEMPLATE, schedule_version, target_workload,
                mode, row["complete_config_entity"],
                config_index=row["debug"]["config_index"],
            )
            candidates.append({
                "candidate_id": identity["candidate_id"],
                "identity": identity["identity"],
                "debug": identity["debug"],
                "workload_id": args.workload_id,
                "family_id": family_id,
                "family_ordinal": ordinal,
                "residence_mode": mode,
                "mode_number": number,
                "selection_rank_sha256": row["sampling_rank_sha256"],
                "hardware_predicate": row["hardware_predicate"],
                "local_lower_status": "not_run_by_contract_generator",
                "board_status": "not_dispatched",
                "performance_label": None,
            })
    if len(candidates) != 32:
        raise AssertionError("expected eight families by four legacy modes")

    replay_proof = None
    if replay_reference is not None:
        replay_root = Path(args.cost_replay_of).resolve()
        reference_candidates = [
            json.loads(line)
            for line in (replay_root / "candidates.jsonl").read_text().splitlines()
            if line.strip()
        ]
        reference_source = replay_reference.get("source_model_implementation", {})
        current_source_sha256 = base.sha256_file(model_source)
        if reference_source.get("sha256") != current_source_sha256:
            raise ValueError("cost replay model source hash differs from frozen reference")
        current_assets = {
            key: base.sha256_file(Path(value))
            for key, value in {
                "darknet_weights": args.darknet_weights,
                "darknet_lib": args.darknet_lib,
            }.items() if value
        }
        reference_assets = {
            key: row.get("sha256")
            for key, row in replay_reference.get("source_model_assets", {}).items()
        }
        if current_assets != reference_assets:
            raise ValueError("cost replay model asset hashes differ from frozen reference")
        if candidates != reference_candidates:
            raise ValueError("cost replay candidate identities differ from frozen reference")
        replay_proof = {
            "mode": "retrospective_cost_reexecution_only",
            "reference_contract_path": str(replay_root),
            "reference_contract_sha256": base.sha256_file(replay_root / "contract.json"),
            "reference_candidates_sha256": base.sha256_file(replay_root / "candidates.jsonl"),
            "candidate_rows_exactly_equal": True,
            "target_performance_labels_read": False,
        }

    contract = {
        "schema": "c3_model_conv_adaptive_holdout_contract_v1",
        "status": (
            "retrospective_cost_reexecution_exact_identity"
            if replay_reference is not None else
            "frozen_before_static_fsim_fpga_or_performance_observation"
        ),
        "workload_id": args.workload_id,
        "source_model": args.model,
        "source_layer": args.layer,
        "source_model_implementation": {
            "path": str(model_source.relative_to(REPO)),
            "sha256": base.sha256_file(model_source),
            "derivation": args.layer_derivation,
        },
        "source_model_assets": {
            key: {"path": str(path.resolve()), "sha256": base.sha256_file(path)}
            for key, value in {
                "darknet_weights": args.darknet_weights,
                "darknet_lib": args.darknet_lib,
            }.items() if value for path in [Path(value)]
        },
        "model_source_geometry": model_source_geometry,
        "explicit_conv": {"ci": args.ci, "co": args.co, "height": args.height,
                          "width": args.width, "kernel": args.kernel,
                          "stride": args.stride,
                          "padding": args.kernel // 2 if args.padding is None else args.padding},
        "workload": target_workload,
        "geometry_signature": signature,
        "exposed_performance_label_collision": replay_reference is not None,
        "geometry_reservation_check": "legacy sources plus all historical *contract*.json; conservatively includes unmeasured contracts",
        "geometry_catalog_source_sha256": base.sha256_file(HERE / "vta_geometry_catalog.py"),
        "prior_performance_label_exposure": exposure,
        "hardware_fingerprint": fingerprint,
        "template": base.TEMPLATE,
        "schedule_source_sha256": source_hashes,
        "schedule_version": schedule_version,
        "complete_config_domain": mode_domains,
        "complete_original_domain_count": len(domain),
        "label_free_eligible_count": sum(row["hardware_predicate"]["passed"]
                                         for row in domain),
        "selection": {
            "seed": args.selection_seed,
            "families": args.families,
            "modes": base.MODE_NUMBERS,
            "rule": "capacity/reuse predicate, then SHA256(seed|workload|ConfigEntity)",
            "uses_static_dma_label": False,
            "uses_fsim": False,
            "uses_board_or_runtime_measurement": False,
            "uses_tophub": False,
            "candidate_commitment_sha256": base.canonical_sha256(candidates),
        },
        "failure_policy": "all failures retained; no label-driven replacement",
        "cost_replay_proof": replay_proof,
        "claim_boundary": (
            "Retrospective cost reexecution of an already exposed workload. Candidate rows must "
            "match the frozen reference exactly; no target performance label is consumed."
            if replay_reference is not None else
            "candidate freeze only; no target outcome or performance claim"
        ),
    }
    output.mkdir(parents=True)
    base.write_json(output / "contract.json", contract)
    (output / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
        encoding="utf-8",
    )
    (output / "complete_domain.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in domain),
        encoding="utf-8",
    )
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    artifacts = {path.name: base.sha256_file(path) for path in sorted(output.iterdir())
                 if path.is_file() and path.name != "artifact_hashes.json"}
    base.write_json(output / "artifact_hashes.json", {
        "artifacts": artifacts,
        "source_sha256": {str(Path(__file__).resolve()): base.sha256_file(__file__)},
    })
    print(json.dumps({
        "workload_id": args.workload_id,
        "geometry_signature": signature,
        "complete_original_domain_count": len(domain),
        "label_free_eligible_count": contract["label_free_eligible_count"],
        "candidate_count": len(candidates),
        "commitment": contract["selection"]["candidate_commitment_sha256"],
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--model-source", default=str(RESNET_SOURCE))
    parser.add_argument("--layer-derivation", required=True)
    parser.add_argument("--darknet-weights")
    parser.add_argument("--darknet-lib")
    parser.add_argument("--ci", type=int, required=True)
    parser.add_argument("--co", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--kernel", type=int, choices=(1, 3), required=True)
    parser.add_argument("--stride", type=int, choices=(1, 2), default=1)
    parser.add_argument("--padding", type=int, choices=(0, 1))
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--selection-seed", required=True)
    parser.add_argument(
        "--cost-replay-of",
        help=(
            "Permit an exposed geometry only for an exact, fail-closed timing replay of an "
            "existing frozen target contract. The output is not a new holdout."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
