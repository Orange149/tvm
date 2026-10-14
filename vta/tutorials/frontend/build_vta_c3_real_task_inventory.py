#!/usr/bin/env python3
"""Build the C3 real-model call/task/exposure inventory without board contact.

The inventory deliberately separates four identities that older C3 records often
collapsed: a graph call instance, a context-bearing execution contract, the exact
AutoTVM workload used by the current dispatch mechanism, and a geometric exposure
signature.  Geometry exposure never implies measured latency.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
C3_ROOT = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
DEFAULT_YOLO_CFG = Path("/home/orange/.tvm_test_data/darknet/yolov3-tiny.cfg")
DEFAULT_OUTPUT = C3_ROOT / "07_grouped_holdout/20260924_c3_real_task_inventory_run01"
SCHEMA = "c3_real_task_inventory_v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def normalized(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "value"):
        return normalized(value.value)
    if hasattr(value, "items"):
        return {str(key): normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) or hasattr(value, "__iter__"):
        return [normalized(item) for item in value]
    raise TypeError("unsupported value: {}".format(type(value).__name__))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def parse_darknet_cfg(path):
    """Parse Darknet cfg sections while preserving repeated section names."""

    raw = Path(path).read_text(encoding="utf-8")
    sections = []
    current = None
    for line_number, original in enumerate(raw.splitlines(), 1):
        line = original.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = {"type": line[1:-1].strip(), "line": line_number, "options": {}}
            sections.append(current)
            continue
        if current is None or "=" not in line:
            raise ValueError("invalid Darknet cfg line {}: {}".format(line_number, original))
        key, value = (item.strip() for item in line.split("=", 1))
        current["options"][key] = value
    if not sections or sections[0]["type"] != "net":
        raise ValueError("Darknet cfg must start with [net]")
    return sections


def _int(options, key, default=None):
    if key not in options:
        if default is None:
            raise KeyError(key)
        return int(default)
    return int(options[key])


def darknet_call_instances(path):
    """Derive convolution calls and tensor geometry directly from a Darknet cfg."""

    sections = parse_darknet_cfg(path)
    net = sections[0]["options"]
    state = {"channels": _int(net, "channels"), "height": _int(net, "height"), "width": _int(net, "width")}
    outputs = []
    calls = []
    for layer_index, section in enumerate(sections[1:]):
        kind = section["type"]
        options = section["options"]
        input_state = dict(state)
        if kind == "convolutional":
            filters = _int(options, "filters")
            size = _int(options, "size")
            stride = _int(options, "stride", 1)
            padding = size // 2 if _int(options, "pad", 0) else _int(options, "padding", 0)
            out_h = (state["height"] + 2 * padding - size) // stride + 1
            out_w = (state["width"] + 2 * padding - size) // stride + 1
            state = {"channels": filters, "height": out_h, "width": out_w}
            contract = {
                "batch": 1,
                "ci": input_state["channels"],
                "co": filters,
                "input_height": input_state["height"],
                "input_width": input_state["width"],
                "output_height": out_h,
                "output_width": out_w,
                "kernel_height": size,
                "kernel_width": size,
                "strides": [stride, stride],
                "padding": [padding, padding, padding, padding],
                "dilation": [1, 1],
                "groups": 1,
                "source_layout": "NCHW/OIHW",
                "input_dtype": "float32_source",
                "weight_dtype": "float32_source",
                "output_dtype": "float32_source",
            }
            context = {
                "batch_normalize": bool(_int(options, "batch_normalize", 0)),
                "activation": options.get("activation", "linear"),
                "cfg_layer_index": layer_index,
            }
            calls.append(
                {
                    "model": "yolov3_tiny_416_darknet",
                    "source": "darknet_cfg",
                    "source_layer": "conv{}".format(layer_index),
                    "call_ordinal": len(calls),
                    "operator_contract": contract,
                    "context_contract": context,
                }
            )
        elif kind == "maxpool":
            size = _int(options, "size")
            stride = _int(options, "stride")
            # Darknet uses padding=size-1 and integer division for maxpool.
            out_h = (state["height"] + size - 1 - size) // stride + 1
            out_w = (state["width"] + size - 1 - size) // stride + 1
            state = {"channels": state["channels"], "height": out_h, "width": out_w}
        elif kind == "upsample":
            stride = _int(options, "stride")
            state = {
                "channels": state["channels"],
                "height": state["height"] * stride,
                "width": state["width"] * stride,
            }
        elif kind == "route":
            indices = [int(item.strip()) for item in options["layers"].split(",")]
            resolved = [index if index >= 0 else layer_index + index for index in indices]
            selected = [outputs[index] for index in resolved]
            if len({(item["height"], item["width"]) for item in selected}) != 1:
                raise ValueError("route combines different spatial shapes at layer {}".format(layer_index))
            state = {
                "channels": sum(item["channels"] for item in selected),
                "height": selected[0]["height"],
                "width": selected[0]["width"],
            }
        elif kind in ("yolo", "shortcut"):
            pass
        else:
            raise ValueError("unsupported Darknet layer type: {}".format(kind))
        outputs.append(dict(state))

    for index, row in enumerate(calls):
        layer = row["context_contract"]["cfg_layer_index"]
        following = []
        for section in sections[layer + 2 :]:
            if section["type"] == "convolutional":
                break
            following.append(section["type"])
        row["context_contract"]["following_nonconv_layers"] = following
        row["call_instance_id"] = digest(
            {"model": row["model"], "source_layer": row["source_layer"], "ordinal": index}
        )
    return calls


def tensor_shape(typ):
    return [int(value) for value in typ.shape]


def call_op_name(call):
    import tvm

    return call.op.name if isinstance(call.op, tvm.ir.Op) else type(call.op).__name__


def resnet18_call_instances():
    """Extract the exact quantized and graph-packed ResNet18 call instances."""

    import tvm
    from tvm import relay
    import vta

    from build_vta_resnet50_relay_residency_dispatch_pair import make_relay_testing_resnet_program

    env = vta.get_env()
    function, _ = make_relay_testing_resnet_program(env, 18, device_annot=False)
    module = relay.transform.InferType()(tvm.IRModule.from_expr(function))
    calls = []
    relay.analysis.post_order_visit(
        module["main"],
        lambda expr: calls.append(expr)
        if isinstance(expr, relay.Call)
        and isinstance(expr.op, tvm.ir.Op)
        and expr.op.name == "nn.conv2d"
        else None,
    )
    all_calls = []
    relay.analysis.post_order_visit(
        module["main"], lambda expr: all_calls.append(expr) if isinstance(expr, relay.Call) else None
    )
    rows = []
    for ordinal, call in enumerate(calls):
        direct_consumers = []
        for consumer in all_calls:
            if any(
                isinstance(argument, relay.Call) and argument.same_as(call)
                for argument in consumer.args
            ):
                direct_consumers.append(call_op_name(consumer))
        data_shape = tensor_shape(call.args[0].checked_type)
        weight_shape = tensor_shape(call.args[1].checked_type)
        output_shape = tensor_shape(call.checked_type)
        packed = str(call.attrs.data_layout) == "NCHW1n16c"
        if packed:
            batch = data_shape[0] * data_shape[4]
            ci = data_shape[1] * data_shape[5]
            co = weight_shape[0] * weight_shape[4]
            in_h, in_w = data_shape[2:4]
            out_h, out_w = output_shape[2:4]
            kh, kw = weight_shape[2:4]
            input_dtype = str(call.args[0].checked_type.dtype)
            weight_dtype = str(call.args[1].checked_type.dtype)
            output_dtype = str(call.checked_type.dtype)
        else:
            batch, ci, in_h, in_w = data_shape
            co, _, kh, kw = weight_shape
            _, _, out_h, out_w = output_shape
            input_dtype = str(call.args[0].checked_type.dtype)
            weight_dtype = str(call.args[1].checked_type.dtype)
            output_dtype = str(call.checked_type.dtype)
        contract = {
            "batch": batch,
            "ci": ci,
            "co": co,
            "input_height": in_h,
            "input_width": in_w,
            "output_height": out_h,
            "output_width": out_w,
            "kernel_height": kh,
            "kernel_width": kw,
            "strides": [int(value) for value in call.attrs.strides],
            "padding": [int(value) for value in call.attrs.padding],
            "dilation": [int(value) for value in call.attrs.dilation],
            "groups": int(call.attrs.groups),
            "source_layout": "{}/{}".format(call.attrs.data_layout, call.attrs.kernel_layout),
            "input_dtype": input_dtype,
            "weight_dtype": weight_dtype,
            "output_dtype": output_dtype,
        }
        context = {
            "packed_vta_region": packed,
            "direct_consumer_ops": sorted(direct_consumers),
            "final_fusion_contract_status": "direct_consumers_only_not_a_lowered_fusion_certificate",
        }
        row = {
            "model": "relay_testing_resnet18",
            "source": "quantized_graph_pack_relay",
            "source_layer": "packed_conv_call_{:02d}".format(ordinal),
            "call_ordinal": ordinal,
            "operator_contract": contract,
            "context_contract": context,
        }
        row["call_instance_id"] = digest(
            {"model": row["model"], "source_layer": row["source_layer"], "ordinal": ordinal}
        )
        rows.append(row)
    return rows


def vta_workload(contract, block_in=16, block_out=16):
    if contract["ci"] % block_in or contract["co"] % block_out:
        return None
    if contract["input_dtype"] not in ("int8", "float32_source"):
        return None
    return [
        "conv2d_packed.vta",
        [
            "TENSOR",
            [
                contract["batch"],
                contract["ci"] // block_in,
                contract["input_height"],
                contract["input_width"],
                1,
                block_in,
            ],
            "int8",
        ],
        [
            "TENSOR",
            [
                contract["co"] // block_out,
                contract["ci"] // block_in,
                contract["kernel_height"],
                contract["kernel_width"],
                block_out,
                block_in,
            ],
            "int8",
        ],
        contract["strides"],
        contract["padding"],
        contract["dilation"],
        "NCHW1n16c",
        "int32",
    ]


def geometry_signature(workload):
    if workload is None:
        return None
    data, weight = workload[1][1], workload[2][1]
    return [
        int(data[0] * data[4]),
        int(data[1] * data[5]),
        int(data[2]),
        int(data[3]),
        int(weight[0] * weight[4]),
        int(weight[2]),
        int(weight[3]),
        *[int(item) for item in workload[3]],
        *[int(item) for item in workload[4]],
    ]


def annotate_calls(rows):
    result = []
    for row in rows:
        item = dict(row)
        workload = vta_workload(item["operator_contract"])
        item["vta_workload"] = workload
        item["geometry_signature"] = geometry_signature(workload)
        item["dispatch_scope_id"] = digest(workload) if workload is not None else None
        item["execution_contract_id"] = digest(
            {
                "workload": workload,
                "operator_contract": item["operator_contract"],
                "context_contract": item["context_contract"],
            }
        )
        item["vta_candidate_status"] = (
            "eligible_for_local_generation"
            if workload is not None
            and item["operator_contract"]["input_dtype"] in ("int8", "float32_source")
            else "not_representable_by_current_packed_template"
        )
        result.append(item)
    return result


def group_rows(calls, key_name, schema):
    groups = defaultdict(list)
    for row in calls:
        if row[key_name] is not None:
            groups[row[key_name]].append(row)
    result = []
    for identity, rows in sorted(groups.items()):
        result.append(
            {
                "schema": schema,
                key_name: identity,
                "model": rows[0]["model"],
                "call_instance_ids": [row["call_instance_id"] for row in rows],
                "call_count": len(rows),
                "source_layers": [row["source_layer"] for row in rows],
                "vta_workload": rows[0]["vta_workload"],
                "geometry_signature": rows[0]["geometry_signature"],
                "execution_contract_ids": sorted({row["execution_contract_id"] for row in rows}),
                "fusion_context_status": (
                    "single_observed_context"
                    if len({canonical(row["context_contract"]) for row in rows}) == 1
                    else "multiple_observed_contexts"
                ),
            }
        )
    return result


def exposure_catalog(c3_root, dispatch_tasks):
    from vta_geometry_catalog import audit_sources

    rows = audit_sources(c3_root, [])
    by_signature = defaultdict(list)
    for row in rows:
        path = Path(row["source"])
        lower_parts = [part.lower() for part in path.parts]
        board_like = any(
            token in part
            for part in lower_parts
            for token in ("board", "timing", "latency", "rpc")
        )
        for signature in row["geometry_signatures"]:
            by_signature[tuple(signature)].append(
                {
                    "source": row["source"],
                    "source_sha256": row["source_sha256"],
                    "board_or_timing_named_source": board_like,
                    "interpretation": (
                        "conservative_possible_latency_exposure"
                        if board_like
                        else "geometry_reserved_not_latency_proof"
                    ),
                }
            )
    output = []
    for task in dispatch_tasks:
        sources = by_signature.get(tuple(task["geometry_signature"]), [])
        possible_latency = [row for row in sources if row["board_or_timing_named_source"]]
        output.append(
            {
                "dispatch_scope_id": task["dispatch_scope_id"],
                "model": task["model"],
                "geometry_signature": task["geometry_signature"],
                "reserved_source_count": len(sources),
                "possible_latency_source_count": len(possible_latency),
                "strict_latency_unseen_status": (
                    "not_eligible_conservative_exposure_detected"
                    if possible_latency
                    else "eligible_pending_manual_evidence_review"
                ),
                "sources": sources,
                "scope": (
                    "path-name based conservative screen; board/timing name is possible exposure, "
                    "not proof that this exact candidate was timed"
                ),
            }
        )
    return output


def build_inventory(yolo_cfg, c3_root):
    calls = annotate_calls(resnet18_call_instances() + darknet_call_instances(yolo_cfg))
    execution = group_rows(calls, "execution_contract_id", "c3_execution_task_v1")
    dispatch = group_rows(calls, "dispatch_scope_id", "c3_dispatch_task_v1")
    exposure = exposure_catalog(c3_root, dispatch)
    exposure_by_id = {row["dispatch_scope_id"]: row for row in exposure}
    for row in dispatch:
        row["exposure"] = {
            key: value
            for key, value in exposure_by_id[row["dispatch_scope_id"]].items()
            if key not in ("sources", "dispatch_scope_id", "model", "geometry_signature")
        }
        row["incumbent"] = {
            "status": "pending_exact_evidence_binding",
            "candidate_id": None,
            "reason": "inventory phase does not infer incumbent from geometry alone",
        }
    counts = {
        "call_instances": len(calls),
        "vta_representable_calls": sum(row["vta_workload"] is not None for row in calls),
        "unique_execution_contracts": len(execution),
        "unique_dispatch_scopes": len(dispatch),
        "dispatch_scopes_by_model": dict(Counter(row["model"] for row in dispatch)),
        "strict_latency_unseen_pending_review": sum(
            row["strict_latency_unseen_status"] == "eligible_pending_manual_evidence_review"
            for row in exposure
        ),
    }
    return calls, execution, dispatch, exposure, counts


def render_status(counts, dispatch, exposure):
    lines = [
        "# C3 real-model task inventory",
        "",
        "Status: `INVENTORY_BUILT_INCUMBENTS_AND_FUSION_CERTIFICATES_PENDING`.",
        "",
        "- Call instances: {}".format(counts["call_instances"]),
        "- VTA-representable calls: {}".format(counts["vta_representable_calls"]),
        "- Unique execution contracts: {}".format(counts["unique_execution_contracts"]),
        "- Unique dispatch scopes: {}".format(counts["unique_dispatch_scopes"]),
        "- Conservative latency-unseen candidates pending manual review: {}".format(
            counts["strict_latency_unseen_pending_review"]
        ),
        "",
        "No board was contacted and no new latency was read. Geometry exposure is not treated as latency proof.",
        "Incumbents are intentionally unbound until exact candidate and comparable timing evidence are joined.",
        "Direct Relay consumers are recorded, but they are not presented as lowered fusion certificates.",
        "",
        "## Dispatch scopes",
        "",
        "| model | calls | contexts | exposure status | source layers |",
        "|---|---:|---:|---|---|",
    ]
    exposure_by_id = {row["dispatch_scope_id"]: row for row in exposure}
    for row in dispatch:
        seen = exposure_by_id[row["dispatch_scope_id"]]
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                row["model"],
                row["call_count"],
                len(row["execution_contract_ids"]),
                seen["strict_latency_unseen_status"],
                ", ".join(row["source_layers"]),
            )
        )
    return "\n".join(lines) + "\n"


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    yolo_cfg = Path(args.yolo_cfg).resolve()
    if not yolo_cfg.is_file():
        raise FileNotFoundError(yolo_cfg)
    calls, execution, dispatch, exposure, counts = build_inventory(yolo_cfg, args.c3_root)
    output.mkdir(parents=True)
    write_jsonl(output / "call_instances.jsonl", calls)
    write_jsonl(output / "execution_tasks.jsonl", execution)
    write_jsonl(output / "dispatch_tasks.jsonl", dispatch)
    write_jsonl(output / "exposure_ledger.jsonl", exposure)
    summary = {
        "schema": SCHEMA,
        "status": "inventory_built_incumbents_and_fusion_certificates_pending",
        "counts": counts,
        "sources": {
            "resnet18": "relay.testing.resnet18 quantized and graph-packed Relay",
            "yolov3_tiny": str(yolo_cfg),
            "yolov3_tiny_sha256": sha256_file(yolo_cfg),
            "historical_geometry": str(Path(args.c3_root).resolve()),
        },
        "board_contacted": False,
        "performance_values_read": False,
        "claim_boundary": (
            "real call/task identity and conservative exposure inventory only; no incumbent, "
            "fusion-equivalence, local legality, FPGA correctness, latency, or oracle claim"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "STATUS.md").write_text(
        render_status(counts, dispatch, exposure), encoding="utf-8"
    )
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(
        output / "artifact_hashes.json",
        {"artifacts": artifacts, "source_sha256": sha256_file(Path(__file__).resolve())},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-cfg", type=Path, default=DEFAULT_YOLO_CFG)
    parser.add_argument("--c3-root", type=Path, default=C3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
