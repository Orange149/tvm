#!/usr/bin/env python3
"""Split Gluon ResNet18 into coarse pipeline-friendly stages.

This script implements the graph-level split strategies discussed for
fine-grained CPU/VTA partitioning research. It does not apply VTA graph_pack
itself. Instead it creates stage-local Gluon blocks and optionally lowers each
stage to Relay independently.

Recommended first scheme:
  three_stage_a
    stage0_cpu : stem + layer1
    stage1_vta : layer2 + layer3
    stage2_cpu : layer4 + head
  three_stage_d
    stage0_cpu : stem + layer1_block0
    stage1_vta : layer1_block1 + layer2 + layer3
    stage2_cpu : layer4 + head
  three_stage_e
    stage0_cpu : stem + layer1_block0
    stage1_vta : layer1_block1 + layer2 + layer3 + layer4_block0
    stage2_cpu : layer4_block1 + head
"""

from __future__ import absolute_import, print_function

import argparse
import math
import os
from xml.sax.saxutils import escape

from mxnet.gluon import HybridBlock, nn
from mxnet.gluon.model_zoo import vision
import numpy as np

import tvm
from tvm import relay
import vta


def _make_stage(name, device, unit_names):
    return {"name": name, "device": device, "kind": "units", "unit_names": list(unit_names)}


def _block_units(block_prefix, has_skip):
    names = [block_prefix + "_main_preadd"]
    if has_skip:
        names.append(block_prefix + "_skip_proj")
    names.append(block_prefix + "_add_relu_tail")
    return names


BLOCK_UNIT_GROUPS = {
    "layer1_block0": _block_units("layer1_block0", False),
    "layer1_block1": _block_units("layer1_block1", False),
    "layer2_block0": _block_units("layer2_block0", True),
    "layer2_block1": _block_units("layer2_block1", False),
    "layer3_block0": _block_units("layer3_block0", True),
    "layer3_block1": _block_units("layer3_block1", False),
    "layer4_block0": _block_units("layer4_block0", True),
    "layer4_block1": _block_units("layer4_block1", False),
}

LAYER1_UNITS = BLOCK_UNIT_GROUPS["layer1_block0"] + BLOCK_UNIT_GROUPS["layer1_block1"]
LAYER2_UNITS = BLOCK_UNIT_GROUPS["layer2_block0"] + BLOCK_UNIT_GROUPS["layer2_block1"]
LAYER3_UNITS = BLOCK_UNIT_GROUPS["layer3_block0"] + BLOCK_UNIT_GROUPS["layer3_block1"]
LAYER4_UNITS = BLOCK_UNIT_GROUPS["layer4_block0"] + BLOCK_UNIT_GROUPS["layer4_block1"]

AUTO_RESOURCE_AWARE_SCHEME = "auto_resource_aware"
UNIT_CONV_COUNTS = {
    "stem": 1,
    "layer1_block0_main_preadd": 2,
    "layer1_block0_add_relu_tail": 0,
    "layer1_block1_main_preadd": 2,
    "layer1_block1_add_relu_tail": 0,
    "layer2_block0_main_preadd": 2,
    "layer2_block0_skip_proj": 1,
    "layer2_block0_add_relu_tail": 0,
    "layer2_block1_main_preadd": 2,
    "layer2_block1_add_relu_tail": 0,
    "layer3_block0_main_preadd": 2,
    "layer3_block0_skip_proj": 1,
    "layer3_block0_add_relu_tail": 0,
    "layer3_block1_main_preadd": 2,
    "layer3_block1_add_relu_tail": 0,
    "layer4_block0_main_preadd": 2,
    "layer4_block0_skip_proj": 1,
    "layer4_block0_add_relu_tail": 0,
    "layer4_block1_main_preadd": 2,
    "layer4_block1_add_relu_tail": 0,
    "head": 0,
}
UNIT_ORDER = ["stem"] + LAYER1_UNITS + LAYER2_UNITS + LAYER3_UNITS + LAYER4_UNITS + ["head"]

SCHEMES = {
    "all_vta": [_make_stage("stage0_vta", "vta", UNIT_ORDER)],
    "three_stage_a": [
        _make_stage("stage0_cpu", "cpu", ["stem"] + LAYER1_UNITS),
        _make_stage("stage1_vta", "vta", LAYER2_UNITS + LAYER3_UNITS),
        _make_stage("stage2_cpu", "cpu", LAYER4_UNITS + ["head"]),
    ],
    "two_stage_a": [
        _make_stage("stage0_cpu", "cpu", ["stem"] + LAYER1_UNITS + LAYER2_UNITS),
        _make_stage("stage1_vta", "vta", LAYER3_UNITS + LAYER4_UNITS + ["head"]),
    ],
    "three_stage_b": [
        _make_stage("stage0_cpu", "cpu", ["stem"] + LAYER1_UNITS + LAYER2_UNITS),
        _make_stage("stage1_vta", "vta", LAYER3_UNITS),
        _make_stage("stage2_cpu", "cpu", LAYER4_UNITS + ["head"]),
    ],
    "three_stage_c": [
        _make_stage("stage0_cpu", "cpu", ["stem"]),
        _make_stage("stage1_vta", "vta", LAYER1_UNITS + LAYER2_UNITS + LAYER3_UNITS),
        _make_stage("stage2_cpu", "cpu", LAYER4_UNITS + ["head"]),
    ],
    "three_stage_d": [
        _make_stage("stage0_cpu", "cpu", ["stem"] + BLOCK_UNIT_GROUPS["layer1_block0"]),
        _make_stage(
            "stage1_vta",
            "vta",
            BLOCK_UNIT_GROUPS["layer1_block1"] + LAYER2_UNITS + LAYER3_UNITS,
        ),
        _make_stage("stage2_cpu", "cpu", LAYER4_UNITS + ["head"]),
    ],
    "three_stage_e": [
        _make_stage("stage0_cpu", "cpu", ["stem"] + BLOCK_UNIT_GROUPS["layer1_block0"]),
        _make_stage(
            "stage1_vta",
            "vta",
            BLOCK_UNIT_GROUPS["layer1_block1"]
            + LAYER2_UNITS
            + LAYER3_UNITS
            + BLOCK_UNIT_GROUPS["layer4_block0"],
        ),
        _make_stage(
            "stage2_cpu",
            "cpu",
            BLOCK_UNIT_GROUPS["layer4_block1"] + ["head"],
        ),
    ],
    "block_stage_a": [
        _make_stage("stage0_cpu", "cpu", ["stem"] + LAYER1_UNITS),
        _make_stage(
            "stage1_vta",
            "vta",
            LAYER2_UNITS + BLOCK_UNIT_GROUPS["layer3_block0"],
        ),
        _make_stage(
            "stage2_cpu",
            "cpu",
            BLOCK_UNIT_GROUPS["layer3_block1"] + LAYER4_UNITS + ["head"],
        ),
    ],
    "block_stage_b": [
        _make_stage(
            "stage0_cpu",
            "cpu",
            ["stem"] + LAYER1_UNITS + BLOCK_UNIT_GROUPS["layer2_block0"],
        ),
        _make_stage(
            "stage1_vta",
            "vta",
            BLOCK_UNIT_GROUPS["layer2_block1"] + BLOCK_UNIT_GROUPS["layer3_block0"],
        ),
        _make_stage(
            "stage2_cpu",
            "cpu",
            BLOCK_UNIT_GROUPS["layer3_block1"] + LAYER4_UNITS + ["head"],
        ),
    ],
    "block_stage_c": [
        _make_stage("stage0_cpu", "cpu", ["stem"] + LAYER1_UNITS),
        _make_stage("stage1_vta", "vta", LAYER2_UNITS),
        _make_stage("stage2_cpu", "cpu", LAYER3_UNITS + LAYER4_UNITS + ["head"]),
    ],
    "block_stage_d": [
        _make_stage("stage0_cpu", "cpu", ["stem"]),
        _make_stage(
            "stage1_vta",
            "vta",
            LAYER1_UNITS + LAYER2_UNITS + BLOCK_UNIT_GROUPS["layer3_block0"],
        ),
        _make_stage(
            "stage2_cpu",
            "cpu",
            BLOCK_UNIT_GROUPS["layer3_block1"] + LAYER4_UNITS + ["head"],
        ),
    ],
}

RESOURCE_AWARE_SCORE_WEIGHTS = {
    "tile_input_load": 1.0,
    "tile_weight_load": 0.25,
    "tile_output_store": 1.0,
    "tile_spill": 1.0,
    "tile_count": 25000.0,
    "cpu_prefix": 0.35,
    "boundary": 0.35,
    "stage_imbalance": 0.35,
    "stage0_dominance": 0.75,
    "edge": 0.35,
    "fusion": 0.5,
    "span": 0.25,
}
RESOURCE_AWARE_CPU_WORKERS = 4
RESOURCE_AWARE_SHARED_CPU_WORKERS = 4
RESOURCE_AWARE_CALIBRATION_MANDATORY = [
    "three_stage_a",
    "three_stage_b",
    "block_stage_c",
    "all_vta",
]


class FeatureStage(HybridBlock):
    """A stage composed only of slices of model.features."""

    def __init__(self, blocks, prefix=None, params=None):
        super(FeatureStage, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.body = nn.HybridSequential()
            for block in blocks:
                self.body.add(block)

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        return self.body(x)


class HeadStage(HybridBlock):
    """A stage composed of tail feature blocks and optional dense output."""

    def __init__(self, feature_blocks, output_block=None, prefix=None, params=None):
        super(HeadStage, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.body = nn.HybridSequential()
            for block in feature_blocks:
                self.body.add(block)
            self.output = output_block

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        x = self.body(x)
        if self.output is not None:
            x = self.output(x)
        return x


class MainPreAddUnit(HybridBlock):
    """Residual main path before the residual add."""

    def __init__(self, body_block, prefix=None, params=None):
        super(MainPreAddUnit, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.body = body_block

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        return self.body(x), x


class SkipProjUnit(HybridBlock):
    """Residual skip projection branch for downsampled blocks."""

    def __init__(self, downsample_block, prefix=None, params=None):
        super(SkipProjUnit, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.body = downsample_block

    def hybrid_forward(self, F, main, residual):  # pylint: disable=arguments-differ
        return main, self.body(residual)


class AddReluTailUnit(HybridBlock):
    """Residual add + relu tail."""

    def hybrid_forward(self, F, main, residual):  # pylint: disable=arguments-differ
        return F.Activation(main + residual, act_type="relu")


class UnitPipelineStage(HybridBlock):
    """Compose a sequential pipeline of tensor or tuple-valued units."""

    def __init__(self, units, input_arity=1, prefix=None, params=None):
        super(UnitPipelineStage, self).__init__(prefix=prefix, params=params)
        self.input_arity = int(input_arity)
        self._unit_names = []
        with self.name_scope():
            for idx, unit in enumerate(units):
                name = "unit{}".format(idx)
                setattr(self, name, unit)
                self._unit_names.append(name)

    def hybrid_forward(self, F, data0, data1=None):  # pylint: disable=arguments-differ
        state = data0 if self.input_arity == 1 else (data0, data1)
        for unit_name in self._unit_names:
            unit = getattr(self, unit_name)
            if isinstance(state, tuple):
                state = unit(*state)
            else:
                state = unit(state)
        return state


def _unit_kind(unit_name):
    if unit_name == "stem":
        return "stem"
    if unit_name == "head":
        return "head"
    if unit_name.endswith("_main_preadd"):
        return "main_preadd"
    if unit_name.endswith("_skip_proj"):
        return "skip_proj"
    if unit_name.endswith("_add_relu_tail"):
        return "add_relu_tail"
    raise KeyError("Unknown unit kind for {}".format(unit_name))


def _block_prefix_from_unit(unit_name):
    if unit_name in ["stem", "head"]:
        return unit_name
    for suffix in ["_main_preadd", "_skip_proj", "_add_relu_tail"]:
        if unit_name.endswith(suffix):
            return unit_name[: -len(suffix)]
    raise KeyError("Cannot infer block prefix for {}".format(unit_name))


def classify_vta_window_category(unit_names, scheme_name=None):
    unit_names = list(unit_names)
    if scheme_name == "all_vta" or tuple(unit_names) == tuple(UNIT_ORDER):
        return "all_vta_extreme"

    if tuple(unit_names) in [
        tuple(LAYER1_UNITS),
        tuple(LAYER2_UNITS),
        tuple(LAYER3_UNITS),
        tuple(LAYER4_UNITS),
    ]:
        return "whole_layer"

    for block_name, block_units in BLOCK_UNIT_GROUPS.items():
        if tuple(unit_names) == tuple(block_units):
            if any(name.endswith("_skip_proj") for name in block_units):
                return "downsample_full_block"
            return "non_downsample_full_block"

    if unit_names and _unit_kind(unit_names[-1]) in ["main_preadd", "skip_proj"]:
        return "tuple_output_partial"

    if any(_unit_kind(name) == "add_relu_tail" for name in unit_names):
        return "tail_like_partial"

    return "mixed_multi_block"


def classify_downsample_window_relation(unit_names):
    category = classify_vta_window_category(unit_names)
    if category == "downsample_full_block":
        return "downsample_full_block"
    if category == "non_downsample_full_block":
        return "non_downsample_full_block"
    if category == "whole_layer":
        if any(_unit_kind(name) == "skip_proj" for name in unit_names):
            return "whole_layer_with_downsample"
        return "whole_layer_no_downsample"
    return "other"


def build_resnet18_unit_blocks(feature_blocks, output_block):
    """Return an ordered subchain-level unit map for ResNet18."""

    unit_blocks = {
        "stem": FeatureStage(feature_blocks[:4], prefix="stem_unit_"),
        "head": HeadStage([feature_blocks[8]], output_block=output_block, prefix="head_unit_"),
    }
    layer_map = {
        "layer1": feature_blocks[4],
        "layer2": feature_blocks[5],
        "layer3": feature_blocks[6],
        "layer4": feature_blocks[7],
    }
    for layer_name, layer_block in layer_map.items():
        for block_idx in range(len(layer_block)):
            block_name = "{}_block{}".format(layer_name, block_idx)
            block = layer_block[block_idx]
            unit_blocks[block_name + "_main_preadd"] = MainPreAddUnit(
                block.body,
                prefix=block_name + "_main_preadd_",
            )
            if getattr(block, "downsample", None) is not None:
                unit_blocks[block_name + "_skip_proj"] = SkipProjUnit(
                    block.downsample,
                    prefix=block_name + "_skip_proj_",
                )
            unit_blocks[block_name + "_add_relu_tail"] = AddReluTailUnit(
                prefix=block_name + "_add_relu_tail_"
            )
    return unit_blocks


def summarize_units(unit_blocks):
    print("========== Atomic Units ==========")
    unit_names = list(unit_blocks.keys())
    print("len(units) =", len(unit_names))
    for idx, name in enumerate(unit_names):
        print("unit[{}] = {}".format(idx, name))
    print("==================================")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheme",
        default="three_stage_a",
        choices=sorted(list(SCHEMES.keys()) + [AUTO_RESOURCE_AWARE_SCHEME]),
        help="Stage split scheme to instantiate",
    )
    parser.add_argument(
        "--model",
        default="resnet18_v1",
        choices=["resnet18_v1"],
        help="Only resnet18_v1 is wired for now",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size used for Relay input shape generation",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square input size used for Relay input shape generation",
    )
    parser.add_argument(
        "--save-relay-dir",
        default="",
        help="If set, lower every stage to Relay and save <stage>.relay there",
    )
    parser.add_argument(
        "--print-relay",
        action="store_true",
        help="Print Relay text for every stage after lowering",
    )
    parser.add_argument(
        "--save-partition-viz",
        default="",
        help="If set, save an SVG visualization of the resolved CPU/VTA partition",
    )
    return parser.parse_args()


def _make_schema(slot_specs):
    slots = []
    for idx, (role, shape, dtype) in enumerate(slot_specs):
        slots.append(
            {
                "slot_index": idx,
                "role": role,
                "shape": tuple(int(dim) for dim in shape),
                "dtype": dtype,
            }
        )
    return {"slots": slots, "arity": len(slots), "kind": "tensor" if len(slots) == 1 else "tuple"}


def _tensor_schema(shape, role="data", dtype="float32"):
    return _make_schema([(role, shape, dtype)])


def _tuple_schema(slot0_role, slot0_shape, slot1_role, slot1_shape, dtype="float32"):
    return _make_schema(
        [
            (slot0_role, slot0_shape, dtype),
            (slot1_role, slot1_shape, dtype),
        ]
    )


def build_resnet18_unit_io_schemas(batch, image_size):
    h4 = image_size // 4
    h8 = image_size // 8
    h16 = image_size // 16
    h32 = image_size // 32
    block_shapes = {
        "layer1_block0": {"inp": (batch, 64, h4, h4), "out": (batch, 64, h4, h4), "skip": False},
        "layer1_block1": {"inp": (batch, 64, h4, h4), "out": (batch, 64, h4, h4), "skip": False},
        "layer2_block0": {"inp": (batch, 64, h4, h4), "out": (batch, 128, h8, h8), "skip": True},
        "layer2_block1": {"inp": (batch, 128, h8, h8), "out": (batch, 128, h8, h8), "skip": False},
        "layer3_block0": {"inp": (batch, 128, h8, h8), "out": (batch, 256, h16, h16), "skip": True},
        "layer3_block1": {"inp": (batch, 256, h16, h16), "out": (batch, 256, h16, h16), "skip": False},
        "layer4_block0": {"inp": (batch, 256, h16, h16), "out": (batch, 512, h32, h32), "skip": True},
        "layer4_block1": {"inp": (batch, 512, h32, h32), "out": (batch, 512, h32, h32), "skip": False},
    }
    schemas = {
        "stem": {
            "input_schema": _tensor_schema((batch, 3, image_size, image_size)),
            "output_schema": _tensor_schema((batch, 64, h4, h4), role="out"),
        },
        "head": {
            "input_schema": _tensor_schema((batch, 512, h32, h32)),
            "output_schema": _tensor_schema((batch, 1000), role="out"),
        },
    }
    for block_name, info in block_shapes.items():
        inp = info["inp"]
        out = info["out"]
        schemas[block_name + "_main_preadd"] = {
            "input_schema": _tensor_schema(inp),
            "output_schema": _tuple_schema("main", out, "residual", inp),
        }
        if info["skip"]:
            schemas[block_name + "_skip_proj"] = {
                "input_schema": _tuple_schema("main", out, "residual", inp),
                "output_schema": _tuple_schema("main", out, "residual", out),
            }
        schemas[block_name + "_add_relu_tail"] = {
            "input_schema": _tuple_schema("main", out, "residual", out),
            "output_schema": _tensor_schema(out, role="out"),
        }
    return schemas


def _shape_numel(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel)


def _shape_nbytes(shape, dtype="float32"):
    return _shape_numel(shape) * np.dtype(dtype).itemsize


def _shape_bits_nbytes(shape, bit_width):
    return (_shape_numel(shape) * int(bit_width)) // 8


def _block_param_nbytes(block):
    total = 0
    for param in block.collect_params().values():
        try:
            shape = tuple(int(x) for x in param.shape)
        except Exception:
            continue
        total += _shape_numel(shape) * np.dtype(param.dtype).itemsize
    return int(total)


def schema_nbytes(schema):
    return sum(_shape_nbytes(slot["shape"], slot["dtype"]) for slot in schema["slots"])


def build_resnet18_unit_metadata(unit_blocks, batch, image_size):
    env = vta.get_env()
    io_schemas = build_resnet18_unit_io_schemas(batch, image_size)
    metadata = {}
    for name, block in unit_blocks.items():
        input_schema = io_schemas[name]["input_schema"]
        output_schema = io_schemas[name]["output_schema"]
        primary_input_shape = input_schema["slots"][0]["shape"]
        primary_output_shape = output_schema["slots"][0]["shape"]
        param_bytes = _block_param_nbytes(block)
        vta_inp_bytes_est = _shape_bits_nbytes(primary_input_shape, env.INP_WIDTH)
        vta_out_bytes_est = _shape_bits_nbytes(primary_output_shape, env.OUT_WIDTH)
        vta_acc_bytes_est = _shape_bits_nbytes(primary_output_shape, env.ACC_WIDTH)
        vta_wgt_bytes_est = int(round(float(param_bytes) * float(env.WGT_WIDTH) / 32.0))
        is_residual = name.startswith("layer")
        residual_live_bytes_est = 0
        if _unit_kind(name) == "main_preadd":
            residual_live_bytes_est = _shape_bits_nbytes(
                output_schema["slots"][1]["shape"], env.OUT_WIDTH
            )
        metadata[name] = {
            "name": name,
            "kind": _unit_kind(name),
            "input_schema": input_schema,
            "output_schema": output_schema,
            "input_shape": primary_input_shape,
            "output_shape": primary_output_shape,
            "input_bytes": schema_nbytes(input_schema),
            "output_bytes": schema_nbytes(output_schema),
            "param_bytes": param_bytes,
            "conv_count": UNIT_CONV_COUNTS.get(name, 0),
            "vta_inp_bytes_est": int(vta_inp_bytes_est),
            "vta_out_bytes_est": int(vta_out_bytes_est),
            "vta_acc_bytes_est": int(vta_acc_bytes_est),
            "vta_wgt_bytes_est": int(vta_wgt_bytes_est),
            "residual_live_bytes_est": int(residual_live_bytes_est),
            "shape_change": bool(tuple(primary_input_shape[1:]) != tuple(primary_output_shape[1:])),
            "partial_layer_boundary": bool(
                _unit_kind(name) in ["skip_proj", "add_relu_tail"] or name.endswith("_main_preadd")
            ),
            "is_residual": bool(is_residual),
        }
    return metadata


def _expand_feature_indices_to_unit_names(feature_indices):
    unit_names = []
    if any(idx in [0, 1, 2, 3] for idx in feature_indices):
        unit_names.append("stem")
    for idx in feature_indices:
        if idx == 4:
            unit_names.extend(LAYER1_UNITS)
        elif idx == 5:
            unit_names.extend(LAYER2_UNITS)
        elif idx == 6:
            unit_names.extend(LAYER3_UNITS)
        elif idx == 7:
            unit_names.extend(LAYER4_UNITS)
        elif idx == 8:
            unit_names.append("head")
    return unit_names


def stage_unit_names(stage_cfg):
    if "unit_names" in stage_cfg:
        return list(stage_cfg["unit_names"])
    return _expand_feature_indices_to_unit_names(stage_cfg["feature_indices"])


def _shape_to_text(shape):
    return "x".join(str(int(dim)) for dim in shape)


def _schema_to_text(schema):
    parts = []
    for slot in schema["slots"]:
        parts.append("{}:{}".format(slot["role"], _shape_to_text(slot["shape"])))
    return " | ".join(parts)


def _bytes_to_text(num_bytes):
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB"]
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return "{} {}".format(int(value), units[unit_idx])
    return "{:.1f} {}".format(value, units[unit_idx])


def build_partition_visualization_rows(scheme_cfg, unit_metadata):
    rows = []
    unit_to_stage = {}
    for stage in scheme_cfg:
        for unit_name in stage_unit_names(stage):
            unit_to_stage[unit_name] = stage
    for unit_name in UNIT_ORDER:
        stage = unit_to_stage.get(unit_name)
        meta = unit_metadata[unit_name]
        rows.append(
            {
                "unit_name": unit_name,
                "stage_name": stage["name"] if stage else "unassigned",
                "device": stage["device"] if stage else "unknown",
                "input_schema": meta["input_schema"],
                "output_schema": meta["output_schema"],
                "input_bytes": meta["input_bytes"],
                "output_bytes": meta["output_bytes"],
                "param_bytes": meta["param_bytes"],
                "conv_count": meta["conv_count"],
                "is_residual": meta["is_residual"],
                "shape_change": meta["shape_change"],
                "kind": meta["kind"],
            }
        )
    return rows


def build_partition_boundaries(scheme_cfg, unit_metadata):
    boundaries = []
    for idx in range(len(scheme_cfg) - 1):
        src_stage = scheme_cfg[idx]
        dst_stage = scheme_cfg[idx + 1]
        src_units = stage_unit_names(src_stage)
        dst_units = stage_unit_names(dst_stage)
        if not src_units or not dst_units:
            continue
        src_last = src_units[-1]
        dst_first = dst_units[0]
        boundaries.append(
            {
                "src_stage": src_stage["name"],
                "dst_stage": dst_stage["name"],
                "src_last_unit": src_last,
                "dst_first_unit": dst_first,
                "schema": unit_metadata[src_last]["output_schema"],
                "bytes": unit_metadata[src_last]["output_bytes"],
            }
        )
    return boundaries


def save_partition_visualization(
    out_path,
    display_scheme_name,
    scheme_cfg,
    unit_metadata,
    selected=None,
):
    rows = build_partition_visualization_rows(scheme_cfg, unit_metadata)
    boundaries = build_partition_boundaries(scheme_cfg, unit_metadata)
    stage_colors = {
        "cpu": {"fill": "#dbeafe", "stroke": "#1d4ed8", "text": "#0f172a"},
        "vta": {"fill": "#dcfce7", "stroke": "#15803d", "text": "#14532d"},
        "unknown": {"fill": "#e5e7eb", "stroke": "#6b7280", "text": "#111827"},
    }

    left_margin = 40
    top_margin = 92
    unit_box_w = 220
    unit_box_h = 112
    gap = 20
    stage_gap = 40
    inner_pad = 10

    stage_spans = []
    cursor_x = left_margin
    for stage in scheme_cfg:
        units = stage_unit_names(stage)
        span_w = len(units) * unit_box_w + max(0, len(units) - 1) * gap
        stage_spans.append(
            {
                "name": stage["name"],
                "device": stage["device"],
                "units": units,
                "x": cursor_x,
                "w": span_w,
            }
        )
        cursor_x += span_w + stage_gap
    svg_w = max(1200, cursor_x - stage_gap + left_margin)
    svg_h = top_margin + unit_box_h + 180

    pieces = []
    pieces.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(
            w=svg_w, h=svg_h
        )
    )
    pieces.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    pieces.append(
        '<text x="{x}" y="34" font-size="24" font-family="monospace" fill="#111827">Partition Visualization: {name}</text>'.format(
            x=left_margin, name=escape(display_scheme_name)
        )
    )
    subtitle = "Current TVM/VTA stack. Single CPU -> VTA -> CPU partition. Colors: CPU blue, VTA green."
    pieces.append(
        '<text x="{x}" y="58" font-size="14" font-family="monospace" fill="#4b5563">{text}</text>'.format(
            x=left_margin, text=escape(subtitle)
        )
    )
    if selected is not None:
        extra = "selected={} score={} vta_units={}".format(
            selected["scheme_name"], selected["score"], selected["vta_unit_names"]
        )
        pieces.append(
            '<text x="{x}" y="78" font-size="13" font-family="monospace" fill="#6b7280">{text}</text>'.format(
                x=left_margin, text=escape(extra)
            )
        )

    current_x = left_margin
    unit_x = {}
    for span in stage_spans:
        color = stage_colors[span["device"]]
        pieces.append(
            '<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" fill-opacity="0.28" stroke="{stroke}" stroke-width="2"/>'.format(
                x=span["x"] - 12,
                y=top_margin - 28,
                w=span["w"] + 24,
                h=unit_box_h + 56,
                fill=color["fill"],
                stroke=color["stroke"],
            )
        )
        pieces.append(
            '<text x="{x}" y="{y}" font-size="18" font-family="monospace" font-weight="700" fill="{text}">{label}</text>'.format(
                x=span["x"],
                y=top_margin - 6,
                text=color["text"],
                label=escape("{} [{}]".format(span["name"], span["device"].upper())),
            )
        )
        x = span["x"]
        for unit_name in span["units"]:
            row = next(item for item in rows if item["unit_name"] == unit_name)
            unit_x[unit_name] = x
            pieces.append(
                '<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'.format(
                    x=x,
                    y=top_margin,
                    w=unit_box_w,
                    h=unit_box_h,
                    fill=color["fill"],
                    stroke=color["stroke"],
                )
            )
            pieces.append(
                '<text x="{x}" y="{y}" font-size="15" font-family="monospace" font-weight="700" fill="{text}">{label}</text>'.format(
                    x=x + inner_pad,
                    y=top_margin + 22,
                    text=color["text"],
                    label=escape(row["unit_name"]),
                )
            )
            lines = [
                "conv={} residual={}".format(row["conv_count"], "yes" if row["is_residual"] else "no"),
                "kind={}".format(row["kind"]),
                "in : {}".format(_schema_to_text(row["input_schema"])),
                "out: {}".format(_schema_to_text(row["output_schema"])),
                "io : {} / {}".format(_bytes_to_text(row["input_bytes"]), _bytes_to_text(row["output_bytes"])),
            ]
            if row["shape_change"]:
                lines.append("shape_change=yes")
            elif row["param_bytes"] > 0:
                lines.append("param={}".format(_bytes_to_text(row["param_bytes"])))
            for line_idx, line in enumerate(lines, start=1):
                pieces.append(
                    '<text x="{x}" y="{y}" font-size="12" font-family="monospace" fill="#334155">{text}</text>'.format(
                        x=x + inner_pad,
                        y=top_margin + 22 + line_idx * 18,
                        text=escape(line),
                    )
                )
            x += unit_box_w + gap

    arrow_y = top_margin + unit_box_h + 34
    for boundary in boundaries:
        src_x = unit_x[boundary["src_last_unit"]] + unit_box_w
        dst_x = unit_x[boundary["dst_first_unit"]]
        mid_x = (src_x + dst_x) / 2.0
        pieces.append(
            '<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#111827" stroke-width="2" marker-end="url(#arrowhead)"/>'.format(
                x1=src_x + 4,
                x2=dst_x - 10,
                y=arrow_y,
            )
        )
        pieces.append(
            '<text x="{x}" y="{y}" text-anchor="middle" font-size="12" font-family="monospace" fill="#111827">{text}</text>'.format(
                x=mid_x,
                y=arrow_y - 10,
                text=escape(
                    "{} -> {} | {} | {}".format(
                        boundary["src_stage"],
                        boundary["dst_stage"],
                        _schema_to_text(boundary["schema"]),
                        _bytes_to_text(boundary["bytes"]),
                    )
                ),
            )
        )

    legend_y = top_margin + unit_box_h + 82
    pieces.append(
        '<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#111827"/></marker></defs>'
    )
    pieces.append(
        '<text x="{x}" y="{y}" font-size="14" font-family="monospace" fill="#111827">Cut summary: stage boundaries are the black arrows; each box is one atomic subchain unit; tuple boundaries are shown as multi-slot shape text.</text>'.format(
            x=left_margin, y=legend_y
        )
    )
    pieces.append("</svg>")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(pieces))


def _count_shape_change_units(unit_names, unit_metadata):
    count = 0
    for name in unit_names:
        meta = unit_metadata[name]
        in_shape = meta["input_shape"]
        out_shape = meta["output_shape"]
        if tuple(in_shape[1:]) != tuple(out_shape[1:]):
            count += 1
    return count


def _count_partial_layer_boundaries(unit_names):
    if not unit_names:
        return 0
    count = 0
    if _unit_kind(unit_names[0]) != "main_preadd":
        count += 1
    if _unit_kind(unit_names[-1]) != "add_relu_tail":
        count += 1
    return count


def build_vta_capacity_spec():
    env = vta.get_env()
    return {
        "inp_cap_bytes": int(env.INP_BUFF_SIZE * env.INP_ELEM_BYTES),
        "wgt_cap_bytes": int(env.WGT_BUFF_SIZE * env.WGT_ELEM_BYTES),
        "acc_cap_bytes": int(env.ACC_BUFF_SIZE * env.ACC_ELEM_BYTES),
        "out_cap_bytes": int(env.OUT_BUFF_SIZE * env.OUT_ELEM_BYTES),
    }


def _ceil_div(lhs, rhs):
    lhs = int(lhs)
    rhs = max(1, int(rhs))
    return (lhs + rhs - 1) // rhs


def _estimate_unit_tile_cost(meta, capacity_spec):
    inp_cap_bytes = capacity_spec["inp_cap_bytes"]
    wgt_cap_bytes = capacity_spec["wgt_cap_bytes"]
    acc_cap_bytes = capacity_spec["acc_cap_bytes"]
    out_cap_bytes = capacity_spec["out_cap_bytes"]
    inp_tiles = _ceil_div(meta["vta_inp_bytes_est"], inp_cap_bytes)
    wgt_tiles = _ceil_div(meta["vta_wgt_bytes_est"], wgt_cap_bytes)
    acc_tiles = _ceil_div(meta["vta_acc_bytes_est"], acc_cap_bytes)
    out_tiles = _ceil_div(meta["vta_out_bytes_est"], out_cap_bytes)
    tile_count_est = max(inp_tiles, wgt_tiles, acc_tiles, out_tiles)
    conv_scale = max(1, int(meta["conv_count"]))
    tile_input_load_bytes_est = meta["vta_inp_bytes_est"] * conv_scale * max(
        inp_tiles, acc_tiles, out_tiles
    )
    # Weight movement matters, but first-order cost is dominated by feature tiling.
    tile_weight_load_bytes_est = (meta["vta_wgt_bytes_est"] * max(1, wgt_tiles)) // 4
    tile_output_store_bytes_est = meta["vta_out_bytes_est"] * max(
        1, 1 + int(bool(meta["shape_change"]))
    )
    tile_acc_live_bytes_peak_est = min(meta["vta_acc_bytes_est"], acc_cap_bytes)
    tile_spill_penalty = 0
    tile_spill_penalty += conv_scale * max(0, inp_tiles - 1) * meta["vta_inp_bytes_est"]
    tile_spill_penalty += max(0, wgt_tiles - 1) * (meta["vta_wgt_bytes_est"] // 4)
    tile_spill_penalty += 2 * conv_scale * max(0, acc_tiles - 1) * meta["vta_acc_bytes_est"]
    tile_spill_penalty += max(0, out_tiles - 1) * meta["vta_out_bytes_est"]
    return {
        "inp_tiles": int(inp_tiles),
        "wgt_tiles": int(wgt_tiles),
        "acc_tiles": int(acc_tiles),
        "out_tiles": int(out_tiles),
        "tile_count_est": int(tile_count_est * conv_scale),
        "tile_input_load_bytes_est": int(tile_input_load_bytes_est),
        "tile_weight_load_bytes_est": int(tile_weight_load_bytes_est),
        "tile_output_store_bytes_est": int(tile_output_store_bytes_est),
        "tile_acc_live_bytes_peak_est": int(tile_acc_live_bytes_peak_est),
        "tile_spill_penalty": int(tile_spill_penalty),
    }


def _estimate_cpu_stage_cost(unit_names, unit_metadata, tail_weight=False):
    cpu_cost_est = 0
    tail_penalty = 0
    for name in unit_names:
        meta = unit_metadata[name]
        base = _estimate_cpu_unit_cost(name, meta)
        if tail_weight:
            depth_idx = UNIT_ORDER.index(name)
            scale = 1.0 + (0.15 * depth_idx)
            cpu_cost_est += int(base * scale)
            tail_penalty += int(
                ((_estimate_cpu_unit_cost(name, meta, apply_early_weight=False) // 3) + (meta["output_bytes"] // 16))
                * scale
            )
        else:
            cpu_cost_est += int(base)
    return int(cpu_cost_est), int(tail_penalty)


def _estimate_cpu_unit_cost(unit_name, meta, apply_early_weight=True):
    base = (
        meta["conv_count"] * 40000
        + (meta["input_bytes"] // 8)
        + (meta["output_bytes"] // 8)
        + (meta["param_bytes"] // 64)
    )
    if not apply_early_weight:
        return int(base)
    if unit_name == "stem" or unit_name.startswith("layer1_"):
        scale = 2.5
    elif unit_name.startswith("layer2_"):
        scale = 1.75
    elif unit_name.startswith("layer3_"):
        scale = 1.25
    elif unit_name.startswith("layer4_"):
        scale = 1.0
    else:
        scale = 0.75
    return int(base * scale)


def _estimate_cpu_bucket_costs(stage0_unit_names, stage2_unit_names, unit_metadata, num_buckets=4):
    bucket_costs = [0 for _ in range(num_buckets)]
    bucket_units = [[] for _ in range(num_buckets)]
    ordered_units = list(stage0_unit_names) + list(stage2_unit_names)
    for name in ordered_units:
        meta = unit_metadata[name]
        cost = _estimate_cpu_unit_cost(name, meta)
        bucket_idx = min(range(num_buckets), key=lambda idx: bucket_costs[idx])
        bucket_costs[bucket_idx] += int(cost)
        bucket_units[bucket_idx].append(name)
    return [int(cost) for cost in bucket_costs], bucket_units


def _scaled_stage_cpu_cost(raw_cost, num_workers):
    workers = max(1, int(num_workers))
    return int(math.ceil(float(raw_cost) / float(workers)))


def _describe_candidate_reasons(candidate):
    reasons = []
    bottleneck_cost = candidate["pipeline_bottleneck_cost_est"]
    stage0_cost = candidate["cpu_stage0_cost_est"]
    stage1_cost = candidate["vta_stage_cost_est"]
    stage2_cost = candidate["cpu_stage2_cost_est"]
    if bottleneck_cost == stage0_cost and stage0_cost > 0:
        reasons.append("stage0_cpu_heavy")
    elif bottleneck_cost == stage1_cost and stage1_cost > 0:
        reasons.append("stage1_vta_heavy")
    elif bottleneck_cost == stage2_cost and stage2_cost > 0:
        reasons.append("stage2_cpu_heavy")
    if candidate["cpu_prefix_penalty"] > 0:
        reasons.append("late_vta_start")
    if candidate["stage0_dominance_penalty"] > 0:
        reasons.append("stage0_dominates")
    if candidate["boundary_penalty"] > 500000:
        reasons.append("boundary_heavy")
    if candidate["fusion_disruption_penalty"] > 0 or candidate["partial_layer_boundaries"] > 0:
        reasons.append("fusion_disruption")
    if candidate["stage_imbalance_penalty"] > 0:
        reasons.append("pipeline_imbalance")
    if not reasons:
        reasons.append("balanced")
    return reasons


def _score_resource_aware_candidate(vta_unit_names, unit_metadata, capacity_spec):
    conv_count = sum(unit_metadata[name]["conv_count"] for name in vta_unit_names)
    shape_change_units = _count_shape_change_units(vta_unit_names, unit_metadata)
    partial_layer_boundaries = _count_partial_layer_boundaries(vta_unit_names)
    span = len(vta_unit_names)
    span_penalty = (max(0, span - 2) ** 2) * 300000
    estimated_load_bytes = sum(unit_metadata[name]["input_bytes"] for name in vta_unit_names)
    estimated_store_bytes = sum(unit_metadata[name]["output_bytes"] for name in vta_unit_names)
    start_idx = UNIT_ORDER.index(vta_unit_names[0])
    end_idx = UNIT_ORDER.index(vta_unit_names[-1])
    stage0_unit_names = UNIT_ORDER[:start_idx]
    stage2_unit_names = UNIT_ORDER[end_idx + 1 :]
    boundary_count = 0
    total_boundary_bytes = 0
    if stage0_unit_names:
        boundary_count += 1
        total_boundary_bytes += unit_metadata[vta_unit_names[0]]["input_bytes"]
    if stage2_unit_names:
        boundary_count += 1
        total_boundary_bytes += unit_metadata[vta_unit_names[-1]]["output_bytes"]
    boundary_penalty = int(total_boundary_bytes + (boundary_count * 100000))
    cpu_bucket_costs_est, cpu_bucket_unit_names = _estimate_cpu_bucket_costs(
        stage0_unit_names,
        stage2_unit_names,
        unit_metadata,
        num_buckets=RESOURCE_AWARE_CPU_WORKERS,
    )
    cpu_work_bucket_max_est = max(cpu_bucket_costs_est) if cpu_bucket_costs_est else 0
    edge_penalty = 0
    if vta_unit_names[0] == "stem":
        edge_penalty += 3000000
    layer2_start_idx = UNIT_ORDER.index("layer2_block0_main_preadd")
    if start_idx < layer2_start_idx:
        edge_penalty += (layer2_start_idx - start_idx) * 2500000
    if vta_unit_names[-1].startswith("layer4_") or vta_unit_names[-1] == "head":
        edge_penalty += 2000000
    if any(name.startswith("layer4_") for name in vta_unit_names):
        edge_penalty += 5000000
    if all(name.startswith("layer4_") or name == "head" for name in vta_unit_names):
        edge_penalty += 4000000
    if not stage0_unit_names:
        edge_penalty += 1000000
    if not stage2_unit_names:
        edge_penalty += 1000000
    peak_inp_bytes_est = 0
    peak_wgt_bytes_est = 0
    peak_acc_bytes_est = 0
    peak_out_bytes_est = 0
    total_internal_boundary_bytes = 0
    total_residual_live_bytes = 0
    max_inp_ratio = 0.0
    max_wgt_ratio = 0.0
    max_acc_ratio = 0.0
    max_out_ratio = 0.0
    tile_input_load_bytes_est = 0
    tile_weight_load_bytes_est = 0
    tile_output_store_bytes_est = 0
    tile_acc_live_bytes_peak_est = 0
    tile_count_est = 0
    tile_spill_penalty = 0
    fusion_disruption_penalty = 0
    inp_cap_bytes = capacity_spec["inp_cap_bytes"]
    wgt_cap_bytes = capacity_spec["wgt_cap_bytes"]
    acc_cap_bytes = capacity_spec["acc_cap_bytes"]
    out_cap_bytes = capacity_spec["out_cap_bytes"]
    hard_reject = False
    reject_reasons = set()
    if span == 1 and any(
        _unit_kind(name) in ["skip_proj", "add_relu_tail"] for name in vta_unit_names
    ):
        hard_reject = True
        reject_reasons.add("partial_layer_single_block")
    block_tail_penalty = 0
    if any(_unit_kind(name) == "add_relu_tail" for name in vta_unit_names):
        block_tail_penalty += 500000 * sum(
            1 for name in vta_unit_names if _unit_kind(name) == "add_relu_tail"
        )
    if any(_unit_kind(name) == "skip_proj" for name in vta_unit_names):
        fusion_disruption_penalty += 350000 * sum(
            1 for name in vta_unit_names if _unit_kind(name) == "skip_proj"
        )
    for idx, name in enumerate(vta_unit_names):
        meta = unit_metadata[name]
        tile_info = _estimate_unit_tile_cost(meta, capacity_spec)
        inp_est = meta["vta_inp_bytes_est"]
        wgt_est = meta["vta_wgt_bytes_est"]
        acc_est = meta["vta_acc_bytes_est"]
        out_est = meta["vta_out_bytes_est"]
        residual_live_est = meta["residual_live_bytes_est"]
        peak_inp_bytes_est = max(peak_inp_bytes_est, inp_est)
        peak_wgt_bytes_est = max(peak_wgt_bytes_est, wgt_est)
        peak_acc_bytes_est = max(peak_acc_bytes_est, acc_est)
        peak_out_bytes_est = max(peak_out_bytes_est, out_est)
        inp_ratio = float(inp_est) / float(inp_cap_bytes)
        wgt_ratio = float(wgt_est) / float(wgt_cap_bytes)
        acc_ratio = float(acc_est) / float(acc_cap_bytes)
        out_ratio = float(out_est) / float(out_cap_bytes)
        max_inp_ratio = max(max_inp_ratio, inp_ratio)
        max_wgt_ratio = max(max_wgt_ratio, wgt_ratio)
        max_acc_ratio = max(max_acc_ratio, acc_ratio)
        max_out_ratio = max(max_out_ratio, out_ratio)
        tile_input_load_bytes_est += tile_info["tile_input_load_bytes_est"]
        tile_weight_load_bytes_est += tile_info["tile_weight_load_bytes_est"]
        tile_output_store_bytes_est += tile_info["tile_output_store_bytes_est"]
        tile_acc_live_bytes_peak_est = max(
            tile_acc_live_bytes_peak_est, tile_info["tile_acc_live_bytes_peak_est"]
        )
        tile_count_est += tile_info["tile_count_est"]
        tile_spill_penalty += tile_info["tile_spill_penalty"]
        if tile_info["inp_tiles"] > 8:
            hard_reject = True
            reject_reasons.add("inp_tile_excess")
        if tile_info["wgt_tiles"] > 8:
            hard_reject = True
            reject_reasons.add("wgt_tile_excess")
        if tile_info["acc_tiles"] > 8:
            hard_reject = True
            reject_reasons.add("acc_tile_excess")
        if tile_info["out_tiles"] > 8:
            hard_reject = True
            reject_reasons.add("out_tile_excess")
        if idx < len(vta_unit_names) - 1:
            total_internal_boundary_bytes += out_est
        if residual_live_est > 0:
            total_residual_live_bytes += residual_live_est
    fusion_disruption_penalty += max(0, total_internal_boundary_bytes - out_cap_bytes)
    fusion_disruption_penalty += max(0, total_residual_live_bytes - acc_cap_bytes)
    cpu_stage0_cost_est, _ = _estimate_cpu_stage_cost(stage0_unit_names, unit_metadata, tail_weight=False)
    cpu_stage2_cost_est, cpu_tail_penalty = _estimate_cpu_stage_cost(
        stage2_unit_names, unit_metadata, tail_weight=True
    )
    cpu_stage0_parallel_cost_est = _scaled_stage_cpu_cost(
        cpu_stage0_cost_est, RESOURCE_AWARE_SHARED_CPU_WORKERS
    )
    cpu_stage2_parallel_cost_est = _scaled_stage_cpu_cost(
        cpu_stage2_cost_est, RESOURCE_AWARE_SHARED_CPU_WORKERS
    )
    cpu_shared_parallel_cost_est = int(cpu_stage0_parallel_cost_est + cpu_stage2_parallel_cost_est)
    vta_stage_cost_est = int(
        tile_input_load_bytes_est * RESOURCE_AWARE_SCORE_WEIGHTS["tile_input_load"]
        + tile_weight_load_bytes_est * RESOURCE_AWARE_SCORE_WEIGHTS["tile_weight_load"]
        + tile_output_store_bytes_est * RESOURCE_AWARE_SCORE_WEIGHTS["tile_output_store"]
        + tile_spill_penalty * RESOURCE_AWARE_SCORE_WEIGHTS["tile_spill"]
        + tile_count_est * RESOURCE_AWARE_SCORE_WEIGHTS["tile_count"]
    )
    layer2_start_idx = UNIT_ORDER.index("layer2_block0_main_preadd")
    active_stage_costs = [
        int(cost)
        for cost in [cpu_shared_parallel_cost_est, vta_stage_cost_est]
        if int(cost) > 0
    ]
    pipeline_bottleneck_cost_est = max(active_stage_costs) if active_stage_costs else 0
    pipeline_min_cost_est = min(active_stage_costs) if active_stage_costs else 0
    cpu_prefix_penalty = int(
        (cpu_stage0_parallel_cost_est * RESOURCE_AWARE_SCORE_WEIGHTS["cpu_prefix"])
        + max(0, start_idx - layer2_start_idx) * 250000
    )
    stage0_dominance_penalty = int(
        max(0, cpu_shared_parallel_cost_est - vta_stage_cost_est)
        * RESOURCE_AWARE_SCORE_WEIGHTS["stage0_dominance"]
    )
    stage_imbalance_penalty = int(
        max(0, pipeline_bottleneck_cost_est - pipeline_min_cost_est)
        * RESOURCE_AWARE_SCORE_WEIGHTS["stage_imbalance"]
    )
    score = (
        pipeline_bottleneck_cost_est
        + cpu_prefix_penalty
        + stage0_dominance_penalty
        + boundary_penalty * RESOURCE_AWARE_SCORE_WEIGHTS["boundary"]
        + stage_imbalance_penalty
        + edge_penalty * RESOURCE_AWARE_SCORE_WEIGHTS["edge"]
        + fusion_disruption_penalty * RESOURCE_AWARE_SCORE_WEIGHTS["fusion"]
        + span_penalty * RESOURCE_AWARE_SCORE_WEIGHTS["span"]
        + block_tail_penalty
    )
    return {
        "estimated_load_bytes": int(estimated_load_bytes),
        "estimated_store_bytes": int(estimated_store_bytes),
        "span_penalty": int(span_penalty),
        "block_tail_penalty": int(block_tail_penalty),
        "boundary_penalty": int(boundary_penalty),
        "edge_penalty": int(edge_penalty),
        "boundary_count": int(boundary_count),
        "total_boundary_bytes": int(total_boundary_bytes),
        "shape_change_units": int(shape_change_units),
        "partial_layer_boundaries": int(partial_layer_boundaries),
        "span": int(span),
        "conv_count": int(conv_count),
        "stage0_unit_names": list(stage0_unit_names),
        "stage2_unit_names": list(stage2_unit_names),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "cpu_stage0_cost_est": int(cpu_stage0_cost_est),
        "cpu_stage2_cost_est": int(cpu_stage2_cost_est),
        "cpu_stage0_parallel_cost_est": int(cpu_stage0_parallel_cost_est),
        "cpu_stage2_parallel_cost_est": int(cpu_stage2_parallel_cost_est),
        "cpu_shared_parallel_cost_est": int(cpu_shared_parallel_cost_est),
        "cpu_tail_penalty": int(cpu_tail_penalty),
        "cpu_bucket_costs_est": [int(item) for item in cpu_bucket_costs_est],
        "cpu_bucket_unit_names": [list(item) for item in cpu_bucket_unit_names],
        "cpu_work_bucket_max_est": int(cpu_work_bucket_max_est),
        "cpu_prefix_penalty": int(cpu_prefix_penalty),
        "stage0_dominance_penalty": int(stage0_dominance_penalty),
        "peak_inp_bytes_est": int(peak_inp_bytes_est),
        "peak_wgt_bytes_est": int(peak_wgt_bytes_est),
        "peak_acc_bytes_est": int(peak_acc_bytes_est),
        "peak_out_bytes_est": int(peak_out_bytes_est),
        "total_internal_boundary_bytes": int(total_internal_boundary_bytes),
        "total_residual_live_bytes": int(total_residual_live_bytes),
        "max_inp_ratio": float(max_inp_ratio),
        "max_wgt_ratio": float(max_wgt_ratio),
        "max_acc_ratio": float(max_acc_ratio),
        "max_out_ratio": float(max_out_ratio),
        "tile_input_load_bytes_est": int(tile_input_load_bytes_est),
        "tile_weight_load_bytes_est": int(tile_weight_load_bytes_est),
        "tile_output_store_bytes_est": int(tile_output_store_bytes_est),
        "tile_acc_live_bytes_peak_est": int(tile_acc_live_bytes_peak_est),
        "tile_count_est": int(tile_count_est),
        "tile_spill_penalty": int(tile_spill_penalty),
        "vta_stage_cost_est": int(vta_stage_cost_est),
        "pipeline_bottleneck_cost_est": int(pipeline_bottleneck_cost_est),
        "stage_imbalance_penalty": int(stage_imbalance_penalty),
        "fusion_disruption_penalty": int(fusion_disruption_penalty),
        "hard_reject": bool(hard_reject),
        "reject_reasons": sorted(list(reject_reasons)),
        "capacity_spec": dict(capacity_spec),
        "score": int(score),
    }


def _build_scheme_from_vta_window(vta_unit_names):
    start_idx = UNIT_ORDER.index(vta_unit_names[0])
    end_idx = UNIT_ORDER.index(vta_unit_names[-1])
    left_units = UNIT_ORDER[:start_idx]
    right_units = UNIT_ORDER[end_idx + 1 :]
    scheme_cfg = []
    if left_units:
        scheme_cfg.append(
            {
                "name": "stage0_cpu",
                "device": "cpu",
                "kind": "units",
                "unit_names": list(left_units),
            }
        )
    scheme_cfg.append(
        {
            "name": "stage1_vta" if left_units else "stage0_vta",
            "device": "vta",
            "kind": "units",
            "unit_names": list(vta_unit_names),
        }
    )
    if right_units:
        scheme_cfg.append(
            {
                "name": "stage2_cpu" if left_units else "stage1_cpu",
                "device": "cpu",
                "kind": "units",
                "unit_names": list(right_units),
            }
        )
    return scheme_cfg


def _matching_known_scheme_name(vta_unit_names):
    known = {
        tuple(LAYER2_UNITS + LAYER3_UNITS): "three_stage_a",
        tuple(LAYER3_UNITS): "three_stage_b",
        tuple(LAYER1_UNITS + LAYER2_UNITS + LAYER3_UNITS): "three_stage_c",
        tuple(BLOCK_UNIT_GROUPS["layer1_block1"] + LAYER2_UNITS + LAYER3_UNITS): "three_stage_d",
        tuple(
            BLOCK_UNIT_GROUPS["layer1_block1"]
            + LAYER2_UNITS
            + LAYER3_UNITS
            + BLOCK_UNIT_GROUPS["layer4_block0"]
        ): "three_stage_e",
        tuple(LAYER2_UNITS + BLOCK_UNIT_GROUPS["layer3_block0"]): "block_stage_a",
        tuple(BLOCK_UNIT_GROUPS["layer2_block1"] + BLOCK_UNIT_GROUPS["layer3_block0"]): "block_stage_b",
        tuple(LAYER2_UNITS): "block_stage_c",
        tuple(LAYER1_UNITS + LAYER2_UNITS + BLOCK_UNIT_GROUPS["layer3_block0"]): "block_stage_d",
        tuple(UNIT_ORDER): "all_vta",
    }
    return known.get(tuple(vta_unit_names), None)


def _enumerate_vta_windows():
    windows = []
    total_units = len(UNIT_ORDER)
    for start_idx in range(total_units):
        for end_idx in range(start_idx, total_units):
            unit_names = UNIT_ORDER[start_idx : end_idx + 1]
            if not any(_unit_kind(name) == "main_preadd" for name in unit_names):
                continue
            if unit_names == ["head"]:
                continue
            if _unit_kind(unit_names[0]) not in ["main_preadd", "stem"]:
                continue
            if _unit_kind(unit_names[-1]) not in ["main_preadd", "skip_proj", "add_relu_tail", "head"]:
                continue
            if unit_names == ["stem"]:
                continue
            windows.append(unit_names)
    return windows


def build_resource_aware_candidates(feature_blocks, output_block, batch, image_size):
    unit_blocks = build_resnet18_unit_blocks(feature_blocks, output_block)
    unit_metadata = build_resnet18_unit_metadata(unit_blocks, batch, image_size)
    capacity_spec = build_vta_capacity_spec()
    candidates = []
    for unit_names in _enumerate_vta_windows():
        score_info = _score_resource_aware_candidate(unit_names, unit_metadata, capacity_spec)
        alias_name = _matching_known_scheme_name(unit_names)
        scheme_name = alias_name or "window_{}__{}".format(unit_names[0], unit_names[-1])
        scheme_cfg = SCHEMES[alias_name] if alias_name in SCHEMES else _build_scheme_from_vta_window(unit_names)
        candidates.append(
            {
                "scheme_name": scheme_name,
                "alias_scheme_name": alias_name,
                "scheme_cfg": scheme_cfg,
                "vta_unit_names": unit_names,
                **score_info,
            }
        )
    candidates.sort(
        key=lambda item: (
            1 if item["hard_reject"] else 0,
            item["score"],
            item["start_idx"],
            item["span"],
            item["scheme_name"],
        )
    )
    return candidates


def build_resource_aware_calibration_set(candidates, max_items=14):
    selected = []
    seen = set()

    def add(candidate):
        if candidate["scheme_name"] in seen:
            return
        seen.add(candidate["scheme_name"])
        selected.append(candidate)

    for mandatory in RESOURCE_AWARE_CALIBRATION_MANDATORY:
        for candidate in candidates:
            if candidate["scheme_name"] == mandatory:
                add(candidate)
                break

    for candidate in candidates[: min(8, len(candidates))]:
        add(candidate)

    single_block = next((c for c in candidates if c["span"] == 1), None)
    if single_block is not None:
        add(single_block)

    mid_window = next(
        (
            c
            for c in candidates
            if c["vta_unit_names"][0].startswith("layer2")
            and c["vta_unit_names"][-1].startswith("layer3")
        ),
        None,
    )
    if mid_window is not None:
        add(mid_window)

    tail_window = next(
        (
            c
            for c in candidates
            if c["vta_unit_names"][-1].startswith("layer4_") or c["vta_unit_names"][-1] == "head"
        ),
        None,
    )
    if tail_window is not None:
        add(tail_window)

    return selected[:max_items]


def print_resource_aware_candidates(candidates, selected_scheme_name=None, max_rows=10):
    print("========== Resource-Aware Candidates ==========")
    if candidates:
        spec = candidates[0]["capacity_spec"]
        print(
            "capacity inp_cap_bytes={} wgt_cap_bytes={} acc_cap_bytes={} out_cap_bytes={}".format(
                spec["inp_cap_bytes"],
                spec["wgt_cap_bytes"],
                spec["acc_cap_bytes"],
                spec["out_cap_bytes"],
            )
        )
    max_rows = min(max_rows, len(candidates))
    for idx, candidate in enumerate(candidates[:max_rows], start=1):
        selected_label = "yes" if candidate["scheme_name"] == selected_scheme_name else "no"
        reason_tags = ",".join(_describe_candidate_reasons(candidate))
        print(
            "#{} {} alias={} score={} selected={} hard_reject={} reject_reasons={} reasons={} start_idx={} end_idx={} span={} boundary_bytes={} boundary_count={} shape_change_units={} partial_layer_boundaries={} tile_input_load={} tile_weight_load={} tile_output_store={} tile_acc_peak={} tile_count={} tile_spill_penalty={} vta_stage_cost={} cpu_stage0_cost={} cpu_stage2_cost={} cpu_stage0_parallel_cost={} cpu_stage2_parallel_cost={} cpu_tail_penalty={} cpu_bucket_costs={} cpu_bucket_max={} pipeline_bottleneck_cost={} cpu_prefix_penalty={} stage0_dominance_penalty={} stage_imbalance_penalty={} boundary_penalty={} edge_penalty={} fusion_disruption_penalty={} block_tail_penalty={} est_load={} est_store={} peak_inp={} peak_wgt={} peak_acc={} peak_out={} internal_boundary_total={} residual_live_total={} max_ratio(in/wgt/acc/out)={:.3f}/{:.3f}/{:.3f}/{:.3f} span_penalty={} vta_units={}".format(
                idx,
                candidate["scheme_name"],
                candidate["alias_scheme_name"] or "-",
                candidate["score"],
                selected_label,
                "yes" if candidate["hard_reject"] else "no",
                ",".join(candidate["reject_reasons"]) if candidate["reject_reasons"] else "-",
                reason_tags,
                candidate["start_idx"],
                candidate["end_idx"],
                candidate["span"],
                candidate["total_boundary_bytes"],
                candidate["boundary_count"],
                candidate["shape_change_units"],
                candidate["partial_layer_boundaries"],
                candidate["tile_input_load_bytes_est"],
                candidate["tile_weight_load_bytes_est"],
                candidate["tile_output_store_bytes_est"],
                candidate["tile_acc_live_bytes_peak_est"],
                candidate["tile_count_est"],
                candidate["tile_spill_penalty"],
                candidate["vta_stage_cost_est"],
                candidate["cpu_stage0_cost_est"],
                candidate["cpu_stage2_cost_est"],
                candidate["cpu_stage0_parallel_cost_est"],
                candidate["cpu_stage2_parallel_cost_est"],
                candidate["cpu_tail_penalty"],
                candidate["cpu_bucket_costs_est"],
                candidate["cpu_work_bucket_max_est"],
                candidate["pipeline_bottleneck_cost_est"],
                candidate["cpu_prefix_penalty"],
                candidate["stage0_dominance_penalty"],
                candidate["stage_imbalance_penalty"],
                candidate["boundary_penalty"],
                candidate["edge_penalty"],
                candidate["fusion_disruption_penalty"],
                candidate["block_tail_penalty"],
                candidate["estimated_load_bytes"],
                candidate["estimated_store_bytes"],
                candidate["peak_inp_bytes_est"],
                candidate["peak_wgt_bytes_est"],
                candidate["peak_acc_bytes_est"],
                candidate["peak_out_bytes_est"],
                candidate["total_internal_boundary_bytes"],
                candidate["total_residual_live_bytes"],
                candidate["max_inp_ratio"],
                candidate["max_wgt_ratio"],
                candidate["max_acc_ratio"],
                candidate["max_out_ratio"],
                candidate["span_penalty"],
                candidate["vta_unit_names"],
            )
        )
    if len(candidates) > max_rows:
        print("... {} more candidates omitted".format(len(candidates) - max_rows))
    print("================================================")


def print_resource_aware_calibration_set(candidates, max_items=14):
    calibration_set = build_resource_aware_calibration_set(candidates, max_items=max_items)
    print("========== Resource-Aware Calibration Set ==========")
    print("count =", len(calibration_set))
    for idx, candidate in enumerate(calibration_set, start=1):
        print(
            "#{:02d} {} alias={} span={} score={} vta_units={}".format(
                idx,
                candidate["scheme_name"],
                candidate["alias_scheme_name"] or "-",
                candidate["span"],
                candidate["score"],
                candidate["vta_unit_names"],
            )
        )
    print("====================================================")


def resolve_scheme_config(
    scheme_name,
    feature_blocks,
    output_block,
    batch,
    image_size,
    print_resource_aware=False,
    selected_candidate_name=None,
    calibration_report=False,
    candidate_top_k=10,
):
    if scheme_name != AUTO_RESOURCE_AWARE_SCHEME:
        return SCHEMES[scheme_name], None, None

    candidates = build_resource_aware_candidates(feature_blocks, output_block, batch, image_size)
    selectable_candidates = [item for item in candidates if not item["hard_reject"]]
    if not selectable_candidates:
        raise RuntimeError("auto_resource_aware could not find a non-rejected VTA candidate")
    if selected_candidate_name:
        matched = next(
            (item for item in selectable_candidates if item["scheme_name"] == selected_candidate_name),
            None,
        )
        if matched is None:
            raise RuntimeError(
                "auto_resource_aware candidate {} not found or rejected".format(
                    selected_candidate_name
                )
            )
        selected = matched
    else:
        selected = selectable_candidates[0]
    if print_resource_aware:
        print_resource_aware_candidates(
            candidates,
            selected_scheme_name=selected["scheme_name"],
            max_rows=candidate_top_k,
        )
    if calibration_report:
        print_resource_aware_calibration_set(candidates)
    print(
        "[AUTO-RESOURCE] selected_scheme={} score={} hard_reject={} reject_reasons={} vta_units={}".format(
            selected["scheme_name"],
            selected["score"],
            "yes" if selected["hard_reject"] else "no",
            ",".join(selected["reject_reasons"]) if selected["reject_reasons"] else "-",
            selected["vta_unit_names"],
        )
    )
    return selected["scheme_cfg"], selected, candidates


def relay_shape_for_stage(stage_name, scheme_name, batch, image_size, scheme_cfg=None):
    del scheme_name
    if scheme_cfg is None:
        raise ValueError("scheme_cfg is required")
    stage_cfg = next(stage for stage in scheme_cfg if stage["name"] == stage_name)
    io_schemas = build_resnet18_unit_io_schemas(batch, image_size)
    first_unit = stage_unit_names(stage_cfg)[0]
    schema = io_schemas[first_unit]["input_schema"]
    return schema["slots"][0]["shape"]


def stage_input_schema_for_stage(stage_cfg, unit_metadata):
    unit_names = stage_unit_names(stage_cfg)
    if not unit_names:
        raise ValueError("Empty stage {}".format(stage_cfg["name"]))
    return unit_metadata[unit_names[0]]["input_schema"]


def stage_output_schema_for_stage(stage_cfg, unit_metadata):
    unit_names = stage_unit_names(stage_cfg)
    if not unit_names:
        raise ValueError("Empty stage {}".format(stage_cfg["name"]))
    return unit_metadata[unit_names[-1]]["output_schema"]


def relay_inputs_for_stage(stage_cfg, unit_metadata):
    schema = stage_input_schema_for_stage(stage_cfg, unit_metadata)
    return [("data{}".format(idx), slot["shape"]) for idx, slot in enumerate(schema["slots"])]


def make_stage_block(feature_blocks, output_block, stage_cfg, unit_blocks=None):
    if "unit_names" in stage_cfg:
        if unit_blocks is None:
            raise ValueError("unit_blocks is required for unit-based stage schemes")
        blocks = [unit_blocks[name] for name in stage_cfg["unit_names"]]
        first_kind = _unit_kind(stage_cfg["unit_names"][0])
        input_arity = 1 if first_kind in ["stem", "head", "main_preadd"] else 2
        return UnitPipelineStage(blocks, input_arity=input_arity, prefix=stage_cfg["name"] + "_")
    blocks = [feature_blocks[idx] for idx in stage_cfg["feature_indices"]]
    if stage_cfg["kind"] == "features":
        return FeatureStage(blocks, prefix=stage_cfg["name"] + "_")
    stage_output = output_block if stage_cfg.get("include_output", False) else None
    return HeadStage(blocks, output_block=stage_output, prefix=stage_cfg["name"] + "_")


def summarize_scheme(scheme_name, scheme_cfg, batch, image_size):
    print("========== Stage Split ==========")
    print("scheme      =", scheme_name)
    print("input       =", (batch, 3, image_size, image_size))
    print("model       = resnet18_v1")
    print("=================================")
    for stage in scheme_cfg:
        if "unit_names" in stage:
            print(
                "{}  device={}  units={}".format(
                    stage["name"], stage["device"], stage["unit_names"]
                )
            )
        else:
            print(
                "{}  device={}  features={}".format(
                    stage["name"], stage["device"], stage["feature_indices"]
                )
            )


def summarize_feature_blocks(feature_blocks, output_block):
    print("========== Feature Blocks ==========")
    print("len(features) =", len(feature_blocks))
    for idx, block in enumerate(feature_blocks):
        print("feature[{}] = {}".format(idx, type(block).__name__))
    print("output      =", type(output_block).__name__)
    print("====================================")


def validate_scheme(feature_blocks, scheme_cfg, unit_blocks=None):
    max_idx = len(feature_blocks) - 1
    for stage in scheme_cfg:
        if "unit_names" in stage:
            if unit_blocks is None:
                raise ValueError("unit_blocks is required for unit-based stage schemes")
            for name in stage["unit_names"]:
                if name not in unit_blocks:
                    raise KeyError("Stage {} references unknown unit {}".format(stage["name"], name))
        else:
            for idx in stage["feature_indices"]:
                if idx < 0 or idx > max_idx:
                    raise IndexError(
                        "Stage {} references feature index {} but valid range is [0, {}]".format(
                            stage["name"], idx, max_idx
                        )
                    )


def lower_stage_to_relay(stage_block, stage_inputs):
    shape_dict = {name: shape for name, shape in stage_inputs}
    mod, params = relay.frontend.from_mxnet(stage_block, shape_dict)
    return mod, params


def main():
    args = parse_args()
    assert args.model == "resnet18_v1", "Only resnet18_v1 is supported for now"

    full_model = vision.get_model(args.model, pretrained=True)
    feature_blocks = list(full_model.features)
    output_block = full_model.output
    unit_blocks = build_resnet18_unit_blocks(feature_blocks, output_block)
    unit_metadata = build_resnet18_unit_metadata(unit_blocks, args.batch, args.image_size)
    scheme_cfg, selected, _ = resolve_scheme_config(
        args.scheme,
        feature_blocks,
        output_block,
        args.batch,
        args.image_size,
        print_resource_aware=(args.scheme == AUTO_RESOURCE_AWARE_SCHEME),
        candidate_top_k=10,
    )
    resolved_scheme_name = selected["scheme_name"] if selected is not None else args.scheme
    summarize_scheme(args.scheme, scheme_cfg, args.batch, args.image_size)
    summarize_feature_blocks(feature_blocks, output_block)
    summarize_units(unit_blocks)
    validate_scheme(feature_blocks, scheme_cfg, unit_blocks)
    if args.save_partition_viz:
        save_partition_visualization(
            args.save_partition_viz,
            resolved_scheme_name,
            scheme_cfg,
            unit_metadata,
            selected=selected,
        )
        print("[VIZ] saved partition visualization ->", args.save_partition_viz)

    if args.save_relay_dir:
        os.makedirs(args.save_relay_dir, exist_ok=True)

    for stage in scheme_cfg:
        stage_block = make_stage_block(feature_blocks, output_block, stage, unit_blocks=unit_blocks)
        stage_inputs = relay_inputs_for_stage(stage, unit_metadata)
        print(
            "[STAGE] {} device={} inputs={}".format(
                stage["name"], stage["device"], stage_inputs
            )
        )

        if args.save_relay_dir or args.print_relay:
            mod, params = lower_stage_to_relay(stage_block, stage_inputs)
            relay_text = mod["main"].astext(show_meta_data=False)
            print("[STAGE] {} params={}".format(stage["name"], len(params)))
            if args.print_relay:
                print("\n[RELAY] {}".format(stage["name"]))
                print(relay_text)
            if args.save_relay_dir:
                out_path = os.path.join(args.save_relay_dir, stage["name"] + ".relay")
                with open(out_path, "w") as f:
                    f.write(relay_text)
                print("[STAGE] saved Relay ->", out_path)


if __name__ == "__main__":
    main()
