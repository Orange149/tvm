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
"""

from __future__ import absolute_import, print_function

import argparse
import os

from mxnet.gluon import HybridBlock, nn
from mxnet.gluon.model_zoo import vision

import tvm
from tvm import relay


SCHEMES = {
    # CPU -> VTA -> CPU
    "three_stage_a": [
        {
            "name": "stage0_cpu",
            "device": "cpu",
            "kind": "features",
            "feature_indices": [0, 1, 2, 3, 4],  # stem + layer1
        },
        {
            "name": "stage1_vta",
            "device": "vta",
            "kind": "features",
            "feature_indices": [5, 6],  # layer2 + layer3
        },
        {
            "name": "stage2_cpu",
            "device": "cpu",
            "kind": "head",
            "feature_indices": [7, 8],  # layer4 + global avg
            "include_output": True,  # final dense
        },
    ],
    # CPU -> VTA (+ tiny CPU epilogue in practice)
    "two_stage_a": [
        {
            "name": "stage0_cpu",
            "device": "cpu",
            "kind": "features",
            "feature_indices": [0, 1, 2, 3, 4, 5],  # stem + layer1 + layer2
        },
        {
            "name": "stage1_vta",
            "device": "vta",
            "kind": "head",
            "feature_indices": [6, 7, 8],  # layer3 + layer4 + global avg
            "include_output": True,
        },
    ],
    # More balanced VTA middle only
    "three_stage_b": [
        {
            "name": "stage0_cpu",
            "device": "cpu",
            "kind": "features",
            "feature_indices": [0, 1, 2, 3, 4, 5],  # stem + layer1 + layer2
        },
        {
            "name": "stage1_vta",
            "device": "vta",
            "kind": "features",
            "feature_indices": [6],  # layer3 only
        },
        {
            "name": "stage2_cpu",
            "device": "cpu",
            "kind": "head",
            "feature_indices": [7, 8],  # layer4 + global avg
            "include_output": True,
        },
    ],
    # CPU stem -> VTA backbone -> CPU tail
    "three_stage_c": [
        {
            "name": "stage0_cpu",
            "device": "cpu",
            "kind": "features",
            "feature_indices": [0, 1, 2, 3],  # stem only
        },
        {
            "name": "stage1_vta",
            "device": "vta",
            "kind": "features",
            "feature_indices": [4, 5, 6],  # layer1 + layer2 + layer3
        },
        {
            "name": "stage2_cpu",
            "device": "cpu",
            "kind": "head",
            "feature_indices": [7, 8],  # layer4 + global avg
            "include_output": True,
        },
    ],
}


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheme",
        default="three_stage_a",
        choices=sorted(SCHEMES.keys()),
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
    return parser.parse_args()


def stage_input_shapes(batch, image_size):
    # These are the natural stage interface shapes for ResNet18 with the
    # recommended stage boundaries. For the image sizes used in this project
    # (224/256/320/384), each value is divisible by 32, so integer division
    # matches the actual feature map sizes.
    h4 = image_size // 4
    h8 = image_size // 8
    h16 = image_size // 16
    return {
        "stage0_cpu": (batch, 3, image_size, image_size),
        "stage1_vta": (batch, 64, h4, h4),
        "stage2_cpu_from_three_stage_a": (batch, 256, h16, h16),
        "stage2_cpu_from_three_stage_b": (batch, 256, h8, h8),
        "stage1_vta_from_two_stage_a": (batch, 128, h8, h8),
    }


def relay_shape_for_stage(stage_name, scheme_name, batch, image_size):
    shapes = stage_input_shapes(batch, image_size)
    if scheme_name == "three_stage_a":
        if stage_name == "stage0_cpu":
            return shapes["stage0_cpu"]
        if stage_name == "stage1_vta":
            return shapes["stage1_vta"]
        return shapes["stage2_cpu_from_three_stage_a"]
    if scheme_name == "three_stage_c":
        if stage_name == "stage0_cpu":
            return shapes["stage0_cpu"]
        if stage_name == "stage1_vta":
            return shapes["stage1_vta"]
        return shapes["stage2_cpu_from_three_stage_a"]
    if scheme_name == "three_stage_b":
        if stage_name == "stage0_cpu":
            return shapes["stage0_cpu"]
        if stage_name == "stage1_vta":
            return shapes["stage1_vta_from_two_stage_a"]
        return shapes["stage2_cpu_from_three_stage_b"]
    if scheme_name == "two_stage_a":
        if stage_name == "stage0_cpu":
            return shapes["stage0_cpu"]
        return shapes["stage1_vta_from_two_stage_a"]
    raise ValueError("Unsupported scheme {}".format(scheme_name))


def make_stage_block(feature_blocks, output_block, stage_cfg):
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


def validate_scheme(feature_blocks, scheme_cfg):
    max_idx = len(feature_blocks) - 1
    for stage in scheme_cfg:
        for idx in stage["feature_indices"]:
            if idx < 0 or idx > max_idx:
                raise IndexError(
                    "Stage {} references feature index {} but valid range is [0, {}]".format(
                        stage["name"], idx, max_idx
                    )
                )


def lower_stage_to_relay(stage_block, input_shape):
    shape_dict = {"data": input_shape}
    mod, params = relay.frontend.from_mxnet(stage_block, shape_dict)
    return mod, params


def main():
    args = parse_args()
    assert args.model == "resnet18_v1", "Only resnet18_v1 is supported for now"

    full_model = vision.get_model(args.model, pretrained=True)
    feature_blocks = list(full_model.features)
    output_block = full_model.output
    scheme_cfg = SCHEMES[args.scheme]
    summarize_scheme(args.scheme, scheme_cfg, args.batch, args.image_size)
    summarize_feature_blocks(feature_blocks, output_block)
    validate_scheme(feature_blocks, scheme_cfg)

    if args.save_relay_dir:
        os.makedirs(args.save_relay_dir, exist_ok=True)

    for stage in scheme_cfg:
        stage_block = make_stage_block(feature_blocks, output_block, stage)
        input_shape = relay_shape_for_stage(
            stage["name"], args.scheme, args.batch, args.image_size
        )
        print(
            "[STAGE] {} device={} input_shape={}".format(
                stage["name"], stage["device"], input_shape
            )
        )

        if args.save_relay_dir or args.print_relay:
            mod, params = lower_stage_to_relay(stage_block, input_shape)
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
