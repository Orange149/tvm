from types import SimpleNamespace

import prepare_vta_model_conv_adaptive_holdout as target


class Env:
    BATCH = 1
    BLOCK_IN = 16
    BLOCK_OUT = 16


def test_stride_one_default_preserves_existing_workload_encoding():
    args = SimpleNamespace(ci=256, co=1024, height=14, width=14, kernel=1,
                           stride=1, padding=None)
    workload = target.workload(args, Env())
    assert workload[3] == [1, 1]
    assert workload[4] == [0, 0, 0, 0]


def test_stride_two_three_by_three_is_encoded_in_workload_identity():
    args = SimpleNamespace(ci=128, co=128, height=56, width=56, kernel=3,
                           stride=2, padding=1)
    workload = target.workload(args, Env())
    assert workload[1][1] == [1, 8, 56, 56, 1, 16]
    assert workload[2][1] == [8, 8, 3, 3, 16, 16]
    assert workload[3] == [2, 2]
    assert workload[4] == [1, 1, 1, 1]


def test_channels_must_be_exactly_packable():
    args = SimpleNamespace(ci=3, co=16, height=416, width=416, kernel=3,
                           stride=1, padding=1)
    try:
        target.workload(args, Env())
    except ValueError as error:
        assert "exactly representable" in str(error)
    else:
        raise AssertionError("unpacked channels were accepted")


def test_resnet50_stride_two_is_on_bottleneck_conv1():
    conv1 = target.extract_resnet50_conv("stage2_unit1_conv1")
    conv2 = target.extract_resnet50_conv("stage2_unit1_conv2")
    assert conv1["input_shape_nchw"] == [1, 256, 56, 56]
    assert conv1["weight_shape_oihw"] == [128, 256, 1, 1]
    assert conv1["output_shape_nchw"] == [1, 128, 28, 28]
    assert conv1["strides"] == [2, 2]
    assert conv1["padding"] == [0, 0, 0, 0]
    assert conv2["input_shape_nchw"] == [1, 128, 28, 28]
    assert conv2["weight_shape_oihw"] == [128, 128, 3, 3]
    assert conv2["strides"] == [1, 1]
    assert conv2["padding"] == [1, 1, 1, 1]


def test_yolo_declared_geometry_is_checked_against_imported_graph(monkeypatch):
    actual = {
        "ci": 128, "co": 256, "height": 20, "width": 20,
        "kernel": 3, "stride": 1, "symmetric_padding": 1,
    }
    monkeypatch.setattr(target, "extract_yolov3_tiny_conv", lambda *unused: actual)
    args = SimpleNamespace(
        model="yolov3_tiny_320", layer="conv8", model_source="model.cfg",
        darknet_weights="model.weights", darknet_lib="libdarknet.so",
        ci=128, co=256, height=20, width=20, kernel=3, stride=1, padding=1,
    )
    assert target.verify_declared_model_geometry(args) == actual
    args.height = 26
    try:
        target.verify_declared_model_geometry(args)
    except ValueError as error:
        assert "declared convolution disagrees" in str(error)
    else:
        raise AssertionError("mismatched YOLO geometry was accepted")


def test_yolo_source_verification_requires_assets():
    args = SimpleNamespace(
        model="yolov3_tiny_320", layer="conv8", model_source="model.cfg",
        darknet_weights=None, darknet_lib=None,
        ci=128, co=256, height=20, width=20, kernel=3, stride=1, padding=1,
    )
    try:
        target.verify_declared_model_geometry(args)
    except ValueError as error:
        assert "requires --darknet-weights" in str(error)
    else:
        raise AssertionError("YOLO source verification without assets was accepted")
