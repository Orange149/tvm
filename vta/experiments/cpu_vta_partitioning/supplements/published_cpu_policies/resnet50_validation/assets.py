"""Read-only identity checks for reused ResNet50 preparation assets.

Importing this module does not import TVM or contact the board.
"""
from pathlib import Path
import shutil

from cpu_vta_partitioning.common import read, sha256


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
REPO = ROOT.parents[4]
LEGACY_OUTPUT = ROOT / "artifacts/resnet50_v2_validation_01"
DEFAULT_OUTPUT = ROOT / "artifacts/resnet50_v2_validation_native_history_01"
DEFAULT_LOCAL_CONFIG = PROJECT / "configs/local.json"
PARENT = ROOT / "artifacts/run_scope_tail_comparison_20261112_01/resnet18/capsule"
WEIGHTS = Path("/home/orange/.mxnet/models/resnet50_v2-ecdde353.params")
WEIGHTS_SHA256 = "e00acf01c0ce0767179e2811e18e46176c40815b55d3b6820694614ed62eac71"
EXPECTED = {
    "runner/vta_stage_pipeline_runner": "25833e7f8b47fc8824ff9460afc0b968f24934dcee41c3b1f28619a99b6b6fe6",
    "runtime/libtvm_runtime.so": "724d15640bc5f75548725048f9b6adb1750bce768aa6ea39fe3fbaeb6726af98",
    "runtime/libvta.so": "cd2eb2e3344abfc3cc7b01d8c3600974811900a0c96b0cb79b16841f1736332e",
    "firmware/vta_hpc.bit": "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6",
    "firmware/u-dma-buf.ko": "4c0aa5713225cd9b29548d9c1cd6e6127aeaafcc1f7d8650d68e3a0a527042c0",
    "inputs/000.bin": "78f9e1dede477e9f3acba0d4d92aae86daeb9a7d1f1ba1e95f373db41207b222",
}


def source_path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else REPO / p


def checked_file(path, expected=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None and value["sha256"] != expected:
        raise ValueError("asset identity differs: " + str(path))
    return value


def audit_assets(local_config=DEFAULT_LOCAL_CONFIG):
    local = read(local_config)
    manifest = read(PARENT / "manifest.json")
    reused = {}
    for name, expected in EXPECTED.items():
        if manifest["files"].get(name, {}).get("sha256") != expected:
            raise ValueError("parent capsule does not freeze expected asset: " + name)
        reused[name] = checked_file(PARENT / name, expected)
    tools = {key: checked_file(source_path(local[key])) for key in
             ("sdk_setup", "host_tvm_library", "tuning_history")}
    python = Path(local["python"])
    if not python.is_file():
        raise FileNotFoundError(python)
    qemu = shutil.which("qemu-aarch64-static")
    if not qemu:
        raise RuntimeError("qemu-aarch64-static is required for ARM references")
    return {"status": "LOCAL_ASSETS_IDENTITIES_MATCH", "model": "mxnet_gluon_resnet50_v2",
            "weights": checked_file(WEIGHTS, WEIGHTS_SHA256), "parent_capsule": str(PARENT),
            "parent_manifest_sha256": sha256(PARENT / "manifest.json"),
            "assets": reused, "tools": tools, "python": str(python), "qemu": qemu,
            "input_shape": [1, 3, 224, 224], "new_runtime_builds": 0,
            "new_executor_builds": 0, "board_accessed": False,
            "not_yet_proved": ["ResNet50 ARM outputs", "stage module ABI/loading",
                               "board memory and storage capacity", "current board identity"]}
