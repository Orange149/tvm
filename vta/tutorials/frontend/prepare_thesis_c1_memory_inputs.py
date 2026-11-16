#!/usr/bin/env python3
"""Freeze 64 real images and independent LLVM frontier references, without quantizing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from thesis_c1_memory_ablation import file_hash, read_json, seal, verify_ref, write_new
from thesis_supplement_run import RunStore

FRONTEND = Path(__file__).resolve().parent
ROOT = FRONTEND.parents[2]
PREPARATION = FRONTEND / "report_out/stage_tile_cotuning/thesis_supplement_experiments/results/01/20260916T011355Z_completion_0670cfb1"
REFERENCE = PREPARATION / "raw/canonical_host_qualification_01"
CANONICAL = PREPARATION / "raw/canonical_split_preparation_02/canonical_qir.json"
PREPROCESSING = {
    "resize": [224, 224], "resize_before_rgb": True, "resample": "Pillow default",
    "mean": [123, 117, 104], "scale": [58.395, 57.12, 57.375],
    "dtype": "float32", "layout": "NCHW", "shape": [1, 3, 224, 224],
}


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path)}


def preprocess(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as source:
        source.load()
        value = np.asarray(source.resize((224, 224)).convert("RGB"), dtype="float32")
    value = (value - np.array(PREPROCESSING["mean"], dtype="float32")) / np.array(
        PREPROCESSING["scale"], dtype="float32")
    return np.ascontiguousarray(value.transpose(2, 0, 1)[None], dtype="float32")


def select_images(directory):
    """Select by content hash before inference; skip unreadable/duplicate contents."""
    from PIL import Image
    rows, unique = [], {}
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file():
            continue
        item = reference(path)
        try:
            with Image.open(path) as source:
                source.verify()
            preprocess(path)
        except (OSError, ValueError, SyntaxError) as error:
            rows.append({**item, "status": "unreadable", "reason": str(error)})
            continue
        if item["sha256"] in unique:
            rows.append({**item, "status": "duplicate"})
        else:
            unique[item["sha256"]] = item
            rows.append({**item, "status": "readable_unique"})
    selected = [unique[key] for key in sorted(unique)[:64]]
    if len(selected) != 64:
        raise ValueError("need at least 64 distinct readable real image contents")
    return selected, rows


def worker(directory):
    import numpy as np
    import PIL
    import tvm
    from tvm.contrib import graph_executor
    from thesis_canonical_graph_split import UnitGraph

    store = RunStore(directory)
    frozen = store.contract
    canonical = verify_ref(frozen["canonical_qir"])
    for item in frozen["selected_images"]:
        verify_ref(item)
    for item in frozen["reference_artifacts"]:
        verify_ref(item)
    source = Path(frozen["reference_directory"])
    graph = UnitGraph(tvm.ir.load_json(Path(canonical["path"]).read_text()))
    edges = sorted({n for boundary in range(22) for n in graph.frontier(boundary)},
                   key=lambda n: graph.ids.get(n, -1))
    executors = {}
    for name in ("canonical_qir", "canonical_boundary_probe"):
        prefix = source / "compiled" / name
        exe = graph_executor.create((prefix / "graph.json").read_text(),
                                    tvm.runtime.load_module(str(prefix / "module.so")), tvm.cpu())
        exe.load_params((prefix / "params.bin").read_bytes())
        executors[name] = exe
    if executors["canonical_boundary_probe"].get_num_outputs() != len(edges):
        raise ValueError("reference probe output map differs from canonical frontiers")
    records, input_hashes = [], set()
    for index in range(67):
        target = store.directory / "raw/inputs" / "input_{:02}".format(index)
        target.mkdir(parents=True)
        row = {"id": "real_{:02}".format(index) if index < 64 else ("zeros", "uniform", "normal")[index - 64]}
        if index < 64:
            row["source_image"] = frozen["selected_images"][index]
            value = preprocess(row["source_image"]["path"])
        else:
            row["kind"] = row["id"]
            if index == 64:
                value = np.zeros((1, 3, 224, 224), dtype="float32")
            else:
                row["seed"] = 59001 if index == 65 else 59002
                rng = np.random.Generator(np.random.PCG64(row["seed"]))
                value = (rng.uniform(0, 1, (1, 3, 224, 224)) if index == 65 else
                         rng.normal(0, 1, (1, 3, 224, 224))).astype("float32")
        (target / "input.bin").write_bytes(value.tobytes())
        row["input"] = reference(target / "input.bin")
        if row["input"]["sha256"] in input_hashes:
            raise ValueError("distinct images became duplicate model inputs")
        input_hashes.add(row["input"]["sha256"])
        for exe in executors.values():
            exe.set_input("data", value)
            exe.run()
        full = executors["canonical_qir"].get_output(0).numpy()
        snapshots = {str(graph.ids[n]): executors["canonical_boundary_probe"].get_output(i).numpy()
                     for i, n in enumerate(edges)}
        if full.tobytes() != snapshots[str(graph.ids[graph.mod["main"].body])].tobytes():
            raise ValueError("complete reference and boundary probe differ")
        if any(not np.isfinite(value).all() for value in snapshots.values()):
            raise ValueError("nonfinite common reference")
        row["frontiers"] = {}
        for node, value in snapshots.items():
            path = target / ("node_" + node + ".npy")
            np.save(path, value, allow_pickle=False)
            row["frontiers"][node] = reference(path)
        records.append(row)
        print("independent host reference {}/67 complete".format(index + 1), flush=True)
    # Detect source drift over the entire preparation, not only at entry.
    for item in [canonical, *frozen["selected_images"], *frozen["reference_artifacts"]]:
        verify_ref(item)
    result = seal({"schema": "c1_memory_ablation_inputs_v1",
                   "canonical_qir_sha256": canonical["sha256"],
                   "preprocessing": {**PREPROCESSING, "pillow_version": PIL.__version__, "numpy_version": np.__version__},
                   "selection": "first 64 distinct readable image content SHA256 values; no model outputs used",
                   "source_directory": frozen["images_directory"], "accuracy_evidence": False,
                   "scope": "numerical qualification inputs; supplied directory is not a balanced ImageNet validation set",
                   "reference_artifacts": frozen["reference_artifacts"], "preparation_source": reference(__file__),
                   "real_inputs": records[:64], "diagnostic_inputs": records[64:],
                   "board_calls": 0, "extra_quantize_calls": 0})
    write_new(store.directory / "inputs_manifest.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE)
    parser.add_argument("--canonical-qir", type=Path, default=CANONICAL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.output_dir)
    if not args.images_dir:
        parser.error("--images-dir required")
    selected, inventory = select_images(args.images_dir)
    canonical = reference(args.canonical_qir)
    source = args.reference_dir.resolve()
    if read_json(source / "contract.json")["canonical_qir_sha256"] != canonical["sha256"]:
        raise ValueError("reference was built from a different quantized graph")
    hashes = read_json(source / "artifact_hashes.json")
    refs = [reference(source / name) for name in ("contract.json", "artifact_hashes.json")]
    for name in ("canonical_qir", "canonical_boundary_probe"):
        for filename in ("graph.json", "module.so", "params.bin"):
            relative = "compiled/{}/{}".format(name, filename)
            refs.append(verify_ref({"path": str(source / relative), "sha256": hashes[relative]["sha256"]}))
    store = RunStore.create(args.output_dir, {
        "run_id": args.output_dir.name, "stage": "C1_inputs", "campaign_id": "c1_memory_ablation",
        "canonical_qir": canonical, "images_directory": str(args.images_dir.resolve()),
        "selected_images": selected, "reference_directory": str(source), "reference_artifacts": refs,
        "scope": "new common host references; no board evidence or accuracy claim",
    })
    write_new(store.directory / "raw/image_inventory.json", inventory)
    result = store.execute([sys.executable, str(Path(__file__).resolve()), "--worker", "--output-dir",
                            str(store.directory)], "common_host_reference", "host_reference_preparation", 1800)
    if result["status"] != "ok":
        write_new(store.directory / "FAILED.json", result)
        raise RuntimeError("host reference failed; inspect " + str(store.directory / result["stderr"]))
    write_new(store.directory / "preparation_cost.json", result)
    (store.directory / "REPORT.md").write_text(
        "# C1 共同输入与参考\n\n64 张真实图与 3 个诊断输入已冻结，所有规范图边界参考已生成。\n\n"
        "状态：host_reference_ready；板端资格和性能尚未执行。图片目录不作为均衡 ImageNet 准确率集。\n\n"
        "图片按内容 SHA256 选择，预处理沿用既有单图协议；复用同图 LLVM 参考程序，没有重新量化。\n",
        encoding="utf-8")
    print(store.directory / "inputs_manifest.json")


if __name__ == "__main__":
    main()
