"""Replay summaries only from byte-verified raw files and matching boot/task receipts."""
from pathlib import Path

from cpu_vta_partitioning.common import digest, read, relative_file, sha256, verify_inventory
from cpu_vta_partitioning.measurement import frames
from .measurement import validate_summary


def identified(document, bundle_id):
    value = {k: v for k, v in document.items() if k != "task_id"}
    if value.get("bundle_id", bundle_id) != bundle_id:
        raise ValueError("task belongs to another bundle")
    value["bundle_id"] = bundle_id
    return {**value, "task_id": digest(value)[:24]}


class Evidence:
    def __init__(self, root, bundle_id, bundle=None):
        self.root, self.bundle_id = Path(root), bundle_id
        self.bundle = Path(bundle) if bundle is not None else None

    def summary(self, value, document, boot):
        expected = identified(document, self.bundle_id)
        ref = value.get("_receipt", {})
        name = f"accepted/{boot}/{expected['task_id']}.json"
        if ref.get("path") != name:
            raise ValueError("receipt does not belong to expected boot/task")
        path = relative_file(self.root, name)
        if not path.is_file() or sha256(path) != ref.get("sha256"):
            raise ValueError("receipt missing or changed")
        receipt = read(path)
        if receipt["task"] != expected or receipt["boot_id"] != boot:
            raise ValueError("accepted task identity mismatch")
        files = receipt["files"]
        raw = relative_file(self.root, f"raw/{boot}/{expected['task_id']}")
        if not files or any(not relative_file(self.root, n).is_relative_to(raw) for n in files):
            raise ValueError("receipt raw file scope mismatch")
        verify_inventory(self.root, files)
        matches = [relative_file(self.root, n) for n in files if Path(n).name == "result.jsonl"]
        if len(matches) != 1:
            raise ValueError("missing or duplicate native frame file")
        rawdir = matches[0].parent
        if sum(Path(n).name == "affinity.json" for n in files) != 1:
            raise ValueError("missing or duplicate native affinity file")
        if str((rawdir / "affinity.json").relative_to(self.root.resolve())) not in files:
            raise ValueError("affinity report and frame file are from different sessions")
        if expected["kind"].startswith("qualify"):
            native = frames(matches[0], sum(expected[k] for k in ("warmup", "scored", "tail")))
            for row in native:
                for dump in row["stage_raw_outputs"]:
                    for output in dump["output_files"]:
                        target = rawdir / "outputs" / ("run_" + str(row["completion_index"])) / Path(output["path"]).name
                        if str(target.relative_to(self.root.resolve())) not in files:
                            raise ValueError("correctness output is not bound by the receipt")
        bundle = self.bundle
        state = self.root / "state.json"
        if bundle is None and state.is_file():
            bundle = Path(read(state)["bundle"])
        if bundle is not None:
            manifest = read(bundle / "manifest.json")
            if manifest["bundle_id"] != self.bundle_id:
                raise ValueError("reference bundle identity mismatch")
        calculated = validate_summary(rawdir, expected, bundle)
        if receipt["summary"] != calculated:
            raise ValueError("summary disagrees with native records")
        if receipt["summary"] != {k: v for k, v in value.items() if k != "_receipt"}:
            raise ValueError("round summary differs from accepted receipt")
        return receipt

    def component_details(self, value, document, boot):
        receipt = self.summary(value, document, boot)
        if document["kind"] != "calibration":
            raise ValueError("component details require an isolated calibration")
        summary = receipt["summary"]
        return {f"{operation}_{scope}_ms": summary[f"{operation}_{suffix}"]
                for operation in ("set", "run", "get")
                for scope, suffix in (("wall", "ms"), ("core", "core_ms"))}
