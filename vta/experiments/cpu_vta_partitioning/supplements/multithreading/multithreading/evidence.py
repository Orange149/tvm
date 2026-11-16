"""Accepted receipts bind every reused summary to exact native files and tasks."""
import math
from pathlib import Path

from cpu_vta_partitioning.common import digest,read,sha256,relative_file,verify_inventory
from cpu_vta_partitioning.measurement import frames,input_sequence,affinity,window,calibration


def identified(document,bundle_id):
    value = {k:v for k,v in document.items() if k!="task_id"}
    value["bundle_id"] = bundle_id
    return {**value,"task_id":digest(value)[:24]}


def check_gate(rows,document):
    expected = document["execution_mode"]
    intervals = {}
    for item in rows:
        if item.get("cpu_stage_gate_mode","off") != expected:
            raise ValueError("native execution mode differs from task")
        if expected == "off": continue
        for i,stage in enumerate(document["stages"]):
            if stage["device"]!="cpu": continue
            times = [item[f"stage{i}_cpu_gate_{key}_ms"] for key in ("request","acquire","release")]
            if any(v is None or not math.isfinite(v) for v in times) or not times[0]<=times[1]<=times[2]:
                raise ValueError("invalid resource-group admission timestamps")
            intervals.setdefault(tuple(stage["mask"]),[]).append((times[1],times[2]))
    for spans in intervals.values():
        spans.sort()
        if any(a[1]>b[0]+1e-6 for a,b in zip(spans,spans[1:])):
            raise ValueError("same resource group executed concurrently")


class Evidence:
    def __init__(self,root,bundle_id):
        self.root,self.bundle_id = Path(root),bundle_id
        self.cache = {}

    def summary(self,value,document,boot):
        expected = identified(document,self.bundle_id)
        ref = value.get("_receipt",{})
        name = f"accepted/{boot}/{expected['task_id']}.json"
        if ref.get("path")!=name:
            raise ValueError("receipt does not belong to expected boot/task")
        path = relative_file(self.root,name)
        if not path.is_file() or sha256(path)!=ref.get("sha256"):
            raise ValueError("receipt missing or changed")
        cache_key = (name,ref["sha256"])
        if cache_key not in self.cache:
            receipt = read(path)
            if receipt["task"]!=expected or receipt["boot_id"]!=boot:
                raise ValueError("accepted task identity mismatch")
            files = receipt["files"]
            raw = self.root/"raw"/boot/expected["task_id"]
            if not files or any(not relative_file(self.root,n).is_relative_to(raw) for n in files):
                raise ValueError("receipt raw file scope mismatch")
            verify_inventory(self.root,files)
            def one(filename):
                matches = [relative_file(self.root,n) for n in files if Path(n).name==filename]
                if len(matches)!=1: raise ValueError("missing or duplicate raw "+filename)
                return matches[0]
            native = frames(one("result.jsonl"),sum(expected[k] for k in ("warmup","scored","tail")))
            input_sequence(native,expected)
            affinity(read(one("affinity.json")),expected["stages"])
            check_gate(native,expected)
            if expected.get("detailed"): one("profile.json")
            if expected["kind"]=="calibration":
                calculated = calibration(native,expected["warmup"],expected["scored"])
            elif expected["kind"].startswith("qualify"):
                calculated = {"correct":True}
            else:
                calculated = window(native,expected["warmup"],expected["scored"],expected["tail"])
            if any(receipt["summary"].get(k)!=v for k,v in calculated.items()):
                raise ValueError("summary disagrees with native records")
            self.cache[cache_key] = receipt
        receipt = self.cache[cache_key]
        if receipt["task"]!=expected or receipt["summary"]!={k:v for k,v in value.items() if k!="_receipt"}:
            raise ValueError("round summary differs from accepted receipt")
        return receipt

    def component_details(self,value,document,boot):
        receipt = self.summary(value,document,boot)
        filename = next(name for name in receipt["files"] if Path(name).name=="result.jsonl")
        rows = frames(self.root/filename,document["warmup"]+document["scored"])
        kept = rows[document["warmup"]:]
        import statistics
        return {f"{operation}_{scope}_ms":statistics.mean(r[f"stage0_{operation}_{suffix}"] for r in kept)
                for operation in ("set","run","get")
                for scope,suffix in (("wall","ms"),("core","process_cpu_ms"))}
