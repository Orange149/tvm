"""Validate summary provenance before reusing rounds or making boot-level inferences."""
from pathlib import Path
import re

from cpu_vta_partitioning.common import read, sha256, digest, relative_file, verify_inventory
from cpu_vta_partitioning.domain import signatures
from cpu_vta_partitioning.prepare.payload import EXECUTION_POLICY


def linked_summary(root, path, receipt):
    return {**receipt["summary"], "_receipt": {
        "path": str(path.relative_to(root)), "sha256": sha256(path)}}


class Evidence:
    def __init__(self, envdir):
        from cpu_vta_partitioning.campaign import validate_run_identity
        self.envdir = Path(envdir).resolve()
        self.root = self.envdir.parent
        state = validate_run_identity(self.root)
        if not re.fullmatch(r"env[0-9]+", self.envdir.name):
            raise ValueError("invalid environment directory")
        self.environment = state["environments"][int(self.envdir.name[3:])]
        self.bundle_id = state["bundle_id"]
        self.controller_identity = state["controller_identity"]
        self.bundle = Path(state["bundle"])
        self.warmup = read(self.envdir / "warmup.json")["warmup"]
        self.cache = {}
        self.pools = {}

    def summary(self, summary, boot, **expected):
        ref = summary.get("_receipt")
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ValueError("measurement has no accepted receipt reference")
        path = relative_file(self.root, ref["path"])
        if not path.is_file() or sha256(path) != ref["sha256"]:
            raise ValueError("accepted receipt missing or changed: " + ref["path"])
        cache_key = (ref["path"], ref["sha256"])
        if cache_key not in self.cache:
            receipt = read(path)
            task = receipt["task"]
            task_id = digest({k:v for k,v in task.items() if k != "task_id"})[:24]
            if (task.get("task_id") != task_id or receipt["boot_id"] != boot
                    or ref["path"] != f"accepted/{boot}/{task_id}.json"
                    or task["bundle_id"] != self.bundle_id or task["environment"] != self.environment):
                raise ValueError("accepted task identity/boot/environment mismatch")
            files = receipt["files"]
            raw = self.root / "raw" / boot / task_id
            if not files or any(not relative_file(self.root, f).is_relative_to(raw) for f in files):
                raise ValueError("accepted raw files are missing or belong to another task")
            names = [Path(f).name for f in files]
            required = ["result.jsonl", "affinity.json"] + (["profile.json"] if task.get("detailed") else [])
            if any(names.count(name) != 1 for name in required):
                raise ValueError("accepted raw result inventory is incomplete")
            verify_inventory(self.root, files)
            self.cache[cache_key] = receipt
        receipt = self.cache[cache_key]
        task = receipt["task"]
        if receipt["boot_id"] != boot or receipt["summary"] != {k:v for k,v in summary.items() if k != "_receipt"}:
            raise ValueError("summary differs from its accepted receipt")
        if any(task.get(k) != v for k,v in expected.items()):
            raise ValueError("accepted task does not match its round or diagnostic condition")
        return task

    def pool(self, phase):
        if phase not in self.pools:
            if phase == "calibration":
                programs = read(self.bundle / "plans/programs.json")
                self.pools[phase] = signatures(programs, self.environment["cores"])
            else:
                from cpu_vta_partitioning.search import screening_pool
                pool = screening_pool(read(self.envdir / "ranking.json"))
                if phase == "confirmation":
                    wanted = {key for b in read(self.envdir / "selection.json").values() for key in b.values()}
                    pool = {key:pool[key] for key in wanted}
                self.pools[phase] = pool
        return self.pools[phase]

    def round(self, path):
        path = Path(path)
        record = read(path)
        phase = path.parent.parent.name
        limit = 6 if phase == "confirmation" else 2
        if phase not in ("calibration", "screening", "confirmation"):
            raise ValueError("invalid measurement phase")
        if path.parent.name not in {f"boot{i}" for i in range(limit)} or path.name not in ("round0.json", "round1.json"):
            raise ValueError("unexpected round path")
        boot_index, round_index = int(path.parent.name[4:]), int(path.stem[5:])
        if record["round"] != round_index or record["attempt"] not in range(3):
            raise ValueError("round identity mismatch")
        boot = record["boot_id"]
        common = {"campaign_phase":phase,"boot_index":boot_index,"round":round_index,"attempt":record["attempt"]}
        window = {"warmup":self.warmup,"scored":300,"tail":20,"input_indices":list(range(16,64))}
        for position in ("before", "after"):
            self.summary(record[position], boot, **common, **window, kind="anchor", position=position)
        if abs(record["after"]["ii_ms"] / record["before"]["ii_ms"] - 1) > .05:
            raise ValueError("accepted round exceeds anchor drift limit")
        if set(record["measurements"]) != set(self.pool(phase)):
            raise ValueError("round measurement pool mismatch")
        for key, value in record["measurements"].items():
            if phase == "calibration":
                s = self.pool(phase)[key]
                self.summary(value, boot, **common, kind=phase, signature=key, stages=[s.to_dict()],
                             warmup=5, scored=32, tail=0, input_indices=list(range(16)))
            else:
                self.summary(value, boot, **common, **window, kind=phase, candidate=key,
                             stages=self.pool(phase)[key]["stages"])
        return record

    def rounds(self, phase):
        limit = 6 if phase == "confirmation" else 2
        paths = sorted((self.envdir / phase).glob("boot*/round*.json"))
        wanted = {self.envdir / phase / f"boot{i}" / f"round{r}.json" for i in range(limit) for r in range(2)}
        if set(paths) != wanted:
            raise ValueError("phase requires two complete rounds per independent boot")
        records, boots = [], set()
        for index in range(limit):
            pair = [self.round(self.envdir / phase / f"boot{index}" / f"round{r}.json") for r in range(2)]
            boot = pair[0]["boot_id"]
            if not boot or pair[1]["boot_id"] != boot or boot in boots:
                raise ValueError("duplicate or mismatched independent boot IDs")
            boots.add(boot)
            records.extend(pair)
        return records

    def diagnostic(self, path, boot):
        path = Path(path)
        record = read(path)
        if record["boot_id"] != boot or record.get("execution_revision") != EXECUTION_POLICY["revision"]:
            raise ValueError("diagnostic boot/revision does not match confirmation")
        pairs = {p["islands"]:p for p in read(self.envdir / "diagnostic_selection.json")}
        blocks = record["blocks"]
        if len(blocks) != len(pairs) or {b["islands"] for b in blocks} != set(pairs):
            raise ValueError("diagnostic block coverage mismatch")
        expected_keys = {f"islands{i}_{m}_{d}" for i in pairs for m in ("shared","dedicated") for d in ("plain","profile")}
        if set(record["measurements"]) != expected_keys:
            raise ValueError("diagnostic condition coverage mismatch")
        for block in blocks:
            islands = block["islands"]
            if block["boot_id"] != boot or block["attempt"] not in range(EXECUTION_POLICY["diagnostic_replacement_blocks"] + 1):
                raise ValueError("diagnostic block boot/attempt mismatch")
            common = {"campaign_phase":"confirmation", "boot_index":int(path.parent.name[4:]),
                      "diagnostic_islands":islands, "diagnostic_attempt":block["attempt"]}
            window = {"warmup":self.warmup,"scored":300,"tail":20,"input_indices":list(range(16,64))}
            for position in ("before", "after"):
                self.summary(block[position], boot, **common, **window, kind="anchor", position="diagnostic_" + position)
            if abs(block["after"]["ii_ms"] / block["before"]["ii_ms"] - 1) > EXECUTION_POLICY["diagnostic_anchor_drift"]:
                raise ValueError("accepted diagnostic exceeds anchor drift limit")
            for mode in ("shared", "dedicated"):
                for detailed in (False, True):
                    key = f"islands{islands}_{mode}_{'profile' if detailed else 'plain'}"
                    self.summary(record["measurements"][key], boot, **common, **window, kind="diagnostic",
                                 diagnostic_key=key, detailed=detailed, stages=pairs[islands][mode])
        return record

    def confirmation(self):
        rounds = self.rounds("confirmation")
        paths = set((self.envdir / "confirmation").glob("boot*/diagnostic.json"))
        wanted = {self.envdir / "confirmation" / f"boot{i}" / "diagnostic.json" for i in range(6)}
        if paths != wanted:
            raise ValueError("confirmation requires six complete diagnostic boots")
        diagnostics = [self.diagnostic(self.envdir / "confirmation" / f"boot{i}" / "diagnostic.json",
                                      rounds[2*i]["boot_id"]) for i in range(6)]
        return rounds, diagnostics
