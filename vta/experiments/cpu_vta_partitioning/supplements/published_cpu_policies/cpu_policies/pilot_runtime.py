"""Frozen-runner execution and raw evidence for the single-boot pilot.

CPU affinity describes computation; it never selects a different executable.
Both VTA placements and every pipeline use the same frozen policy runner.
No timing variation, anchor drift, or performance threshold rejects a session.
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import shlex
import time


KINDS = frozenset({"calibration", "qualify_component", "qualify_pipeline", "performance"})
_FIELDS = frozenset({"kind", "configuration", "stages", "environment", "experiment",
                     "warmup", "scored", "tail", "input_indices", "executor",
                     "executor_path", "executor_sha256", "task_id", "bundle_id"})


def _legacy(bundle):
    from .pilot_common import load_legacy
    load_legacy(bundle)


def identified(document, bundle_id):
    """Include every execution setting and the bundle in the task identity."""
    from cpu_vta_partitioning.common import digest
    value = {k: v for k, v in document.items() if k != "task_id"}
    if value.get("bundle_id", bundle_id) != bundle_id:
        raise ValueError("task belongs to a different pilot bundle")
    value["bundle_id"] = bundle_id
    return {**value, "task_id": digest(value)[:24]}


def make_task(kind, config, bundle, *, warmup=0, scored=1, tail=0, indices=None, **labels):
    """Create a task without the old experiment's identity or timing gates."""
    _legacy(bundle)
    from cpu_vta_partitioning.common import read
    from cpu_strategy.policy.domain import validate_candidate
    from .pilot_common import protocol

    if kind not in KINDS:
        raise ValueError("unsupported pilot task kind")
    if any(type(v) is not int or v < 0 for v in (warmup, scored, tail)) or scored < 1:
        raise ValueError("invalid pilot measurement window")
    if kind == "performance" and scored < 2:
        raise ValueError("performance requires at least two completion timestamps")
    indices = [0] if indices is None else list(indices)
    if indices != [0] or type(indices[0]) is not int:
        raise ValueError("the single-input pilot only permits input 0")
    if set(labels) & _FIELDS:
        raise ValueError("task labels override frozen execution fields")
    component = kind in ("calibration", "qualify_component")
    validate_candidate(config, read(Path(bundle) / "plans/programs.json"), component)
    if component and (len(config["stages"]) != 1 or tail):
        raise ValueError("component tasks require one stage and no tail")
    if not component and len(config["stages"]) < 3:
        raise ValueError("pipeline task requires a complete CPU/VTA graph")
    spec = read(Path(bundle) / "plans/executors.json")["policy"]
    settings = protocol()
    document = {"kind": kind, "configuration": deepcopy(config),
                "stages": deepcopy(config["stages"]),
                "environment": deepcopy(settings["environment"]),
                "experiment": settings["experiment"], "warmup": warmup,
                "scored": scored, "tail": tail, "input_indices": indices,
                "executor": "policy", "executor_path": spec["path"],
                "executor_sha256": spec["sha256"], **labels}
    return identified(document, read(Path(bundle) / "manifest.json")["bundle_id"])


def validate_document(document, bundle):
    """Reject stale identities and changes to the requested execution contract."""
    if not isinstance(document, dict):
        raise ValueError("pilot task must be an object")
    try:
        labels = {k: v for k, v in document.items() if k not in _FIELDS}
        expected = make_task(document["kind"], document["configuration"], bundle,
                             warmup=document["warmup"], scored=document["scored"],
                             tail=document["tail"], indices=document["input_indices"], **labels)
    except KeyError as error:
        raise ValueError("pilot task is missing an execution field") from error
    if document != expected:
        raise ValueError("pilot task identity, executor, or configuration differs")
    return document


def native_command(document, bundle, remote_bundle, remote_job):
    """Build arguments for the frozen policy runner, including VTA-only tasks."""
    validate_document(document, bundle)
    from cpu_vta_partitioning.measurement.tasks import native_command as base_command

    config = document["configuration"]
    argv, inputs = base_command(document, bundle, remote_bundle, remote_job)
    original = remote_bundle + "/runner/vta_stage_pipeline_runner"
    argv[argv.index(original)] = remote_bundle + "/" + document["executor_path"]
    argv[argv.index("--cpu-stage-gate") + 1] = config["execution_mode"]
    loader = argv.index(remote_bundle + "/runtime/ld-linux-aarch64.so.1")
    argv[loader:loader] = ["TVM_BIND_THREADS=1", "TVM_EXCLUDE_WORKER0=1"]
    argv += ["--allow-cpu-affinity-superset", "--control-cpu-affinity",
             ",".join(map(str, config["control_mask"]))]
    if config["affinity_mode"] == "pinned":
        argv.append("--cpu-worker-pinning")
    if document["kind"].startswith("qualify"):
        argv.append("--resource-affinity-probe")
    return argv, inputs


def validate_summary(rawdir, document, bundle):
    """Replay numerics, affinities, frame identities and correctly scoped clocks."""
    validate_document(document, bundle)
    from cpu_vta_partitioning.measurement import frames
    from cpu_strategy.policy.evidence import validate_summary as frozen_summary
    rows = frames(Path(rawdir) / "result.jsonl", sum(document[k] for k in ("warmup", "scored", "tail")))
    if any(type(row.get(key)) is not int for row in rows
           for key in ("frame_id", "completion_index", "input_index")):
        raise ValueError("frame and input identities must be integers")
    return frozen_summary(rawdir, document, bundle)


class Evidence:
    """Only byte-bound raw data from the expected task and boot is accepted."""

    def __init__(self, root, bundle):
        _legacy(bundle)
        from cpu_vta_partitioning.common import read
        self.root = Path(root).resolve()
        self.bundle = Path(bundle).resolve()
        self.bundle_id = read(self.bundle / "manifest.json")["bundle_id"]

    def summary(self, value, document, boot):
        from cpu_vta_partitioning.common import read, relative_file, sha256, verify_inventory
        from cpu_vta_partitioning.measurement import frames

        expected = validate_document(document, self.bundle)
        if expected["bundle_id"] != self.bundle_id:
            raise ValueError("wrong pilot reference bundle")
        name = f"accepted/{boot}/{expected['task_id']}.json"
        ref = value.get("_receipt", {})
        path = relative_file(self.root, name)
        if ref.get("path") != name or not path.is_file() or sha256(path) != ref.get("sha256"):
            raise ValueError("receipt missing, altered, or from a different boot/task")
        receipt = read(path)
        if receipt["task"] != expected or receipt["boot_id"] != boot:
            raise ValueError("receipt task or boot identity differs")
        raw = relative_file(self.root, f"raw/{boot}/{expected['task_id']}")
        files = receipt["files"]
        if not files or any(not relative_file(self.root, n).is_relative_to(raw) for n in files):
            raise ValueError("receipt raw scope differs")
        verify_inventory(self.root, files)
        logs = [relative_file(self.root, n) for n in files if Path(n).name == "result.jsonl"]
        if len(logs) != 1:
            raise ValueError("duplicate or missing native frame file")
        rawdir = logs[0].parent
        if str((rawdir / "affinity.json").relative_to(self.root)) not in files:
            raise ValueError("missing bound affinity file")
        if expected["kind"].startswith("qualify"):
            total = sum(expected[k] for k in ("warmup", "scored", "tail"))
            for row in frames(logs[0], total):
                for dump in row["stage_raw_outputs"]:
                    for output in dump["output_files"]:
                        p = rawdir / "outputs" / ("run_" + str(row["completion_index"])) / Path(output["path"]).name
                        if str(p.relative_to(self.root)) not in files:
                            raise ValueError("correctness output not bound by receipt")
        calculated = validate_summary(rawdir, expected, self.bundle)
        if receipt["summary"] != calculated or calculated != {k: v for k, v in value.items() if k != "_receipt"}:
            raise ValueError("summary disagrees with verified raw data")
        return receipt


def _measure(self, document, bundle, remote_bundle, run_dir, boot):
    """Detached jobs resume by identity; accepted windows are never relaunched."""
    validate_document(document, bundle)
    from cpu_vta_partitioning.common import digest, read, write
    from cpu_vta_partitioning.prepare.payload import job_storage_bytes
    from cpu_vta_partitioning.transport import TransportError
    from cpu_vta_partitioning.transport.scripts import session_script

    root = Path(run_dir) / "raw" / boot / document["task_id"]
    root.mkdir(parents=True, exist_ok=True)
    remote = self.job_path(document, run_dir, boot)
    expected = digest(document)
    q = shlex.quote
    taskfile = root / "task.json"
    if taskfile.exists():
        if read(taskfile) != document:
            raise ValueError("task identity collision")
    else:
        write(taskfile, document)
    if self.boot_id() != boot:
        raise ValueError("pilot boot changed; retain incomplete evidence")
    self.command("mkdir -p " + q(remote))
    argv, inputs = native_command(document, bundle, remote_bundle, remote)
    (root / "inputs.txt").write_text(inputs)
    (root / "execute.sh").write_text(session_script(self.config["remote_base"], remote, boot, expected, argv))
    existing = self.command("test -f " + q(remote + "/task_hash") + " && cat " + q(remote + "/task_hash") + " || true").strip()
    if existing and existing != expected:
        raise ValueError("remote task hash differs")
    try:
        if not existing:
            policy = read(Path(bundle) / "plans/board_execution.json")
            required = job_storage_bytes(document, read(Path(bundle) / "plans/programs.json"))
            self.check_space(remote, required + policy["storage_margin_bytes"])
            self.upload(root / "inputs.txt", remote + "/inputs.txt")
            self.upload(root / "execute.sh", remote + "/execute.sh")
            self.command("cd " + q(remote) + " && nohup sh execute.sh </dev/null > launcher.log 2>&1 &")
        deadline = time.monotonic() + 660
        while True:
            result = self.command("test -f " + q(remote + "/exit_code") + " && cat " + q(remote + "/exit_code") + " || true").strip()
            if result:
                break
            if time.monotonic() >= deadline:
                raise TransportError("native pilot session deadline exceeded")
            time.sleep(.5)
    except TransportError as error:
        raise TransportError(str(error) + "; retained " + self.failure_snapshot(remote, root)) from error
    output = self.collect(remote, root)
    if ((output / "boot_id").read_text().strip() != boot
            or (output / "task_hash").read_text().strip() != expected or self.boot_id() != boot):
        raise ValueError("collected task or boot identity differs")
    if result != "0":
        raise ValueError("native pilot failed; raw evidence retained: " + str(output))
    if (output / "inputs.txt").read_text() != inputs:
        raise ValueError("collected pilot input sequence differs")
    return output


@lru_cache(maxsize=4)
def _ssh_class(bundle):
    _legacy(bundle)
    from cpu_strategy.policy.transport import SSH as FrozenSSH

    class PilotSSH(FrozenSSH):
        measure = _measure

    return PilotSSH


def SSH(config, *, execute_board=False, bundle=None):
    """Reuse the audited mount/deploy/collect lifecycle with the pilot builder."""
    from .pilot_common import DEFAULT_BUNDLE
    instance = _ssh_class(str(Path(bundle or DEFAULT_BUNDLE).resolve()))(config, execute_board=execute_board)
    # This changes connection setup overhead only; each board job keeps its
    # existing lock, task identity, boot check, and detached execution lifecycle.
    instance.options += ["-o", "ControlMaster=auto", "-o", "ControlPersist=60",
                         "-o", "ControlPath=/tmp/cpu-vta-pilot-%C"]
    return instance
