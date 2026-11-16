"""Frozen task templates; candidate selection only changes module composition."""
from pathlib import Path
from functools import lru_cache
import shlex

from cpu_vta_partitioning.common import digest, read
from cpu_vta_partitioning.domain import Stage, signatures, configurations, topologies, candidate_id


@lru_cache(maxsize=8)
def _manifest_file(path, modified_ns, size):
    return read(path)


def manifest_file(path):
    path = Path(path).resolve()
    info = path.stat()
    return _manifest_file(str(path), info.st_mtime_ns, info.st_size)


def task(kind, stages, environment, *, warmup=20, scored=300, tail=20, indices=None, **labels):
    stages = [s.to_dict() if isinstance(s, Stage) else s for s in stages]
    if indices is None:
        indices = list(range(16,64))
    document = {"kind": kind, "stages": stages, "environment": environment,
                "input_indices": indices, "warmup": warmup, "scored": scored, "tail": tail, **labels}
    document["task_id"] = digest(document)[:24]
    return document


def calibration_templates(programs, environment):
    return [task("calibration", [s], environment, warmup=5, scored=32, tail=0,
                 indices=list(range(16)), signature=s.signature)
            for s in signatures(programs, environment["cores"]).values()]


def initial_anchor(programs, environment):
    # Deterministic, chosen without looking at any measured costs.
    topology = min(topologies(programs), key=lambda t: (len(t), tuple(p["name"] for p in t)))
    return next(configurations(topology, environment["cores"]))


def native_command(document, bundle, remote_bundle, remote_job):
    programs = manifest_file(Path(bundle) / "plans/programs.json")
    inputs = manifest_file(Path(bundle) / "plans/inputs.json")["records"]
    serial = document["kind"] in ("calibration", "qualify_component")
    correct = document["kind"].startswith("qualify")
    detailed = document.get("detailed", False)
    stage_rows = document["stages"]
    # Invoke the packaged SDK loader too: the board need not supply matching libc.
    args = [remote_bundle + "/runtime/ld-linux-aarch64.so.1", "--library-path", remote_bundle + "/runtime",
            remote_bundle + "/runner/vta_stage_pipeline_runner"]
    for i, s in enumerate(stage_rows):
        p = programs[s["program"]]
        prefix = f"--stage{i}-"
        for key, value in (("name", s["program"]), ("device", s["device"]),
                           ("graph", f"{remote_bundle}/programs/{s['program']}/graph.json"),
                           ("lib", f"{remote_bundle}/programs/{s['program']}/module.so"),
                           ("params", f"{remote_bundle}/programs/{s['program']}/params.bin"),
                           ("input-names", ",".join(p["input_names"])),
                           ("runtime-num-threads", s["threads"]), ("cpu-affinity", ",".join(map(str,s["mask"])))):
            args += [prefix + key, str(value)]
        if i:
            previous = programs[stage_rows[i-1]["program"]]["output_node_ids"]
            sources = [f"stage{i-1}:{previous.index(node)}" for node in p["input_node_ids"]]
            args += [prefix + "input-sources", ",".join(sources)]
    first = programs[stage_rows[0]["program"]]
    lines = []
    for index in document["input_indices"]:
        rec = inputs[index]
        paths = [rec["input"]] if first["begin"] == 0 else [rec["frontiers"][str(n)]["raw"] for n in first["input_node_ids"]]
        lines.append(",".join(remote_bundle + "/" + p for p in paths))
    runs = document["warmup"] + document["scored"] + document["tail"]
    args += ["--input-list", remote_job + "/inputs.txt", "--input-list-multi", "--runs", str(runs),
             "--warmup-runs", "0", "--queue-depth", "2", "--runtime-num-threads", str(stage_rows[0]["threads"]),
             "--process-timeout-s", "600", "--defer-result-write", "--cpu-stage-gate", "off",
             "--output-jsonl", remote_job + "/result.jsonl", "--thread-affinity-report", remote_job + "/affinity.json",
             "--output-mode", "raw_all_stages" if correct else "raw" if serial else "classification"]
    if serial:
        args.append("--serial")
    if correct:
        args += ["--output-dump-dir", remote_job + "/outputs"]
    if detailed:
        args += ["--queue-timing", "--session-profile-json", remote_job + "/profile.json"]
        args += ["--session-profile-window", f"{document['warmup']},{document['scored']}"]
    e = document["environment"]
    env = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LD_LIBRARY_PATH": remote_bundle + "/runtime",
           "LD_PRELOAD": remote_bundle + "/runtime/libvta.so", "TVM_NUM_THREADS": str(stage_rows[0]["threads"]),
           "TVM_THREAD_POOL_SPIN_COUNT": "0", "VTA_REPLAY_POLICY": "disabled", "VTA_QUEUE_DIAGNOSTICS": "0",
           "C3_COMPLETION_TRACE": "0", "AXU5EVB_DRIVER_POLL_SLEEP_NS": str(e["poll_sleep_ns"]),
           "AXU5EVB_DRIVER_POST_START_SLEEP_NS": str(e["post_start_sleep_ns"])}
    return ["env", "-i", *[f"{k}={v}" for k,v in env.items()], *args], "\n".join(lines) + "\n"
