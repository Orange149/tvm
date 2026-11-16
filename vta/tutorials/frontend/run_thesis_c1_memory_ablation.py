#!/usr/bin/env python3
"""Collect one fresh kernel boot of the frozen paired C1 memory study.

Uses the actual native runner, with independent diagnostic and timing processes.
It never promotes a single-island or single-input historical certificate.  New
boots are attached explicitly; the controller does not reboot unrelated work.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import shlex
import tarfile
import uuid

from thesis_c1_memory_ablation import (
    BOOT_SCHEMA, file_hash, read_json, seal, validate_contract, verify_ref, write_new,
)
from thesis_c1_memory_evidence import inputs_manifest, validate_performance, validate_qualification
from thesis_supplement_run import RunStore


def native_environment(contract, remote, diagnostic):
    q = contract["queue_capacity_bytes"]
    return {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LD_LIBRARY_PATH": remote,
            "LD_PRELOAD": remote + "/libvta.so", "TVM_NUM_THREADS": "4",
            "TVM_THREAD_POOL_SPIN_COUNT": "0", "VTA_REPLAY_POLICY": "disabled",
            "VTA_INSN_BUFFER_BYTES": str(q["instruction"]), "VTA_UOP_BUFFER_BYTES": str(q["uop"]),
            "VTA_QUEUE_DIAGNOSTICS": "1" if diagnostic else "0", "C3_COMPLETION_TRACE": "1" if diagnostic else "0",
            "AXU5EVB_DRIVER_POST_START_SLEEP_NS": "1000", "AXU5EVB_DRIVER_POLL_SLEEP_NS": "1000"}


def session_argv(contract, config, remote, input_list, output, diagnostic, capacity=None, boundary=False):
    """Native argv is structured until the final, shell-quoted SSH boundary."""
    obj = lambda ref, suffix="": remote + "/objects/" + ref["sha256"] + suffix
    args = [remote + "/runner"]
    for index, (stage, count) in enumerate(zip(config["identity"]["stages"], config["identity"]["thread_vector"])):
        prefix = "--stage{}-".format(index)
        args += [prefix + "name", "stage{}".format(index), prefix + "device", stage["device"],
                 prefix + "graph", obj(stage["graph"]), prefix + "lib", obj(stage["lib"], ".so"),
                 prefix + "params", obj(stage["params"]), prefix + "input-names", ",".join(stage["input_names"]),
                 prefix + "runtime-num-threads", str(count)]
        if stage.get("input_sources"):
            args += [prefix + "input-sources", ",".join(stage["input_sources"])]
        if stage["device"] == "cpu":
            args += [prefix + "cpu-affinity", ",".join(map(str, stage["cpu_affinity"]))]
    args += ["--input-list", remote + "/" + input_list,
             "--runs", "64" if boundary else "67" if diagnostic else "307",
             # Native --warmup-runs executes on the serial main-thread path.
             # Warm the actual pipeline workers with five explicit prefix rows.
             "--warmup-runs", "0", "--queue-depth", "2",
             "--process-timeout-s", "285",
             "--runtime-num-threads", "4", "--defer-result-write",
             "--output-mode", "raw_all_stages" if diagnostic else "classification",
             "--output-jsonl", output + "/result.jsonl"]
    if config["policy"] != "ordinary_copy":
        args += ["--p8-managed-slots", "2", "--p8-slot-timeout-s", "60",
                 "--shared-buffer-plan", obj(config["identity"]["shared_plan"]),
                 "--shared-slot-budget-bytes", str(config["additional_slot_bytes"] if capacity is None else capacity)]
    if diagnostic:
        args += ["--c1-memory-audit", "--output-dump-dir", output + "/outputs",
                 "--allocation-snapshot", output + "/allocations.jsonl",
                 "--vta-runtime-profile-dir", output + "/profile",
                 "--vta-runtime-profile-events-limit", "0"]
    return args


def safe_extract(archive, destination):
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if (not target.is_relative_to(destination.resolve()) or
                    not (member.isfile() or member.isdir())):
                raise ValueError("unsafe result archive member")
        # Members have already been restricted to ordinary files/directories;
        # do not require Python 3.12's optional extraction-filter API.
        tar.extractall(destination)


def session_sources(directory):
    return {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def reset_script(contract, remote, boot_id):
    firmware = "c1_memory_" + contract["artifacts"]["bitstream"]["sha256"][:16] + ".bit"
    return "\n".join([
        "test \"$(cat /proc/sys/kernel/random/boot_id)\" = " + shlex.quote(boot_id),
        "for p in /proc/[0-9]*/exe; do x=$(readlink \"$p\" 2>/dev/null || true); case \"$x\" in *vta_stage_pipeline_runner*|*/runner) echo 'active VTA native process' >&2; exit 1;; esac; done",
        "for p in /proc/[0-9]*/exe; do x=$(readlink \"$p\" 2>/dev/null || true); case \"$x\" in */tvm_rpc) d=${p%/exe}; n=${d##*/}; c=$(cat \"$d/task/$n/children\" 2>/dev/null || true); test -z \"$c\" || { echo 'active RPC child' >&2; exit 1; };; esac; done",
        "ln -sf {} {}".format(shlex.quote(remote + "/vta_hpc.bit"), shlex.quote("/lib/firmware/" + firmware)),
        "printf '%s\\n' {} > /sys/class/fpga_manager/fpga0/firmware".format(shlex.quote(firmware)),
        "test \"$(cat /sys/class/fpga_manager/fpga0/state)\" = operating",
        "printf '%s\\n' " + shlex.quote("C1_CONTEXT_RESET_READY " + boot_id + " " + contract["artifacts"]["bitstream"]["sha256"]),
    ])


def native_session_script(boot_id, output, args, env):
    command = ["env", "-i"] + [k + "=" + env[k] for k in sorted(env)] + args
    return "\n".join([
        "test \"$(cat /proc/sys/kernel/random/boot_id)\" = " + shlex.quote(boot_id),
        "mkdir " + shlex.quote(output),
        "unset VTA_INSN_SUBMIT_THRESHOLD_BYTES VTA_COMMAND_MANIFEST_ID VTA_RUNTIME_TRACE THESIS_OUTPUT_STABILITY_DIR",
        "if {} > {} 2>&1; then code=0; else code=$?; fi".format(shlex.join(command), shlex.quote(output + "/native.log")),
        "tar -cf - -C {} .".format(shlex.quote(output)),
        "exit \"$code\"",
    ])


class Collector:
    def __init__(self, contract, store, host, index, records):
        self.contract, self.store, self.index, self.records = contract, store, index, records
        self.ssh = ssh_argv(host)
        self.remote = "/var/volatile/c1_memory_{}_b{}_{}".format(contract["artifact_sha256"][:12], index, uuid.uuid4().hex[:8])
        self.boot_id = None

    def execute(self, script, name, cost="functional_qualification", timeout=60, check=True):
        path = self.store.directory / "raw" / (name + ".sh")
        with path.open("x") as stream:
            stream.write("set -eu\n" + script + "\n")
        result = self.store.execute(self.ssh + ["sh", "-s"], name, cost, timeout, path)
        if check and result["status"] != "ok":
            raise RuntimeError("{}: {}; see {}".format(name, result["status"], result["stderr"]))
        return result

    def identity(self):
        result = self.execute("cat /proc/sys/kernel/random/boot_id", "kernel_identity", "preparation")
        self.boot_id = (self.store.directory / result["stdout"]).read_text().strip()
        uuid.UUID(self.boot_id)
        return self.boot_id

    def deploy(self):
        files = {}
        for name in ("runner", "runtime", "driver", "bitstream"):
            ref = self.contract["artifacts"][name]
            filename = {"runtime": "libtvm_runtime.so", "driver": "libvta.so", "bitstream": "vta_hpc.bit"}.get(name, name)
            files[filename] = ref
        for config in self.contract["support"]:
            for stage in config["identity"]["stages"]:
                for field in ("graph", "lib", "params"):
                    ref = stage[field]
                    files["objects/" + ref["sha256"] + (".so" if field == "lib" else "")] = ref
            plan = config["identity"].get("shared_plan")
            if plan:
                files["objects/" + plan["sha256"]] = plan
        for row in self.records:
            ref = row["input"]
            files["objects/" + ref["sha256"]] = ref
        bundle = self.store.directory / "raw/deployment.tar"
        with tarfile.open(bundle, "w") as tar:
            for target, ref in sorted(files.items()):
                verify_ref(ref)
                tar.add(ref["path"], arcname=target, recursive=False)
            for filename, records in (("inputs_all.txt", self.records), ("inputs_real.txt", self.records[:64])):
                content = ("\n".join(self.remote + "/objects/" + r["input"]["sha256"] for r in records) + "\n").encode()
                member = tarfile.TarInfo(filename)
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
        mkdir = "test ! -e {0} && mkdir {0}".format(shlex.quote(self.remote))
        self.execute(mkdir, "create_owned_ram_directory", "deployment")
        upload = self.store.execute(self.ssh + ["cat > " + shlex.quote(self.remote + "/deployment.tar")],
                                    "upload_bundle", "deployment", 300, bundle)
        if upload["status"] != "ok":
            raise RuntimeError("deployment transfer failed")
        lines = ["cd " + shlex.quote(self.remote), "tar -xf deployment.tar"]
        for target, ref in sorted(files.items()):
            lines.append("test \"$(sha256sum {} | cut -d ' ' -f 1)\" = {}".format(shlex.quote(target), shlex.quote(ref["sha256"])))
        lines += ["chmod 755 runner", "test \"$(cat /sys/class/u-dma-buf/udmabuf0/size)\" = 201326592"]
        for cpu in range(4):
            lines += ["test \"$(cat /sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor)\" = userspace".format(cpu),
                      "test \"$(cat /sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq)\" = 1066666".format(cpu)]
        # The host RunStore retains this exact archive and its hash. Keep only
        # the verified extracted objects in constrained board RAM.
        lines.append("rm -f deployment.tar")
        self.execute("\n".join(lines), "verify_deployment_and_environment", "deployment")

    def reset(self, token):
        # Native execution needs no RPC server. Refuse to reset while an existing
        # native inference or forked RPC worker is using the device.
        return self.execute(reset_script(self.contract, self.remote, self.boot_id), token + "_reset", "health_and_reset")

    def session(self, config, token, diagnostic, capacity=None, boundary=False):
        self.reset(token)
        local = self.store.directory / "raw" / token
        local.mkdir()
        output = self.remote + "/" + token
        args = session_argv(self.contract, config, self.remote, "inputs_real.txt" if boundary or not diagnostic else "inputs_all.txt",
                            output, diagnostic, capacity, boundary)
        env = native_environment(self.contract, self.remote, diagnostic)
        script = native_session_script(self.boot_id, output, args, env)
        write_new(local / "session_contract.json", {"boot_id": self.boot_id, "boot_index": self.index,
                   "configuration_id": config["configuration_id"], "argv": args, "environment": env,
                   "diagnostic": diagnostic, "capacity_bytes": capacity,
                   "selection_contract_sha256": self.contract["artifact_sha256"]})
        result = self.execute(script, token, "capacity_qualification" if boundary else
                              "functional_qualification" if diagnostic else "performance", 300, check=False)
        archive = self.store.directory / result["stdout"]
        if archive.stat().st_size:
            safe_extract(archive, local)
        if result["status"] != "ok" and not (boundary and result.get("returncode") == 3):
            raise RuntimeError("native session failed; partial raw files and costs retained: " + token)
        return local, result


def ssh_argv(host):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes",
            "-o", "HostKeyAlgorithms=+ssh-rsa", "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa", host]


def collect(contract_path, output, host, index, prior_paths):
    contract = read_json(contract_path)
    validate_contract(contract)
    records = inputs_manifest(contract)
    prior = [read_json(path) for path in prior_paths]
    if len(prior) != index or {b.get("boot_index") for b in prior} != set(range(index)):
        raise ValueError("supply exactly the preceding boot records")
    if any(b.get("schema") != BOOT_SCHEMA or b.get("selection_contract_sha256") != contract["artifact_sha256"] or
           b.get("status") != "complete" for b in prior):
        raise ValueError("previous boots must be complete and bound to the same contract")
    for b in prior:
        from thesis_c1_memory_ablation import verify_seal
        verify_seal(b)
    store = RunStore.create(output, {"run_id": output.name, "stage": "C1_MEMORY_ABLATION",
                            "campaign_id": contract["artifact_sha256"],
                            "selection_contract": {"path": str(contract_path.resolve()), "sha256": file_hash(contract_path)},
                            "boot_index": index, "board_host": host})
    state = {"schema": BOOT_SCHEMA, "boot_index": index, "boot_id": None,
             "selection_contract_sha256": contract["artifact_sha256"], "status": "collecting",
             "configurations": [{"configuration_id": key, "status": "not_attempted"}
                                for key in contract["measurement_orders"][str(index)]],
             "capacity_boundaries": []}
    controller = Collector(contract, store, host, index, records)
    by_id = {r["configuration_id"]: r for r in contract["support"]}
    try:
        state["boot_id"] = controller.identity()
        if state["boot_id"] in {b["boot_id"] for b in prior}:
            raise ValueError("kernel boot is not independent")
        controller.deploy()
        results = {r["configuration_id"]: r for r in state["configurations"]}
        pairs, qualified_count = {}, 0
        # Qualify the entire pool before starting any performance process.
        for key in contract["measurement_orders"][str(index)]:
            config = by_id[key]
            token = "qual_" + key
            results[key]["status"] = "qualifying"
            directory, _ = controller.session(config, token, True)
            proof = validate_qualification(contract, config, directory, records)
            results[key].update(status="qualified", qualification=token,
                                qualification_files=session_sources(directory), qualification_proof=proof)
            if config["pair_id"] in pairs and pairs[config["pair_id"]] != proof["final_output_sha256"]:
                raise ValueError("copy/shared outputs are not byte-identical")
            pairs[config["pair_id"]] = proof["final_output_sha256"]
            qualified_count += 1
            print("boot {} qualification {}/16".format(index, qualified_count), flush=True)
        for key in contract["capacity_boundary_ids"]:
            config = by_id[key]
            need = config["additional_slot_bytes"]
            for budget in (max(0, need - 256), need):
                token = "capacity_{}_{}".format(key, budget)
                directory, action = controller.session(config, token, True, budget, boundary=True)
                log = (directory / "native.log").read_text()
                if budget < need:
                    if action["returncode"] != 3 or "[C1_SLOT_CAPACITY_REJECTED]" not in log or (directory / "result.jsonl").exists():
                        raise ValueError("capacity rejection did not occur before model execution")
                else:
                    # Boundary sessions are 64 real frames; the same complete
                    # qualification validator applies after a derived 64-frame check.
                    validate_capacity_acceptance(contract, config, directory, records[:64])
                state["capacity_boundaries"].append({"configuration_id": key, "budget_bytes": budget,
                    "required_bytes": need, "session": token, "files": session_sources(directory),
                    "outcome": "rejected_before_execution" if budget < need else "accepted_64_outputs"})
        if contract["identifiability"]["ranking_windows_allowed"]:
            for key in contract["measurement_orders"][str(index)]:
                directory, _ = controller.session(by_id[key], "perf_" + key, False)
                metric = validate_performance(directory, by_id[key])
                results[key].update(performance="perf_" + key, performance_files=session_sources(directory), performance_metrics=metric)
                print("boot {} window {} II={:.6f} ms".format(index, key[:12], metric["ii_ms"]), flush=True)
        else:
            state["ranking_windows_skipped"] = "identical_full_and_M1_top5"
        state["status"] = "complete"
    except BaseException as error:
        state["status"], state["failure"] = "incomplete", str(error)
        for record in state["configurations"]:
            if record["status"] == "qualifying":
                from thesis_c1_memory_evidence import NumericMismatch
                record["status"] = "numeric_failed" if isinstance(error, NumericMismatch) else "invalid_session"
                record["failure"] = str(error)
        write_new(output / "invalid_session.json", {"reason": str(error), "raw_and_paid_costs_retained": True})
        raise
    finally:
        state["cost_actions"] = [e for e in store.events() if e["event"] == "action_end"]
        write_new(output / "boot.json", seal(state))
    return state


def validate_capacity_acceptance(contract, config, directory, records):
    # Reuse the full raw audit without fabricating three diagnostic frames.
    from thesis_c1_memory_evidence import validate_qualification
    return validate_qualification(contract, config, directory, records, expected_frames=64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--boot-index", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--prior-boot", type=Path, action="append", default=[])
    parser.add_argument("--host", default="root@192.168.1.247")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        contract = read_json(args.contract)
        validate_contract(contract)
        inputs_manifest(contract)
        print("LOCAL_PREFLIGHT_PASSED; board not contacted")
        return
    collect(args.contract, args.output_dir, args.host, args.boot_index, args.prior_boot)


if __name__ == "__main__":
    main()
