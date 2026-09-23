#!/usr/bin/env python3
"""Install a reboot-volatile taskset helper for the C3 CPU3 experiment."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time

from run_vta_c3_reuse_supplement import append, file_hash, seal, write


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    opts = [
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + args.known_hosts,
        "-o", "HostKeyAlgorithms=+ssh-rsa", "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
    ]

    def command(argv):
        begin = time.monotonic()
        result = subprocess.run(argv, capture_output=True, text=True, timeout=90)
        append(output / "results.jsonl", {
            "argv": argv, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "wall_seconds": time.monotonic() - begin,
        })
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout or str(argv))
        return result.stdout

    def ssh(script):
        return command(["ssh", *opts, "root@" + args.host, script])

    append(output / "timeline.jsonl", {"event": "W0", "wall_unix": time.time()})
    try:
        binary = Path(args.taskset).resolve()
        binary_hash = file_hash(binary)
        boot = ssh("cat /proc/sys/kernel/random/boot_id").strip()
        remote = "/var/volatile/" + output.name
        q = shlex.quote
        write(output / "contract.json", {
            "phase": "reboot_volatile_taskset", "boot_id": boot,
            "source_sha256": file_hash(__file__), "taskset_path": str(binary),
            "taskset_sha256": binary_hash, "remote_directory": remote,
            "rootfs_write": "reboot-volatile symlink only", "SD_writes": False,
        })
        existing = ssh(
            "mount | grep -F 'none on / type rootfs (rw)' && "
            "mount | grep -F 'tmpfs on /var/volatile type tmpfs' && "
            "test \"$(cat /sys/devices/system/cpu/online)\" = 0-3 && "
            "test -x /lib/ld-linux-aarch64.so.1 && test -r /lib/libc.so.6 && "
            "if test -x /usr/bin/taskset; then "
            "test \"$(sha256sum /usr/bin/taskset | cut -d' ' -f1)\" = " + binary_hash
            + " && echo existing; else test ! -e " + q(remote) + " && mkdir " + q(remote)
            + " && echo absent; fi"
        ).splitlines()[-1]
        if existing == "absent":
            command(["scp", "-O", *opts, str(binary), "root@" + args.host + ":" + remote + "/taskset"])
            ssh(
                "test \"$(sha256sum " + q(remote + "/taskset") + " | cut -d' ' -f1)\" = "
                + binary_hash + " && chmod 755 " + q(remote + "/taskset") + " && "
                + q(remote + "/taskset") + " -c 3 true && "
                "ln -s " + q(remote + "/taskset") + " /usr/bin/taskset"
            )
        elif existing != "existing":
            raise ValueError("unexpected taskset prerequisite state: " + existing)
        state = ssh(
            "command -v taskset && test \"$(sha256sum /usr/bin/taskset | cut -d' ' -f1)\" = "
            + binary_hash + " && taskset -c 3 sh -c \"grep '^Cpus_allowed_list:' /proc/self/status\""
        )
        if state.splitlines()[-1].split()[-1] != "3":
            raise ValueError("taskset did not constrain the process to CPU3")
        write(output / "summary.json", {
            "status": "VOLATILE_TASKSET_RESTORED", "boot_id": boot,
            "taskset_sha256": binary_hash, "state": state,
            "wall_seconds": time.monotonic() - started,
            "qualification_claim": False, "SD_writes": False,
        })
        print(state, flush=True)
    except BaseException as error:
        write(output / "invalid_session.json", {"message": str(error), "all_timing_invalid": True})
        raise
    finally:
        append(output / "timeline.jsonl", {"event": "T1", "elapsed_seconds": time.monotonic() - started})
        seal(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--taskset", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    run(parser.parse_args())
