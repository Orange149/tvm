#!/usr/bin/env python3
"""Restore reboot-volatile prerequisites without touching SD files or frozen code."""
import argparse
from pathlib import Path
import subprocess
import time
import shlex

from run_vta_c3_reuse_supplement import ROOT, verify, read, write, append, seal, file_hash


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes",
            "-o", "UserKnownHostsFile=" + args.known_hosts, "-o", "HostKeyAlgorithms=+ssh-rsa",
            "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa"]

    def command(argv):
        begin = time.monotonic()
        result = subprocess.run(argv, capture_output=True, text=True, timeout=90)
        append(output / "results.jsonl", {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "wall_seconds": time.monotonic() - begin})
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout or str(argv))
        return result.stdout

    def ssh(script):
        return command(["ssh", *opts, "root@" + args.host, script])

    append(output / "timeline.jsonl", {"event": "W0", "wall_unix": time.time()})
    try:
        c1 = Path(args.c1_dir).resolve()
        verify(c1)
        before = read(c1 / "preflight.json")
        expected_kernel = next(s.split("Linux version ", 1)[1] for s in before["raw"].splitlines()
                               if "Linux version " in s)
        bit = ROOT / "apps/vta_rpc/firmware/vta_hpc.bit"
        module = ROOT / "u-dma-buf.ko"
        if file_hash(bit) != before["bitstream_sha256"]:
            raise ValueError("bitstream differs from C1")
        if file_hash(module) != "4c0aa5713225cd9b29548d9c1cd6e6127aeaafcc1f7d8650d68e3a0a527042c0":
            raise ValueError("unrecognized host module")
        q = shlex.quote
        kernel = ssh("cat /proc/version").strip()
        if kernel != "Linux version " + expected_kernel:
            raise ValueError("kernel build differs from C1")
        boot = ssh("cat /proc/sys/kernel/random/boot_id").strip()
        if boot == before["boot_id"]:
            raise ValueError("actual reboot required")
        remote = "/var/volatile/" + output.name
        write(output / "contract.json", {"phase": "reboot_prerequisites", "boot_id": boot,
                "c1_directory": str(c1), "c1_manifest_sha256": file_hash(c1 / "artifact_hashes.json"),
                "kernel": kernel, "bitstream_sha256": file_hash(bit), "module_sha256": file_hash(module),
                "module_C1_binary_attested": False, "source_sha256": file_hash(__file__),
                "SD_writes": False, "remote_directory": remote})
        ssh("mount | grep -F 'none on / type rootfs (rw)' && "
            "mount | grep -F 'tmpfs on /var/volatile type tmpfs' && "
            "test ! -e /lib/firmware/vta_hpc.bit && test ! -e /dev/udmabuf0 && "
            "test ! -e " + q(remote) + " && mkdir " + q(remote))
        command(["scp", "-O", *opts, str(bit), str(module), "root@" + args.host + ":" + remote + "/"])
        ssh("test \"$(sha256sum " + q(remote + "/vta_hpc.bit") + " | cut -d' ' -f1)\" = " + file_hash(bit)
            + " && test \"$(sha256sum " + q(remote + "/u-dma-buf.ko") + " | cut -d' ' -f1)\" = " + file_hash(module))
        # / is verified RAM rootfs. The symlink and module live only until reboot.
        ssh("ln -s " + q(remote + "/vta_hpc.bit") + " /lib/firmware/vta_hpc.bit && "
            "/sbin/insmod " + q(remote + "/u-dma-buf.ko") + " udmabuf0=201326592")
        state = ssh("cat /sys/class/u-dma-buf/udmabuf0/size && "
                    "cat /sys/class/u-dma-buf/udmabuf0/phys_addr && "
                    "sha256sum /lib/firmware/vta_hpc.bit && dmesg | tail -n 12")
        if state.splitlines()[0] != "201326592":
            raise ValueError("unexpected shared buffer capacity")
        write(output / "summary.json", {"status": "VOLATILE_PREREQUISITES_RESTORED", "boot_id": boot,
                "wall_seconds": time.monotonic() - started, "state": state,
                "qualification_claim": False, "SD_writes": False})
        print(state, flush=True)
    except BaseException as error:
        write(output / "invalid_session.json", {"message": str(error), "all_timing_invalid": True})
        raise
    finally:
        append(output / "timeline.jsonl", {"event": "T1", "elapsed_seconds": time.monotonic() - started})
        seal(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--c1-dir", required=True)
    p.add_argument("--host", default="192.168.1.247")
    p.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    run(p.parse_args())
