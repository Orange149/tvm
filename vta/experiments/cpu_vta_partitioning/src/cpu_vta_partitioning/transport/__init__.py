"""Explicitly gated SSH transport; complete evidence precedes remote cleanup."""
from contextlib import ExitStack
import hashlib
import io
import re
import shlex
import subprocess
import tarfile
import time
from pathlib import Path

from cpu_vta_partitioning.common import digest, read, sha256, write, event, sync_tree
from cpu_vta_partitioning.measurement.tasks import native_command
from cpu_vta_partitioning.prepare.payload import job_storage_bytes
from cpu_vta_partitioning.transport.scripts import session_lock, session_script, dma_check


class TransportError(RuntimeError):
    pass


def extract_checked(archive, destination):
    destination = Path(destination).resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError("unsafe result archive")
        tar.extractall(destination)


class SSH:
    def __init__(self, config, *, execute_board=False):
        if not execute_board:
            raise ValueError("board access requires --execute-board")
        self.config = config
        for key in ("host", "user"):
            if not re.fullmatch(r"[A-Za-z0-9_.:@-]+", config[key]) or config[key].startswith("-"):
                raise ValueError("invalid SSH endpoint")
        base = Path(config["remote_base"])
        if not base.is_absolute() or "\n" in str(base) or ".." in base.parts or str(base) == "/":
            raise ValueError("remote_base must be an absolute experiment directory")
        mount = config.get("storage_mount")
        if mount and (not Path(mount).is_absolute() or base == Path(mount) or not base.is_relative_to(mount)):
            raise ValueError("remote_base must be beneath storage_mount")
        self.destination = config["user"] + "@" + config["host"]
        self.options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=" + str(config.get("connect_timeout_s",10))]
        if config.get("legacy_rsa"):
            self.options += ["-o", "HostKeyAlgorithms=+ssh-rsa", "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa"]
        if config.get("identity_file"):
            self.options += ["-i", str(Path(config["identity_file"]).expanduser())]

    def command(self, script, *, timeout=60, output=None, input_path=None):
        # Non-interactive PetaLinux SSH sessions may omit /sbin, including insmod.
        script = "export PATH=/usr/sbin:/usr/bin:/sbin:/bin\n" + script
        argv = ["ssh", *self.options, "-p", str(self.config.get("port",22)), self.destination,
                "sh -c " + shlex.quote(script)]
        try:
            with ExitStack() as stack:
                source = stack.enter_context(Path(input_path).open("rb")) if input_path else subprocess.DEVNULL
                target = stack.enter_context(Path(output).open("wb")) if output else subprocess.PIPE
                result = subprocess.run(argv, stdin=source, stdout=target, stderr=subprocess.PIPE, timeout=timeout)
            if result.returncode:
                raise TransportError(result.stderr.decode(errors="replace")[-2000:])
            return result.stdout.decode() if output is None else None
        except subprocess.TimeoutExpired as e:
            raise TransportError("SSH timeout; remote job may still be running; resume without deleting it") from e

    def upload(self, source, target):
        argv = ["scp", *(["-O"] if self.config.get("scp_legacy_protocol") else []), *self.options, "-P", str(self.config.get("port",22)), str(source),
                self.destination + ":" + shlex.quote(target)]
        try:
            subprocess.run(argv, check=True, timeout=1800, capture_output=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise TransportError("deployment transfer failed") from e

    def boot_id(self):
        value = self.command("cat /proc/sys/kernel/random/boot_id").strip()
        if not re.fullmatch(r"[0-9a-f-]{36}", value):
            raise TransportError("invalid kernel boot ID")
        return value

    def check_space(self, remote, required):
        available = int(self.command("df -Pk " + shlex.quote(remote)).splitlines()[-1].split()[3]) * 1024
        if available < required:
            raise TransportError(f"board storage has {available} free bytes, needs {required}; no job started")

    def wait_idle(self, boot, timeout=660):
        """Only lightweight polling is allowed while a detached session owns the board."""
        lock = self.config["remote_base"].rstrip("/") + "/locks/" + boot + "/session.lock"
        deadline = time.monotonic() + timeout
        announced = False
        while True:
            result = self.command(
                'test "$(cat /proc/sys/kernel/random/boot_id)" = ' + shlex.quote(boot) +
                ' || exit 99; if test -d ' + shlex.quote(lock) + '; then echo busy; else echo idle; fi').strip()
            if result == "idle":
                return
            if result != "busy":
                raise TransportError("invalid board session status")
            if not announced:
                print("等待板端会话结束；暂缓回收、部署校验和清理。", flush=True)
                announced = True
            if time.monotonic() >= deadline:
                raise TransportError("board session still owns the lock; no bulk I/O performed; wait or reboot and resume")
            time.sleep(2)

    def idle_command(self, script, *, boot=None, **kwargs):
        """Atomically exclude native sessions, including jobs racing the idle check."""
        boot = boot or self.boot_id()
        guarded = "\n".join([
            "set -eu",
            *session_lock(self.config["remote_base"], boot, "maintenance",
                          "echo 'board session active; bulk I/O deferred' >&2; exit 95"),
            script,
        ])
        return self.command(guarded, **kwargs)

    def deploy(self, bundle, run_dir):
        bundle, run_dir = Path(bundle), Path(run_dir)
        manifest = read(bundle / "manifest.json")
        plan = read(bundle / "plans/board_payload.json")
        policy = read(bundle / "plans/board_execution.json")
        remote = self.config["remote_base"].rstrip("/") + "/bundles/" + manifest["bundle_id"]
        q = shlex.quote
        mount = self.config.get("storage_mount")
        if mount:
            # Fail before mkdir when the SD mount is absent; never fall back to rootfs.
            self.command("awk -v wanted=" + q(mount) + " '$2 == wanted && $4 ~ /(^|,)rw(,|$)/ {ok=1} END {exit !ok}' /proc/mounts && test -w " + q(mount))
        boot = self.boot_id()
        self.wait_idle(boot)
        self.command("mkdir -p " + q(remote))
        files = dict(plan["files"])
        for name in ("plans/board_payload.json", "manifest.json"):
            files[name] = {"sha256":sha256(bundle / name),"bytes":(bundle / name).stat().st_size}
        checksums = "".join(v["sha256"] + "  " + k + "\n" for k,v in sorted(files.items())).encode()
        checksum_hash = hashlib.sha256(checksums).hexdigest()
        verify = "cd " + q(remote) + " && test \"$(sha256sum board.sha256 | cut -d ' ' -f 1)\" = " + q(checksum_hash) + " && sha256sum -c board.sha256 > verification.log"
        present = self.command("test -f " + q(remote + "/.deployed") + " && cat " + q(remote + "/.deployed") + " || true").strip()
        if present != sha256(bundle / "manifest.json"):
            # No transfer archive is stored on the board. On an interrupted upload
            # only missing or changed payload bytes need additional free space.
            missing = self.idle_command("cd " + q(remote) + " && if test -f board.sha256 && test \"$(sha256sum board.sha256 | cut -d ' ' -f 1)\" = " + q(checksum_hash) + "; then sha256sum -c board.sha256 2>/dev/null || true; fi", boot=boot, timeout=600)
            good = {line[:-4] for line in missing.splitlines() if line.endswith(": OK")}
            needed = {k:v for k,v in files.items() if k not in good}
            self.check_space(remote, sum(v["bytes"] for v in needed.values()) + policy["storage_margin_bytes"])
            archive = run_dir / "deployment.tar.gz"
            with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
                # Publish the authenticated inventory first, so even the very
                # first interrupted stream can identify already-copied files.
                info = tarfile.TarInfo("board.sha256"); info.size = len(checksums)
                tar.addfile(info, io.BytesIO(checksums))
                for name in sorted(needed):
                    tar.add(bundle / name, arcname=name, recursive=False)
            self.idle_command("cd " + q(remote) + " && tar -xzf -", boot=boot, input_path=archive, timeout=1800)
            self.idle_command(verify, boot=boot, timeout=600)
            self.command("printf '%s\\n' " + q(sha256(bundle / "manifest.json")) + " > " + q(remote + "/.deployed"))
        else:
            self.idle_command(verify, boot=boot, timeout=600)
        return remote

    def qualify_environment(self, remote, boot, run_dir, bundle):
        policy = read(Path(bundle) / "plans/board_execution.json")
        marker = Path(run_dir) / ("environment-" + boot + ".json")
        identity = {"boot_id":boot, "bundle_id":read(Path(bundle) / "manifest.json")["bundle_id"]}
        q = shlex.quote
        if marker.exists():
            if read(marker)["identity"] != identity:
                raise TransportError("environment qualification identity mismatch")
            # run_current waits for detached jobs before environment revalidation.
            self.command("\n".join(["set -eu", *dma_check(policy["udmabuf_bytes"])]))
            return
        script = "\n".join([
            "set -eu", 'test "$(uname -m)" = aarch64',
            *session_lock(self.config["remote_base"], boot, "environment-" + identity["bundle_id"]),
            "for p in /proc/[0-9]*/exe; do x=$(readlink \"$p\" 2>/dev/null || true); case \"$x\" in *ld-linux*) x=$(tr '\\000' ' ' < \"${p%exe}cmdline\" 2>/dev/null || true);; esac; case \"$x\" in *vta_stage_pipeline_runner*|*/tvm_rpc) echo 'VTA workload active' >&2; exit 1;; esac; done",
            "test -d /sys/module/u_dma_buf || insmod " + q(remote + "/firmware/u-dma-buf.ko") + " udmabuf0=" + str(policy["udmabuf_bytes"]),
            *dma_check(policy["udmabuf_bytes"]),
            "install -m 644 " + q(remote + "/firmware/vta_hpc.bit") + " /lib/firmware/vta_hpc.bit",
            "printf '0' > /sys/class/fpga_manager/fpga0/flags",
            "printf 'vta_hpc.bit\\n' > /sys/class/fpga_manager/fpga0/firmware",
            'test "$(cat /sys/class/fpga_manager/fpga0/state)" = operating',
            "uname -a; cat /proc/cmdline; cat /sys/devices/system/cpu/online",
            "cat /sys/class/u-dma-buf/udmabuf0/size /sys/class/u-dma-buf/udmabuf0/phys_addr",
            "for x in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq /sys/class/thermal/thermal_zone*/temp; do test ! -f \"$x\" || { printf '%s ' \"$x\"; cat \"$x\"; }; done",
        ])
        result = self.command(script)
        write(marker, {"identity":identity,"raw":result})

    def job_path(self, document, run_dir, boot):
        # Only generated identifiers can reach cleanup paths.
        for value in (document["bundle_id"], Path(run_dir).name, boot, document["task_id"]):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise ValueError("invalid remote job identity")
        return self.config["remote_base"].rstrip("/") + "/jobs/" + document["bundle_id"] + "/" + Path(run_dir).name + "/" + boot + "/" + document["task_id"]

    def collect(self, remote, root, *, incomplete=False):
        snapshot = Path(root) / "transfers" / str(time.time_ns())
        snapshot.mkdir(parents=True)
        archive = snapshot / "result.tar"
        error = None
        try:
            self.idle_command("tar -cf - -C " + shlex.quote(remote) + " .", output=archive, timeout=180)
        except TransportError as e:
            error = str(e)
        try:
            extract_checked(archive, snapshot / "collected")
        except (OSError, tarfile.TarError, ValueError) as e:
            error = (error or "") + "; extraction: " + str(e)
        write(snapshot / "capture.json", {"incomplete":incomplete,"transport_error":error})
        sync_tree(snapshot)
        if error and not incomplete:
            raise TransportError("result collection failed: " + error)
        return snapshot / "collected"

    def failure_snapshot(self, remote, root):
        try:
            saved = self.collect(remote, root, incomplete=True)
            error = read(saved.parent / "capture.json")["transport_error"]
            return str(saved) + (" (collection incomplete: " + error + ")" if error else "")
        except (TransportError, OSError, ValueError) as error:
            return "collection unavailable: " + str(error)

    def recover_abandoned(self, run_dir, boot):
        for task_file in sorted((Path(run_dir) / "raw" / boot).glob("*/task.json")):
            document = read(task_file)
            if (Path(run_dir) / "accepted" / boot / (document["task_id"] + ".json")).exists():
                continue
            saved = self.failure_snapshot(self.job_path(document, run_dir, boot), task_file.parent)
            event(run_dir, "abandoned_evidence_recovery", boot_id=boot, task_id=document["task_id"], snapshot=saved)

    def acknowledge(self, document, run_dir, boot):
        """Called only after a hash-verified, durable local acceptance receipt."""
        remote = self.job_path(document, run_dir, boot)
        q = shlex.quote
        script = "\n".join([
            "set -eu", "cd " + q(remote),
            'test "$(cat boot_id)" = ' + q(boot),
            'test "$(cat task_hash)" = ' + q(digest(document)),
            'test "$(cat exit_code)" = 0',
            # Keep small identity/exit tombstones. Cleanup is idempotent even if
            # its SSH response is lost, and cannot relaunch an accepted task.
            "printf '%s\\n' " + q(digest(document)) + " > collected_by_host",
            "rm -rf outputs",
            "rm -f result.jsonl native.log affinity.json profile.json inputs.txt execute.sh launcher.log nice.log",
        ])
        self.idle_command(script)

    def measure(self, document, bundle, remote_bundle, run_dir, boot):
        """Remote jobs survive SSH loss; complete results are fetched idempotently."""
        root = Path(run_dir) / "raw" / boot / document["task_id"]
        root.mkdir(parents=True, exist_ok=True)
        remote = self.job_path(document, run_dir, boot)
        q = shlex.quote
        expected = digest(document)
        if not (root / "task.json").exists():
            write(root / "task.json", document)
        elif read(root / "task.json") != document:
            raise ValueError("task identity collision")
        if self.boot_id() != boot:
            raise TransportError("boot changed during a measurement batch; retain this incomplete batch")
        self.command("mkdir -p " + q(remote))
        argv, inputs = native_command(document, bundle, remote_bundle, remote)
        (root / "inputs.txt").write_text(inputs)
        (root / "execute.sh").write_text(session_script(self.config["remote_base"], remote, boot, expected, argv))
        existing = self.command("test -f " + q(remote + "/task_hash") + " && cat " + q(remote + "/task_hash") + " || true").strip()
        if existing and existing != expected:
            raise TransportError("remote task hash mismatch")
        try:
            if not existing:
                policy = read(Path(bundle) / "plans/board_execution.json")
                required = job_storage_bytes(document, read(Path(bundle) / "plans/programs.json")) + policy["storage_margin_bytes"]
                self.check_space(remote, required)
                self.upload(root / "inputs.txt", remote + "/inputs.txt")
                self.upload(root / "execute.sh", remote + "/execute.sh")
                self.command("cd " + q(remote) + " && nohup sh execute.sh </dev/null > launcher.log 2>&1 &")
            deadline = time.monotonic() + 660
            while True:
                result = self.command("test -f " + q(remote + "/exit_code") + " && cat " + q(remote + "/exit_code") + " || true").strip()
                if result:
                    break
                if time.monotonic() >= deadline:
                    raise TransportError("remote session deadline exceeded")
                time.sleep(2)
        except TransportError as error:
            saved = self.failure_snapshot(remote, root)
            raise TransportError(str(error) + "; incomplete evidence: " + saved) from error
        collected = self.collect(remote, root)
        if (collected / "boot_id").read_text().strip() != boot or (collected / "task_hash").read_text().strip() != expected:
            raise TransportError("result identity/boot mismatch")
        if result != "0":
            raise TransportError("native session failed with exit " + result + "; raw result retained at " + str(root))
        if (collected / "inputs.txt").read_text() != inputs:
            raise TransportError("collected input list differs from frozen task")
        if self.boot_id() != boot:
            raise TransportError("boot changed before result acceptance")
        return collected
