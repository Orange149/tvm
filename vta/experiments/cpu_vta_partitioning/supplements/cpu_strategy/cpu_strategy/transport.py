"""Audited SSH lifecycle with the focused runner argument adaptation."""
from pathlib import Path
import shlex
import time

from cpu_vta_partitioning.common import digest, read, write
from cpu_vta_partitioning.measurement.tasks import native_command as base_command
from cpu_vta_partitioning.prepare.payload import job_storage_bytes
from cpu_vta_partitioning.transport import SSH as BaseSSH, TransportError
from cpu_vta_partitioning.transport.scripts import session_script


def native_command(document, bundle, remote_bundle, remote_job):
    if (document.get("experiment") != "resnet18_cpu_strategy"
            or document.get("execution_mode") != "off"
            or document.get("allow_cpu_affinity_superset") is not True):
        raise ValueError("task is not a focused CPU strategy execution")
    argv, inputs = base_command(document, bundle, remote_bundle, remote_job)
    argv.append("--allow-cpu-affinity-superset")
    return argv, inputs


class SSH(BaseSSH):
    def measure(self, document, bundle, remote_bundle, run_dir, boot):
        """A detached job is never relaunched after SSH loss or successful collection."""
        root = Path(run_dir) / "raw" / boot / document["task_id"]
        root.mkdir(parents=True, exist_ok=True)
        remote = self.job_path(document, run_dir, boot)
        q = shlex.quote
        expected = digest(document)
        if (root / "task.json").exists():
            if read(root / "task.json") != document:
                raise ValueError("task identity collision")
        else:
            write(root / "task.json", document)
        if self.boot_id() != boot:
            raise TransportError("boot changed during a measurement batch")
        self.command("mkdir -p " + q(remote))
        argv, inputs = native_command(document, bundle, remote_bundle, remote)
        (root / "inputs.txt").write_text(inputs)
        (root / "execute.sh").write_text(session_script(
            self.config["remote_base"], remote, boot, expected, argv))
        existing = self.command("test -f " + q(remote + "/task_hash") + " && cat " +
                                q(remote + "/task_hash") + " || true").strip()
        if existing and existing != expected:
            raise TransportError("remote task hash mismatch")
        try:
            if not existing:
                policy = read(Path(bundle) / "plans/board_execution.json")
                required = job_storage_bytes(document, read(Path(bundle) / "plans/programs.json"))
                self.check_space(remote, required + policy["storage_margin_bytes"])
                self.upload(root / "inputs.txt", remote + "/inputs.txt")
                self.upload(root / "execute.sh", remote + "/execute.sh")
                self.command("cd " + q(remote) +
                             " && nohup sh execute.sh </dev/null > launcher.log 2>&1 &")
            deadline = time.monotonic() + 660
            while True:
                result = self.command("test -f " + q(remote + "/exit_code") + " && cat " +
                                      q(remote + "/exit_code") + " || true").strip()
                if result:
                    break
                if time.monotonic() >= deadline:
                    raise TransportError("remote session deadline exceeded")
                time.sleep(2)
        except TransportError as error:
            saved = self.failure_snapshot(remote, root)
            raise TransportError(str(error) + "; incomplete evidence: " + saved) from error
        collected = self.collect(remote, root)
        if ((collected / "boot_id").read_text().strip() != boot
                or (collected / "task_hash").read_text().strip() != expected):
            raise TransportError("result identity/boot mismatch")
        if result != "0":
            raise TransportError("native session failed with exit " + result + "; raw files retained")
        if (collected / "inputs.txt").read_text() != inputs:
            raise TransportError("collected input list differs from the frozen task")
        if self.boot_id() != boot:
            raise TransportError("boot changed before result acceptance")
        return collected

    def mount_storage(self):
        """Mount the previously authorized filesystem; never format or repartition."""
        mount = self.config.get("storage_mount")
        if not mount:
            return
        q = shlex.quote
        identity = self.command("/sbin/blkid /dev/mmcblk1p3")
        if ('UUID="b19d08f2-0178-4a5e-a920-121c4e8a7a9b"' not in identity
                or 'TYPE="ext4"' not in identity):
            raise TransportError("experiment filesystem UUID/type mismatch")
        self.command("\n".join([
            "set -eu",
            'if ! awk -v target=' + q(mount) + " '$2 == target {ok=1} END {exit !ok}' /proc/mounts; then",
            "mkdir -p " + q(mount), "mount -t ext4 -o noatime /dev/mmcblk1p3 " + q(mount), "fi",
            "awk -v target=" + q(mount) +
            " '$2 == target && $1 == \"/dev/mmcblk1p3\" && $3 == \"ext4\" && $4 ~ /(^|,)rw(,|$)/ {ok=1} END {exit !ok}' /proc/mounts",
        ]))

    def verify_payload(self, remote, boot):
        self.idle_command("sync; echo 3 > /proc/sys/vm/drop_caches; cd " + shlex.quote(remote) +
                          " && sha256sum -c board.sha256 > verification-after-cache-drop.log",
                          boot=boot, timeout=600)
