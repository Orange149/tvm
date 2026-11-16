"""Reuse the audited transport; change only task-to-native argument adaptation."""
import shlex
import time
from pathlib import Path

from cpu_vta_partitioning.common import digest,read,write
from cpu_vta_partitioning.measurement.tasks import native_command as base_command
from cpu_vta_partitioning.prepare.payload import job_storage_bytes
from cpu_vta_partitioning.transport import SSH as BaseSSH,TransportError
from cpu_vta_partitioning.transport.scripts import session_script


def native_command(document,bundle,remote_bundle,remote_job):
    argv,inputs = base_command(document,bundle,remote_bundle,remote_job)
    mode = document.get("execution_mode")
    if mode not in ("off","resource_group"):
        raise ValueError("missing or invalid execution mode")
    argv[argv.index("--cpu-stage-gate")+1] = mode
    return argv,inputs


class SSH(BaseSSH):
    def measure(self,document,bundle,remote_bundle,run_dir,boot):
        root = Path(run_dir)/"raw"/boot/document["task_id"]
        root.mkdir(parents=True,exist_ok=True)
        remote = self.job_path(document,run_dir,boot)
        q = shlex.quote
        expected = digest(document)
        if (root/"task.json").exists():
            if read(root/"task.json")!=document: raise ValueError("task identity collision")
        else: write(root/"task.json",document)
        if self.boot_id()!=boot: raise TransportError("boot changed during batch")
        self.command("mkdir -p "+q(remote))
        argv,inputs = native_command(document,bundle,remote_bundle,remote)
        (root/"inputs.txt").write_text(inputs)
        (root/"execute.sh").write_text(session_script(self.config["remote_base"],remote,boot,expected,argv))
        existing = self.command("test -f "+q(remote+"/task_hash")+" && cat "+q(remote+"/task_hash")+" || true").strip()
        if existing and existing!=expected: raise TransportError("remote task hash mismatch")
        try:
            if not existing:
                policy = read(Path(bundle)/"plans/board_execution.json")
                self.check_space(remote,job_storage_bytes(document,read(Path(bundle)/"plans/programs.json"))+
                                 policy["storage_margin_bytes"])
                self.upload(root/"inputs.txt",remote+"/inputs.txt")
                self.upload(root/"execute.sh",remote+"/execute.sh")
                self.command("cd "+q(remote)+" && nohup sh execute.sh </dev/null > launcher.log 2>&1 &")
            deadline = time.monotonic()+660
            while True:
                result = self.command("test -f "+q(remote+"/exit_code")+" && cat "+q(remote+"/exit_code")+" || true").strip()
                if result: break
                if time.monotonic()>=deadline: raise TransportError("remote session deadline exceeded")
                time.sleep(2)
        except TransportError as error:
            saved = self.failure_snapshot(remote,root)
            raise TransportError(str(error)+"; incomplete evidence: "+saved) from error
        collected = self.collect(remote,root)
        if (collected/"boot_id").read_text().strip()!=boot or (collected/"task_hash").read_text().strip()!=expected:
            raise TransportError("result identity/boot mismatch")
        if result!="0": raise TransportError("native session failed with exit "+result+"; raw files retained")
        if (collected/"inputs.txt").read_text()!=inputs: raise TransportError("input list mismatch")
        if self.boot_id()!=boot: raise TransportError("boot changed before acceptance")
        from .evidence import check_gate
        from cpu_vta_partitioning.measurement import frames
        check_gate(frames(collected/"result.jsonl",sum(document[k] for k in ("warmup","scored","tail"))),document)
        return collected

    def mount_storage(self):
        """Mount the already-authorized experiment partition; never repartition."""
        mount = self.config.get("storage_mount")
        if not mount: return
        q = shlex.quote
        identity = self.command("/sbin/blkid /dev/mmcblk1p3")
        if 'UUID="b19d08f2-0178-4a5e-a920-121c4e8a7a9b"' not in identity or 'TYPE="ext4"' not in identity:
            raise TransportError("experiment filesystem UUID/type mismatch")
        self.command("\n".join(["set -eu",
            'if ! awk -v target='+q(mount)+" '$2 == target {ok=1} END {exit !ok}' /proc/mounts; then",
            "mkdir -p "+q(mount),"mount -t ext4 -o noatime /dev/mmcblk1p3 "+q(mount),"fi",
            "awk -v target="+q(mount)+" '$2 == target && $1 == \"/dev/mmcblk1p3\" && $3 == \"ext4\" && $4 ~ /(^|,)rw(,|$)/ {ok=1} END {exit !ok}' /proc/mounts"]))

    def verify_payload(self,remote,boot):
        self.idle_command("sync; echo 3 > /proc/sys/vm/drop_caches; cd "+shlex.quote(remote)+
                          " && sha256sum -c board.sha256 > verification-after-cache-drop.log",boot=boot,timeout=600)
