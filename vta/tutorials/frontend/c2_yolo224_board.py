#!/usr/bin/env python3
"""Logged, isolated native YOLO C2 board deployment and measurement."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from c2_yolo224_prepare import ROOT, HERE, sha, write

SSH = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
       "-o", "HostKeyAlgorithms=+ssh-rsa", "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa"]
BIT = ROOT / "apps/vta_rpc/firmware/vta_hpc.bit"


def read(path):
    return json.loads(Path(path).read_text())


class Board:
    def __init__(self, out, host):
        self.out = Path(out).resolve()
        self.host = host
        self.options=list(SSH)
        connection=self.out/'ssh_connection.json'
        if connection.exists():
            pinned=read(connection)
            if pinned['host']!=host:raise ValueError('connection identity differs from this boot')
            self.options+=['-o','UserKnownHostsFile='+pinned['known_hosts']]
        self.remote = "/var/volatile/c2_yolo224_" + self.out.name
        self.logs = self.out / "transport"
        self.logs.mkdir(exist_ok=True)

    def call(self, argv, label, timeout=60):
        record = self.logs / (label + ".json")
        attempt = 1
        while record.exists():
            attempt += 1
            record = self.logs / (label + f"_attempt{attempt:02d}.json")
        started = time.monotonic()
        try:
            result = subprocess.run(argv, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            write(record, {"argv": argv, "timeout": True, "target_completion_unknown": True,
                "seconds": time.monotonic()-started,
                "stdout": (e.stdout or b"").decode(errors="replace"),
                "stderr": (e.stderr or b"").decode(errors="replace")})
            raise
        write(record, {"argv": argv, "returncode": result.returncode,
            "seconds": time.monotonic()-started, "stdout": result.stdout.decode(errors="replace"),
            "stderr": result.stderr.decode(errors="replace")})
        if result.returncode:
            raise RuntimeError(label + ": " + result.stderr.decode(errors="replace")[-2500:])
        return result.stdout

    def ssh(self, command, label, timeout=60):
        return self.call(["ssh", *self.options, self.host, command], label, timeout)

    def preflight(self, label):
        data = self.ssh("cat /proc/sys/kernel/random/boot_id; uname -r; "
            "cat /sys/class/fpga_manager/fpga0/state; "
            "cat /proc/meminfo | grep Cma; ps; "
            "ls -l /dev/mem /sys/class/u-dma-buf/udmabuf0/size 2>/dev/null; "
            "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq "
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null; "
            "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null; "
            "grep -E 'pl[0-3]_ref|fclk' /sys/kernel/debug/clk/clk_summary 2>/dev/null || true", label)
        return {"boot_id": data.decode().splitlines()[0], "raw": data.decode()}

    def restore(self):
        before = self.preflight("restore_before")
        for line in before["raw"].splitlines():
            if "sh -c " not in line and any(s in line for s in ("tvm_rpc server", "./vta_stage_",
                    "/vta_stage_pipeline_runner --", "/c2_yolo224_runner --")):
                raise RuntimeError("accelerator workload already running: " + line)
        q = shlex.quote
        firmware = "c2_yolo224_" + self.out.name + ".bit"
        command = "\n".join([
            "set -eu", "test \"$(cat /proc/sys/kernel/random/boot_id)\" = " + q(before["boot_id"]),
            "test \"$(sha256sum /media/sd-mmcblk1p2/tvm_deploy/firmware/vta_hpc.bit | cut -d' ' -f1)\" = " + sha(BIT),
            "if [ ! -r /sys/class/u-dma-buf/udmabuf0/size ]; then /sbin/insmod /media/sd-mmcblk1p2/tvm_deploy/hp/u-dma-buf.ko udmabuf0=201326592; fi",
            "test \"$(cat /sys/class/u-dma-buf/udmabuf0/size)\" = 201326592",
            "test ! -e /lib/firmware/" + q(firmware),
            "cp /media/sd-mmcblk1p2/tvm_deploy/firmware/vta_hpc.bit /lib/firmware/" + q(firmware),
            "printf '0' > /sys/class/fpga_manager/fpga0/flags",
            "printf '%s' " + q(firmware) + " > /sys/class/fpga_manager/fpga0/firmware",
            "test \"$(cat /sys/class/fpga_manager/fpga0/state)\" = operating",
            "sha256sum /lib/firmware/" + q(firmware),
            "cat /sys/class/u-dma-buf/udmabuf0/size /sys/class/u-dma-buf/udmabuf0/phys_addr /sys/class/u-dma-buf/udmabuf0/dma_coherent",
            "dmesg | tail -15"])
        self.ssh(command, "restore_hardware")
        write(self.out / "board_state.json", {"host": self.host, "remote": self.remote,
            "boot_id": before["boot_id"], "firmware": "/lib/firmware/" + firmware,
            "bitstream_sha256": sha(BIT), "udmabuf_bytes": 201326592})
        self.preflight("restore_after")

    def deploy(self):
        state = read(self.out / "board_state.json")
        if self.preflight("deploy_before")["boot_id"] != state["boot_id"]:
            raise RuntimeError("boot changed before deployment")
        package = self.out / "package"
        manifest = read(package / "manifest.json")
        for name, path in (("libvta.so", self.out / "runtime/libvta.so"),
                           ("libtvm_runtime.so", ROOT / "build_axu_aarch64/libtvm_runtime.so"),
                           ("vta_stage_pipeline_runner", self.out / "runtime/vta_stage_pipeline_runner")):
            dest = package / name
            if not dest.exists():
                import shutil
                shutil.copy2(path, dest)
            elif sha(dest) != sha(path):
                raise ValueError("packaged runtime drift")
        inventory = {str(p.relative_to(package)): sha(p) for p in package.rglob("*") if p.is_file()}
        write(self.out / "deployment_inventory.json", inventory)
        archive = self.out / "deployment.tar"
        with tarfile.open(archive, "x") as tar:
            tar.add(package, arcname="package")
        q=shlex.quote
        self.ssh("test ! -e " + q(self.remote) + " && mkdir " + q(self.remote), "deploy_mkdir")
        self.call(["scp", *self.options, str(archive), self.host+":"+self.remote+"/deployment.tar"], "deploy_upload", 120)
        self.ssh("cd " + q(self.remote) + " && test \"$(sha256sum deployment.tar | cut -d' ' -f1)\" = "
                 + sha(archive) + " && tar -xf deployment.tar", "deploy_extract")
        self.ssh("cd " + q(self.remote + "/package") + " && sha256sum " +
                 " ".join(q(p) for p in inventory), "deploy_hashes")

    def recover_fpga(self, label):
        """Reset only this experiment's FPGA after a completed failed candidate.

        An unknown/running native process must be resolved before calling this.
        The caller also runs a fresh healthy executor before another candidate.
        """
        state=read(self.out/'board_state.json');before=self.preflight(label+'_before')
        if before['boot_id']!=state['boot_id']:raise RuntimeError('boot changed during recovery')
        for line in before['raw'].splitlines():
            if 'sh -c ' not in line and any(s in line for s in
                    ('/c2_yolo224_runner --','/vta_stage_pipeline_runner --','./tvm_rpc server')):
                raise RuntimeError('refuse FPGA recovery with an active accelerator process')
        q=shlex.quote;firmware=state['firmware']
        self.ssh('\n'.join(['set -eu',
            'test "$(sha256sum '+q(firmware)+" | cut -d' ' -f1)\" = "+state['bitstream_sha256'],
            "printf '0' > /sys/class/fpga_manager/fpga0/flags",
            "printf '%s' "+q(Path(firmware).name)+' > /sys/class/fpga_manager/fpga0/firmware',
            'test "$(cat /sys/class/fpga_manager/fpga0/state)" = operating']),label+'_reload')
        after=self.preflight(label+'_after')
        if after['boot_id']!=state['boot_id']:raise RuntimeError('boot changed during recovery')
        write(self.out/(label+'_recovery.json'),{'boot_id':state['boot_id'],
            'bitstream_sha256':state['bitstream_sha256'],'no_active_accelerator_process':True,
            'same_firmware_reloaded':True,'fresh_executor_health_check_required':True})

    def smoke(self):
        state = read(self.out / "board_state.json")
        before = self.preflight("smoke_before")
        if before["boot_id"] != state["boot_id"]:
            raise RuntimeError("boot changed")
        manifest = read(self.out / "package/manifest.json")
        package = self.remote + "/package"
        dest = self.remote + "/smoke_M0_01"
        q = shlex.quote
        command = [package + "/vta_stage_pipeline_runner"]
        for stage in manifest["stages"]:
            prefix = "--stage" + str(stage["index"])
            for key in ("graph", "lib", "params"):
                command += [prefix + "-" + key, package + "/" + stage[key]]
            command += [prefix + "-name", stage["name"], prefix + "-device", stage["device"],
                        prefix + "-input-names", ",".join(stage["input_names"]),
                        prefix + "-input-sources", ",".join(stage["input_sources"]),
                        prefix + "-runtime-num-threads", str(stage["runtime_num_threads"])]
        command += ["--input", package + "/inputs/000.bin", "--runs", "2", "--serial",
                    "--queue-depth", "2", "--runtime-num-threads", "4", "--output-mode", "raw",
                    "--output-dump-dir", dest + "/outputs", "--output-jsonl", dest + "/frames.jsonl",
                    "--allocation-snapshot", dest + "/allocation.jsonl", "--process-timeout-s", "90"]
        environment = {"LD_LIBRARY_PATH": package, "LD_PRELOAD": package + "/libtvm_runtime.so:" + package + "/libvta.so",
            "TVM_NUM_THREADS": "4", "TVM_THREAD_POOL_SPIN_COUNT": "0", "AXU5EVB_DRIVER_SAFE_COPY": "0",
            "AXU5EVB_DRIVER_POST_START_SLEEP_NS": "1000", "AXU5EVB_DRIVER_POLL_SLEEP_NS": "1000",
            "AXU5EVB_DRIVER_DEBUG": "0", "AXU5EVB_DRIVER_INFO": "0", "VTA_RUNTIME_TRACE": "0",
            "VTA_REPLAY_POLICY": "disabled", "VTA_QUEUE_DIAGNOSTICS": "1"}
        shell = "set -eu\nmkdir " + q(dest) + "\ncd " + q(package) + "\nenv " + shlex.join(
            [k+"="+v for k,v in environment.items()] + command)
        self.ssh(shell, "smoke_M0_run", timeout=120)
        self.ssh("tar -C " + q(dest) + " -cf " + q(dest + ".tar") + " .", "smoke_pack")
        self.call(["scp", *self.options, self.host + ":" + dest + ".tar", str(self.out / "smoke_M0_01.tar")], "smoke_fetch")
        self.preflight("smoke_after")

    def configure(self):
        state=read(self.out/'board_state.json'); package=self.out/'package'
        m=read(package/'manifest.json');base=self.remote+'/package'
        reader=read(self.out/'last_reader_certificate.json')
        stages=[]
        for s in m['stages']:
            r={**s,**s['artifact_sha256']}
            for k in ('graph','lib','params'):r[k]=base+'/'+s[k]
            stages.append(r)
        inputs=[{**x,'path':base+'/'+x['path']} for x in read(package/'inputs.json')]
        plan={'schema':'c2-yolo224-program-v1','stages':stages,'inputs':inputs,
              'boot_id':state['boot_id'],'bitstream':state['firmware'],
              'bitstream_sha256':state['bitstream_sha256'],'runner_sha256':sha(package/'c2_yolo224_runner'),
              'runtime_sha256':sha(package/'libtvm_runtime.so'),'driver_sha256':sha(package/'libvta.so'),
              'shared_buffer_plan':self.remote+'/shared_buffer_plan.json',
              'capacity_certificate':self.remote+'/q1/command_capacity_certificate.json',
              'last_reader':{'status':'qualified','qualification_scope':'static; dynamic gates required',
                  'certificate':self.remote+'/last_reader_certificate.json',
                  'certificate_sha256':sha(self.out/'last_reader_certificate.json'),
                  **{k:reader[k] for k in ('input_node','node','function')}}}
        write(self.out/'program.json',plan)
        for path,remote in ((package/'c2_yolo224_runner',base+'/c2_yolo224_runner'),
                            (self.out/'program.json',self.remote+'/program.json'),
                            (self.out/'shared_buffer_plan.json',self.remote+'/shared_buffer_plan.json'),
                            (self.out/'last_reader_certificate.json',self.remote+'/last_reader_certificate.json')):
            self.call(['scp',*self.options,str(path),self.host+':'+remote],'configure_'+path.name,120)
        self.ssh('chmod +x '+shlex.quote(base+'/c2_yolo224_runner'),'configure_executable')

    def run(self,options):
        state=read(self.out/'board_state.json');q=shlex.quote
        before=self.preflight(options.label+'_before')
        if before['boot_id']!=state['boot_id']:raise RuntimeError('independent boot needs new boot deployment')
        for line in before['raw'].splitlines():
            if 'sh -c ' not in line and any(s in line for s in ('/c2_yolo224_runner --','/vta_stage_pipeline_runner --','./tvm_rpc server')):
                raise RuntimeError('another accelerator process is active: '+line)
        dest=self.remote+'/'+options.label;package=self.remote+'/package'
        local=self.out/'runs'/options.label;local.mkdir(parents=True,exist_ok=False)
        environment={'LD_LIBRARY_PATH':package,'LD_PRELOAD':package+'/libtvm_runtime.so:'+package+'/libvta.so',
            'TVM_NUM_THREADS':'4','TVM_THREAD_POOL_SPIN_COUNT':'0','AXU5EVB_DRIVER_SAFE_COPY':'0',
            'AXU5EVB_DRIVER_POST_START_SLEEP_NS':'1000','AXU5EVB_DRIVER_POLL_SLEEP_NS':'1000',
            'AXU5EVB_DRIVER_DEBUG':'0','AXU5EVB_DRIVER_INFO':'0','VTA_RUNTIME_TRACE':'0',
            'VTA_REPLAY_POLICY':'disabled','VTA_QUEUE_DIAGNOSTICS':'1' if options.diagnostics else '0',
            'VTA_INSN_BUFFER_BYTES':str(options.insn),'VTA_UOP_BUFFER_BYTES':str(options.uop)}
        if getattr(options,'allocation_log',False):environment['AXU5EVB_DRIVER_INFO']='1'
        if getattr(options,'loader_log',False):environment['LD_DEBUG']='libs'
        if options.threshold:environment['VTA_INSN_SUBMIT_THRESHOLD_BYTES']=str(options.threshold)
        if options.mode=='M3':environment['VTA_COMMAND_MANIFEST_ID']=sha(self.out/'q1/command_capacity_certificate.json')
        if getattr(options,'observer',False):
            observer=read(self.out/'address_observer_config.json')
            environment['LD_PRELOAD']=self.remote+'/c2_yolo224_address_observer.so:'+environment['LD_PRELOAD']
            environment.update(C2_ADDRESS_LOG=dest+'/reader_addresses.jsonl',
                C2_ADDRESS_RETURN_PC0=observer['return_pcs'][0],C2_ADDRESS_RETURN_PC1=observer['return_pcs'][1])
        command=[package+'/c2_yolo224_runner','--plan',self.remote+'/'+options.program,'--output-dir',dest,
            '--mode',options.mode,'--frames',str(options.frames),'--warmup',str(options.warmup),'--settle',str(options.settle)]
        if options.serial:command.append('--serial')
        if options.detailed:command.append('--detailed')
        if options.proxy_off:command.append('--proxy-off')
        if options.control:command+=['--control',options.control]
        write(local/'contract.json',{'environment':environment,'argv':command,'boot_id':state['boot_id'],
            'runner_sha256':sha(self.out/'package/c2_yolo224_runner'),'program_sha256':sha(self.out/options.program),
            'output_acceptance':'all eight outputs byte identical; no numeric tolerance',
            'independent_statistical_unit':'Linux boot, not block/frame/process'})
        shell='\n'.join(['set -eu','mkdir '+q(dest),'cd '+q(package),'set +e',
            'env '+shlex.join([k+'='+v for k,v in environment.items()]+command)+' > '+q(dest+'/stdout')+' 2> '+q(dest+'/stderr'),
            'result=$?','printf "%s\\n" "$result" > '+q(dest+'/exitcode'),'exit 0'])
        self.ssh(shell,options.label+'_run',timeout=max(180,(options.frames+options.warmup+options.settle)*3+60))
        archive=dest+'.tar'
        self.ssh('tar -C '+q(dest)+' -cf '+q(archive)+' .',options.label+'_pack')
        self.call(['scp',*self.options,self.host+':'+archive,str(local/'raw.tar')],options.label+'_fetch',120)
        with tarfile.open(local/'raw.tar') as tar:
            for member in tar.getmembers():
                if not member.isfile():continue
                name=member.name.removeprefix('./')
                if name not in ('stdout','stderr','exitcode','result.json','outputs.bin','reader_addresses.jsonl'):
                    raise ValueError('unexpected board artifact: '+name)
                (local/name).write_bytes(tar.extractfile(member).read())
        after=self.preflight(options.label+'_after')
        if after['boot_id']!=state['boot_id']:raise RuntimeError('boot changed during run')
        code=int((local/'exitcode').read_text())
        print(json.dumps({'label':options.label,'returncode':code,'local':str(local)}),flush=True)
        if code and not options.expect_invalid:
            raise RuntimeError(options.label+' failed: '+(local/'stderr').read_text()[-1600:])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("restore", "deploy", "smoke", "configure", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="root@192.168.1.247")
    parser.add_argument('--label');parser.add_argument('--mode',default='M0')
    parser.add_argument('--frames',type=int,default=64);parser.add_argument('--warmup',type=int,default=0)
    parser.add_argument('--settle',type=int,default=0);parser.add_argument('--serial',action='store_true')
    parser.add_argument('--detailed',action='store_true');parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--proxy-off',action='store_true');parser.add_argument('--control',default='')
    parser.add_argument('--insn',type=int,default=1<<25);parser.add_argument('--uop',type=int,default=1<<25)
    parser.add_argument('--threshold',type=int,default=0);parser.add_argument('--expect-invalid',action='store_true')
    parser.add_argument('--program',default='program.json')
    parser.add_argument('--observer',action='store_true')
    parser.add_argument('--allocation-log',action='store_true')
    parser.add_argument('--loader-log',action='store_true')
    args=parser.parse_args()
    board=Board(args.out,args.host)
    if args.phase=='run':board.run(args)
    else:getattr(board,args.phase)()


if __name__=="__main__":
    main()
