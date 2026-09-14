#!/usr/bin/env python3
"""Q2 fixed-order threshold correctness/traffic qualification; not performance.

Candidates run from largest to smallest using default 32 MiB backing buffers.
On any timeout/error/mismatch, record and stop without retrying smaller thresholds.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from audit_shared_buffer_storage import sha
from make_shared_buffer_plan import make_plan
from run_c3s_allocation_baseline import SSH, TRACE, remote
from run_queue_capacity import queue_records
from run_shared_buffer_qualification import archive_outputs


THRESHOLDS = (2656, 1328, 656, 320)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--host", default="root@192.168.1.247")
    p.add_argument("--board-root", default="/tmp")
    args = p.parse_args()
    base = Path(__file__).resolve().parent / "report_out/stage_tile_cotuning"
    q1 = json.loads((base / "queue_capacity/q1_board_run1/reconciled_summary.json").read_text())
    reference = base / "c3s_buffer_reuse/s1_board_run1"
    build = json.loads((args.build / "build_manifest.json").read_text())
    for name,digest in build["artifacts"].items(): assert sha(args.build/name) == digest
    for path,digest in build["source_sha256"].items(): assert sha(path) == digest
    args.output.mkdir(parents=True, exist_ok=False)
    board = args.board_root.rstrip("/") + "/queue_" + args.output.name
    state_cmd = ("cat /proc/sys/kernel/random/boot_id; cat /sys/class/fpga_manager/fpga0/state; "
        "cat /sys/class/u-dma-buf/udmabuf0/size; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; "
        "sha256sum /lib/firmware/vta_hpc.bit; ps -eo pid,args")
    preflight = remote(args.host, state_cmd); lines = preflight.decode().splitlines()
    assert lines[1:4] == ["operating", "201326592", "1066666"]
    assert lines[4].split()[0] == "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6"
    assert not any(any(x in line for x in ("tvm_rpc", "vta_stage_pipeline_runner", "vta_stage_pair_runner")) for line in lines[5:])
    remote(args.host, "mkdir " + shlex.quote(board))
    def upload(path):
        subprocess.run(["scp", *SSH[1:], str(path), args.host+":"+board+"/"], check=True, capture_output=True)
    for name in build["artifacts"]: upload(args.build/name)
    for item in q1["inputs"]: upload(reference/item["file"])
    upload(reference/"inputs.txt")
    report = {"boot": lines[0], "build": build, "script_sha256": sha(__file__), "board": board,
        "thresholds": list(THRESHOLDS), "backing_bytes": {"insn": 1<<25, "uop": 1<<25},
        "runs": 32, "warmups": 2, "order": "largest threshold first; stop globally on first failure",
        "scope": "varied-input byte correctness and traffic; NOT performance or minimum capacity",
        "status": "running", "results": []}
    (args.output/"preregistered.json").write_text(json.dumps(report,indent=2)+"\n")
    try:
        for threshold in THRESHOLDS:
            for topology,rank in zip("ABCD",(1,5,7,13)):
                assert remote(args.host,"cat /proc/sys/kernel/random/boot_id").decode().strip()==lines[0]
                audit=json.loads((base/f"c3s_buffer_reuse/s0_static_run1/{topology}.json").read_text())
                plan=make_plan(audit,build); plan_path=args.output/f"{topology}_plan.json"
                if not plan_path.exists(): plan_path.write_text(json.dumps(plan,indent=2)+"\n"); upload(plan_path)
                pkg=f"/media/sd-mmcblk1p2/v1_p7_top20/legacy_profile_rank{rank:02d}"
                expected_package={k:v for k,v in audit["package_sha256"].items() if k!="manifest.json"}
                actual=remote(args.host,"cd "+shlex.quote(pkg)+" && sha256sum "+" ".join(map(shlex.quote,expected_package))).decode()
                assert {r.split()[1]:r.split()[0] for r in actual.splitlines()}==expected_package
                original=shlex.split(next(x for x in (TRACE/f"rank{rank:02d}/shared_command.sh").read_text().splitlines() if x.startswith("exec ")))[1:]
                original[0]=board+"/vta_stage_pipeline_runner"; original[original.index("--runs")+1]="32"
                i=original.index("--input"); original[i:i+2]=["--input-list",board+"/inputs.txt"]
                stem=f"t{threshold}_{topology}"; original[original.index("--output-jsonl")+1]=board+"/"+stem+".jsonl"
                original += ["--warmup-runs","2","--shared-buffer-plan",board+"/"+plan_path.name,
                    "--output-dump-dir",board+"/"+stem+"_outputs"]
                env={"LD_LIBRARY_PATH":pkg,"LD_PRELOAD":pkg+"/libtvm_runtime.so:"+board+"/libvta.so",
                    "TVM_NUM_THREADS":"4","TVM_THREAD_POOL_SPIN_COUNT":"0",
                    "AXU5EVB_DRIVER_POST_START_SLEEP_NS":"1000","AXU5EVB_DRIVER_POLL_SLEEP_NS":"1000",
                    "VTA_QUEUE_DIAGNOSTICS":"1","VTA_QUEUE_BOUNDARY_AUDIT":"0",
                    "VTA_INSN_BUFFER_BYTES":str(1<<25),"VTA_UOP_BUFFER_BYTES":str(1<<25),
                    "VTA_INSN_SUBMIT_THRESHOLD_BYTES":str(threshold)}
                shell="cd "+shlex.quote(pkg)+" && env "+" ".join(shlex.quote(k+"="+v) for k,v in env.items())+" "+shlex.join(original)
                (args.output/f"{stem}_command.sh").write_text(shell+"\n")
                print("[RUN]",stem,flush=True)
                proc=subprocess.run(SSH+[args.host,shell],capture_output=True,timeout=180)
                (args.output/f"{stem}.stdout").write_bytes(proc.stdout); (args.output/f"{stem}.stderr").write_bytes(proc.stderr)
                proc.check_returncode()
                raw=remote(args.host,"cat "+shlex.quote(board+"/"+stem+".jsonl")); (args.output/f"{stem}.jsonl").write_bytes(raw)
                rows=[json.loads(r) for r in raw.splitlines()]; assert len(rows)==32
                outputs=remote(args.host,"tar -C "+shlex.quote(board+"/"+stem+"_outputs")+" -cf - .")
                (args.output/f"{stem}_outputs.tar").write_bytes(outputs)
                assert archive_outputs(outputs)==archive_outputs((reference/f"{topology}_external_outputs.tar").read_bytes())
                records=queue_records(proc.stderr); assert records and any(r["reason"]=="threshold_auto_sync" for r in records)
                assert all(r["submit_threshold"]==threshold and r["submit_threshold_configured"] for r in records)
                q1_topology=next(r for r in q1["results"] if r["topology"]==topology)
                reference_records=q1_topology["modes"]["default_diag"]["queue_records"]
                assert sum(r["load_bytes"] for r in records)==sum(r["load_bytes"] for r in reference_records)
                assert sum(r["store_bytes"] for r in records)==sum(r["store_bytes"] for r in reference_records)
                item={"threshold":threshold,"topology":topology,"frames":32,"byte_exact":True,
                    "submits_including_warmup":len(records),"threshold_submits":sum(r["reason"]=="threshold_auto_sync" for r in records),
                    "explicit_submits":sum(r["reason"]=="explicit_sync" for r in records),
                    "insn_peak":max(r["insn_bytes"] for r in records),"uop_peak":max(r["uop_bytes"] for r in records),
                    "load_bytes":sum(r["load_bytes"] for r in records),"store_bytes":sum(r["store_bytes"] for r in records)}
                report["results"].append(item); (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
                print("[PASS]",stem,item["submits_including_warmup"],flush=True)
        report["status"]="complete_correctness_screen"
    except Exception as error:
        report["status"]="failed_stop_no_retry"; report["error"]=repr(error); raise
    finally:
        (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    post=remote(args.host,state_cmd); assert post.decode().splitlines()[0]==lines[0]; (args.output/"board_after.txt").write_bytes(post)


if __name__ == "__main__": main()
