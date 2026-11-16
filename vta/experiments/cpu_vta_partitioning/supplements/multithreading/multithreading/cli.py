import argparse
import json
from pathlib import Path

from cpu_vta_partitioning.common import read,write
from .domain import ROOT,PROJECT,protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description="ResNet18 单任务多线程与共享核心竞争补充实验")
    commands = parser.add_subparsers(dest="command",required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--base-bundle",type=Path,default=PROJECT/"artifacts/bundles/62a5cff7a2439cf3a7828476")
    prep.add_argument("--local-config",type=Path,default=PROJECT/"configs/local.json")
    prep.add_argument("--config",type=Path,default=ROOT/"configs/resnet18.json")
    val = commands.add_parser("validate");val.add_argument("--bundle",type=Path,required=True)
    for name in ("status","analyze"):
        p = commands.add_parser(name);p.add_argument("--run-dir",type=Path,required=True)
    board = commands.add_parser("board").add_subparsers(dest="action",required=True)
    start = board.add_parser("run")
    start.add_argument("--bundle",type=Path,required=True)
    start.add_argument("--run-id",default="resnet18_multithread_mapping_01")
    start.add_argument("--local-config",type=Path,default=PROJECT/"configs/local.json")
    resume = board.add_parser("resume");resume.add_argument("--run-dir",type=Path,required=True)
    for p in (start,resume): p.add_argument("--execute-board",action="store_true")
    args = parser.parse_args(argv)
    if args.command=="prepare":
        from .prepare import prepare
        print(prepare(args.base_bundle,args.local_config,args.config));return 0
    if args.command=="validate":
        from .prepare import validate
        print(json.dumps({"status":"VALIDATED_OFFLINE","bundle_id":validate(args.bundle)["bundle_id"]}));return 0
    if args.command=="status":
        from .campaign import validate_state,phase_at
        state = validate_state(args.run_dir)
        print(json.dumps({"status":state["status"],"completed_boots":state["boot_index"],"total_boots":10,
                          "phase":phase_at(state["boot_index"]),"active_boot":state["active_boot"]},ensure_ascii=False,indent=2));return 0
    if args.command=="analyze":
        from .analysis import analyze
        value = analyze(args.run_dir)
        print(json.dumps({"direction":value["direction"],"report":str(args.run_dir/"analysis/REPORT_zh.txt")},ensure_ascii=False));return 0
    if not args.execute_board: parser.error("board access requires --execute-board")
    from .campaign import create,run_current,validate_state
    from .transport import SSH
    if args.action=="run":
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]+",args.run_id): parser.error("invalid run ID")
        root = ROOT/"runs"/args.run_id
        connection = read(args.local_config)["board"]
        # Same remote base and lock as the main campaign: no simultaneous VTA jobs.
        if root.exists(): parser.error("run already exists; use board resume")
        from .preflight import qualify
        preflight = qualify(args.bundle,SSH(connection,execute_board=True))
        create(args.bundle,root,connection,excluded_boot=preflight["excluded_boot_id"])
    else:
        root = args.run_dir
        connection = validate_state(root)["connection"]
    result = run_current(root,SSH(connection,execute_board=True))
    print(json.dumps({"status":result["status"],"completed_boots":result["boot_index"],"run_dir":str(root)},ensure_ascii=False))
    if result["status"]=="WAITING_NEW_BOOT":
        print("本次启动已完成。请手动重启并联网；随后由助手执行续跑。")
    return 0
