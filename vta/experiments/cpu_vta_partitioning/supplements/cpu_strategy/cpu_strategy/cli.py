"""Explicit local preparation and opt-in board orchestration."""
import argparse
import json
from pathlib import Path
import re

from cpu_vta_partitioning.common import read
from .domain import PROJECT, ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description="ResNet18 阶段级 CPU 线程与核心组策略")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--base-bundle", type=Path, default=PROJECT / "artifacts/bundles/62a5cff7a2439cf3a7828476")
    prepare.add_argument("--local-config", type=Path, default=PROJECT / "configs/local.json")
    prepare.add_argument("--config", type=Path, default=ROOT / "configs/resnet18.json")
    validate = commands.add_parser("validate")
    validate.add_argument("--bundle", type=Path, required=True)
    for name in ("status", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
    board = commands.add_parser("board").add_subparsers(dest="action", required=True)
    start = board.add_parser("run")
    start.add_argument("--bundle", type=Path, required=True)
    start.add_argument("--run-id", default="resnet18_cpu_strategy_02")
    start.add_argument("--local-config", type=Path, default=PROJECT / "configs/local.json")
    resume = board.add_parser("resume")
    resume.add_argument("--run-dir", type=Path, required=True)
    for command in (start, resume):
        command.add_argument("--execute-board", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .prepare import prepare
        print(prepare(args.base_bundle, args.local_config, args.config))
        return 0
    if args.command == "validate":
        from .prepare import validate
        print(json.dumps({"status": "VALIDATED_OFFLINE", "bundle_id": validate(args.bundle)["bundle_id"]}))
        return 0
    if args.command == "status":
        from .campaign import validate_state
        state = validate_state(args.run_dir)
        print(json.dumps({"status": state["status"], "completed_boots": state["boot_index"],
                          "total_boots": 4, "active_boot": state["active_boot"],
                          "phase": "exploration" if state["boot_index"] == 0 else "confirmation"},
                         ensure_ascii=False, indent=2))
        return 0
    if args.command == "analyze":
        from .analysis import analyze
        value = analyze(args.run_dir)
        print(json.dumps({"report": str(args.run_dir / "analysis/REPORT_zh.txt"),
                          "status": value.get("status", "ANALYZED")}, ensure_ascii=False))
        return 0
    if not args.execute_board:
        parser.error("board access requires --execute-board")
    from .campaign import create, preflight, run_current, validate_state
    from .transport import SSH
    if args.action == "run":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
            parser.error("invalid run ID")
        root = ROOT / "runs" / args.run_id
        if root.exists():
            parser.error("run already exists; use board resume")
        connection = read(args.local_config)["board"]
        transport = SSH(connection, execute_board=True)
        result = preflight(args.bundle, transport)
        state = create(args.bundle, root, connection, excluded_boot=result["excluded_boot_id"])
    else:
        root = args.run_dir.resolve()
        connection = validate_state(root)["connection"]
        state = run_current(root, SSH(connection, execute_board=True))
    print(json.dumps({"status": state["status"], "completed_boots": state["boot_index"],
                      "run_dir": str(root)}, ensure_ascii=False, indent=2))
    if state["status"] == "WAITING_NEW_BOOT":
        print("数据已回收并同步。请按约定手动重启；联网后由助手续跑。")
    return 0
