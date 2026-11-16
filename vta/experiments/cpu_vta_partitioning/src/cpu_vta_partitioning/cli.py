"""CLI boundary: prepare/validate never instantiate transport."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from cpu_vta_partitioning.common import PROJECT, read, write, inventory, digest, now


def parser():
    p = argparse.ArgumentParser(description="CPU–VTA two-stage partition experiments")
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="local-only compilation, tests and package")
    prep.add_argument("--config", type=Path, default=PROJECT / "configs/resnet18.json")
    prep.add_argument("--local-config", type=Path, default=PROJECT / "configs/local.json")
    val = commands.add_parser("validate", help="local-only package validation")
    val.add_argument("--bundle", type=Path, required=True)
    board = commands.add_parser("board")
    sub = board.add_subparsers(dest="action", required=True)
    start = sub.add_parser("run")
    start.add_argument("--bundle", type=Path, required=True)
    start.add_argument("--run-id", required=True)
    start.add_argument("--local-config", type=Path, default=PROJECT / "configs/local.json")
    resume = sub.add_parser("resume")
    resume.add_argument("--run-dir", type=Path, required=True)
    for item in (start,resume):
        item.add_argument("--execute-board", action="store_true", required=True)
    for command in ("status", "analyze"):
        item = commands.add_parser(command)
        item.add_argument("--run-dir", type=Path, required=True)
    return p


def local_prepare(config, local):
    from cpu_vta_partitioning.prepare.build import initialize, prepare_inputs, compile_programs, compile_runtime
    from cpu_vta_partitioning.prepare.package import runtime_closure, create_plans, seal
    with initialize(config.resolve(), local.resolve()) as (workspace, settings):
        prepare_inputs(workspace, settings)
        compile_programs(workspace, settings)
        compile_runtime(workspace, settings)
        runtime_closure(workspace, settings)
        create_plans(workspace)
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CPUVTA_WORKSPACE=str(workspace))
        log = workspace / "plans/local_tests.log"
        report = workspace / "plans/pytest.xml"
        argv = [settings["python"], "-m", "pytest", "-c", str(PROJECT / "pyproject.toml"),
                "--confcutdir=" + str(PROJECT), str(PROJECT / "tests"), "--junitxml=" + str(report)]
        with log.open("w") as stream:
            result = subprocess.run(argv, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("local tests failed; see " + str(log))
        import xml.etree.ElementTree as ET
        suites = list(ET.parse(report).getroot().iter("testsuite"))
        totals = {k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ("tests","errors","failures","skipped")}
        if not totals["tests"] or any(totals[k] for k in ("errors","failures","skipped")):
            raise RuntimeError("offline acceptance requires all tests executed without skips: " + str(totals))
        evidence = {"passed":True,"totals":totals,"argv":argv,"utc":now(),"log":"plans/local_tests.log","junit":"plans/pytest.xml"}
        return seal(workspace,evidence)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = {"status":"PREPARED_OFFLINE","bundle":str(local_prepare(args.config,args.local_config)),
                      "board_validation":"pending; no board connected"}
        elif args.command == "validate":
            from cpu_vta_partitioning.prepare.package import validate
            result = validate(args.bundle)
        elif args.command == "board":
            from cpu_vta_partitioning.campaign import create, run_current
            from cpu_vta_partitioning.transport import SSH
            if args.action == "run":
                import re
                if not re.fullmatch(r"[a-zA-Z0-9_-]+",args.run_id):
                    raise ValueError("run-id must contain only letters, digits, _ or -")
                root = PROJECT / "runs" / args.run_id
                from cpu_vta_partitioning.prepare.package import validate
                validate(args.bundle)
                state = create(args.bundle,root,read(args.local_config)["board"])
            else:
                root = args.run_dir.resolve()
                state = read(root / "state.json")
            result = run_current(root,SSH(state["connection"],execute_board=args.execute_board))
            if result["status"] == "WAITING_NEW_BOOT":
                print("等待真实重启板子。重启测试进程不算新启动。")
                print(f"续跑: {sys.executable} {PROJECT / 'run.py'} board resume --run-dir {root} --execute-board")
        elif args.command == "status":
            result = read(args.run_dir / "state.json")
            result = {k:v for k,v in result.items() if k != "connection"}
            result["accepted_windows"] = len(list((args.run_dir / "accepted").glob("*/*.json")))
            result["analysis_files"] = [str(p) for p in sorted(args.run_dir.glob("env*/analysis.json"))]
        elif args.command == "analyze":
            from cpu_vta_partitioning.analysis.export import export
            result = export(args.run_dir)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(type(error).__name__ + ": " + str(error),file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
