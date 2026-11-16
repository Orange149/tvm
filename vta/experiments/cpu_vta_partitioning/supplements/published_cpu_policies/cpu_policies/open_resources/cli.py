"""The explicit open-resource entry point; importing it never touches SSH."""
import argparse
import json
from pathlib import Path
import re

from .common import DEFAULT_LOCAL_CONFIG, DEFAULT_RUN_ID, ROOT


def _read(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description="九种CPU/VTA资源上下文：准备、完整DP和单次上板")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--base-bundle", type=Path)
    prepare.add_argument("--profile-source", type=Path)
    prepare.add_argument("--calibration-source", type=Path,
                         help="reuse audited 261-component costs and any valid traffic records from a prior batch")
    validate = commands.add_parser("validate")
    validate.add_argument("--bundle", type=Path, required=True)
    for name in ("rank", "status", "analyze"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-dir", type=Path, required=True)
    boards = commands.add_parser("board").add_subparsers(dest="action", required=True)
    run = boards.add_parser("run")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--run-id", default=DEFAULT_RUN_ID)
    run.add_argument("--local-config", type=Path, default=DEFAULT_LOCAL_CONFIG)
    resume = boards.add_parser("resume")
    resume.add_argument("--run-dir", type=Path, required=True)
    for sub in (run, resume):
        sub.add_argument("--execute-board", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .prepare import prepare
        result = {"status": "PREPARED_OFFLINE", "bundle": str(prepare(
            args.base_bundle, args.profile_source, calibration_source=args.calibration_source))}
    elif args.command == "validate":
        from .prepare import validate_bundle
        manifest = validate_bundle(args.bundle, replay_imports=True)
        result = {"status": "VALIDATED_OFFLINE", "bundle_id": manifest["bundle_id"]}
    elif args.command == "rank":
        from .campaign import rank
        rankings, pool, order = rank(args.run_dir)
        result = {"status": "RANKINGS_FROZEN", "pure_static_count": len(rankings["pure_static20"]),
                  "diverse_count": len(rankings["diverse20"]), "pipeline_pool_count": len(pool),
                  "order_count": len(order), "ranking_uses_pipeline_feedback": False}
    elif args.command == "status":
        from .campaign import status
        result = status(args.run_dir)
    elif args.command == "analyze":
        from .analysis import analyze
        result = analyze(args.run_dir)
    else:
        if not args.execute_board:
            parser.error("board access requires --execute-board")
        from .campaign import create, run_current, validate_state
        from .runtime import SSH
        if args.action == "run":
            if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
                parser.error("invalid run identity")
            connection = _read(args.local_config)["board"]
            root = ROOT / "runs" / args.run_id
            state = create(args.bundle, root, connection)
        else:
            root = args.run_dir.resolve()
            state = validate_state(root)
            connection = state["connection"]
        state = run_current(root, SSH(connection, execute_board=True, bundle=state["bundle"]))
        result = {k: state.get(k) for k in ("status", "phase", "boot_id", "imported_components",
                  "source_pilot_components", "reused_calibration_components", "new_components",
                  "completed_components", "completed_traffic", "imported_traffic", "new_traffic",
                  "completed_pipeline_configurations")}
        result["run_dir"] = str(root)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
