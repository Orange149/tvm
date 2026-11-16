#!/usr/bin/env python3
"""Independent focused experiment; no changes to sealed controllers."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
base = ROOT.parents[1] / "src"
if not base.is_dir():
    base = ROOT.parent / "base_controller/src"
sys.path[:0] = [str(ROOT), str(base)]

if __name__ == "__main__":
    from cpu_strategy.policy.cli import main
    raise SystemExit(main())
