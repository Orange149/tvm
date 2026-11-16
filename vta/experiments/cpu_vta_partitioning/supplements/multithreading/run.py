#!/usr/bin/env python3
"""Independent entrypoint; does not change the sealed main controller."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
base_source = ROOT.parents[1]/"src"
if not base_source.is_dir():
    base_source = ROOT.parent/"base_controller/src"
sys.path.insert(0,str(base_source))
sys.path.insert(0,str(ROOT))

if __name__=="__main__":
    from multithreading.cli import main
    raise SystemExit(main())
