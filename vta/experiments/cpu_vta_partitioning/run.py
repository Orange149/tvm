#!/usr/bin/env python3
"""Repository-local entry point; installation is optional."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cpu_vta_partitioning.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
