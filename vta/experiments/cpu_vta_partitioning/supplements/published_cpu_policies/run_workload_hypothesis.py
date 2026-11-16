#!/usr/bin/env python3
"""Run a small frozen offline hypothesis; no board action is provided."""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


if __name__=='__main__':
    from workload_model.experiments.host_exposure import run
    print(json.dumps(run(),ensure_ascii=False))
