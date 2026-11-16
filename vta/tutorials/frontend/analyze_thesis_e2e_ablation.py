#!/usr/bin/env python3
"""Validate and analyze a complete Stage-06 five-boot performance payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_e2e_ablation_protocol import analyze_performance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_performance(json.loads(Path(args.input).read_text(encoding="utf-8")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
