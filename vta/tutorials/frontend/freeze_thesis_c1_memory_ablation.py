#!/usr/bin/env python3
"""Freeze the 3/3/2-topology copy/shared pool before new candidate timing."""

import argparse
from pathlib import Path

from thesis_c1_memory_ablation import file_hash, freeze, read_json, write_new
from thesis_c1_memory_evidence import inputs_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite a frozen run")
    result = freeze(read_json(args.input), args.input.resolve().parent)
    inputs_manifest(result)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_new(args.output_dir / "selection_contract.json", result)
    write_new(args.output_dir / "source_inventory.json", {
        "path": str(args.input.resolve()), "sha256": file_hash(args.input)})
    print("Frozen 16 configurations / 16 feature masks / 20 random orders; ranking windows allowed: {}".format(
        result["identifiability"]["ranking_windows_allowed"]))


if __name__ == "__main__":
    main()
