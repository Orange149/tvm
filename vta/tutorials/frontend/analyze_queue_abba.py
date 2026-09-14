#!/usr/bin/env python3
"""Boot-level noninferiority intervals; never treat individual frames as replicates."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t


def interval(boot_log_ratios):
    if len(boot_log_ratios) < 3:
        return {"independent_boots": len(boot_log_ratios), "upper95_ii_change_pct": None,
                "noninferiority_pass": None, "reason": "fewer than three complete independent boots"}
    values = np.asarray(boot_log_ratios)
    upper = float(np.mean(values) + t.ppf(.95, len(values)-1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    pct = math.expm1(upper) * 100
    return {"independent_boots": len(values), "upper95_ii_change_pct": pct,
            "mean_ii_change_pct": math.expm1(float(np.mean(values))) * 100,
            "noninferiority_pass": pct <= 2.0}


def summarize(directories):
    seen = set()
    boots, aggregate = [], defaultdict(list)
    frozen = None
    for directory in directories:
        data = json.loads((directory / "summary.json").read_text())
        assert data["boot"] not in seen, "multiple directories from the same boot are not independent"
        seen.add(data["boot"])
        identity = (data["reference_sha256"], data["build"]["artifacts"], data["F"])
        if frozen is None:
            frozen = identity
        assert identity == frozen, "different candidate/binary/reference configurations cannot be pooled"
        groups = defaultdict(list)
        for block in data["blocks_completed"]:
            groups[(block["topology"], block["contrast"])].append(block)
        cells = []
        for name in "ABCD":
            for contrast in ("EP", "PF"):
                records = groups[(name, contrast)]
                ids = [r["block"] for r in records]
                assert len(ids) == len(set(ids))
                mean = float(np.mean([r["log_ii_ratio"] for r in records])) if records else None
                complete = data["status"] == "complete_one_boot" and sorted(ids) == list(range(5))
                if complete:
                    aggregate[(name, contrast)].append(mean)
                cells.append({"topology": name, "contrast": contrast, "completed_blocks": len(records),
                    "mean_log_ii_ratio": mean, "descriptive_ii_change_pct": math.expm1(mean)*100 if mean is not None else None,
                    "complete_for_inference": complete})
        boots.append({"boot": data["boot"], "status": data["status"], "cells": cells})
    return {"boots": boots, "inference": [{"topology": name, "contrast": contrast,
            **interval(aggregate[(name, contrast)])} for name in "ABCD" for contrast in ("EP", "PF")],
            "scope": "one-sided per-topology/contrast 95% t bounds across boot means; 2% II margin"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.directories), indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(result)
    print(result, end="")
