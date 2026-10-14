"""Local baseline completion evidence; no FPGA, runtime mutation or latency.

TPS below reconstructs Appendix A of arXiv:2111.15024, not author search
code. Restricted to frozen node24 (stride=1, symmetric padding=1). Paper
element counts are converted with explicit INP/WGT/ACC byte widths. These
are analytical payloads, not measured DDR traffic or final allocations.
"""

import ast
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess


SOFTWARE_COMMIT = "9bc19a8a90b6fef53ef05a4320032414e545f22f"
HARDWARE_COMMIT = "66bcb00b58ff583b724a827752b057e2589aac11"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
OLD = GROUPED / "20261012_c3_banerjee_double_buffer_local_run01"
FILES = ("vta/python/vta/top/vta_conv2d.py", "src/tir/transforms/inject_virtual_thread.cc",
         "vta/python/vta/transform.py", "vta/python/vta/build_module.py", "vta/runtime/runtime.cc")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True, order=True)
class Tile:
    b: int
    h: int
    w: int
    ci: int
    co: int
    oc_nthread: int
    h_nthread: int

    @property
    def identity(self):
        return "b%dh%dw%dci%dco%doc%dhth%d" % tuple(asdict(self).values())


def divisors(n):
    return tuple(i for i in range(1, n + 1) if n % i == 0)


def paper_cost(tile, capacity):
    """Necessary analytical conditions only; full-fusion allocation may differ.

    Local exact-tiling restriction: virtual lanes must divide outer groups.
    This is separately reported, not attributed to the paper's SRAM equation.
    ACC includes the bias term from Eq. 6. Eq. 2's l_acc is recorded as a
    bias-load proxy, not ALL fusion ACC/OUT traffic. Padding in s_inp is a
    logical allocation estimate, not actual DRAM padding loads.
    """
    dimensions = (1, 14, 14, 8, 16)
    inner = (tile.b, tile.h, tile.w, tile.ci, tile.co)
    if any(type(v) is not int or v <= 0 or n % v for n, v in zip(dimensions, inner)):
        raise ValueError("only integral frozen node24 tiles are supported")
    if tile.oc_nthread not in (1, 2) or tile.h_nthread not in (1, 2):
        raise ValueError("invalid virtual thread count")
    bo, ho, wo, cio, coo = (n // v for n, v in zip(dimensions, inner))
    threads = tile.oc_nthread * tile.h_nthread
    inp = tile.b * tile.ci * 16 * (tile.h + 2) * (tile.w + 2) * threads
    wgt = tile.co * tile.ci * 16 * 16 * 3 * 3 * threads
    acc = (tile.b * tile.co * 16 * tile.h * tile.w + tile.co * 16) * 4 * threads
    reasons = []
    if threads == 4:
        reasons.append("paper_both_threads_two")
    if coo % tile.oc_nthread or ho % tile.h_nthread:
        reasons.append("local_exact_outer_thread_divisibility")
    usage = {"INP": inp, "WGT": wgt, "ACC": acc}
    reasons += [name + "_paper_capacity" for name, value in usage.items() if value > capacity[name]]
    # Evaluate only exact lane domains: no fractional cost silently truncated.
    multiplier = None if coo % tile.oc_nthread or ho % tile.h_nthread else (
        bo * (ho // tile.h_nthread) * (coo // tile.oc_nthread) * wo * cio)
    loads = None if multiplier is None else {
        "INP": multiplier * inp, "WGT": multiplier * wgt,
        "ACC_bias_proxy": bo * ho * wo * 256 * 4}
    return {"id": tile.identity, "tile": asdict(tile), "paper_allocation_bytes": usage,
            "paper_payload_bytes": loads, "paper_score_bytes": None if loads is None else sum(loads.values()),
            "necessary_conditions_passed": not reasons, "rejections": reasons,
            "full_fusion_qualified": False}


def enumerate_node24(capacity):
    rows = [paper_cost(Tile(*values), capacity) for values in product(
        *(divisors(n) for n in (1, 14, 14, 8, 16)), (1, 2), (1, 2))]
    ranked = sorted((r for r in rows if r["necessary_conditions_passed"]),
                    key=lambda r: (r["paper_score_bytes"], r["id"]))
    return rows, ranked


def function_ast(path, name):
    tree = ast.parse(Path(path).read_text())
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError("function missing or ambiguous: " + name)
    return ast.dump(matches[0], include_attributes=False)


def verify_checkout(root, commit):
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit or subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("author source checkout differs from frozen clean commit")


def prepare(software, hardware, output):
    verify_checkout(software, SOFTWARE_COMMIT)
    verify_checkout(hardware, HARDWARE_COMMIT)
    output.mkdir(parents=True, exist_ok=False)
    # Preserve author source, license and complete tracked-file inventories.
    for name, root, paths in (
        ("software", software, FILES + ("LICENSE",)),
        ("hardware", hardware, ("config/vta_config.json", "config/pkg_config.py", "LICENSE")),
    ):
        for rel in paths:
            dest = output / "author_sources" / name / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / rel, dest)
        files = subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True).splitlines()
        write_new(output / (name + "_tracked_files.json"), files)
    config_path = REPO / "3rdparty/vta-hw/config/vta_config.json"
    config = json.loads(config_path.read_text())
    required = {"LOG_BATCH": 0, "LOG_BLOCK": 4, "LOG_INP_WIDTH": 3, "LOG_WGT_WIDTH": 3, "LOG_ACC_WIDTH": 5}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("paper enumerator unit contract changed")
    capacity = {name: 1 << config["LOG_" + name + "_BUFF_SIZE"] for name in ("INP", "WGT", "ACC")}
    rows, ranked = enumerate_node24(capacity)
    write_new(output / "tps_space.json", {"origin": "paper_Appendix_A_reconstruction_NOT_author_TPS_code",
        "capacity_bytes": capacity, "all_candidates": rows, "ranked_necessary_domain": ranked,
        "tie_rule": "paper_score_bytes then candidate ID; no latency used",
        "qualification_warning": "logical padding, fusion allocations and final addresses require compiler checks"})
    author_runtime = (software / "vta/runtime/runtime.cc").read_text()
    local_runtime = (REPO / "vta/runtime/runtime.cc").read_text()
    author_config = json.loads((hardware / "config/vta_config.json").read_text())
    features = {}
    for name in ("GREEDY_UOP_LOADS", "VALUE_BASED_UOP_CONSOLIDATION"):
        flag = "ENABLE_RUNTIME_" + name
        features[name] = {"author_default": author_config[flag],
                          "author_source_contains_flag": "VTA_" + flag in author_runtime,
                          "local_source_contains_flag": "VTA_" + flag in local_runtime,
                          "ported_and_qualified": False}
    fold_equal = function_ast(software / "vta/python/vta/transform.py", "FoldUopLoop") == function_ast(
        REPO / "vta/python/vta/transform.py", "FoldUopLoop")
    # Record exact-source search scope rather than asserting TPS is not public.
    matches = []
    import re
    pattern = re.compile(r"\bTPS\b|tiling parameter search|under.utilization", re.I)
    for root in (software, hardware):
        for rel in subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True).splitlines():
            path = root / rel
            if path.is_file() and path.suffix in (".py", ".cc", ".h", ".md", ".sh"):
                for no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                    if pattern.search(line):
                        matches.append({"repository": root.name, "file": rel, "line": no, "text": line})
    protected = {str(path): digest(path) for path in (
        REPO / "vta/runtime/runtime.cc", REPO / "vta/python/vta/top/vta_conv2d.py",
        REPO / "vta/python/vta/transform.py", config_path,
        HERE / "c3_banerjee_local.py", HERE / "run_c3_banerjee_local.py",
        OLD / "local_manifest.json", OLD / "local_qualification.json")}
    write_new(output / "protected.json", protected)
    write_new(output / "audit.json", {"status": "LOCAL_BASELINE_COMPLETION_IN_PROGRESS_NOT_BOARD_READY",
        "board_contacted": False, "latency_ms": None, "software_commit": SOFTWARE_COMMIT,
        "hardware_commit": HARDWARE_COMMIT, "FoldUopLoop_AST_equal": fold_equal,
        "runtime_features": features, "TPS_source_search_matches": matches,
        "TPS_source_search_scope": "tracked py/cc/h/md/sh in the two pinned checkouts; no entry located by these terms",
        "author_VerifyDep_call_commented": "// VerifyDep(dst_index);" in author_runtime,
        "local_VerifyDep_retained": "    VerifyDep(dst_index);" in local_runtime,
        "raw_candidate_count": len(rows), "necessary_domain_count": len(ranked),
        "selected_no_thread": next(r for r in ranked if r["tile"]["oc_nthread"] == r["tile"]["h_nthread"] == 1),
        "selected_oc_thread": next(r for r in ranked if r["tile"]["oc_nthread"] == 2),
        "selected_h_thread": next(r for r in ranked if r["tile"]["h_nthread"] == 2),
        "remaining_gates": ["author dthread pass equivalence", "runtime UOP optimizations safe adaptation",
                            "full-fusion local qualification", "new common-runtime FPGA qualification"],
        "source_hashes": {str(p.relative_to(output)): digest(p) for p in (output / "author_sources").rglob("*") if p.is_file()}})


def verify_protected(output):
    for name, value in json.loads((Path(output) / "protected.json").read_text()).items():
        if digest(name) != value:
            raise ValueError("protected historical/production source changed: " + name)
    return True
