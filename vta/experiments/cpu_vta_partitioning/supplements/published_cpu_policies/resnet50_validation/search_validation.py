"""Static-only acceptance of exact search before any ResNet50 board campaign.

Small complete domains use the existing independent C++ exhaustive scorer.
The historical ResNet18 replay executes the new DP, with frozen static paths
used only as rescored incumbent upper bounds.  No measured FPS is read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from cpu_policies.open_resources import domain
from cpu_policies.open_resources.costs import Costs, ddr_stage_service, integer_weights, quantize_ns
from . import search


ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _keys(rows):
    return [(row["score_ns"], domain.canonical_tuple(row)) for row in rows]


def sources():
    """The acceptance gate seals both search and its independent verification."""
    files = [Path(search.__file__), Path(__file__), Path(__file__).with_name("search_scalar.cc"),
             Path(__file__).with_name("search_equivalence.md")]
    files += [ROOT / name for name in ("cpu_policies/open_resources/domain.py",
        "cpu_policies/open_resources/costs.py", "cpu_policies/open_resources/picojson.h",
        "model_ablation/model.py", "model_ablation/common.py", "resource_envelope/model.py",
        "staged_ablation/model.py", "staged_ablation/common.py")]
    return {str(p.resolve()): _sha(p) for p in files}


def verify_selected_cpp(rankings, costs, output, *, unit_count, max_vta_islands=3):
    """Independently re-score exactly the selected paths, not the global domain."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("search_scalar.cc")
    header = ROOT / "cpu_policies/open_resources/picojson.h"
    compiler = shutil.which("g++") or shutil.which("c++")
    if not compiler:raise RuntimeError("independent C++ scalar compiler is required")
    implementation = {str(p): _sha(p) for p in (source, header, Path(compiler).resolve())}
    fingerprint = search._digest(implementation)[:24]
    executable = output / ("selected_scalar_" + fingerprint)
    if not executable.exists():
        build = subprocess.run([compiler, "-O2", "-std=c++17", "-I", str(header.parent),
                                str(source), "-o", str(executable)], capture_output=True,
                               text=True, timeout=120)
        if build.returncode:raise RuntimeError("independent scalar build failed: " + build.stderr)
    payload, expected, identities = [], [], []
    for method, value in rankings.items():
        rows = value["rankings"] if isinstance(value, dict) else value
        for cfg in rows:
            domain.validate_candidate(cfg, costs.programs, unit_count=unit_count,
                                      max_vta_islands=max_vta_islands)
            stages = []
            for row in cfg["stages"]:
                cost = costs.get(row, cfg)
                stages.append(dict(device=row["device"], begin=row["begin"], end=row["end"],
                    mask=row["mask"], service_ns=quantize_ns(cost["service_ms"]),
                    run_ns=quantize_ns(cost["run_core_ms"]),
                    boundary_ns=quantize_ns(cost["boundary_core_ms"]),
                    ddr_ns=quantize_ns(ddr_stage_service(costs.ddr, row)),
                    weights=list(integer_weights(cost["cpu_core_weights"], threads=row["threads"]))))
            payload.append(dict(method=method, gate=cfg["execution_mode"],
                host_mask=cfg["host_mask"], compute_pool=cfg["compute_pool"],
                unit_count=unit_count, max_vta_islands=max_vta_islands, stages=stages))
            expected.append(cfg["score_ns"]); identities.append([method, cfg["id"]])
    raw = subprocess.run([str(executable.resolve())], input=json.dumps({"configurations": payload}),
                         capture_output=True, text=True, timeout=120)
    if raw.returncode:raise ValueError("independent scalar rejected path: " + raw.stderr)
    actual = json.loads(raw.stdout)["scores_ns"]
    if actual != expected:raise ValueError("independent C++ scores differ from selected DP paths")
    result = dict(passed=True, selected_paths=len(expected), scores_ns=actual,
                  configuration_ids=identities, implementation=implementation,
                  input_sha256=search._digest(payload), independent_global_enumeration=False,
                  scope="independent_integer_scalar_scores_for_selected_paths",
                  pipeline_FPS_used=False)
    _write(output / "SELECTED_CPP_VERIFICATION.json", result)
    return result


def _small(kind):
    n = 7 if kind == "four_CPU_stages" else 5
    if kind == "four_CPU_stages":
        intervals = [("cpu" if i % 2 == 0 else "vta", i, i + 1) for i in range(n)]
    else:
        intervals = [(device, begin, end) for device in ("cpu", "vta")
                     for begin in range(n) for end in range(begin + 1, n + 1)]
    programs = {}
    for device, begin, end in intervals:
        name = f"{device}_{begin:02}_{end-1:02}"
        programs[name] = dict(name=name, device=device, begin=begin, end=end)
    records = {}
    zero = kind == "all_zero_ties"
    equal = kind == "service_ties"
    for key, cfg in domain.requirements(programs).items():
        row = cfg["stages"][0]; t = row["threads"]
        cpu = row["device"] == "cpu"; length = row["end"] - row["begin"]
        service = 0 if zero else 5 if equal else ((31 + length * 7 + row["begin"]) / t
                                                 if cpu else 4 + .51 * length)
        run = 0 if zero or equal else (30 + length * 6 + t * .07 if cpu else 1.3)
        records[key] = dict(configuration=cfg, cost=dict(service_ms=service,
            run_core_ms=run, boundary_core_ms=0 if zero or equal else .017),
            session_thread_weights=[(i + 1) / (t * (t + 1) / 2) for i in range(t)])
    ddr = {"stage_service_ms": {name: 0 if zero else .031 * (p["end"] - p["begin"])
                                for name, p in programs.items()}}
    return n, programs, records, ddr


def verify_small_cpp(output):
    """Run four adversarial domains, both models, against independent C++."""
    from model_ablation.search import engine as full_engine
    from staged_ablation.search import engine as staged_engine
    output = Path(output)
    cases = []
    started = time.monotonic()
    contexts = [domain.context("shared4"), domain.context("host_shared_2"),
                domain.context("host_reserved_3")]
    for kind in ("unequal_work", "service_ties", "all_zero_ties", "four_CPU_stages"):
        n, programs, records, ddr = _small(kind)
        costs = Costs(records, programs=programs, ddr=ddr, distribution="session_weights")
        new = search.rank(programs, costs, unit_count=n, contexts=contexts,
                          output_dir=output / kind / "new")
        if new["status"] != "COMPLETE":raise ValueError("small-domain DP did not finish")
        for method in search.METHODS:
            engine = (full_engine(output / kind / method / "source", "complete")
                      if method == "complete" else staged_engine(output / kind / method / "source", "tarnawski"))
            old_costs = Costs(records, programs=programs, ddr=ddr,
                              distribution="session_weights" if method == "complete" else "balanced")
            old = engine.rank_all(programs, old_costs, ddr=ddr, unit_count=n,
                                    contexts=contexts, output_dir=output / kind / method / "oracle",
                                    verify=True)
            key = "enhanced" if method == "complete" else "single_core"
            same = _keys(new["methods"][method]["rankings"]) == _keys(old["methods"][key]["rankings"])
            if not same or not old["verification"]["passed"]:
                raise ValueError("new DP differs from independent exhaustive C++: " + kind + "/" + method)
            cases.append(dict(case=kind, method=method, independent_Cpp_enumeration=True,
                              ordered_Top20_identical=True,
                              complete_domain_count=new["counts"][method],
                              new_statistics=new["statistics"],
                              oracle_implementation=old.get("implementation")))
        verify_selected_cpp(new["methods"], costs, output / kind / "selected_cpp", unit_count=n)
    result = dict(status="passed", cases=cases, elapsed_seconds=time.monotonic() - started,
                  new_board_sessions=0, new_inferences=0, pipeline_FPS_used=False,
                  implementation=search._implementation())
    _write(output / "SMALL_CPP_VERIFICATION.json", result)
    return result


def _resnet18_static_lists():
    frozen = ROOT / "artifacts/run_scope_common_validation_01"
    seal = _read(frozen / "FREEZE.json")
    if not seal["no_pipeline_FPS_input"]:raise ValueError("static reference lacks feedback isolation")
    lists_path = frozen / "LISTS.json"
    if _sha(lists_path) != seal["files"]["LISTS.json"]:
        raise ValueError("frozen static rankings changed")
    lists = _read(lists_path)["resnet18"]
    return dict(complete=lists["original_complete"], tarnawski=lists["tarnawski"]), {
        str(lists_path): _sha(lists_path), str(frozen / "FREEZE.json"): _sha(frozen / "FREEZE.json")}


def resnet18_inputs(*, include_expected=True):
    """Load only isolated costs, qualified modules and frozen static lists."""
    plan_path = ROOT / "artifacts/run_scope_rank_01/PLAN.json"
    plan = _read(plan_path)
    if plan["model_switches"] != dict(host_CPU_demand=True, host_concentration=True,
                                      team_guard=True, thread_weights=True):
        raise ValueError("ResNet18 reference is not the four-on strongest complete model")
    spec = plan["networks"]["resnet18"]
    sources = {str(plan_path): _sha(plan_path)}
    for stem in ("cost", "ddr", "programs"):
        p = Path(spec[stem + "_path"])
        if _sha(p) != spec[stem + "_sha256"]:
            raise ValueError("frozen ResNet18 static input changed: " + stem)
        sources[str(p)] = _sha(p)
    expected = None
    if include_expected:
        expected, refs = _resnet18_static_lists(); sources.update(refs)
    programs = _read(spec["programs_path"])
    costs = Costs(_read(spec["cost_path"]), programs=programs,
                  ddr=_read(spec["ddr_path"]), distribution="session_weights")
    return programs, costs, expected, sources


def replay_resnet18(output, *, budget_seconds=7200, memory_bytes=8 * 1024**3,
                    resume=False, max_expansions=None, use_static_incumbents=False):
    output = Path(output)
    programs, costs, expected, input_sources = resnet18_inputs(include_expected=use_static_incumbents)
    started = time.monotonic()
    result = search.rank(programs, costs, unit_count=21, max_vta_islands=3,
                          output_dir=output, budget_seconds=budget_seconds,
                          memory_bytes=memory_bytes, resume=resume,
                          incumbents=expected if use_static_incumbents else None,
                          max_expansions=max_expansions)
    # Search output is already frozen on disk before the cold comparator opens
    # historical lists.  Expected rankings cannot affect candidate ordering.
    if expected is None:
        expected, refs = _resnet18_static_lists(); input_sources.update(refs)
    report = dict(status="passed" if result["status"] == "COMPLETE" else "incomplete",
                  actual_new_DP_executed=True,
                  static_lists_used_to_seed=use_static_incumbents,
                  old_static_paths_only_as_rescored_upper_bounds=use_static_incumbents,
                  static_source_hashes=input_sources, search_result_sha256=_sha(output / "search_result.json"),
                  search_status=result["status"], models={},
                  actual_elapsed_seconds=time.monotonic() - started,
                  total_DP_elapsed_seconds=result['elapsed_seconds'],
                  statistics=result['statistics'],
                  new_board_sessions=0, new_inferences=0, pipeline_FPS_used=False,
                  implementation=search._implementation())
    for method in search.METHODS:
        rows = result["methods"][method]["rankings"]
        identical = _keys(rows) == _keys(expected[method])
        report["models"][method] = dict(ordered_Top20_identical=identical,
             complete_domain_count=result["counts"][method], statistics=result["statistics"],
             first_id=rows[0]["id"] if rows else None,
             first_score_ns=rows[0]["score_ns"] if rows else None)
        if result["status"] == "COMPLETE" and not identical:
            report["status"] = "failed"
    if result["status"] == "COMPLETE":
        report["selected_Cpp_verification"] = verify_selected_cpp(result["methods"], costs,
            output / "selected_cpp", unit_count=21)
    if result["counts"]["complete"] != 966240306 or result["counts"]["tarnawski"] != 6121596:
        report["status"] = "failed"
    _write(output / "RESNET18_REPLAY.json", report)
    if report["status"] == "failed":raise ValueError("ResNet18 exact replay differs")
    return report


def acceptance(output, *, budget_seconds=7200, memory_bytes=8 * 1024**3, resume=False):
    """Write the board gate only after actual independent tests and full replay."""
    output = Path(output)
    version = search._digest(sources())[:20]
    work = output / "dp_validation" / version
    small_path = work / "small/SMALL_CPP_VERIFICATION.json"
    if resume and small_path.exists():
        small = _read(small_path)
        if small["implementation"] != search._implementation():
            raise ValueError("small-domain acceptance uses another search implementation")
    else:
        small = verify_small_cpp(small_path.parent)
    replay_output = work / "resnet18"
    replay = replay_resnet18(replay_output, budget_seconds=budget_seconds,
                             memory_bytes=memory_bytes,
                             resume=resume and (replay_output / "search_checkpoint.jsonl.gz").exists())
    passed = small["status"] == replay["status"] == "passed"
    result = dict(status="passed" if passed else "incomplete", passed=passed,
                  small_independent_Cpp_enumeration=True,
                  ResNet18_fresh_DP_replay=replay["status"],
                  ResNet18_cold_replay=True, historical_rankings_used_to_seed=False,
                  independent_global_enumeration_r50=False,
                  implementation=search._implementation(),
                  sources=sources(),
                  files={str(small_path): _sha(small_path),
                         str(replay_output / "RESNET18_REPLAY.json"):
                             _sha(replay_output / "RESNET18_REPLAY.json")},
                  new_board_sessions=0, new_inferences=0, pipeline_FPS_used=False)
    _write(output / "SEARCH_ACCEPTANCE.json", result)
    return result
