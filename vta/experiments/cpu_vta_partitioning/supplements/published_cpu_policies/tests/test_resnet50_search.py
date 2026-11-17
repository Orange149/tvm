"""New DP correctness is checked against the existing independent C++ oracle."""
from copy import deepcopy
import gzip
from pathlib import Path

import pytest

from cpu_policies.open_resources import domain
from cpu_policies.open_resources.costs import Costs
from resnet50_validation import search


def fixture(n=5, equal=False, zero=False):
    programs = {}
    for device in ("cpu", "vta"):
        for begin in range(n):
            for end in range(begin + 1, n + 1):
                name = f"{device}_{begin:02}_{end-1:02}"
                programs[name] = dict(name=name, device=device, begin=begin, end=end)
    records = {}
    for key, cfg in domain.requirements(programs).items():
        row = cfg["stages"][0]; threads = row["threads"]
        cpu = row["device"] == "cpu"
        length = row["end"] - row["begin"]
        wall = (0 if zero else 5 if equal else
                (11 + row["begin"] + length * 7) / threads + .031 * threads
                if cpu else 3 + .37 * length)
        run = 0 if equal or zero else (8 + length * 6 + threads * .13 if cpu else .9)
        boundary = 0 if equal or zero else .007 * (row["begin"] + 1)
        weights = [i + 1 for i in range(threads)]
        records[key] = dict(configuration=cfg, cost=dict(service_ms=wall,
            run_core_ms=run, boundary_core_ms=boundary),
            session_thread_weights=[w / sum(weights) for w in weights])
    ddr = {"stage_service_ms": {name: {str(t): .019 * (p["end"] - p["begin"]) * t
                   for t in (range(1, 5) if p["device"] == "cpu" else [1])}
                   for name, p in programs.items()}}
    if zero:ddr = {"stage_service_ms": {name: 0 for name in programs}}
    return programs, Costs(records, programs=programs, ddr=ddr, distribution="session_weights")


def compare_keys(rows):
    return [(row["score_ns"], domain.canonical_tuple(row)) for row in rows]


@pytest.mark.parametrize("method", ["complete", "tarnawski"])
@pytest.mark.parametrize("equal,zero", [(False, False), (True, False), (False, True)])
def test_small_full_domain_matches_independent_cpp(tmp_path, method, equal, zero):
    programs, costs = fixture(5, equal=equal, zero=zero)
    contexts = [domain.context("shared4"), domain.context("host_reserved_3")]
    result = search.rank(programs, costs, unit_count=5, max_vta_islands=2,
                         methods=[method], contexts=contexts, output_dir=tmp_path / "new")
    assert result["status"] == "COMPLETE"
    assert result["counts"]["topologies"] == 7
    assert not result["pipeline_FPS_used"]
    assert not result["independent_global_enumeration"]
    # The old frozen implementation's C++ exhaustive scorer independently
    # enumerates this full domain, including all resource placements and ties.
    if method == "complete":
        from model_ablation.search import engine
        module = engine(tmp_path / "old", "complete")
    else:
        from staged_ablation.search import engine
        module = engine(tmp_path / "old", "tarnawski")
    old_costs = Costs(costs.records, programs=programs, ddr=costs.ddr,
                      distribution="session_weights" if method == "complete" else "balanced")
    old = module.rank_all(programs, old_costs, ddr=costs.ddr, unit_count=5,
                           max_vta_islands=2, contexts=contexts,
                           output_dir=tmp_path / "old_search", verify=True)
    assert old["verification"]["passed"]
    selector = "enhanced" if method == "complete" else "single_core"
    assert compare_keys(result["methods"][method]["rankings"]) == compare_keys(old["methods"][selector]["rankings"])


def test_checkpoint_resume_is_exact_and_rejects_changed_inputs(tmp_path):
    programs, costs = fixture(5)
    kwargs = dict(unit_count=5, max_vta_islands=2,
                  contexts=[domain.context("shared4")], output_dir=tmp_path / "resume")
    partial = search.rank(programs, costs, max_expansions=7, **kwargs)
    assert partial["status"] == "SEARCH_BUDGET_EXCEEDED"
    assert partial["partial_rankings_are_not_certified"]
    changed = Costs(deepcopy(costs.records), programs=programs, ddr=costs.ddr,
                    distribution="session_weights")
    next(iter(changed.records.values()))["cost"]["service_ms"] += 1
    with pytest.raises(ValueError, match="input or implementation"):
        search.rank(programs, changed, resume=True, **kwargs)
    resumed = search.rank(programs, costs, resume=True, **kwargs)
    fresh = search.rank(programs, costs, unit_count=5, max_vta_islands=2,
                        contexts=kwargs["contexts"])
    assert resumed["status"] == "COMPLETE"
    for method in search.METHODS:
        assert compare_keys(resumed["methods"][method]["rankings"]) == compare_keys(fresh["methods"][method]["rankings"])
    assert resumed["statistics"]["transitions"] == fresh["statistics"]["transitions"]
    path = Path(resumed["checkpoint"])
    lines = gzip.open(path, "rt").readlines()
    lines[-1] = lines[-1].replace('"sha256": "', '"sha256": "broken')
    with gzip.open(path, "wt") as f:f.writelines(lines)
    with pytest.raises(ValueError, match="content hash"):
        search.rank(programs, costs, resume=True, **kwargs)


def test_warm_start_is_only_a_rescored_upper_bound():
    programs, costs = fixture(5)
    kwargs = dict(unit_count=5, max_vta_islands=2, contexts=[domain.context("shared4")])
    first = search.rank(programs, costs, **kwargs)
    seeds = {method: [{**row, "score_ns": 1} for row in info["rankings"]]
             for method, info in first["methods"].items()}
    seeded = search.rank(programs, costs, incumbents=seeds, **kwargs)
    unpruned = search.rank(programs, costs, prune=False, **kwargs)
    for method in search.METHODS:
        expected = compare_keys(first["methods"][method]["rankings"])
        assert compare_keys(seeded["methods"][method]["rankings"]) == expected
        assert compare_keys(unpruned["methods"][method]["rankings"]) == expected
    assert seeded["statistics"]["strict_bound_prunes"] > 0


def test_exact_equivalence_keeps_twenty_paths_and_zero_cost_group_identity():
    programs, costs = fixture(5, zero=True)
    result = search.rank(programs, costs, unit_count=5, max_vta_islands=2,
                         methods=["complete"], contexts=[domain.context("shared4")])
    assert len(result["methods"]["complete"]["rankings"]) == 20
    assert result["statistics"]["exact_labels_merged"] > 0
    assert result["statistics"]["equivalent_prefixes_discarded"] > 0
    assert not search._compatible(1 << 3, 6)
    assert search._compatible(1 << 3, 3)
    assert search._compatible(1 << 3, 12)


def test_inconsistent_frontier_and_engineering_budget_rejected(tmp_path):
    programs, costs = fixture(3)
    p = deepcopy(programs)
    p["cpu_00_00"].update(output_node_ids=[1], output_types=[dict(shape=[1], dtype="int8")])
    p["vta_01_01"].update(input_node_ids=[2], input_types=[dict(shape=[1], dtype="int8")])
    with pytest.raises(ValueError, match="inconsistent frontier"):
        search.rank(p, costs, unit_count=3)
    result = search.rank(programs, costs, unit_count=3, memory_bytes=1,
                         output_dir=tmp_path)
    assert result["status"] == "SEARCH_BUDGET_EXCEEDED" and result["stop_reason"] == "memory_budget"


def test_scalar_label_formula_and_bounds_agree():
    from model_ablation.model import score as full_score
    from staged_ablation.model import score as base_score
    programs, costs = fixture(3)
    for ctx in domain.contexts():
        for method in search.METHODS:
            opts = search._options(programs, costs, ctx, method)
            wanted = [o for o in opts if (o.row["device"], o.row["begin"], o.row["end"])
                      in (("cpu", 0, 1), ("vta", 1, 2), ("cpu", 2, 3))]
            first = [o for o in wanted if o.row["begin"] == 0]
            second = [o for o in wanted if o.row["device"] == "vta"][0]
            third = [o for o in wanted if o.row["begin"] == 2]
            for a in first:
                for b in third:
                    for gate in (("off", "resource_group") if method == "complete" else ("resource_group",)):
                        if gate == "resource_group" and a.bits != b.bits and a.bits & b.bits:continue
                        label = search.ZERO_COMPLETE if method == "complete" else search.ZERO_TARNAWSKI
                        for opt in (a, second, b):label = search._advance(label, opt, method, gate)
                        cfg = domain.candidate([a.row, second.row, b.row], ctx, gate)
                        bc = Costs(costs.records, programs=programs, ddr=costs.ddr, distribution="balanced")
                        expected = (full_score(cfg, costs, "complete") if method == "complete"
                                    else base_score(cfg, bc, "tarnawski"))["score_ns"]
                        assert search._value(label, ctx, method, gate) == expected


def test_more_than_twenty_one_units_and_independent_selected_cpp(tmp_path):
    from resnet50_validation.search_validation import verify_selected_cpp
    original, costs = fixture(23)
    names = ("cpu_00_00", "vta_01_21", "cpu_22_22")
    programs = {name: original[name] for name in names}
    costs.programs = programs
    result = search.rank(programs, costs, unit_count=23,
                         contexts=[domain.context("host_shared_2")])
    assert result["status"] == "COMPLETE" and result["counts"]["topologies"] == 1
    report = verify_selected_cpp(result["methods"], costs, tmp_path / "scalar", unit_count=23)
    assert report["passed"] and report["selected_paths"] == sum(
        len(v["rankings"]) for v in result["methods"].values())
    assert not report["independent_global_enumeration"]


def test_current_scalar_order_is_not_a_safe_state_merge():
    ctx = domain.context("host_shared_3")
    # Prefix A is locally faster, but puts its demand on the future hot core.
    a, b, future = [], [], []
    for begin, core, value in ((0, 0, 10), (0, 1, 11), (2, 0, 100)):
        row = domain.stage("cpu", begin, begin + 1, 1, (core,))
        option = search.Option(row, (), value, value, 0, 0,
                               tuple(value if i == core else 0 for i in range(4)), 1 << core)
        (a if not a else b if not b else future).append(option)
    first = search._advance(search.ZERO_COMPLETE, a[0], "complete", "off")
    second = search._advance(search.ZERO_COMPLETE, b[0], "complete", "off")
    assert search._value(first, ctx, "complete", "off") < search._value(second, ctx, "complete", "off")
    assert first != second
    assert search._value(search._advance(first, future[0], "complete", "off"), ctx, "complete", "off") > search._value(
        search._advance(second, future[0], "complete", "off"), ctx, "complete", "off")


def test_search_never_imports_tvm_or_reads_pipeline_results(tmp_path, monkeypatch):
    import builtins
    programs, costs = fixture(3)
    original = Path.open
    def guard(path, *args, **kwargs):
        if path.name in ("FPS.json", "performance.json", "configurations.csv", "LISTS.json", "rankings.json") or "/accepted/" in str(path):
            raise AssertionError("search opened measured pipeline results")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    original_import = builtins.__import__
    def guard_import(name, *args, **kwargs):
        if name == 'tvm' or name.startswith('tvm.') or name == 'vta' or name.startswith('vta.'):
            raise AssertionError('pure search imported a compiler/runtime module')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard_import)
    result = search.rank(programs, costs, unit_count=3, output_dir=tmp_path)
    assert result["status"] == "COMPLETE"
    initializer = result['statistics']['one_island_initializer']
    assert initializer['evaluated_configurations'] > 0 and not initializer['historical_rankings_used']
    assert result['statistics']['warm_start_paths'] == {'tarnawski': 0, 'complete': 0}


def test_current_cost_initializer_does_not_restrict_the_search():
    programs, costs = fixture(5)
    options = dict(unit_count=5, contexts=[domain.context('host_shared_2')])
    seeded = search.rank(programs, costs, **options)
    plain = search.rank(programs, costs, initialize_one_island=False, **options)
    for method in search.METHODS:
        assert compare_keys(seeded['methods'][method]['rankings']) == compare_keys(plain['methods'][method]['rankings'])
    assert seeded['counts'] == plain['counts']


def test_four_on_algebra_preserves_all_original_constraints():
    import random
    rng = random.Random(20261116)
    for ctx in domain.contexts():
        masks = domain.nonempty_masks(ctx["compute_pool"])
        for _ in range(120):
            label = list(search.ZERO_COMPLETE)
            used = 0
            gate = rng.choice(("off", "resource_group"))
            for stage in range(rng.randrange(1, 5)):
                choices = [m for m in masks if gate == "off" or
                           search._compatible(used, sum(1 << c for c in m))]
                mask = rng.choice(choices); bits = sum(1 << c for c in mask)
                used |= 1 << bits
                service = rng.randrange(0, 10**8)
                label[7] = max(label[7], service)
                for core in mask:
                    label[core] += rng.randrange(0, 10**8)
                    label[8 + core] += service
                if gate == "resource_group":label[11 + bits] += service
            label[4] = rng.randrange(0, 10**9)
            label[5] = rng.randrange(0, 10**9)
            label[6] = rng.randrange(0, 10**9)
            capacities = []
            host_bits = sum(1 << c for c in ctx["host_mask"])
            for subset in range(1, 16):
                work = sum(label[c] for c in range(4) if subset & (1 << c))
                if subset & host_bits == host_bits:work += label[4]
                width = subset.bit_count(); capacities.append((work + width - 1) // width)
            host_guard = max(label[c] + (label[4] if c in ctx["host_mask"] else 0) for c in range(4))
            original = max(label[5], label[6], label[7], *capacities, *label[8:12], host_guard,
                           *(label[12:27] if gate == "resource_group" else [0]))
            assert search._value(tuple(label), ctx, "complete", gate) == original
