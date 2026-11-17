"""Structural-domain tests; no compilation, inference or transport."""
from itertools import combinations
import json
from pathlib import Path

import pytest

from resnet50_validation.domain import programs, summary, topology_counts


def residual_units(blocks, projections):
    kinds = ['stem']
    for block in range(blocks):
        kinds.append('main_preadd')
        if block in projections:
            kinds.append('projection')
        kinds.append('residual_merge')
    kinds.append('head')
    return [{'index': index, 'kind': kind} for index, kind in enumerate(kinds)]


def test_same_structural_rule_exactly_recovers_the_old_resnet18_domain():
    value = summary(residual_units(8, (2, 4, 6)))
    assert value['unit_count'] == 21
    assert value['topologies_per_island_count'] == {'1': 87, '2': 990, '3': 3546}
    assert (value['search_CPU_stages'], value['VTA_stages']) == (85, 87)
    assert value['total_basic_signature_count'] == 775
    assert value['complete_resource_configurations'] == 966_240_306
    assert value['tarnawski_resource_configurations'] == 6_121_596


def test_resnet50_predeclared_structural_domain_and_no_atomic_extras():
    units = residual_units(16, (0, 3, 7, 13))
    value = summary(units)
    assert value['unit_count'] == 38
    assert value['topologies_per_island_count'] == {'1': 299, '2': 13504, '3': 217254}
    assert value['topology_count'] == 231057
    assert (value['search_CPU_stages'], value['VTA_stages']) == (314, 299)
    assert (value['CPU_signature_count'], value['VTA_signature_count']) == (1256, 1495)
    assert value['total_basic_signature_count'] == 2751
    assert value['complete_resource_configurations'] == 58_267_668_578
    assert value['tarnawski_resource_configurations'] == 354_289_812
    rows = programs(units)
    assert len({r['name'] for r in rows}) == len(rows) == 613
    assert all(r['eligible_search'] and r['purpose'] == 'search_stage' for r in rows)
    starts = {u['index'] for u in units if u['kind'] == 'main_preadd'}
    assert {r['begin'] for r in rows if r['device'] == 'vta'} == starts


@pytest.mark.parametrize('blocks,projections', [(1, (0,)), (2, ()), (3, (0, 2))])
def test_reachable_edges_and_path_counts_match_independent_transition_enumeration(blocks, projections):
    units = residual_units(blocks, projections)
    n = len(units)
    starts = {u['index'] for u in units if u['kind'] == 'main_preadd'}
    edges, counts = set(), {}
    for islands in range(1, 4):
        count = 0
        for cuts in combinations(range(1, n), 2 * islands):
            if not all(cuts[i] in starts for i in range(0, len(cuts), 2)):
                continue
            bounds = (0, *cuts, n)
            count += 1
            edges.update(('cpu' if index % 2 == 0 else 'vta', begin, end)
                         for index, (begin, end) in enumerate(zip(bounds, bounds[1:])))
        counts[str(islands)] = count
    assert topology_counts(units) == counts
    actual = {(r['device'], r['begin'], r['end']) for r in programs(units)}
    assert actual == edges


def test_domain_rejects_unowned_or_nonconsecutive_metadata():
    with pytest.raises(ValueError, match='consecutively'):
        summary([{'index': 0, 'kind': 'stem'}, {'index': 2, 'kind': 'main_preadd'}, {'index': 3, 'kind': 'head'}])
    with pytest.raises(ValueError, match='main-path'):
        summary([{'index': i, 'kind': 'stem'} for i in range(4)])


@pytest.fixture(scope='module')
def actual_graph():
    path = Path(__file__).resolve().parents[1] / 'artifacts/resnet50_v2_validation_01/graph/canonical_qir.json'
    if not path.is_file():
        pytest.skip('requires the immutable local common graph')
    tvm = pytest.importorskip('tvm')
    from resnet50_validation.units import UnitGraph
    return UnitGraph(tvm.ir.load_json(path.read_text()))


def test_actual_v2_graph_has_unique_ownership_and_preserves_pre_activation(actual_graph):
    graph = actual_graph
    assert len(graph.units) == 38
    assert len([u for u in graph.units if u['kind'] == 'main_preadd']) == 16
    assert len([u for u in graph.units if u['kind'] == 'projection']) == 4
    assert all(node in graph.owner for node in graph.calls)
    assert sum(u['operator_counts'].get('nn.conv2d', 0) for u in graph.units) == 53
    assert all(u['operator_counts'].get('nn.relu', 0) == 0
               for u in graph.units if u['kind'] == 'residual_merge')
    assert graph.recompose([(i, i + 1) for i in range(38)])


def test_projection_frontier_keeps_live_main_output_and_shared_activation(actual_graph):
    graph = actual_graph
    mod, incoming, outgoing = graph.stage(2, 3)
    assert len(incoming) == len(outgoing) == 2
    assert len(set(incoming).intersection(outgoing)) == 1
    assert len(mod['main'].params) == 2
    # The first projection is stride1, not the old v1 downsampling assumption.
    conv = next(node for node in graph.calls
                if graph.owner[node] == 2 and node.op.name == 'nn.conv2d')
    assert list(map(int, conv.attrs.strides)) == [1, 1]
    assert graph.recompose([(0, 2), (2, 3), (3, 38)])


def test_stage_and_recomposition_reject_invalid_partitions(actual_graph):
    graph = actual_graph
    for begin, end in [(-1, 1), (2, 2), (0, 39)]:
        with pytest.raises(ValueError, match='invalid stage'):
            graph.stage(begin, end)
    with pytest.raises(ValueError, match='continuous'):
        graph.recompose([(0, 2), (3, 38)])


def test_prepared_interval_proof_is_bound_to_actual_graph():
    base = Path(__file__).resolve().parents[1] / 'artifacts/resnet50_v2_validation_01'
    status_path = base / 'units/UNIT_STATUS.json'
    if not status_path.is_file():
        pytest.skip('all-interval preparation is still pending')
    from resnet50_validation.graph import file_hash
    status = json.loads(status_path.read_text())
    assert status['source_sha256'] == file_hash(base / 'graph/canonical_qir.json')
    assert status['interval_checks'] == 741 and status['all_interval_recompositions_exact']
    assert status['extra_quantize_calls'] == 0 and not status['board_accessed']
