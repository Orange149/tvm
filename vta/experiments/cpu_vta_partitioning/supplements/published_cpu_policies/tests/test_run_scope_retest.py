from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_scope_retest import subset, parent, METHODS
from cpu_vta_partitioning.common import read


def test_old_new_union_preserves_rank_order_and_removes_only_extra_baseline():
    old = read(parent.OUT / 'LISTS.json')
    lists, pool, order = subset(old, read(parent.OUT / 'POOL.json'), read(parent.OUT / 'ORDER.json'))
    assert {n: len(v) for n, v in pool.items()} == {'resnet18': 20, 'yolov3_tiny416': 28}
    assert len(order) == 48
    for network in lists:
        assert set(lists[network]) == set(METHODS)
        for method in METHODS:
            assert lists[network][method] == old[network][method]
    assert lists['resnet18'][METHODS[0]] == lists['resnet18'][METHODS[1]]


def test_missing_or_duplicate_execution_order_is_rejected():
    old = read(parent.OUT / 'LISTS.json')
    pool = read(parent.OUT / 'POOL.json')
    order = read(parent.OUT / 'ORDER.json')
    _, _, selected = subset(old, pool, order)
    with pytest.raises(ValueError, match='incomplete or repeated'):
        subset(old, pool, selected[:-1])
    with pytest.raises(ValueError, match='incomplete or repeated'):
        subset(old, pool, selected + selected[:1])
