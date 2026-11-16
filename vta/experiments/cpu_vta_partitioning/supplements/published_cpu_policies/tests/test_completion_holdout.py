from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from prepare_completion_holdout import choose
from cpu_policies.open_resources.domain import candidate,stage


def cfg(mask):
    return candidate([stage('cpu',0,3,len(mask),mask),stage('vta',3,20,1,[0,1,2,3]),stage('cpu',20,21,1,[0])])


def test_selection_unites_three_frozen_first_choices_with_bounded_budget():
    items={c['id']:dict(configuration=c) for c in [cfg([0]),cfg([1]),cfg([2])]}
    ids=list(items);scores={cid:dict(static=i,aggregate=(i+1)%3,midpoint=(i+2)%3,lower=i,upper=i) for i,cid in enumerate(ids)}
    ranks,selected=choose(scores,items)
    assert set(selected)==set(ids) and selected[0]==ranks['static'][0]


def test_identical_first_choices_add_one_disagreement_without_arbitrary_fill():
    items={c['id']:dict(configuration=c) for c in [cfg([0]),cfg([1]),cfg([2])]}
    scores={cid:dict(static=i,aggregate=i,midpoint=i,lower=i,upper=i) for i,cid in enumerate(items)}
    ranks,selected=choose(scores,items)
    assert len(selected)==2 and selected[0]==ranks['static'][0]


def test_equal_scores_use_full_physical_configuration_not_hash_order():
    configs=[cfg([3]),cfg([0]),cfg([2])];items={c['id']:dict(configuration=c) for c in configs}
    scores={cid:dict(static=1,aggregate=1,midpoint=1,lower=1,upper=1) for cid in items}
    ranks,_=choose(scores,items)
    assert ranks['static'][0]==configs[1]['id']
