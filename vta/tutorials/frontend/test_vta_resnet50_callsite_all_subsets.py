from itertools import product

from build_vta_resnet50_callsite_marginal_graphs import mask_name


def test_four_node_mask_domain_has_sixteen_unique_names():
    masks = list(product((0, 1), repeat=4))
    assert len(masks) == 16
    assert len({mask_name(mask) for mask in masks}) == 16
    assert masks[0] == (0, 0, 0, 0)
    assert masks[-1] == (1, 1, 1, 1)
