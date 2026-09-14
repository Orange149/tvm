from build_vta_resnet50_r50a_callsite_subsets import mask_name


def test_holdout_variant_name_is_exact_three_bit_mask():
    assert mask_name((0, 0, 0)) == "mask000"
    assert mask_name((1, 1, 1)) == "mask111"
