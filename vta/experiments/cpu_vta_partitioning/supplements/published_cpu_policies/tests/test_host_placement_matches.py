from audit_host_placement_matches import matches


def test_component_or_invalid_configuration_does_not_match():
    assert not matches(None)
    assert not matches({})
    assert not matches({'stages':None})
    assert not matches({'stages':[None]})
