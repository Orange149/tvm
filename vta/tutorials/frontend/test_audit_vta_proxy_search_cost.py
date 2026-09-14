from audit_vta_proxy_search_cost import replay, target


def test_prebuilt_second_candidate_is_paid_before_first_target():
    rows={'best':dict(status='passed',ratio=1.,build_seconds=10.,board_call_seconds=2.,logical_dma_bytes=5),
          'bad':dict(status='rejected',ratio=None,build_seconds=20.,board_call_seconds=3.,logical_dma_bytes=6)}
    eager=replay(['best','bad'],rows,4.,1.,['best','bad'])
    lazy=replay(['best','bad'],rows,4.,1.)
    assert target(eager,1.,0.)['measured_phase_seconds']==37.
    assert target(lazy,1.,0.)['measured_phase_seconds']==17.
    assert eager[-1]['measured_phase_seconds']==lazy[-1]['measured_phase_seconds']==40.
    reverse=replay(['bad','best'],rows,4.,1.)
    assert reverse[0]['best_ratio'] is None
    assert target(reverse,1.,0.)['dispatches']==2
    assert reverse[-1]['logical_dma_bytes']==11
