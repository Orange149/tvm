from pathlib import Path
from test_resource_campaign import setup
from cpu_strategy.policy import campaign
from cpu_strategy.policy.analysis import analyze


def test_complete_report_uses_last_three_boots_and_all_top20(setup):
    root,_,board=setup
    campaign.run_current(root,board)
    partial=analyze(root)
    assert partial['status']=='INCOMPLETE' and partial['policy_support'] is None
    for i in range(1,6):
        board.boot='boot'+str(i+1);campaign.run_current(root,board)
    report=analyze(root)
    assert report['status']=='COMPLETE' and len(report['top20'])==20
    assert all(row['boots']==3 for row in report['methods'])
    assert not report['formal_significance']
    assert (root/'analysis/top20.svg').exists()
    assert (root/'analysis/thread_wall_core_curves.svg').exists()
