import pytest

from cpu_strategy.analysis import aggregate, comparison, analyze
from test_campaign import setup as campaign_setup


def sample(boot, values):
    return {"boot_id":boot,"measurements":{key:{"ii_ms":value,
             "process_core_ms_per_completion":value*2,"average_cpu_cores":2.}
             for key,value in values.items()}}


def test_equal_weight_boots_and_reciprocal_completion_interval():
    rows=[sample("a",{"x":10}),sample("a",{"x":14}),sample("b",{"x":20})]
    value=aggregate(rows)["x"]
    assert value["boots"]["a"]["ii_ms"]==12
    assert value["ii_ms"]==16
    assert value["fps"]==62.5
    assert value["core_ms_per_completion"]==32


def test_pairing_and_same_configuration_are_explicit():
    values=aggregate([sample("a",{"x":10,"y":20}),sample("b",{"x":12,"y":18})])
    point=comparison(values,"x","y")
    assert point["paired_throughput_gains"]=={"a":1.,"b":.5}
    assert not point["formal_significance"]
    identical=comparison(values,"x","x")
    assert identical["same_configuration"]
    assert identical["mean_paired_gain"]==0
    assert not identical["all_boots_faster"]
    values["y"]["boots"].pop("a")
    with pytest.raises(ValueError,match="identical boot"):
        comparison(values,"x","y")


def test_invalid_evidence_cannot_create_analysis_output(tmp_path,monkeypatch):
    from cpu_strategy import campaign
    def reject(_):raise ValueError("tampered experiment identity")
    monkeypatch.setattr(campaign,"validate_state",reject)
    with pytest.raises(ValueError,match="tampered"):
        analyze(tmp_path)
    assert not (tmp_path/"analysis").exists()


def test_complete_report_uses_static_dp_and_independent_boots(campaign_setup):
    from cpu_strategy import campaign
    root, _, board = campaign_setup
    campaign.run_current(root,board)
    interim=analyze(root)
    assert interim['status']=='INCOMPLETE'
    assert interim['dp_ranking']['scalar_enumeration_checked']
    assert not interim['methods']
    for i in range(1,4):
        board.boot=f'confirmation-{i}'
        campaign.run_current(root,board)
    report=analyze(root)
    assert report['status']=='COMPLETE'
    assert len(report['dp_ranking']['rankings'])==20
    assert all(row['boots']==3 for row in report['methods'])
    assert any('静态DP' in row['method'] for row in report['methods'])
    assert not report['formal_significance']
    assert (root/'analysis/measured_cpu_configurations.svg').is_file()
