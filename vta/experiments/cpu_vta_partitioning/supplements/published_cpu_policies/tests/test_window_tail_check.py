from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_window_tail_check as check
from cpu_vta_partitioning.common import read
from common_top20.runtime import task,command


def test_representatives_are_only_the_previously_frozen_top1_and_short_winner():
    old=read(check.OLD/'analysis/RESULTS.json');pool=read(check.OLD/'POOL.json')
    selected,lists,order=check.representatives(old,pool)
    assert {n:len(rows) for n,rows in selected.items()}=={'resnet18':2,'yolov3_tiny416':2}
    assert len(order)==4
    for network in selected:
        assert lists[network]['static_first'][0]['id']==next(r['configuration'] for r in old['budgets']
            if r['network']==network and r['method']=='original_complete' and r['budget']==1)


def test_native_is_blocked_without_approval_and_wrong_boot_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(check,'OUT',tmp_path)
    with pytest.raises(ValueError,match='not approved'):
        check.board(execute_board=True)


def test_prepared_performance_window_is_8_16_24_and_command_preserves_16_timestamps():
    if not (check.OUT/'PREPARATION.json').exists():
        pytest.skip('capsule not yet prepared')
    prepared=check.assert_sources()
    protocol=read(check.OUT/'PROTOCOL.json')
    assert protocol['performance']==protocol['short_window']
    assert [protocol[k] for k in ('maximum_distinct_configurations','native_sessions','DNN_inferences')]==[4,8,196]
    assert protocol['additional_native_campaign_requires_user_approval']
    for network in prepared['bundles']:
        bundle=check.OUT/network/'capsule';check.validate(bundle)
        cfg=next(iter(read(check.OUT/'POOL.json')[network].values()))
        doc=task(bundle,cfg,'short')
        assert [doc[k] for k in ('warmup','scored','tail')]==[8,16,24]
        argv,_=command(doc,bundle,str(bundle),'/tmp/tail-window-command')
        assert '--queue-timing' not in argv
        assert argv[argv.index('--output-mode')+1]=='raw'
