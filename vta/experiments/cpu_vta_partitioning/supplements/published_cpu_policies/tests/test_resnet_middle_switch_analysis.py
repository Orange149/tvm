"""Critical-path evidence remains limited when the original campaign rejects it."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze_resnet_middle_switch as audit
from test_resnet_middle_switch import profile_fixture


def retired_fixture():
    profile,doc,affinity=profile_fixture()
    doc.update(warmup=0,scored=2,tail=0)
    roles=audit.campaign.old_events.role_map(doc,affinity)
    producer=next(s for s in profile['streams'] if roles[s['tid']]['role']=='producer')
    producer['alive_at_stop']=False
    feeds=[dict(stage='_producer',operation='feed',call=i,tid=producer['tid'],
        wall_start_ns=5000+i*100,wall_end_ns=5050+i*100) for i in range(2)]
    return profile,doc,affinity,feeds


def test_only_completed_producer_can_be_retired_in_diagnostic_view():
    p,d,a,feeds=retired_fixture();r=audit.diagnostic_events(p,d,a,feeds)
    assert len(r['streams'])==13 and len(r['offline_lifecycle_exception'])==1
    assert r['original_campaign_rejection_preserved']
    assert next(s for s in p['streams'] if not s['alive_at_stop'])['alive_at_stop'] is False
    bad=deepcopy(p);bad['streams'][0]['alive_at_stop']=False
    with pytest.raises(ValueError,match='critical thread exited'):
        audit.diagnostic_events(bad,d,a,feeds)


@pytest.mark.parametrize('fault',['missing','duplicate','wrong_tid','unfinished'])
def test_retired_producer_requires_all_finite_feed_events(fault):
    p,d,a,feeds=retired_fixture()
    if fault=='missing':feeds.pop()
    elif fault=='duplicate':feeds[1]['call']=0
    elif fault=='wrong_tid':feeds[0]['tid']+=999
    else:feeds[-1]['wall_end_ns']=p['stop_start_ns']+1
    with pytest.raises(ValueError,match='finite feed evidence'):
        audit.diagnostic_events(p,d,a,feeds)


def test_critical_interval_reconstructs_wall_and_retains_unknowns():
    a=dict(tid=1,role=dict(role='target'),
        on=[dict(start_ns=0,end_ns=10,cpu=1,tid=1),dict(start_ns=90,end_ns=100,cpu=1,tid=1)],
        off=[dict(start_ns=10,end_ns=90,from_cpu=1,to_cpu=1,preempt=True,tid=1)])
    b=dict(tid=2,role=dict(role='other_stage'),on=[dict(start_ns=10,end_ns=70,cpu=1,tid=2)],off=[])
    r=audit.interval(a,'return',0,100,audit.Attribution({'1':a,'2':b}),1)
    assert r['on_CPU_ns']==20 and r['preempt_off_ns']==80
    assert r['actor_CPU_ns']=={'other_stage':60} and r['unobserved_actor_ns']==20
    assert r['wall_ns']==r['on_CPU_ns']+r['off_CPU_ns']
    with pytest.raises(ValueError,match='not fully covered'):
        audit.interval(a,'return',0,101,audit.Attribution({'1':a,'2':b}),1)
