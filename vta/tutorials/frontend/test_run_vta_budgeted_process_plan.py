import pytest
import json
import sys
import time
from run_vta_budgeted_process_plan import execute, validate, run_step


def plan():
    return dict(schema='c3_budgeted_process_plan_v1',budget_seconds=5.,
                schedule='one_candidate_build_then_measure',steps=[
        dict(name='build_a',kind='build',candidate_id='a',argv=['true'],cwd='/tmp'),
        dict(name='measure_a',kind='measure',candidate_id='a',argv=['true'],cwd='/tmp',
             completion_receipt=dict(path='/tmp/a-result.json',required_values=dict(candidate_id='a',status='passed'))),
        dict(name='build_b',kind='build',candidate_id='b',argv=['true'],cwd='/tmp'),
        dict(name='measure_b',kind='measure',candidate_id='b',argv=['true'],cwd='/tmp',
             completion_receipt=dict(path='/tmp/b-result.json',required_values=dict(candidate_id='b',status='passed')))])


def test_late_measurement_is_charged_but_never_eligible(tmp_path):
    moments=iter([1.,2.,3.,6.,6.1])
    result=execute(plan(),tmp_path,0.,clock=lambda:next(moments),
                   action=lambda s,p:dict(returncode=0,completion_verified=s['kind']=='measure'))
    assert result['budget_eligible_completed_candidates']==[]
    assert result['status']=='budget_overrun_current_step_drained'
    assert result['unstarted_steps']==2
    assert result['steps'][-1]['overrun_seconds']==1.
    assert result['elapsed_seconds']==6.1


def test_failed_build_cannot_produce_or_measure_label(tmp_path):
    moments=iter([1.,2.,2.1])
    result=execute(plan(),tmp_path,0.,clock=lambda:next(moments),
                   action=lambda s,p:dict(returncode=3,completion_verified=False))
    assert result['status']=='failed_closed'
    assert result['unstarted_steps']==3


def test_setup_exhausts_budget_before_child_launch(tmp_path):
    def forbidden(*args):
        raise AssertionError('must not launch')
    result=execute(plan(),tmp_path,0.,clock=lambda:8.,action=forbidden)
    assert result['status']=='budget_exhausted_before_next_step'
    assert result['elapsed_seconds']==8.
    assert not result['steps']


def test_reject_prebuilt_front_and_wrong_identity():
    p=plan(); p['steps'][1],p['steps'][2]=p['steps'][2],p['steps'][1]
    with pytest.raises(ValueError,match='previous candidate'):
        validate(p)
    p=plan();p['steps'][1]['completion_receipt']['required_values']['candidate_id']='b'
    with pytest.raises(ValueError,match='same candidate'):
        validate(p)


def test_real_child_process_and_fresh_receipt(tmp_path):
    p=plan();p['budget_seconds']=30.;p['steps']=p['steps'][:2]
    for step in p['steps']:
        step['cwd']=str(tmp_path)
    p['steps'][0]['argv']=[sys.executable,'-c','import time; time.sleep(0.05); print("fixture build")']
    receipt=tmp_path/'fresh.json'
    p['steps'][1]['completion_receipt']['path']=str(receipt)
    p['steps'][1]['argv']=[sys.executable,'-c',
        'import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(candidate_id="a",status="passed")))',str(receipt)]
    result=execute(p,tmp_path,time.monotonic())
    assert result['budget_eligible_completed_candidates']==['a']
    assert result['elapsed_seconds']>=.05
    assert result['steps'][1]['receipt_sha256']
    assert json.loads(receipt.read_text())['candidate_id']=='a'
    with pytest.raises(ValueError,match='stale'):
        run_step(p['steps'][1],tmp_path/'stale.log')


def test_zero_exit_without_matching_receipt_is_failure(tmp_path):
    p=plan();p['budget_seconds']=30.;p['steps']=p['steps'][:2]
    for step in p['steps']:
        step['cwd']=str(tmp_path)
        step['argv']=[sys.executable,'-c','pass']
    p['steps'][1]['completion_receipt']['path']=str(tmp_path/'missing.json')
    result=execute(p,tmp_path,time.monotonic())
    assert result['status']=='failed_closed'
    assert not result['budget_eligible_completed_candidates']
