"""Reuse the unchanged native holdout runner under a new frozen batch."""
from pathlib import Path
from types import FunctionType
from cpu_vta_partitioning.common import read,sha256
import finite_service_holdout_board as parent
import prepare_completion_holdout as ranking

ROOT=parent.ROOT
POOL=ranking.OUTPUT
RUN=ROOT/'runs/device_completion_holdout_01'


def protocol():
    value=parent.protocol();value.update(experiment='device_completion_holdout_01',
        only_local_pool_validation=True,known_topologies_unmeasured_resource_layouts=True)
    return value


def sources():
    files=[Path(__file__),ROOT/'run_completion_holdout.py',ROOT/'tests/test_completion_holdout_board.py',
        ROOT/'audit_completion_cost_refresh.py']
    return {**parent.sources(),**{str(p):sha256(p) for p in files}}


def selection():
    ranking.assert_plan(POOL);seal=read(POOL/'SELECTION_FREEZE.json')
    if seal['plan_sha256']!=sha256(POOL/'PLAN.json') or not seal['no_pipeline_performance_read']:
        raise ValueError('completion holdout selection seal differs')
    for leaf,h in seal['prediction_files'].items():
        if sha256(POOL/leaf)!=h:raise ValueError('completion holdout prediction differs')
    if sum(len(v) for v in seal['selected_configurations'].values())>6:
        raise ValueError('completion holdout configuration budget exceeded')
    known=set(read(POOL/'KNOWN_EXECUTIONS.json')['fingerprints'])
    refresh=read(POOL/'COST_REFRESH_AUDIT.json')
    if (not refresh['no_pipeline_FPS_read'] or refresh['source_sha256']!=sha256(ROOT/'audit_completion_cost_refresh.py')
        or refresh['input_sha256']!=sha256(POOL/'INPUTS.json')):
        raise ValueError('cost-refresh-only premeasurement counterfactual differs')
    inputs=read(POOL/'INPUTS.json')['networks'];seen=set()
    for network,configs in seal['selected_configurations'].items():
        for c in configs:
            if ranking.fingerprint(c) in known:raise ValueError('fresh configuration is already known')
            if c['id'] not in inputs[network] or c!=inputs[network][c['id']]['configuration']:
                raise ValueError('selected native configuration differs from scored input')
            if (network,c['id']) in seen:raise ValueError('duplicate native configuration')
            seen.add((network,c['id']))
    if seal['DNN_budget']!=len(seen)*29:raise ValueError('fresh inference budget mismatch')
    return seal


def adapt(name):
    original=getattr(parent,name)
    fn=FunctionType(original.__code__,{**original.__globals__,**globals()},original.__name__,original.__defaults__)
    fn.__kwdefaults__=original.__kwdefaults__
    return fn


def specs():return adapt('specs')()
def prepare():return adapt('prepare')()
def validate(bundle):return adapt('validate')(bundle)
def make_task(bundle,spec):return adapt('make_task')(bundle,spec)
def native_command(doc,bundle,remote_bundle,remote_job):return adapt('native_command')(doc,bundle,remote_bundle,remote_job)
def matched_ssh(config,*,execute_board=False):return adapt('matched_ssh')(config,execute_board=execute_board)
def qemu():return adapt('qemu')()
def board(*,execute_board=False):return adapt('board')(execute_board=execute_board)
summary=parent.summary
accept=parent.accept
