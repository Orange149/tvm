"""Inference must be backed by accepted raw evidence from matching independent boots."""
from pathlib import Path
import shutil

import pytest

from test_campaign import SyntheticBoard
from cpu_vta_partitioning.campaign import create, run_current, controller_identity, analyze_environment
from cpu_vta_partitioning.common import PROJECT, read, write
from cpu_vta_partitioning.analysis.export import export


@pytest.fixture(scope='module')
def completed_campaign(tmp_path_factory):
    import cpu_vta_partitioning.campaign as campaign
    import cpu_vta_partitioning.prepare.package as package
    base=tmp_path_factory.mktemp('accepted-campaign')
    bundle=base/'bundle';bundle.mkdir()
    programs={p['name']:p for p in read(PROJECT/'tests/fixtures/small_domain.json')}
    write(bundle/'manifest.json',{'bundle_id':'synthetic','controller_identity':controller_identity()})
    write(bundle/'plans/programs.json',programs)
    write(bundle/'plans/protocol.json',{'environment':{'cores':4,'poll_sleep_ns':1000,'post_start_sleep_ns':1000},'seed':13})
    write(bundle/'plans/sensitivity.json',{'cores':[2,4],'poll_sleep_ns':[0,1000,100000],'post_start_sleep_ns':1000})
    root=base/'run';create(bundle,root,{})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(package,'validate',lambda _:None)
        patch.setattr(campaign,'correctness',lambda *args:True)
        board=SyntheticBoard()
        for i in range(10):
            board.boot=f'00000000-0000-0000-0000-{i+1:012}'
            state=run_current(root,board)
    assert state['status']=='COMPLETE'
    return root


@pytest.fixture
def campaign_copy(tmp_path,completed_campaign):
    root=tmp_path/'run'
    shutil.copytree(completed_campaign,root)
    return root


@pytest.mark.parametrize('fault', ['duplicate_diagnostic_boot','wrong_diagnostic_boot','duplicate_round_boot',
                                    'missing_receipt','missing_raw','changed_raw','changed_summary',
                                    'changed_receipt','wrong_receipt_condition','missing_reference'])
def test_analysis_rejects_identity_and_evidence_faults(campaign_copy,fault):
    root=campaign_copy;env=root/'env0'
    round_path=env/'confirmation/boot0/round0.json'
    record=read(round_path)
    key=next(iter(record['measurements']))
    summary=record['measurements'][key]
    receipt_path=root/summary['_receipt']['path']
    receipt=read(receipt_path)
    if fault in ('duplicate_diagnostic_boot','wrong_diagnostic_boot'):
        p=env/'confirmation/boot1/diagnostic.json';value=read(p)
        value['boot_id']=record['boot_id'] if fault=='duplicate_diagnostic_boot' else 'unrelated-boot'
        write(p,value,replace=True)
    elif fault=='duplicate_round_boot':
        for r in range(2):
            p=env/f'confirmation/boot1/round{r}.json';value=read(p)
            value['boot_id']=record['boot_id'];write(p,value,replace=True)
    elif fault=='missing_receipt':
        receipt_path.unlink()
    elif fault in ('missing_raw','changed_raw'):
        raw=root/next(p for p in receipt['files'] if p.endswith('result.jsonl'))
        if fault=='missing_raw': raw.unlink()
        else: raw.write_text(raw.read_text()+'\n')
    elif fault=='changed_receipt':
        receipt['summary']['ii_ms']*=.9;write(receipt_path,receipt,replace=True)
    else:
        if fault=='changed_summary': summary['ii_ms']*=.9
        elif fault=='wrong_receipt_condition': record['measurements'][key]=record['before']
        elif fault=='missing_reference': del summary['_receipt']
        write(round_path,record,replace=True)
    with pytest.raises((ValueError,FileNotFoundError)):
        analyze_environment(env)
    with pytest.raises((ValueError,FileNotFoundError)):
        export(root)


def test_incomplete_boots_do_not_generate_inference(campaign_copy):
    root=campaign_copy
    (root/'env0/confirmation/boot5/round1.json').unlink()
    with pytest.raises(ValueError,match='two complete rounds'):
        analyze_environment(root/'env0')
    assert export(root)['environments']['env0']['status']=='INCOMPLETE'


def test_valid_relocated_campaign_retains_six_boot_inference(campaign_copy):
    result=analyze_environment(campaign_copy/'env0')
    state = read(campaign_copy/'state.json')
    assert result['provenance'] == {k:state[k] for k in ('bundle_id', 'controller_identity')}
    assert all(v['n_boots']==6 for v in result['comparisons'].values())
    assert all(v['host_isolation']['n_boots']==6 for v in result['mechanism_diagnostics'].values())
    assert export(campaign_copy)['environments']['env0']['status']=='ANALYZED'


@pytest.mark.parametrize('incomplete', [False, True])
def test_changed_controller_cli_cannot_overwrite_analysis(campaign_copy, tmp_path, incomplete):
    import os
    import subprocess
    import sys
    from cpu_vta_partitioning.common import inventory
    root = campaign_copy
    if incomplete:
        (root/'env0/confirmation/boot5/round1.json').unlink()
    controller = tmp_path/'changed-controller'
    shutil.copytree(PROJECT/'src', controller/'src', ignore=shutil.ignore_patterns('__pycache__'))
    source = controller/'src/cpu_vta_partitioning/analysis/__init__.py'
    source.write_text(source.read_text() + '\n# Different analysis implementation.\n')
    before = inventory(root)
    result = subprocess.run([sys.executable, '-m', 'cpu_vta_partitioning.cli', 'analyze',
                             '--run-dir', str(root)], cwd=controller,
                            env=dict(os.environ, PYTHONPATH=str(controller/'src')),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and 'controller changed' in result.stderr
    assert inventory(root) == before


@pytest.mark.parametrize('fault', ['dependency_version', 'bundle_id', 'bundle_controller', 'missing_state'])
def test_analysis_identity_faults_leave_outputs_untouched(campaign_copy, tmp_path, monkeypatch, fault):
    from cpu_vta_partitioning.common import inventory
    root = campaign_copy
    if fault == 'dependency_version':
        import importlib.metadata
        version = importlib.metadata.version
        monkeypatch.setattr(importlib.metadata, 'version',
                            lambda name: 'different-version' if name == 'scipy' else version(name))
    elif fault == 'missing_state':
        (root/'state.json').unlink()
    else:
        state = read(root/'state.json')
        bundle = tmp_path/'different-bundle'
        shutil.copytree(state['bundle'], bundle)
        state['bundle'] = str(bundle)
        write(root/'state.json', state, replace=True)
        manifest = read(bundle/'manifest.json')
        manifest['bundle_id' if fault == 'bundle_id' else 'controller_identity'] = 'different'
        write(bundle/'manifest.json', manifest, replace=True)
    before = inventory(root)
    for entry in (lambda: export(root), lambda: analyze_environment(root/'env0')):
        with pytest.raises((ValueError, FileNotFoundError)):
            entry()
    assert inventory(root) == before


def test_resume_rechecks_completed_round_raw_files(campaign_copy,monkeypatch):
    import cpu_vta_partitioning.prepare.package as package
    root=campaign_copy;state=read(root/'state.json')
    round_path=root/'env0/confirmation/boot5/round0.json'
    record=read(round_path);boot=record['boot_id']
    receipt=read(root/next(iter(record['measurements'].values()))['_receipt']['path'])
    (root/next(p for p in receipt['files'] if p.endswith('result.jsonl'))).unlink()
    state.update(status='RUNNING',phase='confirmation',boot_index=5,active_boot=boot)
    state['completed_boots'].remove(boot)
    write(root/'state.json',state,replace=True)
    monkeypatch.setattr(package,'validate',lambda _:None)
    board=SyntheticBoard();board.boot=boot
    with pytest.raises(ValueError,match='missing or changed'):
        run_current(root,board)
    assert not board.executed
