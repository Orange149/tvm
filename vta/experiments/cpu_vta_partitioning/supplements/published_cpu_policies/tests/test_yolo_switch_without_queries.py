from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_yolo_switch_without_queries as campaign


def test_counter_free_observer_preserves_binary_layout_waits_and_short_window():
    docs = [campaign.make_task(campaign.BUNDLE, spec) for spec in campaign.specs()]
    assert sum(sum(doc[k] for k in ('warmup', 'scored', 'tail')) for doc in docs) == 57
    assert len({doc['executor_sha256'] for doc in docs}) == 1
    assert len({doc['configuration']['id'] for doc in docs}) == 1
    for doc in docs:
        assert not doc['queue_counter_reads']
        argv, _ = campaign.native_command(doc, campaign.BUNDLE, '/bundle', '/job')
        assert '--cpu-worker-pinning' in argv and '--process-cpu-affinity' not in argv
        assert '/bundle/runner/vta_stage_pipeline_runner_phase' in argv
        if doc['spec']['probe']:
            assert 'VTA_QUEUE_COUNTERS=0' in argv and 'VTA_QUEUE_COUNTERS=1' not in argv
            assert 'VTA_WAIT_POLICY=original' in argv
            assert '--session-profile-json' in argv
        else:
            assert '--session-profile-json' not in argv
        if doc['kind'].startswith('qualify'):
            assert argv[argv.index('--output-mode') + 1] == 'raw_all_stages'
        else:
            assert argv[argv.index('--output-mode') + 1] == 'raw'


def test_native_execution_requires_explicit_switch_and_unplanned_sessions_fail():
    with pytest.raises(ValueError, match='execute-board'):
        campaign.board()
    wrong = {**campaign.specs()[0], 'case': 'split'}
    with pytest.raises(ValueError, match='unplanned'):
        campaign.make_task(campaign.BUNDLE, wrong)
