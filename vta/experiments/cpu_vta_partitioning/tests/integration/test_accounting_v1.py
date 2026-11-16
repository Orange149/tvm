"""Failed, partial and retransferred sessions still consume experiment budget."""
import json

import pytest
from cpu_vta_partitioning.analysis.export import export
from cpu_vta_partitioning.campaign import measured, controller_identity
from cpu_vta_partitioning.common import read, write
from cpu_vta_partitioning.domain import Stage
from cpu_vta_partitioning.measurement.tasks import task
from cpu_vta_partitioning.transport import TransportError


class Board:
    def __init__(self):
        self.count = 340
        self.bad_affinity = True
        self.calls = 0

    def measure(self, document, bundle, remote, root, boot):
        self.calls += 1
        result = root/'raw'/boot/document['task_id']/'transfers'/str(self.calls)/'collected'
        result.mkdir(parents=True)
        rows = [{'frame_id':i, 'completion_index':i, 'input_index':i % len(document['input_indices']),
                 'completion_ms':float(i+1), 'total_latency_ms':1.} for i in range(self.count)]
        (result/'result.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
        write(result/'affinity.json', {'stages':[{'requested_cpus':[0], 'thread_ids':[1],
            'active_runtime_threads':1, 'effective_affinities':[{'tid':1, 'cpus':[int(self.bad_affinity)]}]}]})
        if self.count < 340:
            # Simulate a partially transferred last line as well as missing rows.
            with (result/'result.jsonl').open('a') as stream:
                stream.write('\n{"frame_id":')
            raise TransportError('partial result after disconnect')
        return result


def setup(tmp_path):
    identity = {'bundle_id':'cost-test', 'controller_identity':controller_identity()}
    bundle = tmp_path/'bundle'; write(bundle/'manifest.json', identity)
    root = tmp_path/'run'; root.mkdir()
    write(root/'state.json', {**identity, 'bundle':str(bundle)})
    document = task('screening', [Stage('cpu',0,21,1,(0,))], {'cores':4})
    return bundle, root, document


def test_rejected_complete_sessions_count_once_per_boot_even_after_resume(tmp_path):
    bundle, root, doc = setup(tmp_path)
    board = Board()
    for boot in ('boot1', 'boot1', 'boot2'):
        with pytest.raises(ValueError, match='affinity escapes'):
            measured(doc, bundle, '/local', root, boot, board)
    counts = export(root)['cost_accounting']
    assert counts['actual_native_frames'] == counts['observed_completed_frames'] == 680
    assert counts['accepted_native_frames'] == counts['accepted_sessions'] == 0
    assert counts['distinct_observed_boots'] == 2 and counts['distinct_accepted_boots'] == 0
    assert counts['failed_session_events'] == 3 and counts['native_frames_status'] == 'complete'
    board.bad_affinity = False
    measured(doc, bundle, '/local', root, 'boot1', board)
    measured(doc, bundle, '/local', root, 'boot1', board)
    counts = export(root)['cost_accounting']
    assert counts['actual_native_frames'] == 680 and counts['accepted_native_frames'] == 340
    assert counts['observed_sessions'] == 2 and counts['accepted_sessions'] == 1
    assert board.calls == 4


def test_partial_capture_is_lower_bound_until_complete_recovery(tmp_path):
    bundle, root, doc = setup(tmp_path)
    board = Board(); board.count = 17
    with pytest.raises(TransportError):
        measured(doc, bundle, '/local', root, 'boot1', board)
    counts = export(root)['cost_accounting']
    assert counts['observed_completed_frames'] == 17 and counts['actual_native_frames'] is None
    assert counts['sessions_with_unknown_frame_count'] == 1 and counts['native_frames_status'] == 'lower_bound'
    board.count = 340; board.bad_affinity = False
    measured(doc, bundle, '/local', root, 'boot1', board)
    counts = export(root)['cost_accounting']
    assert counts['observed_completed_frames'] == counts['actual_native_frames'] == 340
    assert counts['sessions_with_unknown_frame_count'] == 0 and counts['failed_session_events'] == 1


def test_unrecovered_session_is_unknown_not_zero(tmp_path):
    bundle, root, doc = setup(tmp_path)
    class Disconnected:
        def measure(self, *args):
            raise TransportError('no result recovered')
    with pytest.raises(TransportError):
        measured(doc, bundle, '/local', root, 'boot1', Disconnected())
    counts = export(root)['cost_accounting']
    assert counts['observed_completed_frames'] == 0 and counts['actual_native_frames'] is None
    assert counts['distinct_observed_boots'] == 1 and counts['sessions_with_unknown_frame_count'] == 1
    # Atomic event writes are not assumed when reading an interrupted event log.
    with (root/'events.jsonl').open('a') as stream:
        stream.write('{"event":')
    assert export(root)['cost_accounting']['malformed_event_lines'] == 1
