"""Monitor behaviour with temporary files and fake probes; no board access."""
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'monitor_resnet50_status.py'
SPEC = importlib.util.spec_from_file_location('resnet50_health_monitor', SOURCE)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)
BOOT = '11111111-2222-3333-4444-555555555555'


def resources(memory=1024**3, disk=1024**3):
    return dict(state='IDLE_SNAPSHOT', boot_id=BOOT,
        memory_bytes=dict(MemTotal=4 * 1024**3, MemAvailable=memory),
        storage_available_bytes=disk, DMA_buffer_bytes=192 * 1024**2)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.output, self.run, self.destination = root / 'artifacts', root / 'run', root / 'monitor'
        self.campaign = dict(status='COMPONENTS', boot_id=BOOT, updated_at='old stage timestamp')
        monitor.save(self.run / 'run.json', dict(boot_id=BOOT))
        monitor.save(self.run / 'state.json', self.campaign)
        monitor.save(self.run / 'components/state.json', dict(accepted=343, required=2751, status='PARTIAL'))
        monitor.save(self.run / 'traffic/traffic.json', dict(accepted=3, required=299, status='PARTIAL'))
        monitor.save(self.output / 'operations/continuation_state.json', dict(phase='WAITING'))
        self.live = dict(controllers=[dict(pid=123, argv=['fake controller'])], continuation=dict(pid=456))
        self.calls = []
        self.sample = resources()

        def probe(config, boot):
            self.calls.append(boot)
            if isinstance(self.sample, Exception):
                raise self.sample
            return self.sample

        self.subject = monitor.Monitor(self.output, self.run, self.destination,
            execute_board=True, probe=probe, find_actors=lambda *_: self.live)

    def set_campaign(self, **updates):
        self.campaign.update(updates)
        monitor.save(self.run / 'state.json', self.campaign)

    def codes(self, result):
        return {x['code'] for x in result['issues']}

    def test_progress_advances_despite_old_campaign_timestamp(self):
        self.subject.tick(1000)
        monitor.save(self.run / 'components/state.json', dict(accepted=344, required=2751, status='PARTIAL'))
        result = self.subject.tick(2500)
        self.assertEqual(result['seconds_since_progress'], 0)
        self.assertNotIn('NO_RECENT_PROGRESS', self.codes(result))

    def test_stall_is_warning_without_restart_or_measurement(self):
        self.subject.tick(1000)
        result = self.subject.tick(2201)
        self.assertIn('NO_RECENT_PROGRESS', self.codes(result))
        self.assertEqual(monitor.load(self.run / 'components/state.json')['accepted'], 343)
        self.assertFalse(monitor.POLICY['changes_experiment'])

    def test_missing_controller_is_reported_after_handoff_grace(self):
        self.subject.tick(1000)
        self.live['controllers'] = []
        self.assertNotIn('NO_CONTROLLER', self.codes(self.subject.tick(1060)))
        self.assertIn('NO_CONTROLLER', self.codes(self.subject.tick(1180)))

    def test_known_rank_handoff_is_not_unrelated_controller_error(self):
        self.set_campaign(status='STOPPED_ON_ERROR', error_type='ValueError', error='no checkpoint to resume')
        monitor.save(self.output / 'operations/continuation_state.json', dict(phase='RANK_RUNNING'))
        result = self.subject.tick(1000)
        self.assertEqual(result['phase'], 'HANDOFF_RANK_RUNNING')
        self.assertNotIn('CONTROLLER_ERROR', self.codes(result))

    def test_actual_controller_error_and_search_cap_are_reported(self):
        self.set_campaign(status='STOPPED_ON_ERROR', error_type='TransportError', error='SSH timeout')
        self.assertIn('CONTROLLER_ERROR', self.codes(self.subject.tick(1000)))
        self.set_campaign(status='SEARCH_INCOMPLETE')
        self.assertIn('SEARCH_INCOMPLETE', self.codes(self.subject.tick(1060)))

    def test_complete_rank_does_not_require_retired_handoff(self):
        monitor.save(self.output / 'ranking/SEARCH_STATUS.json', dict(status='COMPLETE'))
        monitor.save(self.output / 'operations/continuation_state.json', dict(phase='STOPPED', error='prior board attempt'))
        self.live['continuation'] = None
        result = self.subject.tick(1000)
        self.assertNotIn('CONTINUATION_ERROR', self.codes(result))
        self.assertNotIn('NO_CONTINUATION', self.codes(result))
        self.assertEqual(result['prior_handoff_phase'], 'STOPPED')
        self.assertFalse(result['first_rank_handoff_required'])

    def test_retired_handoff_does_not_hide_current_controller_failure(self):
        monitor.save(self.output / 'ranking/SEARCH_STATUS.json', dict(status='COMPLETE'))
        monitor.save(self.output / 'operations/continuation_state.json', dict(phase='STOPPED', error='prior board attempt'))
        self.live['continuation'] = None
        self.live['controllers'] = []
        self.set_campaign(status='STOPPED_ON_ERROR', error_type='TransportError', error='hash mismatch')
        self.assertIn('CONTROLLER_ERROR', self.codes(self.subject.tick(1000)))

    def test_pipeline_never_launches_extra_resource_probe(self):
        self.set_campaign(status='PIPELINE')
        result = self.subject.tick(1000)
        self.assertEqual(self.calls, [])
        self.assertEqual(result['resource_probe_status'], 'DEFERRED_FORMAL_PIPELINE')

    def test_busy_does_not_turn_missing_memory_into_zero(self):
        self.sample = dict(state='BUSY', boot_id=BOOT)
        result = self.subject.tick(1000)
        self.assertIsNone(result['last_board_snapshot'])
        self.assertNotIn('LOW_MEMORY', self.codes(result))
        self.subject.tick(1030)
        self.assertEqual(len(self.calls), 1)
        self.subject.tick(1060)
        self.assertEqual(len(self.calls), 2)

    def test_successful_resource_reads_are_ten_minutes_apart(self):
        self.subject.tick(1000)
        self.subject.tick(1060)
        self.subject.tick(1599)
        self.assertEqual(len(self.calls), 1)
        self.subject.tick(1600)
        self.assertEqual(len(self.calls), 2)

    def test_resource_failure_stays_open_until_successful_recheck(self):
        self.sample = RuntimeError('offline')
        self.assertIn('RESOURCE_READ_FAILED', self.codes(self.subject.tick(1000)))
        self.assertIn('RESOURCE_READ_FAILED', self.codes(self.subject.tick(1060)))
        self.sample = resources()
        self.assertNotIn('RESOURCE_READ_FAILED', self.codes(self.subject.tick(1600)))

    def test_alerts_are_deduplicated_and_resolutions_recorded(self):
        self.sample = resources(memory=100 * 1024**2, disk=100 * 1024**2)
        self.subject.tick(1000)
        self.subject.tick(1060)
        self.sample = resources()
        self.subject.tick(1600)
        events = [json.loads(x) for x in (self.destination / 'alerts.jsonl').read_text().splitlines()]
        self.assertEqual(sorted((x['event'], x['code']) for x in events),
            [('OPEN', 'LOW_MEMORY'), ('OPEN', 'LOW_STORAGE'), ('RESOLVED', 'LOW_MEMORY'), ('RESOLVED', 'LOW_STORAGE')])

    def test_critical_boot_change_does_not_invent_valid_metrics(self):
        self.sample = dict(state='BOOT_CHANGED', boot_id='another boot')
        self.assertIn('BOARD_BOOT_CHANGED', self.codes(self.subject.tick(1000)))

    def test_duplicate_monitor_lock_is_rejected(self):
        with monitor.monitor_lock(self.destination / 'monitor.lock'):
            with self.assertRaises(RuntimeError):
                with monitor.monitor_lock(self.destination / 'monitor.lock'):
                    pass

    def test_probe_parser_discards_busy_or_raced_metrics(self):
        for state in ('BUSY', 'RACED'):
            with self.subTest(state=state):
                result = monitor.parse_probe('BOOT ' + BOOT + '\nMEM MemAvailable 0\nSTATE ' + state, BOOT)
                self.assertEqual(result, dict(state=state, boot_id=BOOT))

    def test_probe_parser_rejects_missing_fields(self):
        with self.assertRaises(ValueError):
            monitor.parse_probe('BOOT ' + BOOT + '\nSTATE IDLE_SNAPSHOT\nMEM MemAvailable 0', BOOT)

    def test_probe_parser_and_read_only_command_scope(self):
        raw = ('BOOT ' + BOOT + '\nSTATE IDLE_SNAPSHOT\nMEM MemAvailable 12345\n'
               'MEM MemTotal 54321\nDISK_FREE_KB 789\nDMA_BYTES 201326592\n')
        result = monitor.parse_probe(raw, BOOT)
        self.assertEqual(result['memory_bytes']['MemAvailable'], 12345 * 1024)
        script = monitor.probe_script(dict(remote_base='/board/experiment', storage_mount='/media/data'), BOOT)
        for mutation in ('mkdir ', 'rm ', 'kill ', 'reboot', 'sysctl', 'insmod'):
            self.assertNotIn(mutation, script)
        tokens = shlex.shlex(script, posix=True, punctuation_chars='<>')
        tokens.whitespace_split = True
        self.assertFalse({'<', '>', '>>', '<<'} & set(tokens))
        self.assertIn('STATE BUSY; exit 0', script)

    def test_host_only_mode_does_not_access_board(self):
        self.subject.execute_board = False
        self.subject.tick(1000)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
