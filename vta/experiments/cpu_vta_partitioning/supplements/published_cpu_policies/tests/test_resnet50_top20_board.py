"""New-boot execution references old profiles without reranking or mixing boots."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / 'src'))
SPEC = importlib.util.spec_from_file_location('frozen_r50_board', ROOT / 'run_resnet50_top20_board.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class BoardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source, self.output, self.run = root / 'source', root / 'new-output', root / 'new-run'
        self.bundle = self.source / 'pipeline_bundle'
        runner.write(self.source / 'CONTROLLERS.json', {'sealed': True})
        runner.write(self.bundle / 'manifest.json', {'bundle_id': 'unchanged-native-bundle'})
        runner.write(self.bundle / 'provenance/static_inputs/isolated_costs.json',
                     {'costs': {'first': {'source': {'boot_id': 'old-profile-boot'}}}})
        runner.write(self.source / 'ranking/rankings.json', {'frozen': 'forty'})
        digest = runner.sha256(self.source / 'ranking/rankings.json')
        runner.write(self.source / 'ranking/FREEZE.json',
                     {'candidate_count': 40, 'pipeline_FPS_used': False, 'files': {'rankings.json': digest}})
        runner.write(self.source / 'ranking/SEARCH_STATUS.json', {'status': 'COMPLETE'})
        self.science = patch.object(runner, 'assert_controllers', return_value={'sources': {'old.py': 'unchanged'}})
        self.validation = patch.object(runner, 'validate_bundle', return_value={'bundle_id': 'unchanged-native-bundle'})
        self.science.start()
        self.validation.start()
        self.addCleanup(self.science.stop)
        self.addCleanup(self.validation.stop)

    def test_prepare_preserves_old_boot_and_exact_selection(self):
        value = runner.prepare(self.output, self.source)
        self.assertEqual(value['profile_source_boot_ids'], ['old-profile-boot'])
        self.assertEqual(value['new_calibration_sessions'], 0)
        self.assertTrue(value['cross_boot_cost_reference'])
        self.assertEqual(runner.sha256(self.output / 'ranking/rankings.json'),
                         runner.sha256(self.source / 'ranking/rankings.json'))
        self.assertEqual(runner.prepare(self.output, self.source), value)

    def test_prepare_rejects_overwriting_original_experiment(self):
        with self.assertRaisesRegex(ValueError, 'must not overwrite'):
            runner.prepare(self.source, self.source)

    def test_changed_rank_is_rejected(self):
        runner.write(self.source / 'ranking/rankings.json', {'changed': True}, replace=True)
        with self.assertRaisesRegex(ValueError, 'ranking changed'):
            runner.prepare(self.output, self.source)

    def test_board_access_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'execute-board'):
            runner.run(self.output, self.run, self.source, Path('not-read'))

    def test_resume_keeps_new_boot_and_rejects_another_restart(self):
        config = Path(self.temp.name) / 'local.json'
        runner.write(config, {'board': {}})
        class FakeSSH:
            value = 'new-measurement-boot'
            def boot_id(self):
                return self.value
            def qualify_environment(self, *args):
                pass
        ssh = FakeSSH()
        with patch.object(runner.runtime, 'transport', return_value=ssh), \
             patch.object(runner, 'ensure_storage', return_value='existing SD'), \
             patch.object(runner, 'timed_deploy', return_value='/board/frozen'), \
             patch.object(runner.campaign, 'pipeline', return_value={'status': 'COMPLETE'}) as measure, \
             patch.object(runner, 'analyze', return_value={'status': 'COMPLETE'}):
            runner.run(self.output, self.run, self.source, config, execute_board=True)
            identity = runner.read(self.run / 'run.json')
            self.assertEqual(identity['boot_id'], 'new-measurement-boot')
            self.assertEqual(identity['profile_source_boot_ids'], ['old-profile-boot'])
            self.assertEqual(measure.call_args.args[-1], 'new-measurement-boot')
            ssh.value = 'another-new-boot'
            with self.assertRaisesRegex(ValueError, 'immutable artifact differs'):
                runner.run(self.output, self.run, self.source, config, execute_board=True)
            self.assertEqual(measure.call_count, 1)


if __name__ == '__main__':
    unittest.main()
