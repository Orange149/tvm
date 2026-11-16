import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from thesis_completion_dependency_audit import inventory, verify_resets
from thesis_supplement_run import RunStore, write_new_json, sha256


class ContextAcknowledgementTests(unittest.TestCase):
    def test_retained_tail_ack_is_safe_with_matching_host_pending_pop(self):
        first = {'full_call_id': 'b1_O_zeros_health', 'balance': [0, 0, 1, 0],
                 'compute_store_pops': 1, 'last_real_store_index': 0,
                 'last_real_store_ack_ordinal': 1}
        following = {'full_call_id': 'b1_O_zeros_r00', 'balance': [0, 0, 0, 0],
                     'compute_store_pops': 2, 'last_real_store_index': 1,
                     'last_real_store_ack_ordinal': 1}
        result = inventory([first, following], False)
        self.assertTrue(all(x['store_ack_bound_sufficient'] for x in result))
        self.assertEqual(result[-1]['final_inventory_upper_bound'], [0, 0, 1, 0])

    def test_fresh_host_context_cannot_consume_old_ack_as_new_store_completion(self):
        blocks = [{'full_call_id': f'b1_{mode}_zeros_health', 'balance': [0, 0, 1, 0],
                   'compute_store_pops': 1, 'last_real_store_index': 0,
                   'last_real_store_ack_ordinal': 1} for mode in ('O', 'I')]
        without_reset = inventory(blocks, False)
        with_reset = inventory(blocks, True)
        self.assertFalse(without_reset[1]['store_ack_bound_sufficient'])
        self.assertTrue(all(x['store_ack_bound_sufficient'] for x in with_reset))

    def test_reset_does_not_hide_missing_ack_on_real_store(self):
        blocks = [{'full_call_id': 'b1_W_zeros_health', 'balance': [0, 0, 0, 0],
                   'compute_store_pops': 0, 'last_real_store_index': 0,
                   'last_real_store_ack_ordinal': None}]
        self.assertFalse(inventory(blocks, True)[0]['store_ack_bound_sufficient'])

    def test_source_and_output_success_cannot_replace_actual_reset_action(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            store = RunStore.create(root, {'run_id': 'fixture', 'campaign_id': 'fixture', 'stage': '01',
                'completion': {'execution_context_reset': {'required_before_each_new_command_queue': True}}})
            boot = root / 'raw/boot1'
            boot.mkdir()
            (boot / 'calls.jsonl').write_text(json.dumps({'boot_id': 'test-boot', 'full_call_id': 'b1_O_zeros_health'})+'\n')
            write_new_json(boot / 'identity.json', {'boot_id': 'test-boot', 'board_directory': '/ram/fixture'})
            write_new_json(root / 'analysis/board_artifact_bindings.json', {'vta_hpc.bit': {'sha256': 'fixture-hash'}})
            receipt = root / 'analysis/context_reset_b1_O.json'
            write_new_json(receipt, {'action_id': 'missing-executed-action'})
            ready = store.event('context_reset_ready', session_id='b1_O', boot_id='test-boot',
                                receipt='analysis/context_reset_b1_O.json', receipt_sha256=sha256(receipt))
            store.event('context_reset_claimed', session_id='b1_O', boot_id='test-boot', reset_event_id=ready)
            store.event('full_call_start', session_id='b1_O', boot_id='test-boot')
            with self.assertRaisesRegex(ValueError, 'Missing reset action'):
                verify_resets(root, 1)


if __name__ == '__main__':
    unittest.main()
