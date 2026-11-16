import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from c3_tps_sync_release_local import sha
from c3_yolo416_comparison_protocol import manifest
from c3_tps_sync_release_local import read
from c3_yolo416_two_node_comparison import OBJECTS, SOURCE, select_pool, source_bindings, identity_alias


def fixture():
    return [{'object_id': oid, 'kind': 'operator', 'programs': {'TH': {}, 'P': {}},
             'representatives': {'TH': 'TH', 'P': 'P'}} for oid in OBJECTS]


class TwoNodeTests(unittest.TestCase):
    def test_exact_selection_and_budget(self):
        selected = select_pool(manifest('source', fixture()))
        ledger = manifest('attempt', selected)
        self.assertEqual((ledger['unique_programs'], ledger['total_target_calls'],
                          len(ledger['sessions']), ledger['formal_samples']), (4, 240, 40, 72))
        counts = Counter((r['object_id'], r['variant'], r['seed']) for r in ledger['calls'] if r['timed'])
        self.assertEqual(len(counts), 12)
        self.assertEqual(set(counts.values()), {6})

    def test_missing_node_and_ledger_mutation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'EXACT_USER_SELECTED'):
            select_pool(manifest('source', fixture()[:1]))
        ledger = manifest('source', fixture()); ledger['calls'][0]['seed'] = -1
        with self.assertRaisesRegex(ValueError, 'SOURCE_LEDGER_CHANGED'): select_pool(ledger)

    def test_only_ssh_file_may_be_rebound(self):
        with tempfile.TemporaryDirectory() as folder:
            host, binary = Path(folder)/'known_hosts', Path(folder)/'module.so'
            host.write_text('old'); binary.write_bytes(b'program')
            frozen = {'known_hosts': str(host), 'bindings': {str(host): sha(host), str(binary): sha(binary)}}
            host.write_text('new trusted connection configuration')
            bindings, adoption = source_bindings(frozen)
            self.assertTrue(adoption['changed']); self.assertEqual(bindings[str(host)], sha(host))
            binary.write_bytes(b'changed program')
            with self.assertRaisesRegex(ValueError, 'FROZEN_IDENTITY_CHANGED'): source_bindings(frozen)

    def test_no_extra_I_calls(self):
        ledger = manifest('source', fixture())
        self.assertEqual({r['variant'] for r in ledger['calls']}, {'TH', 'P'})
        for i in range(1, 7):
            rows = [r for r in ledger['calls'] if r['round'] == i and r['timed']]
            self.assertEqual(len(rows), 12)

    def test_archived_actual_P_functions_are_no_op_aliases(self):
        artifacts = read(SOURCE/'artifacts.json')
        for oid in OBJECTS:
            proof = identity_alias(artifacts['programs'][oid+'/P'])
            self.assertEqual(proof['status'], 'I_EQUALS_P_EXACT_FUNCTION_NO_OP')
            self.assertEqual(proof['method_proof']['changes'], [])
            self.assertEqual(proof['method_proof']['census']['MAC_identities'], 1)


if __name__ == '__main__': unittest.main()
