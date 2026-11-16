"""Read sealed captures; exercise adapter and failures without target calls."""
from copy import deepcopy
import json
import unittest

import tvm
from c3_official_window_method import audit_function
from c3_tps_aggregation_prepare import HOLD, QUALIFIED
from c3_tps_notification_local import depths
from c3_tps_notification_board_support import certify_call, for_storage, split_capture, BEGIN, END

LOCAL = HOLD / '20261015_c3_tps_notification_local_run02'


class CarrierBoardCertificates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = {}
        for variant in 'BNG':
            parent = tvm.ir.load_json((QUALIFIED / f'TPS_{variant}_detailed/parent.json').read_text())
            cls.events[variant] = audit_function(list(parent.functions.values())[0])['events']

    def capture(self, variant='G', index=1, mode='detailed'):
        directory = LOCAL / f'{variant}_{mode}'
        dump = (directory / f'call_{index:02d}_capture.txt').read_text()
        ledger = json.loads((directory / f'call_{index:02d}_end.json').read_text())['submit_ledger']
        return dump, ledger

    def test_cold_and_warm_both_modes_three_programs(self):
        for variant, total, nops in [('B', 76, 3), ('N', 83, 5), ('G', 77, 5)]:
            for mode in ('detailed', 'minimal'):
                definitions = None
                for index in (1, 2):
                    with self.subTest(variant=variant, mode=mode, index=index):
                        dump, ledger = self.capture(variant, index, mode)
                        original = deepcopy(ledger)
                        cert = certify_call(self.events[variant], dump, ledger, definitions, *depths())
                        definitions = cert['cache_definitions']
                        self.assertEqual(cert['organization']['all_instructions'], total)
                        self.assertEqual(cert['organization']['compute_NOP'], nops)
                        self.assertEqual(cert['actual_submit_ledger'], original)
                        self.assertEqual(ledger, original)
                        json.dumps(for_storage(cert), allow_nan=False)

    def test_missing_duplicate_capture_and_extra_submit_rejected(self):
        dump, _ = self.capture()
        for bad in (dump.replace(BEGIN, ''), dump.replace(END, ''), BEGIN + dump, dump + dump):
            with self.subTest(), self.assertRaises(ValueError):
                split_capture(bad)

    def test_incomplete_or_wrong_actual_driver_census_rejected(self):
        dump, ledger = self.capture()
        for change in ({'completion': False}, {'timeout': 1}, {'insns': 83, 'insn_bytes': 1328}):
            bad = deepcopy(ledger)
            bad['submits'][0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'CARRIER_ACTUAL_COMPLETION'):
                certify_call(self.events['G'], dump, bad, None, *depths())

    def test_actual_payload_corruption_rejected(self):
        dump, ledger = self.capture()
        before, actual = dump.split(END)
        self.assertIn('reset_out: 0', actual)
        bad = before + END + actual.replace('reset_out: 0', 'reset_out: 1', 1)
        with self.assertRaisesRegex(ValueError, 'UNEXPLAINED_ENCODING_CHANGE'):
            certify_call(self.events['G'], bad, ledger, None, *depths())


if __name__ == '__main__':
    unittest.main()
