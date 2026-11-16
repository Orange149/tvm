"""Read sealed local dumps only; no target function is invoked."""
import json
from pathlib import Path
import unittest

from c3_tps_aggregation_capacity import organization


class SealedOrganizationTest(unittest.TestCase):
    def test_three_tps_variants_and_dump_identity_rejection(self):
        root = Path(__file__).resolve().parent / ('report_out/stage_tile_cotuning/'
            'c3_dma_residency_autotune/07_grouped_holdout/20261014_c3_tps_window_complete_run03')
        for variant, expected in [('B', (8, 76, 3, 4, 75520, 2)),
                                  ('N', (8, 89, 11, 9, 75520, 2)),
                                  ('G', (4, 83, 11, 7, 75504, 3))]:
            folder = root / f'TPS_{variant}_detailed'
            dump = (folder / 'native_call_07.txt').read_text()
            native = json.loads((folder / 'native_certificate_07.json').read_text())
            report = organization(dump, native)
            self.assertEqual(tuple(report[key] for key in ('MAC_GEMM', 'all_instructions', 'compute_NOP',
                'UOP_LOAD', 'UOP_LOAD_bytes', 'UOP_SRAM_wraps')), expected)
            self.assertEqual(report['serialized_uop_bytes'], 37760)
            self.assertEqual(report['allocation_projection']['status'],
                             'ALLOCATION_PROJECTION_MATCHES_CERTIFIED_REFERENCES_AND_STATISTICS')
            with self.assertRaisesRegex(ValueError, 'DUMP_CERTIFICATE_MISMATCH'):
                organization(dump + '\n', native)


if __name__ == '__main__':
    unittest.main()
