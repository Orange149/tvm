"""Corrupt real saved submission logs to exercise command evidence guards."""
import json,shutil,tempfile,unittest
from pathlib import Path
from c2_yolo224_command import account,reconcile

SOURCE=Path(__file__).resolve().parent/'report_out/stage_tile_cotuning/c2_yolov3_tiny/20261018_run03'


@unittest.skipUnless((SOURCE/'runs/qual_M2/verified.json').exists(),'requires saved board evidence')
class CommandEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='c2_command_evidence_')
        self.root=Path(self.temp.name);self.run=self.root/'runs/qual_M2';self.run.mkdir(parents=True)
        for name in ('result.json','contract.json','stderr','verified.json'):
            shutil.copy2(SOURCE/'runs/qual_M2'/name,self.run/name)
        self.rows=[json.loads(line[len('[VTA_QUEUE] '):]) for line in (self.run/'stderr').read_text().splitlines()
                   if line.startswith('[VTA_QUEUE] ')]

    def tearDown(self):self.temp.cleanup()

    def save(self):
        (self.run/'stderr').write_text(''.join('[VTA_QUEUE] '+json.dumps(row)+'\n' for row in self.rows))

    def test_real_complete_ledger_passes(self):
        value=account(self.run)
        self.assertEqual(value['complete_submissions'],640)
        self.assertEqual(value['complete_tail_peaks'],[745792,264])
        self.assertEqual(value['Q1_capacities'],[1491712,4096])

    def test_missing_tail_record_is_rejected(self):
        self.rows.pop();self.save()
        with self.assertRaisesRegex(ValueError,'incomplete'):account(self.run)

    def test_duplicate_sequence_is_rejected(self):
        self.rows[-1]['submit']-=1;self.save()
        with self.assertRaisesRegex(ValueError,'sequence'):account(self.run)

    def test_lowered_reported_peak_is_rejected(self):
        self.rows[-1]['insn_peak']-=16;self.save()
        with self.assertRaisesRegex(ValueError,'peak accounting'):account(self.run)

    def test_changed_backing_contract_is_rejected(self):
        self.rows[-1]['insn_capacity']-=256;self.save()
        with self.assertRaisesRegex(ValueError,'capacity contract'):account(self.run)

    def test_same_count_but_changed_command_work_is_rejected(self):
        shutil.copytree(self.run,self.root/'runs/qual_M3')
        self.rows[0]['load_bytes']+=16;self.save()
        with self.assertRaisesRegex(ValueError,'command work'):reconcile(self.root)


if __name__=='__main__':unittest.main()
