"""Negative controls for the verifier using real saved YOLO output fixtures."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from c2_yolo224_analyze import reference,metrics
from c2_yolo224_address_observer import verify

SOURCE=Path(os.environ.get('C2_YOLO_EVIDENCE',str(Path(__file__).resolve().parent/
    'report_out/stage_tile_cotuning/c2_yolov3_tiny/20261018_run03')))


@unittest.skipUnless((SOURCE/'dynamic_address_certificate.json').exists(),'requires the saved real-board fixture')
class EvidenceNegativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.reference=reference(SOURCE/'runs/reference_M0_01')

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='c2_yolo_verify_');self.root=Path(self.temp.name)
        self.run=self.root/'runs/address_observed_E1';self.run.mkdir(parents=True)
        for name in ('result.json','outputs.bin','exitcode','stderr','reader_addresses.jsonl'):
            shutil.copy2(SOURCE/'runs/address_observed_E1'/name,self.run/name)
        for name in ('last_reader_certificate.json','c2_yolo224_address_observer.so'):
            shutil.copy2(SOURCE/name,self.root/name)
        self.result=json.loads((self.run/'result.json').read_text())

    def tearDown(self):self.temp.cleanup()

    def save(self): (self.run/'result.json').write_text(json.dumps(self.result))

    def test_real_fixture_passes_both_independent_verifiers(self):
        self.assertTrue(metrics(self.run,self.reference)['all_outputs_byte_exact']);verify(self.root,'address_observed_E1')

    def test_single_bit_output_change_is_rejected(self):
        with (self.run/'outputs.bin').open('r+b') as stream:
            value=stream.read(1);stream.seek(0);stream.write(bytes([value[0]^1]))
        with self.assertRaisesRegex(ValueError,'raw mismatch'):metrics(self.run,self.reference)

    def test_eighth_output_cannot_be_omitted(self):
        self.result['frames'][0]['outputs'].pop();self.save()
        with self.assertRaisesRegex(ValueError,'incomplete'):metrics(self.run,self.reference)

    def test_wrong_frame_token_is_rejected(self):
        self.result['frames'][0]['tokens'][1]['frame']+=1;self.save()
        with self.assertRaisesRegex(ValueError,'token mismatch'):metrics(self.run,self.reference)

    def test_forged_generation_is_rejected(self):
        event=next(x for x in self.result['events'] if x['event']=='consumer_acquired' and x['edge']==1)
        event['generation']+=1;self.save()
        with self.assertRaisesRegex(ValueError,'stale token'):metrics(self.run,self.reference)

    def test_missing_last_read_release_hook_is_rejected(self):
        self.result['events']=[e for e in self.result['events'] if not(e['event']=='validated_early_release' and e['frame']==0)]
        self.save()
        with self.assertRaisesRegex(ValueError,'reader hook'):metrics(self.run,self.reference)

    def test_release_before_reader_return_is_rejected(self):
        returned=next(e['time_ms'] for e in self.result['events'] if e['event']=='last_reader_return' and e['frame']==0)
        next(e for e in self.result['events'] if e['event']=='input_lease_released' and e['edge']==0 and e['frame']==0)['time_ms']=returned-1
        self.save()
        with self.assertRaisesRegex(ValueError,'after the actual last read'):metrics(self.run,self.reference)

    def test_missing_later_argument_checks_is_rejected(self):
        self.result['frames'][0]['reader_argument_checks']=0;self.save()
        with self.assertRaisesRegex(ValueError,'argument checks'):metrics(self.run,self.reference)

    def test_actual_output_range_outside_pool_is_rejected(self):
        captures=[json.loads(x) for x in (self.run/'reader_addresses.jsonl').read_text().splitlines()]
        source=int(next(s for s in self.result['slots'] if s['edge']==0)['physical'],16)
        target=next(x for x in captures[:2] if int(x['physical'],16)!=source)
        target['physical']='0x0'
        (self.run/'reader_addresses.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in captures))
        with self.assertRaisesRegex(ValueError,'outside allocated physical pool'):verify(self.root,'address_observed_E1')


if __name__=='__main__':unittest.main()
