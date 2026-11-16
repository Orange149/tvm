"""Adoption uses complete modes and the existing common artifacts, never times."""
import unittest
from pathlib import Path
from c3_tps_sync_release_local import HOLD, read, sha
from c3_fallback_identity_compare import qualified_unit


class AdoptionTests(unittest.TestCase):
    def test_sealed_fixed_T_U_complete_qualification_and_ARM_bindings(self):
        for node in (34,46):
            root=HOLD/('20261017_c3_node%d_uop_identity_board_run01'%node)
            artifact=read(root/'artifacts/prepared.json')
            for old in ('B','A'):
                d=artifact['programs'][old]
                unit=Path(d['qualified_directory']).parents[1]
                bindings={}
                qualified_unit(unit,bindings)
                self.assertTrue(bindings)
                for key,digest in d['sha256'].items():self.assertEqual(sha(d[key]),digest)
                contract=read(d['arm_contract'])
                self.assertTrue(contract)
            self.assertEqual(artifact['programs']['B']['abi'],artifact['programs']['A']['abi'])
            self.assertEqual(read(artifact['programs']['B']['reference_contract']),
                             read(artifact['programs']['A']['reference_contract']))


if __name__=='__main__':unittest.main()
