"""Read-only physical-range qualification regressions; no board access."""
import unittest
from qualify_vta_cpu_tail_handoff import check_ranges


class RangeTests(unittest.TestCase):
    def row(self,physical=4096,virtual=8192,size=256):
        return {"physical_address":physical,"virtual_address":virtual,"bytes":size}

    def test_adjacent_ranges_allowed(self):
        check_ranges([self.row(),self.row(4352,8448)],4096,512)

    def test_overlapping_ranges_rejected(self):
        with self.assertRaises(AssertionError):
            check_ranges([self.row(size=512),self.row(4352,8448)],4096,1024)

    def test_pool_overrun_rejected(self):
        with self.assertRaises(AssertionError):
            check_ranges([self.row(size=513)],4096,512)

    def test_unaligned_physical_rejected(self):
        with self.assertRaises(AssertionError):
            check_ranges([self.row(4097)],4096,1024)

    def test_unaligned_virtual_rejected(self):
        with self.assertRaises(AssertionError):
            check_ranges([self.row(virtual=8193)],4096,1024)

    def test_below_pool_rejected(self):
        with self.assertRaises(AssertionError):
            check_ranges([self.row(3840)],4096,1024)


if __name__=="__main__":
    unittest.main()
