"""Boundary observation contracts; no VTA target execution."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from c3_tps_cost_probe import (HEADER, HERE, OFFICIAL, instrument_runtime,
                               instrument_helper, strip_observation, split_boundary)


class OverlayTests(unittest.TestCase):
    def test_sealed_runtime_execution_roundtrip(self):
        for folder in ('runtime_local_final', 'runtime_arm'):
            original=(OFFICIAL/folder/'c3_official_window_runtime_generated.cc').read_text()
            output=instrument_runtime(original)
            self.assertEqual(strip_observation(output),original)
            self.assertEqual(output.count('c3_tps_cost::BeginDriver();'),1)
            self.assertEqual(output.count('c3_tps_cost::EndDriver(timeout);'),1)

    def test_ambiguous_runtime_anchor_rejected(self):
        original=(OFFICIAL/'runtime_arm/c3_official_window_runtime_generated.cc').read_text()
        with self.assertRaisesRegex(ValueError,'DRIVER_SEAM'):
            instrument_runtime(original+original)

    def test_helper_preserves_single_invocation_and_existing_timer(self):
        original=(HERE/'c3_official_window_runtime_helper.cc').read_text()
        output=instrument_helper(original)
        self.assertEqual(strip_observation(output),original)
        self.assertEqual(output.count('target.function.CallPacked'),2)  # One per distinct entry.
        self.assertNotIn('time_evaluator',output)


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.h=dict(begin_ns=1000000,end_ns=4000000,started_calls=1,completed_calls=1)
        self.d=dict(mode='on',begins=1,ends=1,timeout=0,clock_reads=2,
                    driver_begin_ns=1500000,driver_end_ns=3500000)

    def test_closed_split(self):
        row=split_boundary(self.h,self.d,'on',1,3.)
        self.assertEqual(row['before_driver_ms'],.5)
        self.assertEqual(row['driver_run_and_wait_ms'],2.)
        self.assertEqual(row['after_driver_ms'],.5)
        self.assertFalse(row['pure_device_compute_time'])

    def test_off_requires_no_clocks(self):
        d={**self.d,'mode':'off','clock_reads':0,'driver_begin_ns':0,'driver_end_ns':0}
        self.assertFalse(split_boundary(self.h,d,'off',1,3.)['split_available'])
        d['clock_reads']=1
        with self.assertRaises(ValueError):split_boundary(self.h,d,'off',1)

    def test_bad_order_and_multiple_driver_calls_rejected(self):
        for update in ({'driver_begin_ns':0},{'driver_end_ns':4000001},{'begins':2},
                       {'ends':0},{'clock_reads':4},{'timeout':1},{'mode':'off'}):
            with self.subTest(update=update),self.assertRaises(ValueError):
                split_boundary(self.h,{**self.d,**update},'on',1)

    def test_nonfinite_and_inconsistent_complete_time(self):
        for elapsed in (float('nan'),float('inf'),0,2.9):
            with self.subTest(elapsed=elapsed),self.assertRaises(ValueError):
                split_boundary(self.h,self.d,'on',1,elapsed)

    def test_stale_helper_count_rejected(self):
        with self.assertRaises(ValueError):split_boundary(self.h,self.d,'on',2)


class CompiledHeaderTests(unittest.TestCase):
    def test_real_clock_modes_clear_and_fixed_mode(self):
        # This is a clock-only C++ fixture. It does not load TVM, a driver or a VTA kernel.
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'clock.cc';exe=Path(folder)/'clock'
            source.write_text('#include "c3_tps_cost_boundary.h"\n#include <iostream>\n'
                'int main(){c3_tps_cost::Clear(); c3_tps_cost::BeginDriver(); '
                'c3_tps_cost::EndDriver(0);std::cout<<c3_tps_cost::JSON()<<"\\n";'
                'setenv("C3_TPS_COST_MODE","off",1);c3_tps_cost::Clear();'
                'std::cout<<c3_tps_cost::JSON()<<"\\n";}\n')
            subprocess.run(['g++','-std=c++17','-O2','-I'+str(HEADER.parent),str(source),'-o',str(exe)],
                           check=True,capture_output=True)
            for mode in ('off','on'):
                result=subprocess.run([str(exe)],env={**os.environ,'C3_TPS_COST_MODE':mode},
                                      check=True,capture_output=True,text=True)
                first,cleared=map(json.loads,result.stdout.splitlines())
                self.assertEqual(first['clock_reads'],2 if mode=='on' else 0)
                self.assertEqual(first['begins'],1)
                self.assertEqual(first['ends'],1)
                self.assertEqual(cleared['mode'],mode)
                self.assertEqual(cleared['begins'],0)
                self.assertEqual(cleared['clock_reads'],0)


if __name__=='__main__':unittest.main()
