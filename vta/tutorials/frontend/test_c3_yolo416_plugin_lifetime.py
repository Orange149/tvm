"""Subprocess teardown regression, CPU fake only and zero DNN/VTA calls."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tvm.contrib import cc
from c3_tps_sync_release_local import HERE, REPO, HOLD, PYTHON, env_for_worker
from c3_yolo416_lifetime_probe import build_fake


class PluginLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.fake = cls.root/'fake.so'
        build_fake(cls.fake)
        cls.legacy = HOLD/'20261017_c3_yolo416_graph_prepare_run01/graph_observer_fsim.so'
        cls.fixed = cls.root/'fixed.so'
        cc.create_shared(str(cls.fixed), [str(HERE/'c3_yolo416_graph_observer.cc'),
            str(HERE/'c3_yolo416_plugin_lifetime.cc')], options=['-std=c++17','-O2','-ldl',
            *['-I'+str(REPO/p) for p in ('include','3rdparty/dmlc-core/include','3rdparty/dlpack/include')]])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_probe(self, library, order):
        return subprocess.run([str(PYTHON), str(HERE/'c3_yolo416_lifetime_probe.py'),
            'probe','--fake',str(self.fake),'--observer',str(library),'--order',order],
            env=env_for_worker(), capture_output=True, text=True, timeout=30)

    def test_legacy_unload_and_frame_return_reproduce_segfault(self):
        for order in ('drop_library_first','frame_return'):
            with self.subTest(order=order):
                r = self.run_probe(self.legacy, order)
                self.assertEqual(r.returncode, -11, r.stdout+r.stderr)
                self.assertIn('Segmentation fault', r.stderr)

    def test_legacy_control_destroys_objects_before_unload(self):
        r = self.run_probe(self.legacy, 'objects_first')
        self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
        self.assertEqual(r.stdout.count('FAKE_MODULE_DESTROYED'), 1)
        self.assertIn('NORMAL_RETURN DNN_CALLS=0 VTA_CALLS=0 GRAPH_CALLS=0', r.stdout)

    def test_fixed_code_survives_all_orders_and_objects_are_destroyed(self):
        for order in ('drop_library_first','frame_return','objects_first'):
            with self.subTest(order=order):
                r = self.run_probe(self.fixed, order)
                self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
                self.assertEqual(r.stdout.count('FAKE_MODULE_DESTROYED'), 1)
                self.assertIn('NORMAL_RETURN DNN_CALLS=0 VTA_CALLS=0 GRAPH_CALLS=0', r.stdout)
                if order == 'drop_library_first':
                    self.assertIn('EMPTY_ROWS=0', r.stdout)


if __name__ == '__main__':
    unittest.main()
