"""Compile/run production startup-rule checks without TVM, an SDK or a board."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class NativePlanTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ is required for native rule checks")
    def test_startup_rules_and_physical_rejections(self):
        source = Path(__file__).resolve().parents[2] / "apps/native_deploy/test_shared_buffer_plan.cc"
        header = source.with_name("vta_shared_buffer_plan.h")
        with tempfile.TemporaryDirectory(prefix="tvm_shared_buffer_test_") as directory:
            binary = Path(directory) / "test_shared_buffer_plan"
            subprocess.run(["g++", "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary), str(header), hashlib.sha256(header.read_bytes()).hexdigest()],
                                    check=True, capture_output=True, text=True)
            self.assertIn("23 shared-buffer C++ checks passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
