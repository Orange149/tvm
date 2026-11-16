import os
from pathlib import Path
import signal
import subprocess
import time
import unittest
from thesis_completion_context_reset import RPC_CHILDREN_CHECK


class RPCChildCheckTests(unittest.TestCase):
    def test_real_live_child_is_found_without_checkpoint_restore_proc_file(self):
        parent = subprocess.Popen(['sh', '-c', 'sleep 20 & wait'],
                                  start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            children = []
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not children:
                command = f'pid={parent.pid}\n' + RPC_CHILDREN_CHECK + '\nrpc_children\n'
                children = subprocess.check_output(['sh', '-c', command], timeout=2).decode().split()
                if not children:
                    time.sleep(0.01)
            self.assertTrue(children)
            for pid in children:
                fields = (Path('/proc') / pid / 'status').read_text().splitlines()
                self.assertIn(parent.pid, [int(line.split()[1]) for line in fields if line.startswith('PPid:')])
        finally:
            os.killpg(parent.pid, signal.SIGTERM)
            parent.wait(timeout=2)

    def test_no_child_does_not_require_optional_proc_children_interface(self):
        # A non-existent parent cannot own any live process; this only exercises
        # the enumeration core. The production reset additionally verifies RPC exe.
        command = 'pid=999999999\n' + RPC_CHILDREN_CHECK + '\nrpc_children\n'
        self.assertEqual(subprocess.check_output(['sh', '-c', command], timeout=2), b'')


if __name__ == '__main__':
    unittest.main()
