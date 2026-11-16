import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from thesis_supplement_run import RunStore


class RunStoreTests(unittest.TestCase):
    def make_run(self, directory):
        return RunStore.create(Path(directory) / "run", {
            "run_id": "test", "stage": "00", "campaign_id": "test_only",
        })

    def test_existing_run_and_changed_contract_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            with self.assertRaises(FileExistsError):
                self.make_run(directory)
            path = run.directory / "contract.json"
            path.write_text(json.dumps({**run.contract, "budget": "modified"}))
            with self.assertRaisesRegex(ValueError, "contract changed"):
                run.execute([sys.executable, "-c", "print('should never execute')"], "bad", "prep", 1)
            self.assertEqual(list((run.directory / "raw").iterdir()), [])

    def test_failed_action_preserves_output_and_paid_wall(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            result = run.execute([sys.executable, "-c", "import sys; print('paid output'); sys.exit(7)"],
                                 "failure", "failed_recovery", 3)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["returncode"], 7)
            self.assertIn("paid output", (run.directory / result["stdout"]).read_text())
            self.assertGreater(result["wall_seconds"], 0)
            events = run.events()
            self.assertEqual([row["event"] for row in events], ["run_created", "action_start", "action_end"])
            self.assertEqual(events[-1]["parent_event_id"], events[-2]["event_id"])

    def test_timeout_cancels_process_group_and_retains_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            marker = Path(directory) / "late_result"
            child = "import time; from pathlib import Path; time.sleep(1); Path(%r).write_text('late')" % str(marker)
            parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); print('started',flush=True); time.sleep(10)" % child
            result = run.execute([sys.executable, "-c", parent], "deadline", "online_tuning", 0.2)
            self.assertEqual(result["status"], "timeout")
            self.assertIn("started", (run.directory / result["stdout"]).read_text())
            self.assertLess(result["wall_seconds"], 3)
            self.assertIn("action_cancel", [row["event"] for row in run.events()])
            time.sleep(1.1)
            self.assertFalse(marker.exists(), "A timed-out child produced a late result")


if __name__ == "__main__":
    unittest.main()
