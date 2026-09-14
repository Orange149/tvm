#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).with_name("audit_vta_p7r_usmp_lifetime_feasibility.py")
SPEC = importlib.util.spec_from_file_location("usmp_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class UsmpLifetimeFeasibilityTest(unittest.TestCase):
    def test_queue_parser_reports_malformed_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.stderr"
            path.write_text(
                "noise\n[VTA_QUEUE] {\"submit\": 1, \"queue_id\": \"q\"}\n"
                "[VTA_QUEUE] {bad}\n", encoding="utf-8"
            )
            records, malformed = MODULE.load_queue_records(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(malformed), 1)

    def test_replay_check_is_exact_manifest_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            attestation = root / "attestation.json"
            negative = root / "negative.json"
            write_json(attestation, {"manifest_id": "m", "runtime": {"replay_policy": "disabled"}})
            write_json(negative, {
                "manifest_id": "m",
                "calls": {"capture": {"rejected": True}, "replay": {"rejected": True}},
            })
            result = MODULE.inspect_replay(attestation, negative)
            self.assertTrue(result["exact_manifest_capture_and_replay_rejected"])
            write_json(negative, {
                "manifest_id": "different",
                "calls": {"capture": {"rejected": True}, "replay": {"rejected": True}},
            })
            result = MODULE.inspect_replay(attestation, negative)
            self.assertFalse(result["exact_manifest_capture_and_replay_rejected"])

    def test_profile_tail_is_not_complete_lifetime_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = root / "events.json"
            meta = root / "meta.json"
            write_json(events, [{
                "seq": index, "ts_us": index, "kind": "load", "bytes": 4,
                "x_size": 1, "y_size": 1, "x_stride": 1,
            } for index in range(2)])
            write_json(meta, {"events_limit": 2})
            result = MODULE.inspect_profile(events, meta)
            self.assertTrue(result["is_bounded_tail_capture"])
            self.assertFalse(result["has_unified_correlation_fields"])
            self.assertFalse(result["has_resource_physical_range"])

    def test_trace_contract_requires_finish_device_done_and_zero_drop(self):
        contract = MODULE.build_trace_contract()
        command = contract["command_submit"]["required_fields"]
        run = contract["clock_and_completeness"]["required_run_header"]
        self.assertIn("decoded_finish_count", command)
        self.assertIn("device_done_ts_ns", command)
        self.assertIn("dropped_event_count", run)
        self.assertIn("enabled replay", " ".join(contract["fail_closed_conditions"]))

    def test_immutable_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.write_outputs(output, {}, {}, "command")


if __name__ == "__main__":
    unittest.main()
