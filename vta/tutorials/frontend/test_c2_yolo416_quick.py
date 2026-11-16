"""Regression checks for variable input sets and unchanged YOLO224 acceptance."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from c2_yolo224_analyze import reference


class FrozenReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        raw = bytearray()
        frames = []
        for frame in range(8):
            entries = []
            for output in range(8):
                entries.append(dict(index=output, offset=len(raw), bytes=4, dtype="float32"))
                raw.extend(struct.pack("<f", float(frame % 4 + output)))
            frames.append(dict(complete=True, input_index=frame % 4, outputs=entries))
        (self.directory / "outputs.bin").write_bytes(raw)
        (self.directory / "exitcode").write_text("0")
        (self.directory / "result.json").write_text(json.dumps(dict(status="passed", frames=frames)))

    def tearDown(self):
        self.temporary.cleanup()

    def test_four_inputs_require_two_identical_cycles(self):
        self.assertEqual(set(reference(self.directory, 4)), set(range(4)))

    def test_224_default_still_requires_64_inputs(self):
        with self.assertRaisesRegex(ValueError, "two complete cycles"):
            reference(self.directory)

    def test_changed_repeated_output_is_rejected(self):
        path = self.directory / "outputs.bin"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "not repeatable"):
            reference(self.directory, 4)

    def test_missing_output_member_is_rejected(self):
        path = self.directory / "result.json"
        value = json.loads(path.read_text())
        value["frames"][-1]["outputs"].pop()
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "incomplete frame/output bundle"):
            reference(self.directory, 4)


if __name__ == "__main__":
    unittest.main()
