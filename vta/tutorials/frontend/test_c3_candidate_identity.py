"""Tests for stable C3 candidate identities and archive adaptation."""

from __future__ import annotations

import math
import unittest

from c3_candidate_identity import (
    FAILURE_CATEGORIES,
    candidate_id,
    canonical_json_bytes,
    failure_schema_record,
    make_failure,
)
from extract_vta_candidate_features import adapt_archive


class CandidateIdentityTest(unittest.TestCase):
    def setUp(self):
        self.hardware = {"target": "axu5evb", "config_sha256": "config-hash"}
        self.workload = ["conv2d_packed.vta", ["TENSOR", [1, 4, 56, 56, 1, 16], "int8"]]
        self.config = {
            "index": 7,
            "code_hash": None,
            "entity": [["tile_h", "sp", [-1, 8]], ["tile_w", "sp", [-1, 56]]],
        }

    def identity(self, hardware=None, workload=None, config=None, mode="original"):
        return candidate_id(
            self.hardware if hardware is None else hardware,
            "conv2d_packed.vta",
            "schedule-v1",
            self.workload if workload is None else workload,
            mode,
            self.config if config is None else config,
        )

    def test_mapping_key_order_does_not_change_identity(self):
        reordered_hardware = {"config_sha256": "config-hash", "target": "axu5evb"}
        reordered_config = {
            "entity": self.config["entity"],
            "code_hash": None,
            "index": 7,
        }
        self.assertEqual(self.identity(), self.identity(reordered_hardware, config=reordered_config))

    def test_config_space_index_is_debug_only(self):
        renumbered = dict(self.config)
        renumbered["index"] = 98123
        self.assertEqual(self.identity(), self.identity(config=renumbered))

    def test_semantic_entity_change_changes_identity(self):
        changed = dict(self.config)
        changed["entity"] = [["tile_h", "sp", [-1, 7]], ["tile_w", "sp", [-1, 56]]]
        self.assertNotEqual(self.identity(), self.identity(config=changed))

    def test_mode_and_hardware_are_semantic(self):
        self.assertNotEqual(self.identity(), self.identity(mode="weight_stationary"))
        self.assertNotEqual(
            self.identity(),
            self.identity(hardware={"target": "axu5evb", "config_sha256": "other"}),
        )

    def test_non_finite_json_is_rejected(self):
        with self.assertRaises(ValueError):
            canonical_json_bytes({"bad": math.nan})

    def test_failure_vocabulary_is_closed(self):
        self.assertEqual(tuple(failure_schema_record()["categories"]), FAILURE_CATEGORIES)
        self.assertEqual(make_failure("compile", "compiler rejected candidate")["category"], "compile")
        with self.assertRaises(ValueError):
            make_failure("mystery", "not frozen")


class ArchiveAdapterTest(unittest.TestCase):
    def test_adapter_emits_identity_aggregate_and_unmeasured_command(self):
        workload = ["conv2d_packed.vta", ["TENSOR", [1, 1, 7, 7, 1, 16], "int8"]]
        config = {"index": 11, "code_hash": None, "entity": [["tile_h", "sp", [-1, 7]]]}
        selected = [{"workload": workload, "candidates": [{"config_index": 11, "config": config}]}]
        archive = {
            "status": "completed",
            "small_request_threshold_bytes": 4096,
            "workload_count": 1,
            "rows": [
                {
                    "workload": workload,
                    "config_index": 11,
                    "unique_tensor_bytes": {"input_bytes": 784},
                    "redundancy": {"input_load": 2.0},
                    "static_dma": {
                        "totals": {"load_buffer_2d_calls": 2, "load_buffer_2d_bytes": 1024},
                        "max_request_bytes_by_memory": {"inp": 512},
                        "request_descriptors": [
                            {
                                "direction": "load",
                                "memory_name": "inp",
                                "multiplicity": 2,
                                "bytes_per_request": 512,
                                "padded": True,
                            }
                        ],
                    },
                }
            ],
        }
        records = adapt_archive(
            archive,
            selected,
            {"target": "axu5evb"},
            "conv2d_packed.vta",
            "schedule-v1",
            "original",
            "source-hash",
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(len(record["candidate_id"]), 64)
        self.assertEqual(record["debug"]["config_index"], 11)
        self.assertEqual(record["command_signature"]["status"], "not_measured")
        aggregate = record["transfer_signature"]["descriptor_aggregates"]
        self.assertEqual(aggregate["expanded_calls"], 2)
        self.assertEqual(aggregate["exact_request_bytes_histogram"]["load"]["512"], 2)


if __name__ == "__main__":
    unittest.main()
