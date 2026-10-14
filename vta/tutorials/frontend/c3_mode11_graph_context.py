"""Pure helpers for the C3 mode11 graph-context correctness diagnostic.

This module deliberately contains no timing API and no board side effects.  It
holds the fail-closed graph contract, the monotonic debug cursor, array
comparison helpers, and the three-argument core CPU reference used by v6.1.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


TARGET_NODE = 61
INPUT_NODE = 58
WEIGHT_NODE = 59
ADD_NODE = 60
SUCCESSOR_NODE = 62
TARGET_SYMBOL = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_10"
EXPECTED_ABI = ("X", "p27", "p28", "Y")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def array_sha256(array):
    value = np.ascontiguousarray(array)
    return sha256_bytes(value.view(np.uint8).tobytes())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def _entry_attr(graph, key, node, output=0):
    row_ptr = graph["node_row_ptr"]
    entry = int(row_ptr[node]) + int(output)
    kind, values = graph["attrs"][key]
    return entry, kind, values[entry]


def graph_tensor_contract(graph, node, output=0):
    entry, _, shape = _entry_attr(graph, "shape", node, output)
    entry2, _, dtype = _entry_attr(graph, "dltype", node, output)
    entry3, _, device = _entry_attr(graph, "device_index", node, output)
    entry4, _, storage = _entry_attr(graph, "storage_id", node, output)
    if len({entry, entry2, entry3, entry4}) != 1:
        raise ValueError("inconsistent tensor-entry mapping")
    dtype_obj = np.dtype(str(dtype))
    contiguous = []
    stride = 1
    for extent in reversed(shape):
        contiguous.append(stride)
        stride *= int(extent)
    return {
        "node_id": int(node),
        "output_index": int(output),
        "entry_id": entry,
        "shape": [int(x) for x in shape],
        "dtype": str(dtype),
        "device_index": int(device),
        "storage_id": int(storage),
        "byte_offset": 0,
        "strides_elements": list(reversed(contiguous)),
        "contiguous": True,
        "nbytes": int(stride * dtype_obj.itemsize),
    }


def frozen_graph_contract(graph):
    """Extract and validate the exact node/entry/ABI contract from graph JSON."""
    if isinstance(graph, (str, bytes)):
        graph = json.loads(graph)
    target = graph["nodes"][TARGET_NODE]
    if target["op"] != "tvm_op" or target["attrs"]["func_name"] != TARGET_SYMBOL:
        raise ValueError("frozen target node or symbol drifted")
    expected_inputs = [[INPUT_NODE, 0, 0], [WEIGHT_NODE, 0, 0], [ADD_NODE, 0, 0]]
    if target["inputs"] != expected_inputs:
        raise ValueError("target ABI input order drifted")
    tensors = {
        "X": graph_tensor_contract(graph, INPUT_NODE),
        "p27": graph_tensor_contract(graph, WEIGHT_NODE),
        "p28": graph_tensor_contract(graph, ADD_NODE),
        "Y": graph_tensor_contract(graph, TARGET_NODE),
        "Y62": graph_tensor_contract(graph, SUCCESSOR_NODE),
    }
    expected = {
        "X": ([1, 32, 16, 16, 1, 16], "int8"),
        "p27": ([16, 32, 1, 1, 16, 16], "int8"),
        "p28": ([16, 1, 1, 1, 16], "int32"),
        "Y": ([1, 16, 16, 16, 1, 16], "int8"),
        "Y62": ([1, 255, 16, 16], "float32"),
    }
    for name, (shape, dtype) in expected.items():
        if tensors[name]["shape"] != shape or tensors[name]["dtype"] != dtype:
            raise ValueError("frozen tensor contract drifted: " + name)
    if tensors["X"]["storage_id"] != 12 or tensors["Y62"]["storage_id"] != 12:
        raise ValueError("expected sid=12 input/successor reuse is absent")
    if tensors["Y"]["storage_id"] == 12:
        raise ValueError("target output unexpectedly aliases sid=12")
    return {
        "target_node": TARGET_NODE,
        "target_symbol": TARGET_SYMBOL,
        "abi_order": list(EXPECTED_ABI),
        "graph_inputs_in_abi_order": [INPUT_NODE, WEIGHT_NODE, ADD_NODE],
        "tensors": tensors,
        "entry_mapping_source": "node_row_ptr[node_id] + output_index",
        "storage_lifetime": {
            "sid12_nodes": [INPUT_NODE, SUCCESSOR_NODE],
            "assert_X_unchanged_after_node62": False,
            "post_node62_target_output": "UNOBSERVABLE_WITH_CURRENT_RUNTIME",
        },
    }


class StrictDebugCursor:
    """Allow only strictly increasing execute-node/read-last-node operations."""

    BANNED = (
        "run",
        "debug_get_output",
        "run_individual",
        "run_individual_node",
        "profile",
        "profile_rpc",
    )

    def __init__(self, module):
        self.module = module
        self.last_executed_node = -1
        self.calls = []

    def execute_node(self, node):
        node = int(node)
        if node <= self.last_executed_node:
            raise ValueError("debug nodes must be strictly increasing")
        self.module._execute_node(node)  # pylint: disable=protected-access
        self.last_executed_node = node
        self.calls.append({"api": "execute_node", "node": node, "executes_operator": True})

    def execute_through(self, node):
        """Execute every graph node after the cursor through ``node``.

        The local GraphExecutorDebug ``execute_node`` packed function executes
        exactly one node.  It does *not* execute a graph prefix.  Prefix
        capture therefore has to visit every intervening node explicitly.
        """
        node = int(node)
        if node <= self.last_executed_node:
            raise ValueError("debug target must be after the last executed node")
        executed = []
        for current in range(self.last_executed_node + 1, node + 1):
            self.execute_node(current)
            executed.append(current)
        return executed

    def get_node_output(self, node, output=0):
        node, output = int(node), int(output)
        if node != self.last_executed_node:
            raise ValueError("get_node_output may only read the last executed node")
        value = self.module._get_node_output(node, output)  # pylint: disable=protected-access
        self.calls.append({
            "api": "get_node_output", "node": node, "output": output,
            "executes_operator": False,
        })
        return value


def install_debug_api_traps(module):
    """Replace Python timing/rerun entry points with fail-closed traps."""
    def banned(name):
        def fail(*_args, **_kwargs):
            raise RuntimeError("forbidden debug API invoked: " + name)
        return fail

    installed = []
    for name in StrictDebugCursor.BANNED:
        # Install even when a particular wrapper version does not publish the
        # name today.  This makes later accidental access fail closed instead
        # of silently exposing a newly-added timing/rerun entry point.
        setattr(module, name, banned(name))
        installed.append(name)
    return installed


def summarize_array(array):
    value = np.asarray(array)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": array_sha256(value),
        "nonzero": int(np.count_nonzero(value)),
        "min": float(value.min()) if value.size else None,
        "max": float(value.max()) if value.size else None,
    }


def first_difference(expected, actual):
    left, right = np.asarray(expected), np.asarray(actual)
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "kind": "metadata",
            "expected_shape": list(left.shape), "actual_shape": list(right.shape),
            "expected_dtype": str(left.dtype), "actual_dtype": str(right.dtype),
        }
    flat = np.flatnonzero(left.reshape(-1) != right.reshape(-1))
    if flat.size == 0:
        return None
    index = int(flat[0])
    coordinate = [int(x) for x in np.unravel_index(index, left.shape)]
    return {
        "kind": "value", "flat_index": index, "index": coordinate,
        "expected": left.reshape(-1)[index].item(),
        "actual": right.reshape(-1)[index].item(),
        "mismatch_count": int(flat.size),
    }


def arrays_equal(expected, actual):
    return first_difference(expected, actual) is None


def ordinary_prefix_tap_graph(graph):
    """Expose nodes 58/61 through normal GraphExecutor without running node 62.

    The storage plan and every node through the target remain byte-for-byte
    unchanged.  Node 62 becomes a no-op only so its legal sid=12 reuse cannot
    overwrite node 58 before the two diagnostic heads are copied out.
    """
    source = json.loads(graph) if isinstance(graph, (str, bytes)) else copy.deepcopy(graph)
    frozen_graph_contract(source)
    tapped = copy.deepcopy(source)
    tapped["nodes"][SUCCESSOR_NODE] = {
        "op": "null", "name": "c3_diagnostic_disabled_successor62", "inputs": []
    }
    tapped["heads"] = [[INPUT_NODE, 0, 0], [TARGET_NODE, 0, 0]]
    if tapped["nodes"][:SUCCESSOR_NODE] != source["nodes"][:SUCCESSOR_NODE]:
        raise ValueError("tap changed a node before the successor")
    for key in ("attrs", "node_row_ptr", "arg_nodes"):
        if tapped[key] != source[key]:
            raise ValueError("tap changed frozen graph storage metadata: " + key)
    return tapped


def successor62_reference(y):
    """CPU reference for frozen node 62, used to validate the ordinary tap."""
    value = np.asarray(y)
    expected = (1, 16, 16, 16, 1, 16)
    if value.shape != expected or value.dtype != np.int8:
        raise ValueError("unexpected node61 tensor contract")
    logical = value.transpose(0, 4, 1, 5, 2, 3).reshape(1, 256, 16, 16)[:, :255]
    scaled = logical.reshape(1, 3, 85, 16, 16).astype(np.float32) * np.float32(0.179688)
    sigmoid = lambda x: np.float32(1.0) / (np.float32(1.0) + np.exp(-x))
    result = np.concatenate(
        (sigmoid(scaled[:, :, :2]), scaled[:, :, 2:4], sigmoid(scaled[:, :, 4:])), axis=2
    )
    return result.reshape(1, 255, 16, 16).astype(np.float32)


def core_reference(x, weight):
    """Packed 1x1 int8 convolution followed by >>8, clip(0,127), int8."""
    x = np.asarray(x)
    weight = np.asarray(weight)
    if x.shape != (1, 32, 16, 16, 1, 16) or x.dtype != np.int8:
        raise ValueError("unexpected packed input contract")
    if weight.shape != (16, 32, 1, 1, 16, 16) or weight.dtype != np.int8:
        raise ValueError("unexpected packed weight contract")
    logical_x = x.transpose(0, 2, 3, 1, 4, 5).reshape(1, 16, 16, 512)
    logical_w = weight[:, :, 0, 0, :, :].transpose(0, 2, 1, 3).reshape(256, 512)
    accum = np.einsum(
        "bhwi,oi->bhwo", logical_x.astype(np.int32), logical_w.astype(np.int32),
        dtype=np.int32,
    )
    shifted = np.right_shift(accum, 8)
    clipped = np.clip(shifted, 0, 127).astype(np.int8)
    return clipped.reshape(1, 16, 16, 16, 16).transpose(0, 3, 1, 2, 4)[:, :, :, :, None, :]


def classify_replay(matrix):
    """Return the earliest reliable boundary plus non-exclusive flags."""
    flags = ["POST_TARGET_OUTPUT_UNOBSERVABLE_WITH_CURRENT_RUNTIME"]
    if matrix.get("R", {}).get("status") != "passed":
        return "BASE_FAILURE_NOT_REPRODUCED", flags
    if matrix.get("S", {}).get("status") == "DEBUG_PREFIX_REFERENCE_INVALID":
        flags.extend(["DEBUG_PATH_CHANGED_REPRODUCTION", "REFERENCE_REPLAY_NOT_RUN"])
        return "DEBUG_PREFIX_REFERENCE_INVALID", flags
    if matrix.get("B", {}).get("incumbent_matches_reference") is False:
        return "REFERENCE_REPLAY_CONTRACT_INVALID", flags
    mode11 = matrix.get("B", {}).get("mode11_matches_reference")
    if mode11 is False:
        repeated = matrix["B"].get("mode11_repeat_matches_reference") is False
        if not repeated:
            flags.append("UNSTABLE_REPRODUCTION")
        zero = matrix["B"].get("mode11_zero_p28_matches_incumbent")
        flags.append("P28_SENSITIVE" if zero is True else "P28_ZERO_STILL_FAILS")
        return "FUSED_FUNCTION_INDEPENDENT_BUFFER_DIVERGENCE", flags
    c = matrix.get("C", {})
    if c.get("entry_matches") is False:
        return "UPSTREAM_ENTRY_DIVERGENCE", flags
    if c.get("exit_matches") is False:
        return "GRAPH_CONTEXT_TARGET_EXIT_DIVERGENCE", flags
    if c.get("successor_matches") is False:
        return "SUCCESSOR_OR_LIVE_STATE_DIVERGENCE", flags
    if c.get("debug_paths_correct") and c.get("full_graph_correct") is False:
        flags.append("DEBUG_PATH_CHANGED_REPRODUCTION")
    return "NO_RELIABLE_DIVERGENCE_LOCALIZED", flags
