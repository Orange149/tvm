"""Zero-target checks for strict two-symbol routing and observation-only changes."""
import unittest
from c3_yolo224_two_node_prepare import (OFFICIAL, HERE, SYMBOLS, precheck_sources,
                                        experimental_collection, experimental_runtime, graph_helper)


class GraphPreparationTests(unittest.TestCase):
    def test_exact_original_graph_symbols(self):
        graph = precheck_sources()
        self.assertEqual(set(SYMBOLS), {34,46})
        self.assertEqual(len(graph['heads']), 8)

    def test_collection_inverse_only_capacity_and_capture_state(self):
        path = OFFICIAL/'runtime_local_final/c3_official_window_runtime_collection.h'
        before = path.read_text();after = experimental_collection(before)
        self.assertEqual(after.replace('std::array<Submit, 32> submits{};\n  bool graph_dump{false};',
                                      'std::array<Submit, 16> submits{};'), before)

    def test_runtime_inverse_preserves_all_execution_code(self):
        before = (OFFICIAL/'runtime_local_final/c3_official_window_runtime_generated.cc').read_text()
        after = experimental_runtime(before)
        for name in ('VTA_DEBUG_DUMP_INSN','VTA_DEBUG_DUMP_UOP'):
            after = after.replace('if ((debug_flag_ & '+name+') || c3_official_window::State().graph_dump) {',
                                  'if (debug_flag_ & '+name+') {')
        self.assertEqual(after.rsplit('\nextern "C" void C3Yolo224SetGraphDump',1)[0], before)

    def test_graph_timer_body_and_target_call_unchanged(self):
        before = (HERE/'c3_yolo416_graph_once.cc').read_text();after = graph_helper(before)
        a = before.split('TVM_REGISTER_GLOBAL("c3.yolo416.graph_timed_once")',1)[1]
        b = after.split('TVM_REGISTER_GLOBAL("c3.yolo416.graph_timed_once")',1)[1]
        self.assertEqual(a,b)
        self.assertNotIn('debug(handle, 6)',after)


if __name__ == '__main__': unittest.main()
