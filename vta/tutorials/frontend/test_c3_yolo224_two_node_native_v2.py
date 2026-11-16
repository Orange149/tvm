import unittest
from c3_valid_window import certify_trace


def load(tensor,domain):
    return {'kind':'load','tensor':tensor,'domain':domain,'offset':0,'sram':0,
            'x':1,'y':1,'stride':1,'pad':[0,0,0,0]}


def kernel(reset):
    return {'kind':'kernel','scope':int(not reset),'mode_reset':[0,reset],
            'uops':[{'args':[0,reset,0,0,0,0,0,0],'loops':[]}]}


class ActualDependencyTests(unittest.TestCase):
    def setUp(self):
        self.events=[load(0,2),load(1,1),kernel(1),kernel(0),
            {'kind':'store','tensor':3,'domain':4,'offset':0,'sram':0,
             'x':1,'y':1,'stride':1,'pad':[0,0,0,0]}]

    def test_unrelated_input_weight_and_reset_need_no_total_order(self):
        result=certify_trace(self.events,ancestors=[0,0,0,7,15])
        self.assertEqual(result['packed_product_count'],1)

    def test_missing_weight_load_edge_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'READ_BEFORE_LOAD_COMPLETION'):
            certify_trace(self.events,ancestors=[0,0,0,5,15])

    def test_unordered_input_overwrite_is_still_rejected(self):
        events=self.events[:4]+[load(0,2)]+self.events[4:]
        with self.assertRaisesRegex(ValueError,'OVERWRITE_BEFORE_LAST_READ'):
            certify_trace(events,ancestors=[0,0,0,7,7,31])

    def test_store_without_compute_completion_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'STORE_BEFORE_REDUCTION_POSTPROCESS'):
            certify_trace(self.events,ancestors=[0,0,0,7,7])


if __name__=='__main__':unittest.main()
