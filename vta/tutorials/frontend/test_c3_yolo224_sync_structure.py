import unittest
from c3_yolo224_sync_structure import classify_events

def kernel(reset=0,inp=0):return {'kind':'kernel','mode_reset':[0,reset], 'uops':[{'args':[0,reset,0,inp,0,0,0,0],'loops':[]}]}
def load(domain,sram):return {'kind':'load','domain':domain,'sram':sram,'x':1,'y':1,'pad':[0,0,0,0]}
def trace(reset=0,inp=1):return [kernel(reset),{'kind':'push','pair':[2,1]},{'kind':'pop','pair':[2,1]},load(2,inp),load(1,0)]
class Tests(unittest.TestCase):
 def test_disjoint_is_only_candidate(self):
  d=classify_events(trace());self.assertEqual(d['waits'][0]['classification'],'ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE');self.assertFalse(d['full_native_safety_certified'])
 def test_mac_same_address_blocked(self):self.assertEqual(classify_events(trace(inp=0))['waits'][0]['classification'],'MAC_INPUT_OVERWRITE_PROTECTION')
 def test_reset_physical_not_called_safe(self):self.assertEqual(classify_events(trace(reset=1,inp=0))['waits'][0]['classification'],'RESET_PHYSICAL_READ_UNRESOLVED')
 def test_internal_push_is_boundary(self):
  t=trace();t.insert(4,{'kind':'push','pair':[1,2]});t.append({'kind':'pop','pair':[1,2]})
  self.assertEqual(classify_events(t)['waits'][0]['classification'],'NO_WEIGHT_TARGET_BEFORE_CONTROL_BOUNDARY')
 def test_unbalanced_rejected(self):
  with self.assertRaisesRegex(ValueError,'UNBALANCED'):classify_events(trace()[:2])
 def test_padding_counted_as_write(self):
  t=trace(inp=1);t[0]=kernel(inp=2);t[3]['pad']=[0,0,1,0]
  self.assertEqual(classify_events(t)['waits'][0]['classification'],'MAC_INPUT_OVERWRITE_PROTECTION')
 def test_sealed_standard_and_tps_differ(self):
  import tvm,vta
  from c3_yolo224_sync_structure import SOURCE,HOLD,project
  standard=tvm.ir.load_json((SOURCE/'G2/tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_3.tir.json').read_text())
  parent=tvm.ir.load_json((HOLD/'20261016_c3_tps_sync_release_local_run04/prepare/R/parent.json').read_text())
  tps=parent[list(parent.functions)[0]]
  for fn,expected in [(standard,0),(tps,2)]:
   events,_,_=project(fn)
   d=classify_events(events)
   self.assertEqual(d['MAC_categories'].get('ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE',0),expected)
if __name__=='__main__':unittest.main()
