"""Fresh-process regression for VTA IR registration, without target calls."""
import json
import os
from pathlib import Path
import subprocess
import unittest
from c3_node46_tps_local import HOLD,HERE,PYTHON,REPO


class FreshReader(unittest.TestCase):
    def test_fresh_process_loads_real_qualified_node46_parent(self):
        path=HOLD/'20261017_c3_node46_tps_local_run01/prepare/R/parent.json'
        code='''import json
from c3_node46_tps_identity_readonly import load_parent
m=load_parent(PATH)
assert len(m.functions)==1
f=m[list(m.functions)[0]]
assert str(f.attrs['global_symbol'])=='tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast'
print(json.dumps({'loaded':True,'target_calls':0}))
'''.replace('PATH',repr(str(path)))
        env=dict(os.environ,PYTHONPATH=':'.join(str(REPO/p) for p in ('python','vta/python','vta/tutorials/frontend')),
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        result=subprocess.run([str(PYTHON),'-c',code],env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),{'loaded':True,'target_calls':0})


if __name__=='__main__':unittest.main()
