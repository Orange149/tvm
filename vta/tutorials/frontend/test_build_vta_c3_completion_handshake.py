import subprocess
import pytest
from build_vta_c3_completion_handshake import source_variant, ROOT, HERE


def test_variant_changes_decision_but_never_writes_unknown_register():
    original = (ROOT / "3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc").read_text()
    observer = source_variant(original, HERE / "c3_fresh_completion.h", False)
    guarded = source_variant(original, HERE / "c3_fresh_completion.h", True)
    assert "if (flag == VTA_DONE) break;" in observer
    assert "if (flag == VTA_DONE) break;" not in guarded
    assert "C3AcceptFreshCompletion(c3_valid, flag)" in guarded
    assert original.count("VTAWriteMappedReg(") == observer.count("VTAWriteMappedReg(") == guarded.count("VTAWriteMappedReg(")


def test_stale_level_rejected_and_short_fresh_event_accepted(tmp_path):
    source = tmp_path / "test.cc"
    source.write_text('#include "' + str(HERE / "c3_fresh_completion.h") + '"\n'
        'int main(){ if(C3AcceptFreshCompletion(0,1)) return 1; '
        'if(C3AcceptFreshCompletion(1,0)) return 2; '
        'if(!C3AcceptFreshCompletion(1,1)) return 3; '
        'if(C3AcceptFreshCompletion(2,1)) return 4; return 0; }')
    binary = tmp_path / "test"
    subprocess.run(["g++", "-std=c++17", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
