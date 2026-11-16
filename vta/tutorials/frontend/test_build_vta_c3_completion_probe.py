from pathlib import Path
import subprocess
import pytest
from build_vta_c3_completion_probe import replace_once, derived_driver, ROOT, HERE


def test_patching_preserves_original_decision_and_no_new_register_write():
    source = (ROOT / "3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc").read_text()
    derived = derived_driver(source, HERE / "c3_completion_probe.h")
    assert source.count("if (flag == VTA_DONE) break;") == derived.count("if (flag == VTA_DONE) break;") == 1
    assert source.count("VTAWriteMappedReg(") == derived.count("VTAWriteMappedReg(")
    assert "DONE_RD_OFFSET +" not in derived


def test_missing_or_duplicate_anchor_rejected():
    for value in ("nothing", "xx"):
        with pytest.raises(ValueError):
            replace_once(value, "x", "y")


def test_probe_records_stale_and_fresh_samples(tmp_path):
    source = tmp_path / "state.cc"
    source.write_text('#include "' + str(HERE / "c3_completion_probe.h") + '"\n'
        'int main(){ C3CompletionProbe p(true); p.before_done=1; p.Observe(0,1); '
        'if(p.saw_zero || p.samples!=1) return 1; p.Observe(1,0); p.Observe(2,1); '
        'if(!p.saw_zero || !p.saw_high_then_zero || p.changes!=2) return 2; '
        'C3CompletionProbe disabled(false); disabled.Observe(0,1); return disabled.samples!=0; }')
    binary = tmp_path / "state"
    subprocess.run(["g++", "-std=c++17", str(source), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True)
