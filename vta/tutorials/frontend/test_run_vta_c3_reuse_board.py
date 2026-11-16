import pytest
from run_vta_c3_reuse_board import validate_state, control_blocks, BITSTREAM


def state(raw="tmpfs on /var/volatile type tmpfs (rw)\n"):
    return dict(fpga_state="operating", udmabuf_size=201326592,
                bitstream_sha256=BITSTREAM, raw=raw)


def test_preflight_valid():
    validate_state(state(), exclusive=True)


@pytest.mark.parametrize("raw", [
    "tmpfs on /var/volatile type tmpfs\n[5] EXT4-fs (mmcblk1p2): error count since last fsck: 48",
    "tmpfs on /var/volatile type tmpfs\n 900 root 0:00 ./tvm_rpc server --port=9090",
    "tmpfs on /run type tmpfs\n",
])
def test_preflight_rejects_unsafe_state(raw):
    with pytest.raises(ValueError):
        validate_state(state(raw), exclusive=True)


def test_control_block_requires_all_four_correct():
    modes = ["weight_resident_barrier", "input_weight_resident_barrier"]
    resident = [dict(family_id="M1", residence_mode=m, candidate_id="r"+m) for m in modes]
    reload = [dict(family_id="M1", residence_mode=m, candidate_id="c"+m) for m in modes]
    passed = {r["candidate_id"] for r in resident + reload}
    blocks = control_blocks(resident, reload, passed)
    assert [r["candidate_id"][0] for r in blocks[0][1]] == ["r", "c", "r", "c"]
    passed.remove(reload[0]["candidate_id"])
    assert control_blocks(resident, reload, passed) == []
