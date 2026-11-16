"""Read-only archive binding for the V4 two-bank experiment.

Bit headers are parsed strictly. Payload equality is an exact byte comparison,
not an assertion that a filename/date identifies a synthesized design. The
ACC interval is issued only for the reviewed, fingerprinted GEMM RTL below;
pipeline latency alone never issues that certificate. Changed RTL fails closed.
"""
from pathlib import Path
import hashlib
import json
import re
import struct
import zipfile
import xml.etree.ElementTree as ET


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse_bit(data):
    def take(n):
        nonlocal offset
        if n < 0 or offset+n > len(data):
            raise ValueError("TRUNCATED_BITSTREAM")
        result = data[offset:offset+n]
        offset += n
        return result
    offset = 0
    magic = take(struct.unpack(">H", take(2))[0])
    if magic != bytes.fromhex("0ff00ff00ff00ff000"):
        raise ValueError("BITSTREAM_MAGIC")
    if take(2) != b"\x00\x01":
        raise ValueError("BITSTREAM_FIELD_PREFIX")
    fields = {}
    for expected in b"abcd":
        if take(1) != bytes([expected]):
            raise ValueError("BITSTREAM_FIELD_ORDER")
        value = take(struct.unpack(">H", take(2))[0])
        if not value.endswith(b"\x00"):
            raise ValueError("BITSTREAM_METADATA_TERMINATOR")
        fields[chr(expected)] = value[:-1].decode("ascii")
    if take(1) != b"e":
        raise ValueError("BITSTREAM_PAYLOAD_FIELD")
    size = struct.unpack(">I", take(4))[0]
    payload_offset = offset
    payload = take(size)
    if offset != len(data) or payload.find(bytes.fromhex("aa995566")) < 0:
        raise ValueError("BITSTREAM_TRAILING_DATA_OR_MISSING_SYNC")
    return {"metadata": fields, "payload_offset": payload_offset,
            "payload_bytes": size, "whole_sha256": sha(data),
            "payload_sha256": sha(payload)}, payload


def audit_archive(repo):
    repo = Path(repo).resolve()
    build = repo/"3rdparty/vta-hw/build/hardware/xilinx/hls/axu5evb_1x16_i8w8a32_15_15_18_17"
    files = [repo/"apps/vta_rpc/firmware/vta_hpc.bit", build/"export/vta.bit",
             build/"vta.runs/impl_1/vta_wrapper.bit"]
    bindings, payloads, bits = {}, [], []
    def bind(path):
        path = Path(path)
        data = path.read_bytes()
        bindings[str(path)] = sha(data)
        return data
    for path in files:
        meta, payload = parse_bit(bind(path))
        bits.append({"path": str(path), **meta})
        payloads.append(payload)
    if not all(p == payloads[0] for p in payloads):
        raise ValueError("BITSTREAM_CONFIG_PAYLOAD_DIFFERENT")
    with zipfile.ZipFile(build/"vta_wrapper.xsa") as archive:
        bind(build/"vta_wrapper.xsa")
        embedded, payload = parse_bit(archive.read("vta_wrapper.bit"))
        if payload != payloads[0]:
            raise ValueError("XSA_BITSTREAM_CONFIG_DIFFERENT")
        hwh_data = archive.read("vta.hwh")
        hwh = ET.fromstring(hwh_data)
    bd_path = build/"vta.srcs/sources_1/bd/vta/vta.bd"
    bd = json.loads(bind(bd_path))["design"]["components"]
    depths = {}
    for name in ("load_queue", "gemm_queue", "store_queue",
                 "l2g_queue", "g2l_queue", "g2s_queue", "s2g_queue"):
        depths[name] = int(bd[name]["parameters"]["Input_Depth_axis"]["value"])
        modules = [m for m in hwh.iter("MODULE") if m.attrib.get("INSTANCE") == name]
        if len(modules) != 1:
            raise ValueError("XSA_FIFO_MODULE_IDENTITY")
        params = {p.attrib["NAME"]:p.attrib["VALUE"] for p in modules[0].iter("PARAMETER")}
        hdepth = params.get("Input_Depth_axis", params.get("C_INPUT_DEPTH_AXIS"))
        if hdepth is None or int(hdepth) != depths[name]:
            raise ValueError("XSA_BD_FIFO_DEPTH_DISAGREEMENT:" + name)
        bind(build/f"vta.srcs/sources_1/bd/vta/ip/vta_{name}_0/vta_{name}_0.xci")
    rtl_paths = [build/"vta_compute/soln/syn/verilog/compute.v",
                 build/"vta_compute/soln/impl/ip/hdl/verilog/compute.v",
                 build/"ip_lib/xilinx_com_hls_compute_1_0_1/hdl/verilog/compute.v"]
    rtl = [bind(p) for p in rtl_paths]
    if not all(r == rtl[0] for r in rtl):
        raise ValueError("COMPUTE_RTL_ARCHIVE_DISAGREEMENT")
    log_path = build/"vta.runs/impl_1/runme.log"
    log = bind(log_path).decode()
    required = ["xilinx_com_hls_compute_1_0_1", "vta_compute_0_0.dcp",
                "write_bitstream -force vta_wrapper.bit", "Bitgen Completed Successfully"]
    if not all(t in log for t in required):
        raise ValueError("IMPLEMENTATION_LOG_BINDING_INCOMPLETE")
    bind(build/"vta.srcs/sources_1/bd/vta/ip/vta_compute_0_0/vta_compute_0_0.dcp")
    bind(build/"vta_compute/soln/.autopilot/db/vta.pragma.0.cc")
    bind(build/"vta_compute/soln/syn/report/compute_csynth.rpt")
    bind(build/"vta_compute/soln/syn/report/compute_csynth.xml")
    bind(repo/"3rdparty/vta-hw/hardware/xilinx/src/vta.cc")
    interval = reviewed_gemm_feedback(build, bind)
    return {"schema":"c3_tps_window_hardware_v1",
            "status":"CONFIG_PAYLOAD_ARCHIVE_AND_GEMM_FEEDBACK_BOUND",
            "bitstreams":bits, "embedded_xsa_bitstream":embedded,
            "payload_exact_bytes_equal":True, "hwh_sha256":sha(hwh_data),
            "fifo_depths":depths, "compute_rtl_exact_bytes_equal":True,
            "source_bindings":bindings,
            "verified_acc_interval":interval['conservative_minimum_issue_distance'],
            "gemm_feedback_proof":interval,
            "limits":["Feedback rule is for this fixed GEMM RTL and its non-stalling pp1 pipeline",
                      "No new ALU/RESET fusion is qualified",
                      "Current source file is inventoried, not asserted identical to archived preprocessed source"],
            "board_contacted":False, "hardware_rebuilt":False}


def reviewed_gemm_feedback(build, bind):
    """Conservative RAW rule from fixed RTL, not II/latency-as-gap inference.

    pp1 has one stage with iter0..iter3 age registers. ACC read is enabled for
    iter2 on port1, and writeback for iter3 on port0. Its BRAM is synchronous
    read-first: a same-edge read/write must NOT be used as forwarding. All
    pp1 blocking expressions are constant false. Four issue positions between
    equal ACC destinations conservatively separate the earlier write from the
    later read. This exceeds the two-position read/write requirement; it is
    not a claim that the hardware's exact feedback latency is four cycles.
    """
    rtl_path=build/'vta_compute/soln/syn/verilog/compute.v'
    raw=bind(rtl_path)
    if sha(raw)!='d6863a81f72908b874fdfa86e783a43a90e2039344fd7b90ec0ccfb1c077e139':
        raise ValueError('UNREVIEWED_GEMM_RTL_FEEDBACK')
    text=raw.decode().replace('\r\n','\n')
    required=[
        "assign ap_block_pp1_stage0 = ~(1'b1 == 1'b1);",
        "assign ap_block_pp1_stage0_11001 = ~(1'b1 == 1'b1);",
        "assign ap_block_pp1_stage0_subdone = ~(1'b1 == 1'b1);",
        'ap_enable_reg_pp1_iter3 <= ap_enable_reg_pp1_iter2;',
        'ap_enable_reg_pp1_iter2 <= ap_enable_reg_pp1_iter1;',
        'ap_enable_reg_pp1_iter1 <= ap_enable_reg_pp1_iter0;',
        "if (((ap_enable_reg_pp1_iter3 == 1'b1) & (1'b0 == ap_block_pp1_stage0))) begin\n        acc_mem_V_address0 = acc_mem_V_addr_1_reg_14305;",
        "end else if (((ap_enable_reg_pp1_iter2 == 1'b1) & (1'b0 == ap_block_pp1_stage0))) begin\n        acc_mem_V_address1 = zext_ln544_2_fu_7707_p1;",
        "(1'b0 == ap_block_pp1_stage0_11001) & (ap_enable_reg_pp1_iter2 == 1'b1)",
        "(1'b0 == ap_block_pp1_stage0_11001) & (icmp_ln253_reg_14223_pp1_iter2_reg == 1'd0) & (ap_enable_reg_pp1_iter3 == 1'b1)",
        "acc_mem_V_we0 = 64'd18446744073709551615;",
    ]
    if not all(item in text for item in required):
        raise ValueError('GEMM_ACC_FEEDBACK_RTL_CLAUSE_MISSING')
    memories=[build/'vta_compute/soln/syn/verilog/compute_acc_mem_V.v',
              build/'vta_compute/soln/impl/ip/hdl/verilog/compute_acc_mem_V.v',
              build/'ip_lib/xilinx_com_hls_compute_1_0_1/hdl/verilog/compute_acc_mem_V.v']
    memory=[bind(p) for p in memories]
    if not all(m==memory[0] for m in memory):
        raise ValueError('ACC_MEMORY_RTL_ARCHIVE_DISAGREEMENT')
    ram=memory[0].decode()
    if not all(item in ram for item in ('always @(posedge clk)',
             'ram[addr0][i*COL_WIDTH +: COL_WIDTH] <= d0[i*COL_WIDTH +: COL_WIDTH];',
             'q1[i*COL_WIDTH +: COL_WIDTH] <= ram[addr1][i*COL_WIDTH +: COL_WIDTH];')):
        raise ValueError('ACC_SYNCHRONOUS_READ_FIRST_RULE_MISSING')
    return {'status':'REVIEWED_BOUND_GEMM_ACC_FEEDBACK_RULE',
            'conservative_minimum_issue_distance':4,
            'read_age_register':'ap_enable_reg_pp1_iter2',
            'write_age_register':'ap_enable_reg_pp1_iter3',
            'pipeline_blocking':'all pp1 blocking expressions constant false',
            'same_edge_forwarding_assumed':False,
            'gemm_rtl_path':str(rtl_path),'gemm_rtl_sha256':sha(raw),
            'acc_memory_rtl_sha256':sha(memory[0]),
            'reviewed_line_groups':[[4984,5030],[5647,5719],[5777,5783],[6235,6243],[6755,6759]],
            'scope':'nonempty MAC GEMM; full expanded addresses and both hardware-loop carries must pass'}

