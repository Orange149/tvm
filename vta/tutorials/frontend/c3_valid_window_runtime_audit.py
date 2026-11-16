#!/usr/bin/env python3
"""Local source/identity checks for the frozen C3 aggregation runtime.

This does not load a VTA DSO, contact a board, mutate the compiler/runtime,
or qualify a new aggregation program. The cache fixture compiles the exact
UopKernelMap and public PushGEMMOp method with host-only test collaborators.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from c3_valid_window_uop_values import certify_uop_values


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HISTORY = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20261013_c3_banerjee_baseline_completion_run02"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run(command, output, label):
    result = subprocess.run(command, capture_output=True, text=True, cwd=REPO)
    (output / (label + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(label + " failed; see saved log")
    return result.stdout


def cache_fixture(text):
    map_start = text.index("class UopKernelMap {")
    map_end = text.index("\nenum PipelineStage", map_start)
    method_start = text.index("  void PushGEMMOp(void** uop_handle,")
    method_end = text.index("\n  void PushALUUop(", method_start)
    kernel_map = text[map_start:map_end]
    push_method = text[method_start:method_end]
    return r'''#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
#define CHECK_LT(a,b) CHECK((a)<(b))
#define CHECK_EQ(a,b) CHECK((a)==(b))
#define VTA_DEBUG_DUMP_UOP 1
double NowMicros() { return 0; }
struct RuntimeProfiler {
  static RuntimeProfiler& Global() { static RuntimeProfiler r; return r; }
  void AddPushGEMMOp(double) {}
};
struct UopKernel {
  std::vector<char> signature;
  int marker{};
  UopKernel(const char* p, int n) { if (n) signature.assign(p,p+n); }
  void Dump() {}
};
''' + kernel_map + r'''
struct CommandQueue {
  enum class ReplayMode { kDisabled, kReplay };
  ReplayMode mode_{ReplayMode::kDisabled};
  UopKernel* record_kernel_{};
  UopKernel* selected{};
  int debug_flag_{};
  void PushGEMMOp(UopKernel* kernel) { selected=kernel; }
  void CheckInsnOverFlow() {}
''' + push_method + r'''
};
CommandQueue* active;
int initialized=0;
int init_a(void*) { ++initialized; active->record_kernel_->marker=1; return 0; }
int init_b(void*) { ++initialized; active->record_kernel_->marker=2; return 0; }
template<class F> bool rejected(F f) { try { f(); } catch (const std::runtime_error&) { return true; } return false; }
int main() {
  UopKernelMap map;
  int sig=0;
  for (sig=0; sig<=98; ++sig) assert(map.Get(&sig,sizeof(sig)) != nullptr);
  sig=99; assert(rejected([&]{ map.Get(&sig,sizeof(sig)); }));
  sig=-2; assert(rejected([&]{ map.Get(&sig,sizeof(sig)); }));
  int two[2]={0,1}; assert(rejected([&]{ map.Get(two,sizeof(two)); }));
  sig=-1; assert(map.Get(nullptr,0)==map.Get(&sig,sizeof(sig)));
  CommandQueue queue; active=&queue;
  void* shared=nullptr; sig=7;
  queue.PushGEMMOp(&shared,init_a,&sig,sizeof(sig));
  queue.PushGEMMOp(&shared,init_b,&sig,sizeof(sig));
  assert(queue.selected->marker==1 && initialized==1);
  void* separate=nullptr;
  queue.PushGEMMOp(&separate,init_b,&sig,sizeof(sig));
  assert(queue.selected->marker==2 && initialized==2);
  sig=8; queue.PushGEMMOp(&shared,init_b,&sig,sizeof(sig));
  assert(queue.selected->marker==2 && initialized==3);
  std::cout << "{\"same_handle_same_signature_different_content_reuses_first\":true,"
               "\"different_handles_separate_content\":true,"
               "\"different_int_signatures_separate_content\":true,"
               "\"ordinary_nonnegative_signature_min\":0,"
               "\"ordinary_nonnegative_signature_max\":98,"
               "\"signature_99_rejected\":true,\"two_int_signature_rejected\":true,"
               "\"signature_minus_one_aliases_empty\":true}\n";
}
'''


def greedy_fixture(queue_test):
    """Use the actual queue header to expose a prior-use overwrite after wrap.

    Small SRAM is only a host fixture, not a modified experiment capacity.
    The dummy opcode=2 instruction records a pending GEMM read of B's range.
    """
    prefix = queue_test.read_text().split("int main() {", 1)[0]
    return "#include <cstdio>\n" + prefix + r'''int main() {
  UopQueue<4096,false,false> q;
  InsnQueue<4096,false,false> insn;
  auto a=make(4,0), b=make(4,10), c=make(4,20), d=make(4,30);
  for (auto k : {&a,&b,&c,&d}) { q.Push(k,[]{assert(false);}); q.FlushUopLoad(insn); }
  auto e=make(4,40), f=make(4,50);
  q.Push(&e,[]{assert(false);}); q.FlushUopLoad(insn);
  assert(b.cached());
  q.Push(&b,[]{assert(false);});
  auto* gemm=insn.CreateMemInsn(0);
  gemm->opcode=2; gemm->sram_base=b.sram_begin_; gemm->x_size=b.size();
  unsigned prior_load=insn.count()-2;
  assert(insn.data()[prior_load].sram_base==0 && insn.data()[prior_load].x_size==4);
  assert(insn.data()[prior_load+1].sram_base==4);
  q.Push(&f,[]{assert(false);}); q.FlushUopLoad(insn);
  assert(insn.data()[prior_load].sram_base==0 && insn.data()[prior_load].x_size==8);
  assert(insn.count()==prior_load+2);
  assert(f.sram_begin_==4 && !b.cached());
  printf("{\"sram_wrap\":true,\"prior_gemm_reads_old_cached_range\":true,"
         "\"greedy_extension_covers_range_before_prior_gemm\":true,"
         "\"general_inflight_safety_is_not_proven\":true}\n");
}
'''


def historical_tps_fifo_audit():
    """Bind actual cold/warm FIFO values to the sealed final TIR trace."""
    calls = [json.loads(line) for line in (HISTORY / "qualification/calls.jsonl").read_text().splitlines()]
    rows = [row for row in calls if row.get("program") == "TPS-oc-reuse-ci4" and row.get("dump_file")]
    assert len(rows) == 4
    cold_session = rows[0]["session_id"]
    trace_path = HISTORY / "binaries/TPS-oc-reuse-ci4/address_audit.json"
    kernels = json.loads(trace_path.read_text())["kernels"]
    paths = [HISTORY / "qualification" / row["dump_file"] for row in rows]
    for row, path in zip(rows, paths):
        assert digest(path) == row["dump_sha256"]
    results = []
    frozen = None
    for row, path in zip(rows, paths):
        certificate = certify_uop_values(path.read_text(), kernels, frozen)
        if frozen is None:
            frozen = certificate["cache_definitions"]
        assert certificate["compute_references"] == 28
        assert certificate["serialized_uop_elements"] == 9440
        assert certificate["uop_sram_wraps"] == 2
        same_session = row["session_id"] == cold_session
        certificate["session_binding"] = "SAME_SESSION" if same_session else "CROSS_SESSION_STRUCTURE_ONLY_NOT_WARM_CACHE_QUALIFICATION"
        certificate["cold_definition_session"] = cold_session
        results.append({"call_kind":row["call_kind"],"call_position":row.get("call_position"),
                        "session_id":row["session_id"],
                        "dump_file":str(path.relative_to(REPO)),"dump_sha256":digest(path),
                        "certificate":certificate})
    def rejected(text, expected, definitions):
        try:
            certify_uop_values(text, expected, definitions)
        except ValueError as error:
            return str(error)
        raise AssertionError("negative fixture unexpectedly accepted")
    cold = paths[0].read_text()
    warm = paths[1].read_text()
    negatives = {"warm_without_cold_definition":rejected(warm,kernels,None),
                 "changed_actual_initializer":rejected(cold.replace("acc=0, inp=0, wgt=0", "acc=0, inp=1, wgt=0",1),kernels,None),
                 "changed_actual_reference":rejected(warm.replace("range (0, 14)","range (1, 15)",1),kernels,frozen)}
    return {"status":"SEALED_TPS_COLD_FIFO_PASSED_WARM_STRUCTURE_MATCHED_SESSION_BINDING_UNRESOLVED",
            "program":"TPS-oc-reuse-ci4","final_trace_path":str(trace_path.relative_to(REPO)),
            "final_trace_sha256":digest(trace_path),"sram_wrap_present":True,"negative_rejections":negatives,
            "warm_cache_qualification":False,
            "warm_binding_gap":"three warmed dumps are session qualification:014; initializer dump is qualification:013; session 014 warmup did not dump initializer values",
            "scope":"actual UOP values, reset/mode/loop/opcode against sealed final TIR; tensor versions/tokens and unprinted ALU immediate remain separate",
            "calls":results}


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    production = REPO / "vta/runtime/runtime.cc"
    header = HERE / "c3_banerjee_uop_queue.h"
    text = production.read_text()
    source_start = text.index("template <int kMaxBytes, bool kCoherent, bool kAlwaysCache>\nclass UopQueue")
    source_end = text.index("// Internal kernel structure\nclass UopKernelMap", source_start)
    overlay = text[:source_start] + header.read_text() + "\n\n" + text[source_end:]
    seam = """      VTAMemInsn* insn = insn_queue_.CreateMemInsn(VTA_MEM_ID_UOP);
      insn->opcode = VTA_OPCODE_LOAD;
      uop_queue_.FlushUopLoad(insn);"""
    assert overlay.count(seam) == 2
    overlay = overlay.replace(seam, "      uop_queue_.FlushUopLoad(insn_queue_);")
    identities = {}
    for name, binary in [("runtime_board_attempt02", "libvta.so"), ("runtime_sim", "libvta_fsim.so")]:
        directory = HISTORY / name
        build = json.loads((directory / "build.json").read_text())
        hashes = {"base_runtime_sha256": digest(production), "header_sha256": digest(header),
                  "overlay_sha256": digest(directory / "runtime.cc"), "binary_sha256": digest(directory / binary)}
        assert build["returncode"] == 0 and build["enabled"]
        assert all(build[k] == v for k, v in hashes.items())
        assert overlay == (directory / "runtime.cc").read_text()
        assert "-DC3_BANERJEE_VALUE_UOPS=1" in build["command"]
        assert "-DC3_BANERJEE_GREEDY_UOPS=1" in build["command"]
        identities[name] = {"hashes": hashes, "build_sha256": digest(directory / "build.json"),
                            "actual_command_defines": [a for a in build["command"] if a.startswith("-D")]}
    fixture = output / "cache_identity_fixture.cc"
    fixture.write_text(cache_fixture(text))
    run(["g++", "-std=c++17", "-O2", str(fixture), "-o", str(output / "cache_identity_fixture")], output, "cache_identity_compile")
    cache = json.loads(run([str(output / "cache_identity_fixture")], output, "cache_identity_run"))
    queue_test = HERE / "c3_banerjee_uop_queue_test.cc"
    run(["g++", "-std=c++17", "-O2", str(queue_test), "-o", str(output / "actual_uop_queue_test")], output, "uop_queue_compile")
    run([str(output / "actual_uop_queue_test")], output, "uop_queue_run")
    greedy = output / "greedy_inflight_counterexample.cc"
    greedy.write_text(greedy_fixture(queue_test))
    run(["g++", "-std=c++17", "-O2", "-I" + str(HERE), str(greedy),
         "-o", str(output / "greedy_inflight_counterexample")], output, "greedy_inflight_compile")
    greedy_result = json.loads(run([str(output / "greedy_inflight_counterexample")], output, "greedy_inflight_run"))
    config = json.loads((REPO / "3rdparty/vta-hw/config/vta_config.json").read_text())
    assert config == {"TARGET":"axu5evb", "HW_VER":"0.0.2", "LOG_INP_WIDTH":3,
                     "LOG_WGT_WIDTH":3, "LOG_ACC_WIDTH":5, "LOG_BATCH":0, "LOG_BLOCK":4,
                     "LOG_UOP_BUFF_SIZE":15, "LOG_INP_BUFF_SIZE":15,
                     "LOG_WGT_BUFF_SIZE":18, "LOG_ACC_BUFF_SIZE":17}
    units = {"UOP":{"element_bytes":4,"sram_elements":8192,"sram_bytes":32768},
             "INP":{"element_bytes":16,"sram_elements":2048,"sram_bytes":32768},
             "WGT":{"element_bytes":256,"sram_elements":1024,"sram_bytes":262144},
             "ACC":{"element_bytes":64,"sram_elements":2048,"sram_bytes":131072},
             "OUT":{"element_bytes":16,"sram_elements":2048,"sram_bytes":32768}}
    sessions = [json.loads(line) for line in (HISTORY / "qualification/sessions.jsonl").read_text().splitlines()]
    bitstream_records = {tuple(sorted(s["bitstream_load_record"].items())) for s in sessions}
    assert len(bitstream_records) == 1
    result = {"schema":"c3_valid_window_runtime_source_audit_v1", "status":"SOURCE_AND_FROZEN_IDENTITY_CHECKS_PASSED_NEW_PROGRAMS_NOT_QUALIFIED",
              "board_contacted":False, "runtime_modified":False, "identities":identities,
              "cache_source_fixture":cache, "cache_fixture_sha256":digest(fixture),
              "actual_queue_regression":{"passed":True,"source_sha256":digest(queue_test),
                    "scope":"actual queue header with fake DMA and reduced test SRAM; does not qualify a new program"},
              "greedy_inflight_counterexample":greedy_result,
              "greedy_inflight_fixture_sha256":digest(greedy),
              "historical_tps_uop_fifo":historical_tps_fifo_audit(),
              "memory_units":units,"historical_bitstream_record":dict(next(iter(bitstream_records))),
              "historical_bitstream_scope":"sealed prior sessions only; current board not observed",
              "source_hashes":{str(p.relative_to(REPO)):digest(p) for p in [production,header,
                    REPO/'src/target/llvm/codegen_cpu.cc',REPO/'3rdparty/vta-hw/include/vta/hw_spec.h',
                    REPO/'3rdparty/vta-hw/include/vta/hw_spec_const.h',REPO/'3rdparty/vta-hw/hardware/xilinx/src/vta.cc',
                    REPO/'3rdparty/vta-hw/hardware/xilinx/src/vta.h',REPO/'vta/runtime/queue_capacity.h']}}
    write_new(output / "runtime_audit.json", result)
    print(json.dumps({"status":result["status"], "result":str(output / "runtime_audit.json"),
                      "board_contacted":False,"cache_alias_counterexample":cache["same_handle_same_signature_different_content_reuses_first"]},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir.resolve())
