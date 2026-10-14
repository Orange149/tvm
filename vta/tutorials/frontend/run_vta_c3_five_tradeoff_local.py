#!/usr/bin/env python3
"""Fixed local worker only: never connects to a board or measures latency."""

import argparse
import ctypes
from pathlib import Path
import time
import traceback

from c3_five_tradeoff_prefetch import make_probe
from c3_residency_tradeoff import write_new
from c3_residency_tradeoff_evidence import check_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dump-commands", action="store_true",
                        help="local native FSim instruction/UOP diagnostic, no FPGA or latency")
    parser.add_argument("--board-compile", action="store_true",
                        help="cross-compile frozen candidate only; never dispatches to FPGA")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if args.dump_commands and args.board_compile:
            raise ValueError("native local dump and ARM compilation must be separate")
        write_new(args.output_dir / "runtime_identity.json", check_runtime())
        probe = make_probe()
        if args.dump_commands:
            from vta.testing import simulator
            library = simulator.LIBS[0]
            handle = library.VTATLSCommandHandle
            handle.restype = ctypes.c_void_p
            debug = library.VTASetDebugMode
            debug.argtypes = (ctypes.c_void_p, ctypes.c_int)
            debug(handle(), 6)  # Same native DumpInsn/UOP flags, never FORCE_SERIAL.
        result = probe(args.output_dir, "node24_partial_input_experimental",
                              resident_groups=12, keep_resident=True, fsim=not args.board_compile,
                              board_compile=args.board_compile,
                              batch_group_load=True, group_gemm=True)
        if not args.board_compile and result["status"] != "LOCAL_FSIM_PASSED_BOARD_NOT_QUALIFIED":
            raise ValueError("new candidate FSim failed; do not dispatch to FPGA")
        write_new(args.output_dir / "attempt.json", {
            "status": "CROSS_COMPILE_COMPLETED_NOT_FPGA_QUALIFIED" if args.board_compile else
                      "LOCAL_FSIM_PASSED_ADDITIONAL_QUEUE_PROOF_REQUIRED",
            "candidate": "N24-R12-G2-PREFETCH", "seconds": time.monotonic() - started,
            "board_contacted": False, "latency_ms": None,
            "command_diagnostic": args.dump_commands,
        })
        print("CROSS_COMPILE_COMPLETED_NOT_FPGA_QUALIFIED" if args.board_compile else
              "LOCAL_FSIM_PASSED_ADDITIONAL_QUEUE_PROOF_REQUIRED")
    except BaseException as error:
        write_new(args.output_dir / "attempt.json", {
            "status": "LOCAL_ATTEMPT_FAILED", "error": str(error),
            "traceback": traceback.format_exc(), "seconds": time.monotonic() - started,
            "board_contacted": False, "latency_ms": None,
        })
        raise


if __name__ == "__main__":
    main()
