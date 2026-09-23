"""Board adapter for the Y00F06 input plus staged-weight experiment."""

import c3_vthread_residency_board as common
import c3_y00_input_weight_protocol as protocol


SCHEMA = "c3_y00_input_weight_staged_v1"


def run(args, out):
    common.protocol = protocol
    common.SCHEMA = SCHEMA
    common.run(args, out)
