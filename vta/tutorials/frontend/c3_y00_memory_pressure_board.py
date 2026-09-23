"""Board adapter for the Y00 same-Tile memory-pressure residency experiment."""

import c3_vthread_residency_board as common
import c3_y00_memory_pressure_protocol as protocol


SCHEMA = "c3_y00_memory_pressure_residency_v1"


def run(args, out):
    common.protocol = protocol
    common.SCHEMA = SCHEMA
    common.run(args, out)
