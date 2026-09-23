"""T7 specialization of the vthread residency development analysis."""

import c3_vthread_residency_results as base
import c3_vthread_residency_t7_protocol as protocol


SCHEMA = "c3_vthread_weight_residency_t7_v1"


def analyze(args, out):
    previous_schema, previous_protocol = base.SCHEMA, base.protocol
    base.SCHEMA, base.protocol = SCHEMA, protocol
    try:
        return base.analyze(args, out)
    finally:
        base.SCHEMA, base.protocol = previous_schema, previous_protocol

