"""Board adapter for the T8 eager/staged full-residency development experiment."""

import c3_vthread_residency_board as common
import c3_vthread_staged_protocol as protocol


SCHEMA = "c3_vthread_staged_residency_v1"


def run(args, out):
    common.protocol = protocol
    common.SCHEMA = SCHEMA
    common.run(args, out)
