"""T7 identity specialization of the three-variant vthread residency protocol."""

from c3_vthread_residency_protocol import (  # noqa: F401
    QUEUE_ENV,
    SEEDS,
    SEQUENCES,
    VARIANTS,
    balance,
    candidate_prefix,
    development_classification,
    logical_signature,
    paired,
    schedule,
    validate_timing,
)


COMPARISONS = (
    ("T7_O2", "T7_VR", "new_framework_effect"),
    ("T7_VR", "T7_VS", "residency_program_effect"),
    ("T7_O2", "T7_VS", "net_vs_o2"),
)

