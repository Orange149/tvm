"""A coarse, scope-corrected isolated CPU share estimator, not a new profile.

Noncaller session counters are used only under an explicit quiet-between-run
assumption. The caller is the remainder of measured run CPU, so serial output
statistics cannot inflate its share. Coarse counter noise and runtime work
outside run remain limitations; negative residuals are rejected, never clipped.
"""
from cpu_policies.open_resources.costs import quantize_ns


def reconstruct(rows,noncaller_session_ms):
    if not rows or not noncaller_session_ms:raise ValueError('isolated multi-thread run counters required')
    run=sum(quantize_ns(row['stage0_run_process_cpu_ms']) for row in rows)
    noncaller=[quantize_ns(value) for value in noncaller_session_ms]
    caller=run-sum(noncaller)
    if run<=0 or caller<0:raise ValueError('run remainder incompatible with coarse session counters')
    vector=[caller,*noncaller]
    if sum(vector)!=run:raise ValueError('reconstructed run CPU not conserved')
    return dict(total_run_CPU_ns=run,estimated_thread_CPU_ns=vector,
        weights=[value/run for value in vector],
        noncaller_outside_run_CPU_assumed_negligible=True,
        coarse_session_counter_noise_not_removed=True,
        newly_measured_per_thread_profile=False,
        weights_estimated_from_original_isolated_receipt=True)
