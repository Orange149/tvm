# C3 real-task board executor amendment

Frozen before the next board session on 2026-09-24.

## Trigger and invalid attempt

`20260924_c3_real_task_board_boot1_run01` is an invalid, non-spliceable attempt. It
completed 90 selection timing samples, froze an unusable decision file, and then
stopped before reference timing when one locally qualified program produced a
wrong FPGA result during `reference_warmup`.

The rejected identity is candidate
`b7accea8b536852ea03df90ee398407110a40e4fe9734e8eb88ed08715e8a96e`, program
`be8e922ed721ea623da3447191adc8afbdce3e00d166d663b27856a17b5b4f9f`, for
`R18-U128C128H28K3S1`. Seed 0 had 55,832 unequal output elements after one
completed 188,170-instruction device submission. The output hash matched neither
the expected result nor any previous output. This is a candidate FPGA correctness
failure, not an RPC timeout or incomplete completion.

No timing row or selection decision from run01 may be reused. Because the invalid
attempt nevertheless exposed selection latency for both tasks, subsequent runs
must be described as non-adaptive reruns of a previously frozen selector, not as
strictly latency-unseen task confirmation.

## Corrected stage order

The candidate identities, binaries, static feature values, method orders, budgets,
seeds, and latency statistics remain unchanged. Only the executor failure policy
and stage order are corrected:

1. Run three-seed, fail-fast FPGA correctness for every locally qualified
   candidate before any timing.
2. A wrong result in this qualification gate rejects only that candidate. It is
   retained in the attempted-candidate and budget accounting and is not replaced.
3. Execute selection timing only for FPGA-correct members of the already frozen
   selection union. A rejected program in a method prefix still consumes its
   additional-candidate budget.
4. Freeze selection decisions before full-pool reference warm-up or reference
   timing.
5. Uniformly warm and time only the FPGA-correct reference pool.
6. Any RPC/device interruption, boot or affinity change, or wrong output after the
   qualification gate invalidates the entire new session. Sessions are never
   spliced.

The reference oracle is therefore the fastest measured member of the frozen,
locally qualified, FPGA-correct pool. It is not an oracle over locally qualified
programs that fail real FPGA correctness.

This amendment follows the existing ResNet18 board-pool convention in
`run_vta_c3_resnet18_board_pool.py`: local lowering and FSim are pre-qualification,
FPGA correctness is the final performance-pool hard gate, and only FPGA-correct
candidates receive latency labels.
