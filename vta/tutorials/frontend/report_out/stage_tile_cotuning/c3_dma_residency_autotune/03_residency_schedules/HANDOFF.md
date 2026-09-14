# P3-local handoff

## Changed implementation surface

- `vta/python/vta/top/vta_conv2d.py`: original schedule extracted into an internal implementation; official registration remains mode 0.
- `vta/python/vta/top/vta_conv2d_residency.py`: isolated experimental AutoTVM template with explicit modes 0--3.
- `vta/tutorials/frontend/test_vta_residency_schedule.py`: frozen-original, static-DMA, and FSim-reference checks.

Do not merge these modes into the production tuning task yet.  Do not describe `paper_inspired_hybrid` as an exact 2026-paper reproduction.

## Re-run

From the repository root:

```sh
VTA_HW_PATH=/tmp/vta-hw-fsim /home/orange/miniconda3/envs/vta-resnet/bin/python -m pytest vta/tutorials/frontend/test_vta_residency_schedule.py -k 'not match_reference_in_fsim' -q
VTA_HW_PATH=/tmp/vta-hw-fsim timeout 30s /home/orange/miniconda3/envs/vta-resnet/bin/python -m pytest vta/tutorials/frontend/test_vta_residency_schedule.py -k match_reference_in_fsim -q
```

## Next allowed step

Treat this run as G3 NO-GO.  A successor may investigate a dependency-safe weight lifetime only in a new run directory and must first satisfy all of: VTA dependency legality, compact/2D DMA legality, SRAM capacity, local FSim correctness, and a measured weight-LOAD reduction.  The failed approach must not be silently restored.
