# HANDOFF

`candidate_pool.json` records the exact 80-to-deduplicated mapping and preserves one original incumbent per workload. `results.jsonl` contains the protected originals, same-tile original controls, and every input/weight/hybrid static result.

Use `summary.json` for lower legality and target-DMA distributions. Do not rank these records by FSim time: no simulator timing was collected. Candidates that lower are eligible only for later correctness and board measurement; they are not performance wins.
