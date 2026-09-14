# Status

Complete as a post-qualification P7Q salvage analysis. The original confirmatory P7 endpoint is not claimed because one of the 80 frozen candidates failed real-FPGA correctness.

The four valid-pool oracles are all protected original TopHub incumbents. B7/B8 retain them at gross dispatch 1 and therefore have zero measured pool regret at every budget. This demonstrates the value of incumbent protection, but not an added request-signature benefit: B6 did not improve over bytes-only B5, and B8 did not improve over B7. Consequently frozen Gate G7 is **NO-GO**, and P8 stage/FPS propagation is not authorized from this result.

The secondary same-ConfigEntity analysis remains useful: 42/56 residency variants were faster than their original schedule control, with a median 3.31% improvement and 22/56 at least 5% faster. Thus residency has a real local mechanism, but the current candidate generator does not combine it with the strongest compute schedule. In particular, it forces experimental `oc_nthread=h_nthread=1`, while all four pool oracles use `oc_nthread=2`.
