# YOLOv3-tiny CPU/VTA/CPU Balance Comparison

| candidate | fps | stage0 ms | stage1 VTA ms | stage2 ms | CPU sum ms | detection gate | note |
|---|---:|---:|---:|---:|---:|---|---|
| `pool2_to_dual_pre_logits` | 4.8715 | 205.8 | 161.0 | 82.3 | 288.2 | True | best throughput |
| `pool1_to_dual_pre_logits` | 4.7758 | 148.3 | 210.3 | 84.4 | 232.7 | True | best CPU/VTA resource balance |
| `pool1_to_dual_pre18_14` | 3.2233 | 184.9 | 185.0 | 305.2 | 490.1 | True | CPU tail too heavy |

## Conclusion

`pool1_to_dual_pre_logits` is the better balanced split when treating CPU stage0+stage2 as the shared CPU-side resource: CPU sum is 232.7 ms and VTA is 210.3 ms. It keeps detection correctness and reaches 4.7758 fps. The earlier head cut `pool1_to_dual_pre18_14` balances stage0 and VTA individually, but moves too much work to CPU tail and drops to 3.2233 fps. The original `pool2_to_dual_pre_logits` still has the best measured throughput at 4.8715 fps, but it is less balanced because CPU sum is 288.2 ms versus VTA 161.0 ms.
