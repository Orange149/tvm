# Static VTA cut-endpoint delta: units 03..15/16/17

Unit indices are zero-based. The VTA start is fixed at unit 03; only its right endpoint
moves. Static convolution DMA comes from the selected TopHub tile and lowered TIR. The
boundary column is the graph-level VTA-to-CPU tensor contract and is a separate cost.

| VTA interval | tail unit | VTA->CPU boundary | tensors | static LOAD calls | static LOAD bytes | static WGT bytes | static STORE bytes | observed LOAD bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 03..15 | `layer4_block0_main_preadd` | 301056 | 2 | 1440 | 11785216 | 7397376 | 1204224 | 11806208 |
| 03..16 | `layer4_block0_skip_proj` | 200704 | 2 | 1504 | 12002816 | 7528448 | 1229312 | 12025856 |
| 03..17 | `layer4_block0_add_relu_tail` | 100352 | 1 | 1504 | 12002816 | 7528448 | 1229312 | 12025856 |

| extension | added unit/workload | static LOAD delta | observed LOAD delta | STORE delta | VTA->CPU boundary delta | classification |
|---|---|---:|---:|---:|---:|---|
| 03..15 -> 03..16 | `layer4_block0_skip_proj`; h14_ci256_co512_k1s2 x1 | +64 calls / +217600 B | +66 calls / +219648 B | +25088 B | -100352 B | `vta_dma_vs_boundary_tradeoff` |
| 03..16 -> 03..17 | `layer4_block0_add_relu_tail`; none | +0 calls / +0 B | +0 calls / +0 B | +0 B | -100352 B | `boundary_only_reduction_no_new_conv_dma` |

## Search interpretation

- `03..15 -> 03..16` moves the layer4 projection to VTA. It adds tile-dependent
  convolution DMA while reducing the boundary by one 100352-byte live tensor. This is
  a genuine DMA-versus-shared-boundary tradeoff, not a scalar-byte dominance result.
- `03..16 -> 03..17` moves the add-ReLU tail to VTA. It adds no convolution workload
  and the observed VTA LOAD/STORE counts and bytes also remain unchanged, while the
  boundary falls by another 100352 bytes. Under memory features the longer endpoint is
  no worse; final pruning must still check CPU/VTA service and pipeline balance.
- These deltas are available before deploying each complete topology: workload DMA is
  extracted once from the incumbent tile, and boundary bytes come from the graph tensor
  contract. Runtime profile is validation evidence, not a per-candidate requirement.
