# P8B 双向单边界串行审查

更新日期：2026-09-06

## 本地结论

- CPU->VTA 与 VTA->CPU 的 shape/dtype/device contract 均通过审计。
- 两个方向复用同一显式 u-dma-buf slot 协议，不实现双 slot 或并行 Pipeline。

## 单 Boot 结果

| 方向 | 张量 bytes | B0 copy ms | CPU mapping penalty ms | 普通物化 bytes | zero-copy bytes | latency 差值 ms | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `cpu_to_vta` | 802816 | 0.726637 | -0.253247 | 1605632 | 0 | +1.204572 | passed |
| `vta_to_cpu` | 100352 | 0.116311 | +0.005370 | 200704 | 0 | +0.846264 | passed |

- 当前 A/B 是 ordinary-copy 与 zero-copy 的同 compiled-graph 路径等价检查；独立数值 reference 继承冻结 Top-20 证据并单独记录。
- 整帧 latency 配对差值包含 stage/runtime 波动，尤其当其大于边界 copy 本身时，不得全部归因于 zero-copy。
- 聚合 P8B gate：`passed`。当前结果不允许声明双缓冲 Pipeline FPS 提升。

## 下一步

停在 P8B review。只有双向 gate 通过并经用户确认，才进入 P8C 的双 slot 状态机与 Pipeline 实验。

本地审计 SHA256：`e323d0cbfed24aaa540ffaeb96adcbc11d27f3481410f819d77c745a36c68ec3`
