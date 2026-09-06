# P8A ABI 与共享内存可行性审查

更新日期：2026-09-06

## 本地结论

- GraphExecutor 的 input/output zero-copy 入口、runner 的 shape/dtype/offset/alignment/物理地址检查均已审计。
- 冻结边界为 `[1,64,56,56] float32`，共 `802816 bytes`；producer 为 CPU view，consumer 为 VTA ext-dev view。
- P8A 只验证单 slot 串行同址绑定，不代表双 slot Pipeline 已实现。

## 单 Boot 结果

- 路径等价性：两种普通 copy control 与 zero-copy 的 A/B 逐帧输出均一致；这是同一 compiled graph 的 ordinary-copy 对照，不冒充新的独立 CPU reference。
- 测量顺序：每帧组成 ordinary/zero-copy matched pair，并平衡 AB/BA 顺序；早期 blocked-order 结果因漂移已废弃。
- 最快正确普通路径：`memcpy`，边界 materialization 中位时间 `0.714217 ms`。
- 同址路径将 framework stage-boundary materialization 从 `1605632` 降为 `0 bytes`；VTA 内部 DDR LOAD 不在此范围。
- CPU 直接写 u-dma-buf 的 producer run 配对差值为 `-0.251582 ms`；`baseline - zero-copy` 单帧串行 latency 配对差值为 `+1.157847 ms`。
- 单 boot gate：`passed`。该结果只能决定是否进入 P8B，不能声明流水线 FPS 提升。
- 一致性证据边界：u-dma-buf `dma_coherent=0`，该 allocator 元数据不能为 PL 的 HPC 路径背书；VTA 路径依据冻结的 HPC bitstream/runtime hash、`VTA_COHERENT_ACCESSES=true` 和既有 HP/HPC 受控实验。

## 下一步

停在 P8A review。只有本次 gate 通过并经用户确认，才进入 P8B 的单边界串行 matched control；不自动实现双 slot Pipeline。

本地审计 SHA256：`afc1f3818db3f9047048e1d363286b46ac8f4add87e00deae20e36b94cfe5a98`
