# P8C 双 Slot Pipeline 审查

更新日期：2026-09-06

## 本地实现

- `BoundarySlotManager` 管理 FREE -> PRODUCER_WRITING -> READY -> CONSUMER_USING -> FREE。
- token 同时携带 edge、slot、generation 和 frame owner；等待有超时，worker 错误会中止全部 manager。
- B1 为单 slot 串行 zero-copy；B2 为每条边双 slot 的三 stage 并行 Pipeline；单 VTA mutex 保留。
- 本地 gate：`passed`。

## 最新 Boot 详情

| 模式 | latency 中位数 (ms) | Pipeline II (ms) | FPS | 框架边界物化 (B/frame) | slot wait P95 (ms) |
|---|---:|---:|---:|---:|---:|
| B0 | 557.624770 | 90.452821 | 11.055 | 1806336 | 0.000000 |
| B1 | 186.931929 | 188.162213 | 5.315 | 0 | 0.005852 |
| B2 | 522.171782 | 91.713839 | 10.903 | 0 | 32.137158 |

- `B0-B2` Pipeline II 差值为 `-1.261018 ms`，`B2-B0` FPS 差值为 `-0.152`。
- 相对 FPS 差值为 `-1.37%`；两个完整测量窗口的平均 II 分别为 B0 `91.323/89.583`、B2 `92.263/91.165` ms。
- 吞吐采用 `(窗口末完成时间-窗口首完成时间)/(N-1)`，再对平衡 block 取中位数；逐帧间隔中位数仅用于观察调度抖动。
- B0 的 VTA-facing host copy 为 `802816 + 100352 bytes/frame`，B2 为 0；VTA 内部 `LOAD=12025856`、`STORE=1229312 bytes/frame` 未改变。
- 最新 boot 的 VTA->CPU producer slot wait 中位数为 `0.001 ms`、P95 为 `25.460 ms`；该等待是可与其他 stage 重叠的背压，不能作为额外时延再次相加，也不能仅凭单个 boot 判定固定瓶颈。
- 正确性、generation/owner、双 slot 使用、终态释放和 preflight 聚合 gate：`passed`。
- 本节只展示最新一个 boot；正式判断使用下方跨 boot 汇总，不能用帧内样本代替独立 boot。

## 跨 Boot 汇总

| Boot | B0 II (ms) | B2 II (ms) | II 减少 (ms) | 相对 FPS 增量 | 边界 API 时间减少 | 功能 gate |
|---|---:|---:|---:|---:|---:|---|
| `823cf8e1-bf33-4e4f-81af-6f3090d34e69` | 90.712107 | 91.392066 | -0.679958 | -0.74% | +97.11% | passed |
| `c7654876-547f-49f8-9356-323f90a9e48a` | 92.013213 | 91.743051 | +0.270163 | +0.29% | +97.17% | passed |
| `efe6fd7c-a4db-4023-a351-421978120af0` | 90.452821 | 91.713839 | -1.261018 | -1.37% | +97.04% | passed |

- 已完成 `3` 个独立 boot；II 减少均值 `-0.556938 ms`，相对 FPS 增量均值 `-0.61%`。
- 两条异构边界的框架 API 服务时间由 B0 平均 `1.446899 ms/frame` 降至 B2 `0.041830 ms/frame`，减少 `1.405069 ms`（`97.11%`）；物化字节由 `1806336` 降至 `0 B/frame`。
- VTA->CPU producer slot wait 的跨 boot 中位数范围为 `0.001--18.541 ms`，说明背压存在但强度不稳定，暂不据此增加固定 penalty 或第三个 slot。
- 当前正式性能 claim：`not allowed`。统计独立单位是 boot，不是帧或同 boot block。

## 下一步

三个 boot 已完成但区间 gate 未通过；只保留字节/组件时间结论，决定是否补 boot 后再进入其他阶段。

本地审计 SHA256：`c867d6b7afb0e618770304c74abc63f6e6cb7977101768d9b862766ca94f1a4c`
