# P7R155--P7R158 exact allowlist 命令容量会话

## 完成结果

- 当前 boot：`aa7a3e5c-021d-4d6b-88ae-7f696faa567c`。
- 最终 exact allowlist：Y00 selected、Y00 same-tile original fallback、Y03 selected，共 3 个身份。
- 静态最大峰值：instruction `13,488 B`，UOP `1,364 B`。
- 部署容量：按 4 KiB 页对齐为 instruction `16,384 B`、UOP `4,096 B`。
- P7R155 正向认证：3 身份 × 3 seed = 9/9 正确；队列观测峰值与静态最大峰值一致，9 次提交。
- 负向认证：instruction 缩小一页至 `12,288 B` 后，13,488 B 身份在设备提交前拒绝；
  `submissions=0`、`driver_run_calls=0`、instruction/UOP observed peak 均为 0。
- 普通 tmpfs RPC 已恢复；FPGA 为 `operating`；当前 boot 未见新增 EXT4/mmc 错误。

主要正向证据：
`07_grouped_holdout/20260912_p7r155_selected_allowlist_capacity_run02`。

最终隔离诊断：
`07_grouped_holdout/20260912_p7r158_bitstream_isolated_capacity_run01`。
P7R158 的 artifact ledger 已逐项重算通过，共 16 个文件。

## 新发现的安全边界

P7R158 使用相同 Y00 selected 身份和相同二进制做 bitstream 隔离的 A/B/A：

| 阶段 | instruction/UOP 容量 | 额外布局补偿 | 结果 |
|---|---:|---:|---|
| A1（重载 bitstream 后） | 16 KiB / 4 KiB | 0 | 正确 |
| B control | 12 KiB / 4 KiB | 4 KiB | 错误 684,302 个元素 |
| A2（不重载） | 16 KiB / 4 KiB | 0 | 错误 684,101 个元素 |
| A3（再次重载 bitstream 后） | 16 KiB / 4 KiB | 0 | 正确 |

B control 实际 instruction 峰值只有 7,648 B、UOP 峰值 460 B，均小于设置容量，且确实完成了一次
设备提交。因此容量检查本身工作正常，但“可容纳命令”不等于“缩容部署保持数值正确”。4 KiB 哑元
VTA 分配用于补偿队列 backing 缩小造成的预期 bump-allocator 偏移；它仍未消除错误，说明当前证据
不能把问题简化为一个张量起始地址平移。B 后的 A2 仍错、重载后的 A3 恢复，表明还存在 bitstream
运行状态耦合。

正确论文口径是：

> 静态 command peak 是安全定容的必要条件，不是充分条件；容量合同还必须绑定 exact allowlist、
> 物理布局、bitstream/runtime 身份，并在最终容量下通过真实 FPGA 多 seed 数值认证。本文最终选择
> 已通过 9/9 认证的 16 KiB/4 KiB，而不采用虽能容纳部分候选但出现状态耦合的 12 KiB/4 KiB。

当前不能声称已经定位为 u-dma-buf 驱动错误，也不能用 W05 健康门代替目标 workload 的正确性门；
W05 在异常前后均为 3/3，只说明板卡和普通 RPC 基本可用。
