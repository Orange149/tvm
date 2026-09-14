# P7R444：Y10 最终入选整网命令缓冲定容与不足一页拒绝

## 结论

P7R444 对 Y10 完整池最终 oracle——Y10F02 input-stationary（`dc939aaf...`）——完成了精确整网
命令容量闭环。实验绑定 P7R435 中实际执行过的 Graph JSON、参数和交叉动态库，并绑定 P7R441
完整池 oracle；不重新选择候选，也不测 latency。

诊断 runtime 在三个确定性 320×320 输入上观察到 36 次设备提交，最大单次提交为：

| 队列 | 实测峰值 | 4 KiB 页对齐容量 |
|---|---:|---:|
| instruction | 1,065,280 B | 1,069,056 B |
| UOP | 1,320 B | 4,096 B |
| 合计 backing | — | 1,073,152 B |

默认 runtime 为 instruction 32 MiB + UOP 32 MiB，共 67,108,864 B。对这一精确静态图，页对齐
定容将命令 backing 减少 98.4009%。它表示减少 u-dma-buf 中的命令队列预留，不表示释放整块
192 MiB，也不表示物理 AXI 流量下降。

## 正向与负向验证

- 正向：以 1,069,056 B instruction + 4,096 B UOP 重启诊断 RPC；三个输入的全部 8 个图输出均
  与峰值运行逐字节一致，36 次提交完成，复测峰值仍为 1,065,280/1,320 B。
- 负向：instruction 再减少一页至 1,064,960 B，UOP 保持 4,096 B；另外补回一页布局 padding，
  避免把地址布局变化混入容量变量。前 5 次合法提交完成，下一条超限命令以
  `queue backing capacity exceeded before submission` 被拒绝；runtime 的 `driver_run_calls=5`
  与队列 `submissions=5` 相等，证明违规提交没有送入 FPGA。
- 恢复：重新加载冻结 bitstream 并恢复默认 tmpfs RPC；同一 seed 的完整图 8 输出再次精确匹配。
  实验前后 boot 均为 `4d232b6f-6e2a-4395-aafe-853a021403e8`，FPGA 为 operating，未出现新的
  EXT4/mmc 错误。

## 与旧容量结论的关系

P7R261 的 155,648+4,096 B 只适用于旧 YOLO-416 Y00 selected graph，不能迁移给 Y10。P7R444
实际测得 Y10 instruction 峰值为 1,065,280 B，约为旧峰值的 7.02 倍。这正好证明本文的部署观点：
命令容量必须由最终 exact allowlist/图/调度重新推导，不能使用一个“VTA 通用小容量”。

## 证据边界

该实验只支持：精确静态图、当前 bitstream/runtime/二进制身份下的逻辑命令队列峰值、正向正确性
与不足容量 fail-closed。它不支持动态 shape、任意候选、模型精度、物理 AXI 或性能提升主张。

原始产物：
`07_grouped_holdout/20260913_p7r444_y10_selected_fullgraph_capacity_run01/`。
