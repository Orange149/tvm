# P7R253--P7R261：u-dma-buf 整图分配与命令容量闭环

日期：2026-09-12  
开发板 boot：`aa7a3e5c-021d-4d6b-88ae-7f696faa567c`  
结论：P7R246 的四图 OOM 已精确归因并由 allowlist 定容消除；最终 Y00 exact graph 已完成容量正例与少一页负例。

## 1. 问题

P7R246 在同一 clean-start RPC 中加载四个 YOLOv3-tiny graph executor 后，首次 VTA 调用申请
32 MiB 时失败。旧证据只能说明 192 MiB u-dma-buf 不够，不能回答 graph、参数、命令队列各占多少，
也不能证明缩小命令 backing 后程序仍正确。

本轮依次回答：

1. u-dma-buf bump allocator 的实际 requested/padding/high-water 是多少；
2. 四个 graph 变体的完整 instruction/UOP 峰值是多少；
3. 按 allowlist 定容是否能消除 P7R246 OOM；
4. 最终 selected graph 少一页时能否在越界 submission 前 fail closed。

## 2. 隔离诊断实现

P7R254 在 AXU 驱动已有只读快照 ABI 上，通过隔离 runtime 注册
`vta.runtime.udmabuf_allocation_snapshot`。它只读取 initialized、capacity、high-water、requested、
padding、allocation count 和虚实基址，不触发初始化或新分配。诊断版仅部署到
`/var/volatile/vta_c3_ram/runtime_allocdiag_v2`；默认 runtime 未覆盖，每个实验结束均重载冻结 bitstream
并恢复 `/var/volatile/vta_c3_ram/runtime`。

## 3. P7R259：OOM 精确重建

六个相互隔离的 clean-start 子会话得到：

| 状态 | u-dma-buf high-water |
|---|---:|
| 单个 graph+params | 34,300,928 B |
| 三个 graph、运行前 | 102,902,784 B |
| 四个 graph、运行前 | 137,203,712 B |
| 三个 graph、首次运行后 | 179,956,736 B |
| 四个 graph再分配首个 32 MiB 队列后 | 170,758,144 B |

每个 graph+params 恰有 38 次分配、34,299,032 B requested 和 1,896 B alignment padding。四图运行前
余 64,122,880 B；首个 32 MiB UOP 队列成功后仅余 30,568,448 B，第二个 32 MiB instruction 队列
因此差 2,985,984 B。这一水位与 P7R246 错误中的 `used=170758144` 完全一致。

四个完整图变体的队列峰值为：

| graph 变体 | instruction peak | UOP peak | submissions/推理 |
|---|---:|---:|---:|
| stock | 151,744 B | 1,384 B | 12 |
| Y00 input | 151,760 B | 1,384 B | 12 |
| Y02 weight barrier | 151,744 B | 7,508 B | 28 |
| Y00 + Y02 | 151,760 B | 7,508 B | 28 |

所以四变体 allowlist 的 4 KiB 对齐容量是 155,648 B instruction + 8,192 B UOP；最终 Y00-only
allowlist 是 155,648 B + 4,096 B。

## 4. P7R258/P7R260：容量作用域与 OOM 消除

P7R258 故意把 Y00-only 的 4 KiB UOP 容量用于四变体测量。stock 与 Y00 通过，执行到 Y02 时因
7,508 B UOP 峰值被 `queue backing capacity exceeded before submission` 拒绝。它证明容量不能从
较小 allowlist 静默迁移到较大 allowlist。

P7R260 改用四变体容量 155,648+8,192 B 后，原先失败的四 Executor 2×2 因子实验完整跑通：12/12
correctness 与 32/32 timing 均正确，八轮平衡顺序全部完成。中位时间为 stock 272.758 ms、Y00
268.461 ms、Y02 350.584 ms、both 347.737 ms；Y00 的收益和 Y02 的退化均保持，逻辑 DMA interaction
仍严格为零。因此缩容只修复空间可执行性，没有改变原性能结论。

## 5. P7R261：最终 exact graph 正负合同

实验绑定 P7R252 manifest `0ad7d0b7...11111b` 及其 exact Y00 graph binary：

- 正例容量：155,648 B instruction + 4,096 B UOP；person 与两组随机输入的八输出摘要 3/3 逐哈希
  匹配历史 P7R241，36 次 submission 的实测峰值为 151,760/1,384 B；
- 负例容量：151,552 B instruction + 4,096 B UOP；首个 9,328 B 合法 submission 已执行，下一段
  因容量比实测峰值少 208 B 而在提交前拒绝；`driver_run_calls == queue.submissions == 1`；
- 默认 32 MiB + 32 MiB 命令 backing 为 67,108,864 B，selected graph 合同为 159,744 B，减少
  99.76196%；
- 正例末端 u-dma-buf high-water 为 44,405,760 B；第二、第三组输入不再增加 high-water；
- 结束后默认 RPC 已恢复，FPGA 为 operating，存储错误列表为空。

## 6. 可写与不可写

可以写：按 exact graph allowlist 的完整命令峰值定容，能把默认命令 backing 从 64 MiB 缩到约
156 KiB；它消除了四替代图测量器的 u-dma-buf OOM，并对最终部署提供少一页 fail-closed 反证。

不能写：159,744 B 是所有 YOLO/VTA 的通用容量；allocator high-water 是物理 AXI 流量；缩容本身
提高 FPS；四 Executor 并存是正常部署必须；单 boot 静态图合同自动覆盖动态形状、replay 或其他
bitstream。

## 7. 证据入口

- `07_grouped_holdout/20260912_p7r254_udmabuf_snapshot_runtime_build_run01`
- `07_grouped_holdout/20260912_p7r259_yolov3_tiny_all_variant_queue_peaks_run01`
- `07_grouped_holdout/20260912_p7r258_yolov3_tiny_route_factorial_reduced_queue_run01`
- `07_grouped_holdout/20260912_p7r260_yolov3_tiny_route_factorial_allowlist_capacity_run02`
- `07_grouped_holdout/20260912_p7r261_yolov3_tiny_selected_graph_capacity_run01`

P7R253、P7R255 run01/run02、P7R260 run01 保留为基础设施失败或宿主执行窗口中断记录，不计入候选
正确性或性能结论。
