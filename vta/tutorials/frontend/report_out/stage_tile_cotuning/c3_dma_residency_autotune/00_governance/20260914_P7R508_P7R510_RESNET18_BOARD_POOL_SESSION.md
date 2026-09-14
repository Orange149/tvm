# P7R508--P7R510：ResNet18 文献对齐完整 FPGA 池

日期：2026-09-14  
开发板 boot：`bf5bc78c-115b-40e0-8853-6684f04ecf3d`

## 身份恢复与无标签冻结

串口给出的 Dropbear RSA 公钥与网络侧主机密钥完整一致，SHA-256 为
`SHA256:u7LMrzTebC0p8iNcXkRoTweS6DEqa0EYmM0uhFpxLX0`。P7R508 在读取任何 R18 目标标签前绑定
严格 `known_hosts`、公钥优先认证和 board-control 源码；P7R509 将冻结哈希匹配的 bitstream、
u-dma-buf 模块与 runtime 只恢复到易失内存。恢复后 FPGA=`operating`，u-dma-buf=201326592 B，
物理基址 `0x68500000`，RPC cwd 为 `/var/volatile/vta_c3_ram/runtime`。没有向 SD 写实验产物。

## 完整池结果

P7R510 对 P7R494 冻结的 214 个身份和 P7R496 的 214 个 ARM 二进制执行一个不可拼接 clean-start
session。候选顺序、三 correctness seeds 与五轮计时次数均未修改。

| 几何 | 候选 | FPGA-correct | FPGA-invalid | pool oracle |
|---|---:|---:|---:|---:|
| R18-H1 | 104 | 104 | 0 | 5.945580 ms |
| R18-H2 | 44 | 43 | 1 | 5.156932 ms |
| R18-H3 | 66 | 60 | 6 | 0.257132 ms |
| 合计 | 214 | 207 | 7 | 按 workload 分开定义 |

七个错误身份均在 seed 0 得到 `wrong_answer` 并首错即停；因此实际 correctness invocation 为 628，
比无 fail-fast 的 642 少 14 次。错误按模式分为 original 2、weight-resident-barrier 4、combined 1；
没有 input-stationary 错误。该分类只适用于当前精确身份，不外推到整个模式。

全部 207 个正确候选各完成五轮平衡计时，共 1035 个样本。H1/H2 oracle 都是 original，H3 oracle
是 input+weight combined residency；这已经提示 minimum-access/驻留不可能被预设为每个几何都
最快，但正式比较必须等 P7R511--P7R513 的只读分析。

## 完整墙钟与运行资源

| 指标 | 实测 |
|---|---:|
| W0→T0 | 7.874622 s |
| T0→T1 | 274.737200 s |
| W0→T1 | 282.611822 s |
| correctness host-call wall | 89.185273 s |
| timing host-call wall | 182.982733 s |
| 逻辑 LOAD bytes | 21,617,020,928 |
| 逻辑 STORE bytes | 319,169,536 |
| 逻辑 DMA calls | 6,562,805 |
| FPGA kernel invocations | 4,937 |

上述 byte/call 是 runtime 的逻辑 VTA LOAD/STORE 计数，不是物理 AXI burst。全部 11 个板池 artifact
重算哈希无差异，`status=complete_non_spliced_board_pool`，不存在 `invalid_session.json`。实验后
boot 未变化，FPGA、192 MiB u-dma-buf 与 tmpfs RPC 均正常，也没有新的 EXT4/MMC 错误。

## 结论边界与下一步

本节点关闭的是“R18 三几何没有完整真实 outcome pool”的缺口，并再次证明 lowering/FSim 正确不
等于真实 FPGA 正确。它尚未给出 Random、stock-XGB、HW-Aware、ML²Tuner、Cheng 与本文方法的
time-to-target 排名，也没有 ResNet18 整图 latency/FPS。

按暂停协议，下一节点先把 P7R510 单向适配为 outcome ledger，再执行六策略 20-seed 等预算分析和
Cheng 四方案汇总；不得再修改候选身份、排序规则或 oracle 定义。
