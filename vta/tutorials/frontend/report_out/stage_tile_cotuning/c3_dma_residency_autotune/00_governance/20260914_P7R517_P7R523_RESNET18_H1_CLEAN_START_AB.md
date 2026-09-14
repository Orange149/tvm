# P7R517--P7R523：ResNet18 H1 三对三从零调优与整图验证

状态：`COMPLETE; PAUSED_BEFORE_P7R524_FINAL_FIGURES`

## 1. 实验口径

在同一开发板 boot `bf5bc78c-115b-40e0-8853-6684f04ecf3d` 上，按预注册顺序串行执行：

```text
P7R517  empty-history AutoTVM-XGB, seed 67001
P7R518  ours, valid run04
P7R519  ours, valid run01
P7R520  empty-history AutoTVM-XGB, seed 67002
P7R521  empty-history AutoTVM-XGB, seed 67003
P7R522  ours, valid run01
```

两侧均在 W0 后核对 runtime/RPC 身份、重载同一 bitstream、启动新 RPC，再以 T0 为共同起点；
T1 是选中配置进入完整 ResNet18，并完成三个确定性输入的全部输出等价检查和七轮交错计时。
XGB 使用空历史 original ConfigSpace、`trials=300` 和600 s搜索预算；本文方法不读取 P7R510/P7R512
性能标签，在线生成、lowering/FSim、交叉编译并派发13个候选。两侧候选空间不同，因此这是
“从零到可部署质量”的系统流程比较，不是同空间算法消融。

## 2. 三对三主结果

| 指标 | AutoTVM-XGB | 本文方法 | 结果 |
|---|---:|---:|---:|
| T0→T1，中位数 | 668.459 s | 309.699 s | -53.67%，约2.16× |
| T0→T1，范围 | 668.165--669.101 s | 301.173--326.265 s | 本文更短，但波动更大 |
| W0→T1，中位数 | 676.359 s | 317.694 s | -53.03% |
| gross/FPGA candidate，中位数 | 240 | 13 | -94.58% |
| 最终 ResNet18 latency，中位数 | 116.912 ms | 117.902 ms | 本文慢0.847% |
| 最终 FPS（由中位 latency 换算） | 8.553 | 8.482 | 本文低0.840% |

六次完整图均通过三个输入的全部1000维输出逐元素相等检查，且每次选中图都在其同次 stock
reference 的2%范围内。由此可以主张：在 R18-H1 上，本文系统流程把从 T0 到“正确且通过2%非劣
线的完整图”所需中位墙钟降低53.67%，代价是最终整图中位 latency 比 XGB 高0.847%。不能写成
本文最终图更快，也不能把2%非劣改写成 FPS 提升。

XGB 三次运行依次考虑231/286/240个候选，其中成功孤立测量93/89/90个；无效比例为
59.74%/68.88%/62.50%。本文三次都派发13个经本地资格的候选，13/13通过板端正确性；每次算子
搜索记录273次 kernel invocation、397,456,384 B逻辑LOAD、33,918,976 B逻辑STORE和540,696次
逻辑DMA调用。XGB runner没有为每个候选保存同口径 runtime DMA profile，因此不能声称两侧搜索期
逻辑DMA总量下降，只能比较可共同审计的候选数与墙钟。

## 3. best-so-far 与 oracle 边界

六个完成会话中观察到的最小孤立算子延迟为5.871189 ms。它只是“六次运行的跨空间 observed
minimum”，不是完整 ConfigSpace oracle，也没有反馈给任一在线策略。XGB 三个 seed 首次进入该
observed minimum +2% 的位置分别为 trial 72/52/65，对应 T0 77.613/43.353/59.783 s；本文首点约
6.19--6.21 ms，三次均未进入这条算子级2%带。这与完整图结果并不矛盾：H1只是整图中的一个算子，
两侧最终完整图仍全部通过相对同次 stock 的2%部署门。

因此本节点给出的核心量不是“本文更快找到全空间算子 oracle”，而是“本文以更小的、包含驻留
语义的候选流程更早完成最终图正确性与质量门”。P7R523 保存了 best-so-far 对 candidate position、
T0墙钟，以及 T0成本/最终整图 latency 三张只读 SVG。

## 4. 失败会话与工程成本

P7R518 run01--run03 因 RPC server 的持久父进程/派生会话对象生命周期问题全部 fail closed，未与
有效 run04 拼接，也未进入3×3中位数。三次失败仍分别支付 T0 181.483/185.823/181.346 s，合计
548.652 s；W0合计572.559 s。它们证明完整墙钟记录确实覆盖候选生成、资格、交叉编译、RPC恢复和
失败，而不是只给成功计时。修正后 P7R518 run04、P7R519、P7R522 均完成13/13正确性且运行后只留
一个 PPid=1 RPC父进程。

这些失败属于实现成熟过程的额外工程成本，不应摊入冻结方法的三次有效性能中位数；写论文时可在
可复现性/失败分析中单列，不能隐藏，也不能把它们归因于搜索算法本身。

## 5. 完整性与下一节点

- P7R523 ledger：`99172237569bd269b66f25713b489abbb3aa38b9fb642d95656431479fd8c206`
- 六个有效 run 与三个失败 run 的 artifact ledger 全部逐文件复核通过；没有 session 拼接。
- P7R523 汇总目录包含 `summary.json`、`results.jsonl`、`contract.json`、`timeline.jsonl`、三张SVG、
  `command.txt` 与 `artifact_hashes.json`。
- 最终图随机输入全等不是 ImageNet accuracy；逻辑 VTA DMA 不是物理 AXI burst。

本节点到此暂停。下一节点 P7R524 只允许从不可变 summary/ledger 生成四张论文总图并同步最终文档，
不得重新上板、修改候选顺序或把 P7R516G/P7R510 标签反馈进已完成搜索。
