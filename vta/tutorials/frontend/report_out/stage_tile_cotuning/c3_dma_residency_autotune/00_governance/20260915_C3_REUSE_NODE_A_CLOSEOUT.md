# C3 输入—权重复用补充实验：节点 A 完成并暂停

状态：`NODE_A_COMPLETE_PAUSED; NODE_B_NOT_STARTED; NO_NEW_BOARD_RESULTS`。

本节点只读整理既有证据，未连接开发板、未重新编译或调优。已实现独立标准库聚合器
`audit_vta_c3_reuse_evidence.py`；未修改旧 runner、旧 oracle 或原始实验目录。

## 1. 正式产物与核验

正式目录：
`../07_grouped_holdout/20260915_c3_reuse_node_a_audit_run02/`。

- [summary.json](../07_grouped_holdout/20260915_c3_reuse_node_a_audit_run02/summary.json)：覆盖与配对统计；
- [reuse_pairs.csv](../07_grouped_holdout/20260915_c3_reuse_node_a_audit_run02/reuse_pairs.csv)：54条完整配对记录；
- [family_coverage.csv](../07_grouped_holdout/20260915_c3_reuse_node_a_audit_run02/family_coverage.csv)：包括缺少original的family，避免只展示完整组；
- [counterexamples.md](../07_grouped_holdout/20260915_c3_reuse_node_a_audit_run02/counterexamples.md)：四类正反案例；
- [reference_checks.json](../07_grouped_holdout/20260915_c3_reuse_node_a_audit_run02/reference_checks.json)：独立索引/区域计数；
- [input_bindings.json](../07_grouped_holdout/20260915_c3_reuse_node_a_audit_run02/input_bindings.json)：12个输入目录、319个文件的校验与链接；
- [artifact_hashes.json](../07_grouped_holdout/20260915_c3_reuse_node_a_audit_run02/artifact_hashes.json)：正式输出账本。

输入哈希在聚合前和产出前各校验一次，全部一致。包含214个保留的R18二进制的核验。
账本 SHA-256：`ddc9763fc024b6aa46b2d7bcfb7dc7c662a5234329af6f542ee66c7d92aa742d`；
summary SHA-256：`5022345a3a18391918db875594d444610c4b473603b50dc0414ffb2d47d0d2ca`。

`run01`保留为工具开发初版；正式结论采用`run02`。初版未提取部分已有命令字段，正式版已支持空值
schema alias、原始DMA字段和FSim命令peak；未覆盖任何历史文件。两版均不产生新的板端标签。

回归命令：

```text
/home/orange/miniconda3/envs/vta-resnet/bin/python -m pytest -q \
  vta/tutorials/frontend/test_audit_vta_c3_reuse_evidence.py \
  vta/tutorials/frontend/test_analyze_vta_c3_cheng_four_schemes.py
```

结果：`23 passed`。包含输入变动、回退、身份/上下文、计时归一化、缺失值与原数据端到端核验。
默认python没有pytest，改用仓库现有测试环境，未安装或修改环境依赖。

## 2. 输入复用：49个实际same-tile配对

| 几何 | 实测配对数 | 原始中位数更快的组 | 延迟改善中位数 | input bytes变化中位数 | weight bytes变化中位数 | 总DMA calls变化中位数 |
|---|---:|---:|---:|---:|---:|---:|
| R18-H1 | 15 | 10 | +2.81% | -50.00% | +50.00% | -44.44% |
| R18-H2 | 11 | 4 | -0.067% | 0.00% | 0.00% | 0.00% |
| R18-H3 | 14 | 9 | +0.49% | -62.50% | +62.50% | -49.24% |
| Y00 | 5 | 5 | +18.29% | -50.00% | 0.00% | -33.33% |
| Y03 | 4 | 4 | +41.48% | -84.38% | 0.00% | -79.41% |

“更快”仅按原始中位数比较，不是统计显著性。多个中位数独立计算，不相互代入。

H1D0580：input 2,889,728→1,444,864 B，weight 14,450,688→21,676,032 B，
总DMA 17,541,120→23,321,600 B，算子97.618→73.318 ms。这是“总bytes增加仍加速”的实际反例。
逐张量calls显示请求合并/组织也发生变化；尚不能把收益单独归因为某一个DMA请求启动成本。

因此支持“输入优先重分配输入/权重复用并改变请求组织，收益依赖tile和几何”，不支持
“输入优先一定减少总流量”或“总bytes是唯一可靠的性能排序指标”。

## 3. 权重复用：三个family、四条会话配对

| 几何/family | 会话 | 延迟改善 | weight bytes变化 | submission变化 |
|---|---|---:|---:|---:|
| Y02B00 | P7R123 | +25.28% | -92.31% | 1→17 |
| Y02B00 | P7R145 | +23.55% | -92.31% | 1→17 |
| Y03 | P7R154 | +80.61% | -99.70% | 1→5 |
| R18-H3D0103 | P7R510 | -4.08% | 0.00% | 1→2 |

Y02的两条记录是同一family的重复会话，不算两个workload。跨启动身份来自既有治理记录；这两份
RPC产物未直接提供boot_id，本节点没有重新观测或伪造boot。Y03提供另一个几何的正例，但也只是一组。

H3D0103的weight请求16→1，总DMA calls33→18，搬运字节不变；driver累计执行时间约改善39.40%，
算子延迟却恶化4.08%。与同步/执行组织代价相符，但没有纯同步控制，不写成精确的因果分解。

## 4. 组合：19 / 1 / 0必须分开

R18-H1/H2/H3的统计分别为：

| 口径 | H1 | H2 | H3 |
|---|---:|---:|---:|
| 四方案可由实测或资源回退解释 | 0 | 18 | 1 |
| 四模式均实际正确性认证并计时 | 0 | 0 | 1 |
| 四模式使用四种不同已核验二进制 | 0 | 0 | 0 |
| 可对照original确认IW同时减少input/weight bytes | 0 | 0 | 0 |

最后一行不表示所有组合候选无效：大量IW候选缺少same-tile original，无法判定双方搬运变化，
在CSV中保持`unknown`。不能拿这个0外推整个搜索空间。

H3D0103中original与input实际二进制相同，进一步说明“四种flag”不等于“四个有效不同程序”。
组合延迟恶化6.33%，没有降低input/weight bytes。R18另有孤立算子快速但图上下文错误的组合候选，
其既有正确性结论保留；本节点不新增或重算其完整图资格。

目前最迫切的缺口仍然是：四种模式都适用、original同时存在输入和权重重复加载、IW确实减少两者
的same-tile四元组。不能用含回退的19组替代该实验。

## 5. 本节点增加了哪些可核验的语义证据

- 独立枚举H1D0580每个输出tile的输入源坐标，对照裁剪padding的矩形面积，累计11,288个
  源空间位置（按tile重复计数）；乘64通道和4个输出通道外tile，精确得到原profile的2,889,728 B。
- Y02整层权重384×256×3×3=884,736 B；原始在13个空间组重复加载，得到11,501,568 B；
  驻留后一次整层加载，精确得到884,736 B。
- 旧Y02 timing缺少完整调用次数字段，正式聚合通过三seed直接正确性profile核对逻辑计数倍数，
  确认为两个完整执行；不按submission数除，也不无证据默认2。
- 已有R18/YOLO本地命令peak被正确提取并标注FSim/静态来源；不冒充本节点新增的FPGA peak测量。

这些检查只覆盖指定矩形加载和权重重复案例，不证明全部UOP消费者、halo驻留或异步覆盖安全。
完整实际SRAM分配跨度仍未知，lowering之后的空Allocate列表不能解释成“SRAM占用0”。
两条Y02会话的命令peak字段仍未从本节点输入提取，维持unknown；历史命令缩容结论不因此被改写。

## 6. 下一节点

按[补充计划](C3_INPUT_WEIGHT_REUSE_SUPPLEMENT_PLAN.md)暂停。节点B尚未开始：需要在三类可容纳
整层权重的机制测试几何上，先做本地四方案资格与实际双重复用核验，再冻结候选。
不自动连接开发板、不新增搜索器、不重复旧大池调优。
