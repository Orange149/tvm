# P1 来源账本

检索日期：2026-09-10；全文补充日期：2026-09-12。以下只记录用户提供的合法全文、公开入口及本地源码。

| ID | 来源 | 类型 | 可验证内容 | 不可验证内容 | 状态 |
|---|---|---|---|---|---|
| S0 | `/home/orange/code/tvm/Design of data access and schedule optimization for VTA compiled.pdf`，SHA-256 `9fdbe490db46d319fb992ff016f99c0f7bd64718f079f84749ff21d3c8839a02` | 用户提供的正式论文全文，9 页 | Algorithm 1/2、Fig. 1--3、Table 1/2、四方案、TIR/runtime 修改点、资源条件、平台与模型结果 | 作者源码、TVM commit、自动调优器实现、重复次数/方差 | 已完整读取并成为方法事实的首要来源 |
| S1 | [ScienceDirect 正式出版页](https://www.sciencedirect.com/science/article/abs/pii/S0167739X25004595) | 出版者一手来源 | 标题、作者、FGCS 176、March 2026、108165、正式 DOI；3 条 highlights；摘要；引言公开片段；Section 3.1 权重覆盖问题概述；YOLOv3 约 10% | 全文页码、图表、算法、实现和完整实验设置 | 部分可访问；全文 403 |
| S2 | [SSRN 预印本记录](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5157285) | 作者提交的预印本元数据 | 标题、作者、13 页、posted 2025-02-26、DOI `10.2139/ssrn.5157285`、摘要、license notice | PDF 正文 | 元数据可访问；页面/PDF 403 |
| S3 | [SSRN DOI](https://doi.org/10.2139/ssrn.5157285) | DOI 标识 | 预印本身份 | 正文 | 可解析到 S2，但正文未取到 |
| S4 | [FGCS DOI](https://doi.org/10.1016/j.future.2025.108165) | DOI 标识 | 正式版本身份和 2025 DOI 字符串 | 正文 | 可解析到 S1 |
| S5 | [ResearchGate 预印本记录](https://www.researchgate.net/publication/389365717_Design_of_Data_Access_and_Schedule_Optimization_for_Vta_Compiled_Instruction_Streams) | 聚合元数据 | 2025 preprint 记录；页面明确 no file available/request PDF | 正文 | 仅作为缺失全文的旁证 |
| S6 | [OUCI 元数据](https://ouci.dntb.gov.ua/works/lxLvqny8/) | 聚合索引 | 正式 DOI、卷、作者、参考文献数量 | 方法正文 | 元数据交叉核对 |
| L1 | `vta/python/vta/top/vta_conv2d.py` | 本地一手源码 | 当前 compute、loop reorder、cache scope/lifetime、AutoTVM knobs、DMA/tensorize pragma | B2026 实现 | P0 哈希冻结，仅读取 |
| L2 | `C3_DMA_RESIDENCY_AUTOTUNING_MASTER_PLAN.md` | 本地治理 | P1 边界、命名和 Gate | 论文事实 | 仅读取 |
| L3 | `00_governance/*` | 本地治理 | frozen platform、claim、预算和降级规则 | 论文事实 | 仅读取 |
| S7 | [TVM OSDI 2018](https://www.usenix.org/conference/osdi18/presentation/chen) | 论文一手来源 | 模板化搜索、学习型 cost model、硬件特征驱动优化 | 本文 VTA 证书实现 | 已核对 |
| S8 | [Ansor OSDI 2020](https://www.usenix.org/system/files/osdi20-zheng.pdf) | 论文一手来源 | sketch、进化搜索与学习型排序 | VTA FPGA 正确性 | 已核对 |
| S9 | [MetaSchedule RFC 0005](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md) | Apache TVM 一手设计文档 | schedule rule、postprocessor、测量数据库 | VTA 专用 verifier | 已核对 |
| S10 | [Timeloop ISPASS 2019](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation) | 论文/作者机构页 | 架构约束、mapspace、数据流与存储层次协同 | TVM 编译结果与板端证书 | 已核对 |
| S11 | [MAESTRO](https://arxiv.org/abs/1805.02566) | 作者论文 | 数据中心式复用、occupancy、性能/能耗 Pareto | VTA runtime/FPGA correctness | 已核对 |
| S12 | [VTA blueprint](https://arxiv.org/abs/1807.04188) | VTA 原论文 | 参数化硬件、任务/微指令 ISA、JIT 与 operator autotuning | 本文驻留证书 | 已核对 |
| L4 | `src/meta_schedule/utils.h`、`src/meta_schedule/postproc/*` | 本地一手源码 | postprocessor 失败在测量前拒绝 trace；GPU/VTCM verifier 先例 | VTA verifier | 已核对 |
| L5 | `python/tvm/autotvm/tuner/model_based_tuner.py` | 本地一手源码 | 失败测量以零吞吐更新模型，说明只在 runner 检错不等于不污染模型 | 本文 allowlist 效果 | 已核对 |
| L6 | `src/tir/transforms/inject_virtual_thread.cc` | 本地一手源码 | touched/written allocation 按 virtual-thread 数扩展并重写索引 | 具体 FPGA 正确性 | 已核对 |

## 证据引用规则

- 方法和实验事实优先引用 S0；公开时间线可同时引用 S1/S2。
- S5/S6 只用于交叉核对元数据，不能用来填补方法细节。
- 代码事实引用 L1 的行号；不能把 L1 的实现反向归因给 B2026。
- 现在可以引用正文页、Figure、Table 和 Algorithm；仍不可臆造作者源码、commit、未报告搜索算法或统计量。
- 搜索结果中的摘要性文字仅可转述，不大段复制；本目录未保存或再分发论文 PDF。

## 检索式

```text
"Design of Data Access and Schedule Optimization for VTA Compiled Instruction Streams" PDF
"10.1016/j.future.2025.108165" PDF
"10.2139/ssrn.5157285"
"input prioritized schedule" "on-chip weight" VTA
"minimum data access" VTA
title + Algorithm / Figure / GitHub / author names
```

检索结果没有发现可审计的作者代码或补充材料。
