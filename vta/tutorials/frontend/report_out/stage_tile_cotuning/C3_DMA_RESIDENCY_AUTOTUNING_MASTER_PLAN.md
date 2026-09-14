# C3：VTA 驻留--请求--命令协同调优主控执行计划

状态：`P7R_SEMANTIC_DISPATCH_AND_COMMAND_RESOURCE_CERTIFICATE_INTEGRATED`  
制定日期：2026-09-09  
最近更新：2026-09-11  
适用论文：《面向深度神经网络的嵌入式异构计算共享内存技术研究》  
预计周期：18--24 个有效工作日；若只剩 10--12 日，执行至 P7 MVP，不启动完整 P8/P9。  
执行方式：后续由主代理按阶段串行派发 subagent；每个阶段通过 gate 后才能进入下一阶段。

当前离线进度以
`c3_dma_residency_autotune/00_governance/progress.json` 为准：P2 本地原路径回归已通过；
P3 已验证 input-stationary，并通过显式 full-sync 与编译器依赖分段使代表 workload 的
weight-residency 正确执行；P4/P4e/P4f 已完成稳定身份、250 候选静态池、165 个 lower-success
候选的三 seed FSim 与 AXU5EVB 交叉编译资格；P4g 将显式屏障模式扩展到 60 个 tile，32 个
合法候选完成三 seed FSim，P4i 又完成 32/32 AXU5EVB 交叉编译资格；P4h 先取得 6 个代表候选的 FSim command dry-run，P4j 再覆盖全部 197 个本地合格候选，量化少 LOAD 与
多同步/多提交/累计 UOP 增长的权衡。P5b 已冻结无标签 B4--B7 shortlist，P5c 又把合规命令画像和 32 个 mode-4 候选纳入 197 项全池 command-aware shortlist；P6a 冻结通用
B7 探索合同，P6b 冻结 W00/W02/W09 各 original/input/barrier 的 9 项机制 canary 合同。
P5 历史数据仍不足以用旧标签无泄漏评价新顺序，完整 G2、G4--G6
仍需开发板上的 runtime DMA、正确性与 latency 标签。

> 2026-09-11 执行更新：开发板已恢复并完成 P6/P7Q。P7 冻结池 79/80 候选通过真实 FPGA 正确性，79 个合法候选取得 395 个完整区组计时样本。四个 pool oracle 均为 TopHub；B6 full-request 未优于 B5 bytes-only，且驻留池沿用了 `oc_nthread=h_nthread=1` 的机制隔离限制，未覆盖四个 `oc_nthread=2` incumbent 的驻留变体。按本计划 G7 与停止条件，当前不进入 P8。完整结果见 `c3_dma_residency_autotune/07_grouped_holdout/20260911_p7q5_search_analysis_run01/RESULTS.md`；任何 virtual-thread 联合扩展必须建立新协议和未见确认集。

> 2026-09-11 方法审计更新：W05 exact allowlist 的 8 KiB manifest/runtime attestation 仅关闭空间与部署安全子模块，不改变 G7 NO-GO。已补做 Rieber/ML²Tuner 启发的 local-validity leave-one-workload-out 和 same-tile 增量性能消融：硬件 validity 模型在 budget=16 时平均得到 14.50 个合法样本，Random 为 10.18；DMA bytes 可改善开发集 top-1 排序，但 full request 和 command/hardware 特征没有额外收益。另一个 79 点 latency-command Pareto 分析中，四个 oracle 已全部使用最小 8 KiB backing，未观察到联合目标权衡。详见 `c3_dma_residency_autotune/00_governance/C3_LITERATURE_METHOD_AND_EXPERIMENT_GAP_AUDIT.md`。

> 同日 P7R113 将目标改为 same-tile `ΔT` 并显式约束物理代价系数非负：bytes/calls 相对 mode baseline 明显改善 MAE 与 Spearman；request-shape 在四个开发折上得到 regret@1=0，但回归误差和相关性下降；command 仍无稳定增量。该混合结果只用于冻结下一次新网络确认协议，不把 post-hoc replay 升格为 G7 通过。

> 同日 P7R112/P7R114 继续验证了两个先前未落实的论文方向。Rieber 2022 在当前稀疏 observed-domain 上的 locality presampling 未优于 Random，且无法忠实复现完整 ConfigSpace/SA，故该分支以负结果停止。USMP 式统一生命周期图因缺逐资源 alloc/free、物理范围、frame/stage-submit、device_done、逐批 FINISH 与 replay lifetime 而 fail closed；已冻结最小 instrumentation contract，在满足前不报告 packing 节省。

> 2026-09-11 P7R 更新：已按新协议开放且仅开放 `input_stationary × oc_nthread=2`。
> W04 的 320 个联合配置经完整 lowering 后仅 27 个满足原/驻留双路径合法且输入 DMA 下降，
> 27/27 又通过三 seed FSim。真实 FPGA 淘汰了连 original 都错误的 config330；config455/461
> 四个配对模块与哨兵通过 18/18 seed。7 个随机完整区组显示 config455 相对同 tile original
> 成对中位提升 8.93%，config461 则下降 12.64%，尽管二者聚合 bytes/calls 相同。差别来自输入
> descriptor 的短行跨距与连续整行形状，说明下一步必须建立 request-shape/overlap-aware 规则，
> 而不是再测 W04 或只按 bytes 排序。两者均显著慢于 TopHub config463，尚无 incumbent、未见
> workload、搜索效率或 FPS 创新结论。详见 `00_governance/P7R_VTHREAD_JOINT_METHOD.md` 与
> `07_grouped_holdout/20260911_p7r9_w04_joint_timing_run01/RESULTS.md`。

> 2026-09-11 P7R 未见验证更新：E00/E01/E02 共 2368 个联合配置经解析证书保留 209 个
>（8.83%）；六个冻结 residency 点均通过三 seed FSim，但只有 E02 通过真实 FPGA 正确性并
> 进入前瞻计时。E02 的 request-shape/continuous 两点相对同 tile original 分别下降 14.80%/
> 13.18%，冻结预测失败。E00/E01 多个 original 在 FPGA 错误；其中 E00 config1081 可由
> original 路径的 runtime UOP `dst_idx` 依赖检查在本地复现，其余点仍表现为 FSim--FPGA
> 差距。故停止 request-shape 性能扩测，第三创新点转入“解析证书 + 原/驻留双路径命令/FSim +
> FPGA canary + TopHub 安全回退”的硬件合法性证书设计。详见
> `00_governance/P7R_UNSEEN_VALIDATION_RESULTS.md`。

> 2026-09-11 P7R DMA-Pareto 更新：E02 的输入字节虽减少 50%，总 DMA 字节却分别增加约
> 42.65%/37.32%，解释了“局部指标变好、真实延迟退化”的根因。新增 exact-same-tile
> DMA Pareto 证书（输入 bytes 降、总 bytes/calls 不增）：P7Q 回顾性选择 11/14 个
> input-stationary 配对，11/11 更快、中位 +8.31%；E02 两个退化点均 abstain。随后冻结未见
> E03：864 个配置经本地证书保留 74 个（过滤 91.44%），两组 original/residency 均通过三
> seed FSim；但 Pareto 首选 config17 在真实 FPGA 为 0/3 正确，合同在计时前停止。这否定了
> “静态 Pareto 足以直接计时”，同时前瞻验证了 FPGA correctness canary 和 original/TopHub
> 回退的必要性。当前仍不能主张未见性能预测或 stage/FPS 提升。
> 修正版算法、公式、证书键与当前主张边界统一见
> `00_governance/P7R_HARDWARE_CERTIFIED_AUTOTUNE_METHOD.md`。

> 2026-09-11 P7R 结构迁移更新：在此前没有 C3 驻留板端标签的 W03/W05/W06 上，候选按
> 开发集已知正收益映射结构和目标 DMA-Pareto 在所有目标标签前冻结。W03/W06 的 original
> 虽通过三 seed FSim，但在 FPGA 均 0/3，合同在 residency 与计时前停止；W05 三段共 9/9
> 正确，input-stationary 相对同 tile original 的成对中位加速为 12.59%、7/7 获胜，但相对
> TopHub 为 -34.15%，最终安全回退。硬件账本已扩至 10 个 exact 证书，并形成测量前派发
> allowlist。该结果给出一项“前瞻局部收益 + 硬件门禁 + 零回归回退”闭环，但不解锁 P8。
> 下一步只允许新的 `oc_nthread=2` TopHub 邻域候选和固定预算效率评价，禁止重复失败配置。

> 2026-09-11 P7R bounded-hybrid 更新：上述 TopHub 邻域已完成。纯 input-stationary 的 27 个
> input-reducing t2 候选全部因总 DMA bytes 增加而 abstain；将 bounded hybrid 的虚线程绑定
> 从组内轴修正为跨组轴后，400 个配置有 19 个全 DMA Pareto，按 TopHub tile 距离冻结 2 个。
> 两候选及同 tile original 通过 12/12 FSim、4/4 AXU 交叉编译和 18/18 FPGA seed。config574/
> 494 相对同 tile 分别 +4.96%/+9.86%，均 7/7 获胜；config574 最终距 TopHub 仅 0.31%，处于
> 2% 等价带内，但严格派发仍回退 TopHub。当前可主张 400→2 的候选派发空间压缩和
> strongest-incumbent 等价带结果；不得把 398 个未测点称作已知 pool oracle，也不进入 P8。

> 2026-09-11 P7R 资源证书集成更新：派发已升级为携带完整 ConfigEntity 的 v2，AutoTune
> 按实体语义而不是易漂移的 config index 过滤，并校验 candidate/resource-certificate hash。
> `{hybrid494, hybrid574, TopHub575}` 调优集合由峰值推导为 12 KiB provisional 计划；最终
> `{hybrid574, TopHub575}` 子集推导为 8 KiB，并用既有同会话 6/6 FPGA 缩容证据升级为
> `qualified_reduced_capacity`。replay 因无可观测资源记录而显式禁用。该差异进一步把本文
> 与 Banerjee 2021 已有的 VTA TPS、冗余 LOAD 消除及分层正确性验证区分开。

> 2026-09-11 P7R manifest attestation 更新：上述资源证书已针对当前 runtime 重新前瞻执行。
> 每个 FSim 签名绑定 worker 实际映射的运行库哈希；板端在 tmpfs 隔离 runtime 下按预先生成的
> manifest ID 执行最终两个身份，6/6 seed 正确，同会话返回 4096/3808 B instruction、
> 4096/1480 B UOP、6 次提交、相同 manifest ID 和 `replay=disabled`。qualifier 通过后 pending
> 与 ready manifest ID 保持不变，默认 RPC 已恢复且没有新增存储错误。正式证据为 p7r98--p7r107。

## 0. 文档地位

本文档是新 C3 的唯一主控计划，取代 `ONE_MONTH_EXECUTION_PLAN.md` 中原来仅 3--4 日的“编译期传输签名引导安全 AutoTVM”任务定义，但不改写其中已经完成的 C1、C2、slot 和 queue 实验事实。

新 C3 与历史 `c3s_buffer_reuse` 没有继承关系：后者已经并入 C2。新实验统一写入：

```text
vta/tutorials/frontend/report_out/stage_tile_cotuning/c3_dma_residency_autotune/
```

任何 subagent 都不得把本计划中的“待验证”改写为“已成立”，也不得自行扩大到 VTA RTL、ISA、u-dma-buf 内核、硬件预取器或重新综合 FPGA。

## 1. 目标、核心假设与创新边界

### 1.1 总目标

在固定 VTA bitstream、阵列规模、量化策略和显式 DMA 共享内存路径下：

1. 复现最接近的 2026 VTA 输入优先/片上权重复用 schedule；
2. 把数据驻留模式作为按 workload 选择的调优变量；
3. 从最终 Lowered TIR 精确提取 DMA 请求形态，而不只估算总访问字节；
4. 用片上容量和候选自身的 instruction/UOP 足迹生成资源安全证书；
5. 在 TopHub 和复现方法双强基线保护下减少无效、错误和灾难性慢候选；
6. 将被安全接受的 workload/stage 服务时间反馈一次给现有 k-best DP，验证是否影响完整 stage 和多帧流水。

建议最终名称：

> **面向 VTA 显式 DMA 共享内存的驻留策略与请求形态协同调优方法**

“命令资源”作为该方法的安全证书和次目标，同时也是 C2 队列定容成果的编译器接口，不重复拆成第四个创新点。

### 1.2 待验证假设

| 编号 | 假设 | 对应证据 |
|---|---|---|
| H1 | 改变 schedule 的 loop order/cache lifetime 能降低输入或权重重复 LOAD | 模式间 TIR、静态 DMA、runtime DMA |
| H2 | 不同卷积几何适合不同驻留模式 | grouped workload 的模式胜者分布 |
| H3 | 请求数、small/stride 和 reload 比总 bytes 更能排除慢配置 | bytes-only 与 full-signature 消融 |
| H4 | 硬件合法性和传输签名能减少板端试验，而不误删强配置 | regret@budget、near-oracle recall、派发总数 |
| H5 | 新 schedule 的命令足迹可按候选确定，并安全缩小 backing | instruction/UOP/FINISH 证书、replay 显式策略与压力测试 |
| H6 | 局部改善在关键 workload 上能传递到完整 stage 或流水 | stage service、DP 重排、II/FPS |

### 1.3 2026 论文的使用方式

Cheng 等的 *Design of Data Access and Schedule Optimization for VTA Compiled Instruction Streams* 正式刊于 FGCS 2026 年卷，但 DOI 为 `10.1016/j.future.2025.108165`，并存在 2025 年公开预印本记录。

2026-09-12 已取得并核对用户提供的 9 页正式全文。正文确认该文提出 output/input-priority ×
原始/权重复用四种方案，并在 TVM passes 和 VTA runtime 中实现不覆盖式权重驻留、按 SRAM 条件启用及
资源不足回退；这些都属于最接近直接基线。正文同时没有定义 AutoTVM ConfigEntity 搜索器、trial
预算、time-to-target、regret 或等预算 tuner 对照。

强制口径：

> B2026 复现用于建立最接近的直接基线，不作为本文原创贡献。课题启动时间和带日期记录只用于说明独立研究过程，不替代相关工作引用和同平台比较。本文核心方法增量限定为：在驻留模式 × tile 联合候选空间中，以最终共享内存/DMA 先验减少达到 FPGA-correct 优解所需的搜索成本。候选证书、强 incumbent、u-dma-buf 布局和 stage/流水验证属于可信执行支撑，不单独承担相对 B2026 的算法创新。

Banerjee 等 2021 年工作已经覆盖 VTA tiling parameter search、DRAM bytes 削减、virtual-thread
冗余 LOAD 消除和 FSim/TSim/DE10 分层验证；AEx 等也研究了处理器综合中的 instruction-memory
定容。因此本文不得宣称这些概念首次出现，创新边界必须落到固定 VTA/ISA 的 exact-identity
三态派发、equal budget、incumbent 保护，以及 host instruction/UOP backing 的 per-allowlist
证书与 runtime attestation。

全文方法规格现已具备；但作者源码和 TVM commit 仍不可得。只有实现论文式权重地址不覆盖、条件
`LOAD WGT` 和 `PushGEMMOp`/UOP 语义后，才可称 method-exact reproduction；不得称 code-exact。
当前 `weight_resident_barrier` 是功能级替代，不是论文方法的精确复现。

### 1.4 不做什么

- 不修改 VTA RTL/ISA/DMA engine，不做提前于 LOAD 的硬件预取；
- 不修改 u-dma-buf ko 模块；
- 不重新联合搜索 FPGA 频率、阵列、SRAM 容量或 AXI 端口；
- 不把逻辑 DMA descriptor 称为物理 AXI transaction 或 compute stall；
- 不把原样复现、普通 SRAM 合法性过滤、一般 weight stationary 单独写成首次创新；
- 不把 C1 的整网划分、C2 的 shared-slot/queue 再计算为 C3 新意；
- 不使用弱 fallback 替代 TopHub/历史高性能基线。

## 2. 当前起点与必须保留的证据

### 2.1 固定硬件

当前配置：AXU5EVB，100 MHz，`BATCH=1`，`BLOCK_IN=BLOCK_OUT=16`，input SRAM 32 KiB，weight SRAM 256 KiB，accumulator SRAM 128 KiB，UOP SRAM 32 KiB，AXI 数据宽度 128 bit。

硬件配置以实际文件和板端 fingerprint 为准，文档数字不能代替运行前哈希：

- `3rdparty/vta-hw/config/vta_config.json`；
- 固定 bitstream；
- u-dma-buf 模块、VTA runtime/driver、RPC server；
- CPU 频率/governor/affinity 和板端 boot ID。

### 2.2 已有软件资产

| 文件 | 可复用能力 | C3 要补的内容 |
|---|---|---|
| `vta/python/vta/top/vta_conv2d.py` | 原始 packed-conv schedule、tile knobs | 独立实验 schedule 和驻留模式 |
| `tune_resnet18_vta.py` | task/case/config 选择，Random/XGB/GA/Grid | mode、稳定候选 ID、本地 shortlist 接口 |
| `vta_autotvm_measure.py` | 顺序交叉编译、NumPy 逐元素参考、计时、runtime profile | 多 seed、环境指纹、错误隔离、静态签名关联 |
| `extract_static_vta_dma.py` | 最终 TIR 的精确逻辑 DMA 提取 | 任意 task/config/mode 批处理、失败分类、紧凑特征 |
| `vta_tuning_history.py` | 显式 TopHub、稀疏 overlay、严格覆盖和哈希 | 适配新模板/模式，双 incumbent |
| `run_stage_tile_iteration.py` | TopHub 不可删除、≥2% overlay、stage 重排 | 接收 C3 安全 overlay 与 B2026 基线 |
| native stage pipeline runner | manifest、JSONL、profile、多帧执行 | 最终 stage/流水验证，不在前期改动 |

### 2.3 已有实验证据

- 自然 Top-20：10.220--11.511 FPS；历史 topology B 可复现约 10.724 FPS；
- 80 个 TopHub 单轴邻域：32 正确、41 编译失败、7 数值失败、0 个安全替换；
- `tile_w` 邻居 LOAD calls/payload/runtime 中位比为 4.5×/4.173×/3.374×；
- 灾难性新 schedule 相对旧 B 的 LOAD calls/payload/device wait 为 14.91×/3.78×/3.39×；
- 8/8 direct-template workload 的六个静态 DMA 字段与 runtime 精确一致；
- 10 类 workload 的 input reload 为 1.72--4.29×，weight reload 为 1.00--7.00×；
- 原冻结 schedule 的 instruction/UOP 峰值为 5312/2792 B；Q1 16 KiB 和 T2656/11008 B 已验证，但**只对旧 schedule 有效**。

## 3. 总体架构与代码策略

### 3.1 四种候选模式

```math
p\in\{original,input\_stationary,weight\_stationary,hybrid2026\}.
```

- `original`：原 VTA schedule，保证 TopHub 和历史二进制兼容；
- `input_stationary`：提升 input cache 生命周期，使输入 tile 跨更多输出通道 tile 复用；
- `weight_stationary`：提升 weight cache 生命周期，使权重 tile 跨更多空间 tile 复用；
- `hybrid2026`：按论文全文复现/参数化输入优先与片上权重复用；细节不足时改名 `paper_inspired_hybrid`。

初始联合变量：

```math
x=(p,tile_b,tile_h,tile_w,tile_{ci},tile_{co},oc\_nthread,h\_nthread,
reuse_h,reuse_w,reuse_{co}).
```

`reuse_*` 仅在论文规格确认和 hybrid 原型通过后开放；首轮强制 `oc_nthread=h_nthread=1`，避免 virtual thread 干扰驻留机制归因。

### 3.2 不污染原模板

推荐实现：

1. 在 `vta_conv2d.py` 中抽取内部 `_schedule_conv2d_packed_impl(cfg, outs, mode)`；
2. 原注册函数 `schedule_conv2d_packed()` 始终调用 `mode=original`，其默认行为必须不变；
3. 新建实验模板名，例如 `conv2d_packed_residency.vta`；
4. 实验模板的 mode knob 使用整数编码 `[0,1,2,3]`，避免 XGB 将字符串实体转换失败；
5. TopHub B0 继续走原模板；新模板显式导入其 tile 实体，不能假设旧 log 原生含 mode 字段。

不得直接给官方 `conv2d_packed.vta` 配置空间追加 knob 后继续使用旧 `config.index`。

### 3.3 稳定候选身份

新增 mode 后 config-space index 会变化。正式候选 ID 定义为规范化 JSON 的 SHA-256：

```text
candidate_id = SHA256({
  hardware_fingerprint,
  template_name,
  schedule_version,
  workload,
  residence_mode,
  complete_config_entity
})
```

`config.index` 只作调试字段，不作为跨版本身份、缓存键或证据连接键。所有 overlay、artifact 和 measurement 都以 `candidate_id` 连接。

### 3.4 模式的初步 loop/cache 设计

当前原模板输出外层约为 `H外→CO外→W外`，卷积在 `x_j0` 下计算，input/weight cache 都在卷积规约 `k_o` 下加载。

- input-stationary 候选：外层优先 `H外→W外→CO外`，把 input cache 提升到空间 tile 层，在多个 CO tile 间复用；input SRAM 只有 32 KiB，是主要合法性风险。
- weight-stationary 候选：外层优先 `CO外→H外→W外`，把 weight cache 提升到 CO tile 层，在多个空间 tile 间复用；weight SRAM 为 256 KiB，预计合法率更高。
- hybrid 候选：在 CO/H/W 外轴增加有界 group，在 group 内提升 weight、子空间内提升 input；必须以论文算法为准，不能凭摘要宣称等价。

所有模式保持 `inp/wgt/acc_scope`、GEMM tensorize 和硬件 ABI 不变；第一阶段只改变 loop order、cache lifetime 和 `compute_at`。

### 3.5 两级筛选

L0 资源/合法性层：

```math
M_{inp}(x)\le32KiB,
\quad M_{wgt}(x)\le256KiB,
\quad M_{acc}(x)\le128KiB,
\quad M_{uop,onchip}(x)\le32KiB.
```

同时检查 tensorize 整除、padding/stride、地址/对齐、静态 shape、lower/build 成功和数值正确性。公式只做预筛，最终以 VTA `StorageRewrite`、`InjectDMAIntrin` 和构建结果为准。

L1 传输签名层：

```text
TransferSignature = {
  load/store calls and bytes,
  input/weight/acc/out 分类,
  average/max/request-size histogram,
  small requests,
  strided/padded requests,
  input/weight/output reload ratio,
  lowered TIR hash
}
```

先用硬约束和 Pareto 支配，不在小样本上随意拟合线性权重。学习排序器只作为后续消融，不是首要交付。

### 3.6 命令资源证书

静态 TIR extractor 当前不能给出精确 instruction/UOP backing 峰值。新 schedule 的命令足迹必须通过候选 dry-run/runtime 诊断获得：

```text
CommandSignature = {
  max instruction bytes,
  max serialized UOP bytes,
  FINISH requirement and derivation,
  replay policy and observed templates,
  submit count and reason
}
```

容量规则为 `C_q(S)=align_A(max_{c∈S,b∈submits(c)} B_q(c,b))`。对齐 `A`、是否增加 headroom、
队列实例数和生命周期组必须写入 manifest，禁止把“两倍余量”、8 KiB 或 36 KiB 固化为算法。
搜索和测量新候选时先使用默认大 backing；只有精确 allowlist 冻结后才生成 provisional 证书，
经实际 runtime capacity status 和多 seed FPGA 正确性验证后才能升级为
`qualified_reduced_capacity`。没有 replay 观测时必须声明 `replay_policy=disabled`。

## 4. 目录、状态与产物规范

### 4.1 固定目录

```text
c3_dma_residency_autotune/
├── 00_governance/
├── 01_paper_reproduction/
├── 02_baselines/
├── 03_residency_schedules/
├── 04_dma_command_signatures/
├── 05_grouped_replay/
├── 06_prospective_board/
├── 07_stage_pipeline/
└── 08_final_evidence/
```

每次执行使用不可覆盖目录：

```text
YYYYMMDD_<phase>_<purpose>_runNN/
```

### 4.2 每个 run 的最小产物

```text
STATUS.md
HANDOFF.md
manifest.json
preregistered.json
command.txt
stdout.log
stderr.log
results.jsonl
summary.json
artifact_hashes.json
```

失败目录不得删除或覆盖；修复后创建新 `runNN`，并在新旧 `STATUS.md` 互相引用。

### 4.3 主状态文件

由主代理创建和唯一维护：

```text
00_governance/progress.json
```

建议 schema：

```json
{
  "schema": "c3_execution_state_v1",
  "current_phase": "P0",
  "phase_status": "planned",
  "last_accepted_gate": null,
  "active_run": null,
  "board_boot_id": null,
  "open_blockers": [],
  "accepted_claim_level": "none"
}
```

subagent 不直接把阶段标记为通过；它只提交证据，由主代理复核后更新状态。

## 5. 分阶段执行计划

## P0：治理冻结与证据盘点

预计：1 日。不得修改 schedule 或运行新板端性能实验。

### 任务

1. 记录 TVM 主仓、VTA submodule、所有相关未提交 diff；不清理、不 reset 用户改动。
2. 对硬件配置、TopHub 文件、历史高性能 binary、bitstream、u-dma-buf ko、runtime、driver、runner、模型和输入生成 SHA-256。
3. 冻结 10 类 ResNet18 workload、开发集、grouped holdout、额外几何 holdout 和 candidate ID schema。
4. 冻结 B0--B8 定义、板端总派发预算、正确性 seed、2%/5%门槛、统计单位和恢复协议。
5. 创建 `SCOPE.md`、`CLAIM_LEDGER.md`、`PUBLICATION_TIMELINE.md`、`FROZEN_MANIFEST.json`、`PREREGISTERED_PROTOCOL.md` 和 `progress.json`。

### Gate G0

- 源码/硬件/TopHub/数据/指标/预算均有哈希或明确版本；
- 数据划分在读取新实验标签前冻结；
- 现有 dirty worktree 已盘点且无覆盖风险；
- 未满足时不得编码。

### 推荐 subagent

`C3-P0 evidence-curator`：只写 `00_governance/`，不改源码。

## P1：2026 论文规格与复现合同

预计：1--2 日。

### 任务

1. 已取得用户提供的正式全文；继续检索作者补充材料或代码，但不得把缺失源码当成正文机制缺口。
2. 已按章节提取 input-prioritized、weight reuse、四方案的伪代码与 loop/cache/runtime 位置；全文没有报告可提取的 AutoTVM 搜索器参数。
3. 建立论文环境与本平台映射表：TVM/VTA revision、开发板、频率、阵列、SRAM、网络、量化和基线。
4. 生成 `PAPER_METHOD_SPEC.md`、`PSEUDOCODE_TO_VTA_MAP.md`、`REPRODUCTION_GAPS.md` 和公开时间线。
5. 当前本地机制命名保持 functional/paper-inspired；实现正文地址与 UOP 语义后才升级为 method-exact。

### Gate G1

- 每个复现机制都有论文页码/图表/伪代码来源；
- 无法确认的细节全部列为假设；
- 若关键 loop/cache 细节不足，允许继续独立重实现，但禁止“精确复现”措辞。

### 推荐 subagent

`C3-P1 paper-spec`：只读论文和源码，只写 `01_paper_reproduction/`。

## P2：强基线恢复

预计：1 日。依赖 G0，不依赖新 schedule。

### 任务

1. 用原模板恢复 10 类 workload 的 TopHub config 和完整 config entity；
2. 生成原始 Lowered TIR、静态 DMA signature、TIR hash；
3. 运行本地既有测试与 reference；
4. 在板端只重放 TopHub canary 和历史 topology B；
5. 确认原始 VTA stage 输出、bitstream/runtime hash 和 10 FPS 量级强基线。

### Gate G2

- TopHub config 全覆盖、无 fallback；
- canary 逐元素正确；
- topology B 与历史 10.724 FPS 的偏差原则上不超过 5%，否则先诊断环境；
- 无法恢复强基线时停止所有“性能提升”实验。

### 推荐 subagent

`C3-P2 baseline-restorer`：允许执行现有脚本，不改核心 schedule。

## P3：原模板兼容重构与固定驻留原型

预计：3--5 日。依赖 G1/G2。

### 代码范围

- `vta/python/vta/top/vta_conv2d.py`；
- 新实验模板模块；
- `tune_resnet18_vta.py` 的模板/mode 入口；
- 新建 `test_vta_residency_schedule.py`。

第一轮不修改 `build_module.py`、`transform.py`、runtime、driver、native runner。

执行记录：第一轮和 P3d schedule-only probe 遵守了上述边界；P3d 证明旧式显式 sync API
与通用依赖检测仍会阻断外层权重驻留后，P3e 才以独立、可回退的 compiler-level proof
获准修改 `transform.py` 和通用 `coproc_sync.cc`。该变更把 full-sync 定义为依赖分段边界，
未修改 runtime、driver、RTL 或 ISA，并通过 50/50 相关回归。它应作为“负结果驱动的
后续编译器增强”报告，不能倒写成第一轮预先完成的设计。

### 实现顺序

1. 抽取内部 schedule 实现，但让原注册入口固定 `original`；
2. 证明重构前后 original 的 TIR、DMA 和 TopHub 行为一致；
3. 仅在 padding=0、stride=1 的代表 workload 上实现 input/weight 两种模式；
4. 再覆盖 1×1/3×3、stride1/2、padding0/1；
5. 按 P1 合同实现 hybrid；
6. 首轮关闭 virtual thread，模式正确后逐项恢复。

### 三类代表 workload

- 大空间尺寸、较小通道的 3×3；
- 中等空间尺寸、常规 3×3；
- 小空间尺寸、大通道或 stride-2 1×1 projection。

最终 ID 由 P0 冻结 workload 清单给出，不在看到结果后更换代表。

### Gate G3：驻留机制成立

- original 重构前后 TIR 结构和六项核心 DMA 字段一致；
- 新模式的 loop order、cache lifetime/`compute_at` 和 TIR hash 确实不同；
- 至少一种模式使目标 input 或 weight LOAD bytes/calls/reload 下降 ≥20%；
- 所有成功 lower/build 的代表配置通过 FSim/独立参考；
- 编译失败、SRAM 超限、padding 限制和数值错误分类保存；
- 若三种新模式都不能产生正确且不同的 TIR，停止驻留主线，降级为既有模板的安全 DMA 筛选。

### 推荐 subagent

`C3-P3 schedule-implementer`：只修改列出的 schedule/template/test 文件；不得上板大规模搜索。

## P4：任意候选传输特征器和稳定身份

预计：2--3 日。依赖 G3。

### 新接口

建议新建：

```text
extract_vta_candidate_features.py
```

示例接口：

```text
--task-file TASKS.json
--candidate-manifest CANDIDATES.json
--output candidate_features.jsonl
--descriptor-detail none|histogram|full
```

### 输出 schema

每行至少包含：

- `candidate_id`、workload、template/schedule version、mode、完整 ConfigEntity；
- lower/build 状态与失败类别；
- SRAM working set 估计与编译器最终判定；
- LOAD/STORE calls/bytes，按 inp/wgt/acc/out 分类；
- average/max、small/stride/padded、请求尺寸直方图；
- input/weight/output reload；
- TIR hash 和工具版本；
- command footprint 状态：`not_measured`、`dry_run` 或 `qualified`。

不要默认保存数百万重复 descriptor；正式筛选使用聚合直方图，只有诊断候选保存 full descriptor。

### 同时修复的测量链问题

1. `vta_autotvm_measure.py` artifact 加入 mode、candidate ID、硬件/boot/bitstream/runtime/build hash；
2. correctness 至少使用 3 个冻结 seed，不只 seed=0；
3. safe overlay 的比较统计统一使用 median；
4. 稀疏 log 必须经 `audited_history` 叠加 TopHub；
5. 检查单算子 benchmark 的 `--tune-log` 是否安全叠加 TopHub及CSV字段一致性。

### Gate G4

- 对原始 8 个 direct-template 样本保持既有六字段精确一致；
- 每个新模式至少 2 个正确候选完成静态/runtime 核心字段核对；
- stable ID 在 config-space 顺序变化后不变；
- lower 动态 extent、编译失败和数值失败都可机读分类；
- 逻辑 DMA 与物理 AXI 的边界在报告中明确。

### 推荐 subagent

`C3-P4 feature-extractor`：修改 extractor/measure/schema tests，不修改 schedule 语义。

## P5：候选生成器与无泄漏离线回放

预计：2--3 日。依赖 G4。

### 候选器

建议新建：

```text
plan_vta_safe_tuning.py
```

输入 `candidate_features.jsonl`，输出冻结 `shortlist.json`，支持预算 `{4,8,16,32}` 和下列策略：

| 编号 | 方法 |
|---|---|
| B0 | 原始 TopHub incumbent |
| B1 | B2026/hybrid 复现最优 |
| B2 | Random AutoTVM |
| B3 | XGB AutoTVM |
| B4 | compile-valid/SRAM only |
| B5 | B4 + DMA total bytes |
| B6 | B4 + full request-shape Pareto/ranking |
| B7 | B6 + residence mode + incumbent protection |
| B8 | B7 + command tie-break + pipeline criticality |

### 数据治理

- 以 workload 为分组单位，禁止同一 workload 的不同配置跨开发/验证集；
- 现有 80 点先用于开发与历史 grouped replay；
- 阈值、small 定义、Pareto 次序和 feature version 在 holdout 前冻结；
- 随机基线在相同候选流上离线重放至少 1000 个排列；
- 所有方法共享同一个板端标注池，不为每种方法重复测相同候选。

### 指标

```math
Regret@b=\frac{T_{best@b}-T_{pool\ oracle}}{T_{pool\ oracle}}.
```

同时报告：

- 达到 strongest incumbent `±2%` 所需总派发数和墙钟；
- compile/lower failure、wrong answer、runtime/device failure 的避免率；
- valid-board-trial ratio；
- near-oracle recall、Top-K recall/nDCG；
- best-so-far 曲线；
- 特征计算、编译、板端运行和设备恢复时间；
- 不把“正确候选数”冒充总试验预算。

### Gate G5

full signature 相对 compile-valid 和 bytes-only 至少满足一项：

- 达到相同 best latency 的总板端派发减少 ≥50%；
- 固定预算下 Regret 明显更小；
- 灾难性慢配置识别明显改善且不误删 near-oracle。

如果只比 Random 好、但不优于 compile-valid/bytes-only，不能把 DMA 请求形态升格为贡献；停止扩大上板，结果降为 case study。

### 推荐 subagent

`C3-P5 shortlist-evaluator`：只写候选器、分析器和 `05_grouped_replay/`，不得根据 holdout 结果反调规则。

## P6：前瞻板端 pilot

预计：2--3 日。依赖 G5。该阶段只有一个 boot 时一律标为 pilot。

### 预算

- 3 个预注册代表 workload；
- 每 workload 最多 12 个唯一候选，包含 TopHub、B2026 和所有策略的去重并集；
- 总计最多 36 个新候选；
- 每候选 3 个冻结 correctness seed，计时 `repeat=5`；
- 同一候选不因多个方法选中而重复编译/测量。

### 板端顺序

1. 环境 preflight 与 TopHub canary；
2. 生成不可覆盖 run 目录和 preregistration；
3. 以阻塞随机顺序测候选，并在前后插入 strongest incumbent；
4. 任一 wrong answer/runtime device error 后停止当前批次，执行恢复协议；
5. 批次结束重载固定 bitstream、重启 RPC、再次验证 canary；
6. 只在正确候选间比较 median，记录全部失败成本。

### Gate G6

- 至少 3 类代表 workload 均有一个新模式正确运行；
- 至少两类 workload 的最优模式不同，或该假设明确失败；
- 新方法相对 strongest of TopHub/B2026 满足：至少一个 workload ≥2% 提升，或相同性能下试验预算明显下降；
- 若没有性能、搜索效率或请求形态的任何正信号，停止正式 312 点扩样。

### 推荐 subagent

`C3-P6 board-pilot-operator`：只运行冻结 shortlist；不得现场换候选或调门槛。

## P7：正式候选池、搜索效率与命令资源

预计：3--5 日。依赖 G6。

### 候选池

- 10 个现有 ResNet18 唯一 workload；
- 至少 3 个预注册几何 holdout，必须是当前 VTA 真正支持的 1×1/3×3、stride/padding 组合；
- 每 workload 最多 24 个唯一候选；
- 理论上限约 312 个唯一候选；实际先做各策略并集去重，达到统计辨识要求后停止，不强制测满。

所有方法在同一测量池上离线重放预算曲线。未测候选不能假装有 oracle 标签；若要声称 pool oracle，必须完整测完该冻结 pool。

### 命令资源子实验

只对以下集合启用 queue diagnostics：

- TopHub；
- B2026；
- 本文每 workload 胜者；
- 与 strongest incumbent 性能相差不超过 2% 的 Pareto 候选。

对其记录 instruction/UOP/FINISH/replay/submit，生成 `resource_certificate.json`。根据候选自身峰值计算 backing 后先做 32 帧拒绝/正确性测试，再做每 workload 或最终 stage 的 1000 次压力。诊断运行与性能运行分开。

### Gate G7：搜索方法成立

- grouped holdout 上，B6/B7 多数 workload 以 ≤1/2 总派发达到 B2/B3 的相同 best latency；
- 最终性能位于 TopHub/B2026 strongest 的 2%范围内或更优；
- full signature 相比 bytes-only 有额外贡献；
- 至少两种几何的最优驻留模式不同，才能保留“按层选择”主张；
- command 证书正确，但其空间收益不得重复冒充新的 C2 创新。

当前判定：**G7 尚未通过**。W05 已提供一项前瞻 same-tile 正收益，但没有击败 TopHub；
W03/W06 被硬件正确性门提前拒绝。允许继续 P7 的 TopHub 邻域验证，禁止进入 P8/stage/FPS。

### 推荐 subagent

`C3-P7 prospective-tuner` 负责冻结 pool 和上板；`C3-P7 stats-reviewer` 在测量结束后只读分析。两者不得同时改实验规则。

## P8：完整 stage 与一次 DP 反馈

预计：2--3 日。依赖 G7。

### 任务

1. 用 audited sparse overlay 编译包含获胜 workload 的完整 Relay stage；
2. 五个冻结 VTA stage 逐元素对量化 LLVM reference；
3. 对比逐 workload DMA 聚合和完整 stage runtime，保留 ACC/ALU/图级残差；
4. 将安全 stage service 写回现有 k-best DP，一次性重排，不循环调参；
5. 对 A/B/C/D 或新的 topology-diverse shortlist 运行 baseline/new 同 boot 交错；
6. pilot 先用 62 帧，最终胜者用 302 帧 ABBA；
7. 记录 single latency、stage service、稳态 II/FPS、DMA totals、mutex wait 和边界 copy。

### Gate G8

- 完整 stage correctness 全部通过；
- 若 DP 排名变化，必须实际编译/上板验证新旧候选；
- 至少两个 stage 或一条完整 pipeline 有可重复收益，才能声称局部改善传递到系统；
- 若不传递，只报告 workload/search-efficiency，不声称 FPS 提升。

### 推荐 subagent

`C3-P8 stage-integrator`：只接受 G7 已冻结 overlay，不重新调单算子规则。

## P9：跨 boot 确认、证据冻结和论文写作

预计：2--3 日，另外需要用户提供独立 reboot 窗口。

### 统计协议

- 只对最终少量胜者做独立 boot，不对 312 个候选全部重复；
- 每个正式结论至少 3 个完整独立 boot；
- 同一 boot 的帧、repeat、ABBA block 不能冒充独立样本；
- 比较采用同 boot 配对，报告每 boot 原始值、均值/中位数、效应量和置信区间；
- 若只有一个 boot，明确写 `pilot only`；
- 不因方向不理想删除 boot 或更换指标。

### 最终产物

- `FINAL_EVIDENCE_LEDGER.md`；
- `FINAL_RESULTS.json`；
- `REPRODUCIBILITY_MANIFEST.json`；
- `FAILURE_CATALOG.md`；
- 基线/消融/预算曲线/请求形态/空间--性能图；
- 对 `MASTER_THESIS_DRAFT.md` 和创新点整合总结的受控更新。

### 创新等级

| 等级 | 条件 | 论文定位 |
|---|---|---|
| A 强独立创新 | G1--G8通过；2--3 workload 相对 TopHub/B2026约5%稳定提升，并有stage收益 | 独立方法章，强调性能与搜索效率 |
| B 小而完整创新 | 无明显stage加速，但前瞻/grouped验证中派发预算下降≥50%，最终性能在强基线2%内 | 独立方法章，强调调优效率和安全性，不强调FPS |
| C 辅助成果 | 仅复现2026、仅过滤非法配置或只在当前ResNet生效 | 相关工作复现/C1增强/案例研究 |

## 6. 板端安全与失败恢复协议

已有实验表明非法候选可能污染 PL 状态，而且只重启进程可能继续得到错误输出。每个板端 subagent 必须执行：

1. 立即停止当前候选批次，保存 candidate ID、日志、最后正确帧和 boot ID；
2. 检查并终止残留 runner/RPC，确认没有实验进程；
3. 重载冻结 bitstream，不能只看 FPGA manager 显示 `operating`；
4. 重启 RPC；
5. 运行冻结 TopHub canary并逐元素核对；
6. canary 正确后创建新 run 继续，旧 run 保持 failed；
7. canary 仍失败则整 boot 作废并等待用户重启，subagent 不得自行重启开发板；
8. 编译失败、lower失败、wrong answer、timeout、RPC/SSH环境失败必须分开分类。

板端 SD 曾写满。允许使用 tmpfs 收集后立即拉回本机，但同一性能对照两侧必须使用相同日志介质和模式。

## 7. Subagent 串行执行规范

### 7.1 主代理职责

- 每次只派发一个会修改同一源码区域的 subagent；
- 只读文献/统计任务可并行，但不得并行操作开发板；
- 派发前读取本计划、`progress.json` 和上一阶段 `HANDOFF.md`；
- 验收 `git diff`、测试、原始产物、manifest 和 gate；
- 只有主代理可以更新 `progress.json`、接受 gate、决定下一阶段或降级；
- 不自动提交 git，不清理用户已有修改。

### 7.2 Subagent 通用任务模板

```markdown
# Subagent Task

任务编号：C3-Px-Ay
阶段：
目标：
前置 gate：
允许读取：
允许修改：
禁止修改：
冻结 manifest/hash：
必须执行的命令：
正确性 oracle：
总预算：
必须生成的产物：
Go 条件：
No-Go 条件：
板端权限与恢复要求：
```

### 7.3 Subagent 交接模板

```markdown
# Subagent Handoff

任务编号：
阶段：
状态：completed / failed / blocked
已完成：
代码变更及原因：
文件清单：
命令与退出码：
测试结果：
实验结果：
失败样本：
与 preregistration 的偏差：
未解决风险：
当前板端 boot/进程状态：
建议下一任务：
```

### 7.4 推荐派发顺序

```text
C3-P0 evidence-curator
  → C3-P1 paper-spec
  → C3-P2 baseline-restorer
  → C3-P3 schedule-implementer
  → C3-P4 feature-extractor
  → C3-P5 shortlist-evaluator
  → C3-P6 board-pilot-operator
  → C3-P7 prospective-tuner
  → C3-P7 stats-reviewer
  → C3-P8 stage-integrator
  → C3-P9 evidence-writer
```

每个 subagent 只能完成一个卡片；不得自行越过 gate、修改门槛或启动下一阶段。

## 8. 第一批派发任务

后续正式启动时，主代理首先派发 `C3-P0 evidence-curator`，而不是直接修改 `vta_conv2d.py`。推荐首条任务：

```markdown
阅读 C3_DMA_RESIDENCY_AUTOTUNING_MASTER_PLAN.md、ONE_MONTH_EXECUTION_PLAN.md、
FPGA_HARDWARE_AWARE_AUTOTUNING_LITERATURE_REVIEW.md 和现有 stage_tile_cotuning 证据。
只执行 P0：创建 c3_dma_residency_autotune/00_governance，盘点当前 git/submodule dirty 状态，
记录硬件、TopHub、历史二进制、workload、数据和工具哈希，生成 scope、claim ledger、
frozen manifest、preregistered protocol 与 progress 草案。不得改源码、上板或清理旧文件。
完成后提交 HANDOFF，等待主代理验收 G0。
```

P0 未通过前，不派发 schedule 实现任务。

## 9. 最终停止条件

满足任一项立即停止对应扩展，不用更多实验掩盖负结果：

1. 不能稳定恢复 TopHub/历史高性能基线；
2. 新模式没有产生不同 TIR 或无法保持正确性；
3. 驻留变化没有带来至少20%的目标 LOAD/reload变化；
4. full request signature 不优于 compile-valid/bytes-only；
5. prospective pilot 没有性能、搜索效率或请求形态的任何正信号；
6. 新 schedule 的 command footprint 无法安全资格化；
7. 完整 stage 发生无法解释的数值差异或设备污染不可恢复；
8. 剩余时间不足以完成与 TopHub/B2026 的公平比较。

停止后保留所有失败产物，并按 C 级成果写为复现、边界或负结果，不改变 C1/C2 已成立贡献。

## 10. 成功时的论文表述

只有达到 A 或 B 级后，才能将第三点写为：

> 针对固定 VTA 数据访问调度难以同时适配不同卷积几何、仅按总传输量难以刻画显式 DMA 请求开销的问题，本文在复现输入优先和片上权重复用调度的基础上，提出面向显式 DMA 共享内存的驻留策略与请求形态协同调优方法。该方法将数据驻留模式与分块参数联合编码，根据非对称片上存储约束排除不可执行候选，从最终 Lowered TIR 中提取分类请求数、粒度、stride 和重复载入特征，并以 TopHub和复现方法作为双重强基线进行有限板端验证；对性能等价候选进一步生成 instruction/UOP 资源证书，并将安全接受的 stage 服务反馈给异构流水线搜索。

具体性能、搜索预算和空间收益数字必须由 P9 最终证据表自动回填，不提前写入。

## 11. 2026-09-11 第三点执行更新

当前不再重复 E00/E01/E03 或旧 stage 基线。W05 `oc_nthread=2` 的新联合空间已完成以下闭环：

1. 固定 FPGA 容量、lowering、全张量 DMA-Pareto 和 TopHub 邻域规则将 400 个候选压到 2 个；
2. 两候选通过 FSim、交叉编译和 FPGA 正确性，且相对同 tile original 分别快 4.96%/9.86%；
3. 最优 hybrid574 距 TopHub575 仅 0.31%，严格派发仍保留 TopHub；
4. 对最终 `{TopHub575, hybrid574}` 生成 instruction/UOP 峰值证书，并在 4096+4096 B backing
   下完成本地及真实 FPGA/u-dma-buf 缩容复验；板端 6/6 seed 正确并直接返回实际容量、峰值和
   提交数，相对原 64 MiB requested backing 减少 99.9878%；
5. 将规则扩到 W00--W09、五模式的 197 个身份，统一 36 KiB 容量复验 197/197 通过；留一法
   9/10，W08 反例证明新增形状必须重新生成证书，不能固化 36 KiB 常数。
6. v2 semantic dispatch 对两个 allow-timing ConfigEntity 完成 2/2 当前空间匹配；调优集合加
   fallback 的 provisional 容量为 8192+4096 B，最终两身份容量为 4096+4096 B，后者已由
   6/6 FPGA 证据升级。两种集合产生不同容量，直接验证了“由 allowlist 推导、不是板级常数”。

第四项仅绑定精确 W05 静态身份和 guarded runtime。下一步若扩展空间结论，应对完整 stage 的
全部提交取峰值并进行新合同验证；在此之前不得将 8 KiB 写成整网容量，也不启动没有 G7 胜者
支持的 P8 stage 性能实验。
