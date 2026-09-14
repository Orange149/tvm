# P7R：虚拟线程与 DMA 驻留联合优化方法及实验契约

状态：`PREREGISTERED_BEFORE_P7R_LABELS`  
冻结日期：2026-09-11  
性质：P7Q 之后的新开发协议；W01/W04/W07/W08 从此只作开发集，不能再次作为未见确认集。

## 1. 为什么停止重复 P7Q

P7Q 已经完整测量旧冻结池：79 个真实 FPGA 正确候选、395 个延迟样本。其候选生成器把
`oc_nthread=h_nthread=1`，但四个受保护 TopHub incumbent 都使用 `oc_nthread=2`。因此 P7Q
只回答了“单虚拟线程、给定弱 tile 时，驻留模式怎样变化”，没有回答第三创新点真正的问题：

> 固定 FPGA 后，怎样利用 SRAM、DMA、虚拟线程和依赖队列信息，在不破坏强计算配置的前提下，
> 剪掉不可能获益或不可能合法的 schedule，并把少量有希望的候选交给 AutoTVM 实测？

P7R 不扩大 P7Q 样本，也不把 P7Q 的开发反馈重新包装成 confirmatory evidence。

## 2. 与 AutoTVM 的关系

AutoTVM 不是傻瓜遍历。它把 schedule knob 组成配置空间，并可用 XGBoost 等代价模型预测候选；
其迭代变量特征比单纯 knob 特征更容易跨 shape 泛化。[1][2] Ansor 进一步用 program sketch、
进化搜索和学习代价模型搜索张量程序。[3] MetaSchedule 则显式提供 schedule rule、mutator 和
postprocessor，可在实测前拒绝硬件非法程序。[4]

P7R 不替代这些搜索器，而是在它们前面增加 VTA 专用但可参数化的“候选生成与证书层”：

1. 从已有高性能 incumbent 出发，而不是重新从弱默认配置开始；
2. 对同一 tile 生成驻留/虚拟线程的相对变换；
3. 用复用、片上容量、DMA 请求形态和依赖合法性先验剪枝；
4. 只把证书通过的候选交给 AutoTVM 代价模型或板端测量；
5. 任一候选没有超过 incumbent 时安全回退。

这与 MAESTRO、Timeloop 和 Interstellar 的共同认识一致：数据流优劣取决于 loop mapping、数据复用、
片上容量、带宽和利用率的联合约束，不存在对全部网络层都最优的固定 dataflow。[5][6][7]

## 3. 时间上限与优化目标

对候选 `x`，不再用“DMA bytes 越少越好”作唯一目标，而采用：

```math
\widehat T(x)=\max\{T_{load}(x),T_{compute}(x),T_{store}(x)\}
             +T_{sync}(x)+T_{launch}(x)+T_{tail}(x).
```

其中：

```math
T_{compute}^{lb}=\frac{MAC(x)}{f\,BATCH\,BLOCK_{in}\,BLOCK_{out}},
```

```math
T_{load}=\sum_q \left(L_{dma}(shape_q)+\frac{bytes_q}{BW_{eff}(shape_q)}\right).
```

当前 AXU5EVB 配置为 100 MHz、`BATCH=1`、`BLOCK_IN=BLOCK_OUT=16`，理论峰值为
25.6 GMAC/s，若一次乘法和一次加法各计一次则为 51.2 GOPS。VTA 的访问--执行解耦和虚拟线程
用于重叠访存与计算、隐藏访存延迟，而不是免费增加 SRAM。[8][9]

优化目标是最小化 `T_hat` 或真实板端延迟，并受下面的硬件证书约束；减少 LOAD 只有在没有引入
更大的 compute tile 损失、同步或尾部开销时才是收益。

## 4. 两个必须同时成立的判据

### 4.1 复用必须在虚拟线程复制后仍然存在

令：

- `Gco = CO_blocks / tile_co`：输出通道外层 tile 数；
- `t`：输出通道虚拟线程数；
- `r`：每次 input residency 覆盖的输出通道 tile 数。

理想情况下，输入 LOAD 次数从每个空间 tile 的 `Gco` 次降为 `Gco/r` 次。但 VTA 的
`cthread` context 不共享被触及的局部缓冲；虚拟线程注入会重映射并扩大相应 allocation。因此
如果所谓复用只是跨 `t` 个 context，输入仍会被复制，不能把它计为真实复用。编译期下界为：

```math
r_{effective}=\frac{r}{d_{ctx}},\qquad
R_{inp}^{ideal}=1-\frac{1}{r_{effective}},
```

其中 `d_ctx` 是同一输入 tile 实际被复制到的 context 数。只有 `r_effective > 1` 才生成候选；
最终值以 Lowered TIR 的逻辑 DMA 请求计数为准，不用公式冒充实际 AXI transaction。

### 4.2 延长驻留生命周期后仍能装入 SRAM

当前硬件的向量深度为：

```text
input: 32768 / 16  = 2048 vectors
weight: 262144 / 256 = 1024 vectors
acc:    131072 / 64  = 2048 vectors
uop:    32768 / 4    = 8192 entries
```

完整证书为：

```math
D_m^{lowered}(x) \le D_m^{hardware},
\quad m\in\{inp,wgt,acc,uop\}.
```

公式只能做快速预筛；`StorageRewrite` 后的实际 allocation、DMA 地址范围、依赖 pass、完整 build 和
数值正确性是权威判据。TVM 的 VTA memory info 会把上述 bit 数作为上限，超限 allocation 应在
构建阶段拒绝。[10]

## 5. 一个具体例子：为何 W04 不能直接改最优配置

W04 的受保护 TopHub 配置为：

```text
tile_h=14, tile_w=14, tile_co=4, tile_ci=1,
oc_nthread=2, h_nthread=1, CO_blocks=16.
```

原 schedule 一次只保持一个 `tile_co` 的卷积结果，每个 context 的 accumulator footprint 为：

```math
14\times14\times4=784\ vectors,
```

两个 context 合计 1568，小于 2048。若 input-stationary 将一次卷积区域提升到全部 16 个输出通道
block，footprint 变为：

```math
14\times14\times16=3136\ vectors>2048.
```

因此“TopHub config463 直接换 input-stationary”在测量前就应被拒绝。可实验的方向是缩小空间 tile，
或只驻留有界 `r` 个输出通道 tile；它们必须与丢失的 tile 计算效率共同比较。这正是 P7R 相比旧
P7 的关键变化：不是多加一个 mode knob，而是把 mode 对 buffer lifetime 的影响纳入合法性和代价。

## 6. P7R 最小实现范围

首轮只实现并验证一个变量，避免再次形成大而不可解释的搜索池：

```text
mode = input_stationary
oc_nthread in {1, 2}
h_nthread = 1
residency_co_group in {1, 2, all}
```

候选从强 incumbent 的单轴邻域产生，包括同 tile 对照、`tile_h/2`、`tile_w/2` 和有界 CO group。
硬剪枝顺序固定为：

1. 整除、tensorize 和 context 数合法；
2. 解析变换后的 loop scope，计算 `r_effective`，拒绝零复用候选；
3. Lowered TIR/StorageRewrite 容量证书；
4. 静态 DMA 必须相对同 tile original 减少目标 LOAD，且没有非法依赖；
5. FSim 三个固定 seed 逐元素完全一致；
6. AXU5EVB 交叉编译；
7. 真实 FPGA 三 seed 正确性；
8. 正确后才计时。

任何一步失败即停止该候选，不重启或关闭开发板，不通过增加重复次数挽救失败结论。

## 7. 开发集、确认集与停止规则

### 7.1 开发集

- W04 是首个正向开发样例，因为 `Gco=4`，存在两组输出通道复用机会；
- W01/W07/W08 的 incumbent 均为 `Gco=2,t=2`，作为“跨 context 后没有额外复用”的负例；
- 旧 P7Q 标签只允许用于解释与调试，不用于 P7R 最终无偏性能结论。

### 7.2 未见确认集

在实现冻结前另行生成至少三个此前未计时的卷积几何，覆盖：

- `Gco/t=1`：规则必须拒绝或回退；
- `Gco/t>=2` 且 SRAM 富余：规则应生成驻留候选；
- 理论有复用但 accumulator 超限：规则必须在上板前拒绝。

确认指标为：非法候选拦截率、near-oracle recall、regret@budget、板端派发数、候选相对同 tile
对照的延迟，以及最终对受保护 incumbent 的胜负。

### 7.3 Gate

- `G7R-local`：三个结构类别均按预期生成/拒绝，原模板 TIR 回归不变，FSim 全部正确；
- `G7R-board`：所有派发候选三 seed 正确，且至少一个未见候选相对同 tile original 有稳定收益；
- `G7R-innovation`：相同测量预算下，证书引导策略比 Random 和 knob-only XGB 降低 regret 或板端
  派发数，同时从不劣化 protected incumbent。

若只有局部同 tile 收益而没有搜索效率或最终 incumbent 收益，结论降级为机制实验；若三项均失败，
终止该路线，不再重复 P7Q。

## 8. 可写入论文的创新边界

若 Gate 通过，创新点表述为：

> 提出一种面向固定显式 DMA 加速器的硬件证书引导驻留调优方法。该方法以已有高性能计算 tile
> 为锚点，联合建模虚拟 context 引起的局部缓冲复制、驻留生命周期造成的 SRAM 占用、实际 Lowered
> TIR 的 DMA 请求形态和命令依赖，在板端测量前过滤无复用或资源非法候选，并由学习型调优器对剩余
> 候选排序；未获益时安全回退到原 incumbent。

这里的新意不是 XGBoost、input-stationary 或 weight-stationary 本身，而是“相对强 incumbent 的
驻留变换 + context-aware 复用判据 + lowering 后资源证书 + 安全回退”组成的闭环。

## 9. Sources

[1] Chen et al., “Learning to Optimize Tensor Programs,” NeurIPS 2018.  
https://proceedings.neurips.cc/paper/2018/hash/8b5700012be65c9da25f49408d959ca0-Abstract.html

[2] Apache TVM, AutoTVM XGBoost cost-model source.  
https://apache.googlesource.com/tvm/+/afbfb7aa7e43732cb716f8e443df696110be6afc/python/tvm/autotvm/tuner/xgboost_cost_model.py

[3] Zheng et al., “Ansor: Generating High-Performance Tensor Programs for Deep Learning,” OSDI 2020.  
https://www.usenix.org/system/files/osdi20-zheng.pdf

[4] Apache TVM RFC 0005, “Meta Schedule (AutoTensorIR).”  
https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md

[5] Kwon et al., “MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings,” 2018.  
https://arxiv.org/abs/1805.02566

[6] Parashar et al., “Timeloop: A Systematic Approach to DNN Accelerator Evaluation,” ISPASS 2019.  
https://accelergy.mit.edu/timeloop.pdf

[7] Yang et al., “Interstellar: Using Halide's Scheduling Language to Analyze DNN Accelerators,” ASPLOS 2020.  
https://mast.stanford.edu/pubs/interstellar_using_halides_scheduling_language_to_analyze_dnn_accelerators/

[8] Moreau et al., “A Hardware–Software Blueprint for Flexible Deep Learning Specialization,” VTA paper.  
https://arxiv.org/abs/1807.04188

[9] Apache TVM, “VTA: An Open, Customizable Deep Learning Accelerator,” official release article.  
https://tvm.apache.org/2018/07/12/vta-release-announcement.html

[10] Local implementation evidence: `3rdparty/vta-hw/config/vta_config.json`,
`3rdparty/vta-hw/include/vta/hw_spec_const.h`, `vta/python/vta/build_module.py`,
`src/tir/transforms/inject_virtual_thread.cc`, and `vta/python/vta/top/vta_conv2d.py`.

