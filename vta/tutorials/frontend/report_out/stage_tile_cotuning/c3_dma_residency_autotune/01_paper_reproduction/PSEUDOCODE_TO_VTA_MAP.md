# 公开机制到当前 VTA schedule 的映射

状态：`paper-inspired mapping`，不是论文伪代码转写。

## 1. 当前源码事实

权威文件：`vta/python/vta/top/vta_conv2d.py`（P0 冻结 SHA-256：`b7fc752b...`）。

| 当前行为 | 源码行 | 含义 |
|---|---:|---|
| 定义 `tile_b/h/w/ci/co` | 170--174 | AutoTVM split 空间；当前函数中 `tile_b` 只定义，未在后续 output schedule 中 apply |
| 定义 `oc_nthread/h_nthread` | 175--176 | virtual-thread knob |
| padding 输入直接设为 input scope；否则 `cache_read` | 192--198 | 建立片上 input producer |
| `cache_read(kernel, wgt_scope)` | 198 | 建立片上 weight producer |
| conv 结果置于 accumulator scope | 199 | partial sum 在 ACC scope |
| output reorder 为 `B → H_outer → CO_outer → W_outer → CO_inner → H_inner → W_inner → ...` | 215--220 | 当前外层空间/通道顺序；`store_pt = W_outer` |
| conv stage 在 output 的 `W_outer` 下计算 | 223 | 每个 store tile 内生成 conv |
| virtual thread 可移动到最外 | 231--241 | 可能改变依赖和缓冲需求，首轮应关闭以隔离驻留机制 |
| conv 内部 reorder 为 `B → K_outer → W → ... → CO → H ...` | 243--245 | `K_outer` 是 reduction channel 外层 |
| 对 `K_outer` 应用 `tile_ci` | 247 | 输入通道分块 |
| input 和 weight cache 都 `compute_at(conv, K_outer)` | 248--249 | 两类数据都在每个 reduction-channel tile 的局部生命周期加载 |
| input/weight 标记 DMA，conv tensorize GEMM，output 标记 DMA | 251--255 | 必须保持 VTA ABI/指令化语义 |

## 2. 公开描述与可测试变换

| 公开机制 | 论文可确认程度 | 当前代码中的冲突/机会 | P3 可测试映射 | 证明方式 |
|---|---|---|---|---|
| input-prioritized | 仅名称、目标和有参数调优 | output 外层 `H→CO→W`，且 input 在 conv 的 `K_outer` 下局部加载；跨 CO 的 input lifetime 很短 | 令空间 group 位于 CO group 之外，并把 input producer 提升到覆盖多个 CO tile 的合法层级 | Lowered TIR 中 input allocation/LOAD loop 边界和 input reload 下降 |
| on-chip weight reuse | 名称、覆盖问题和目标可确认 | weight 与 input 同在 `K_outer` 下加载，conv 又在每个 output `W_outer` tile 内实例化；相同 weight 可能随空间 tile 重载 | 令 CO/reduction group 位于空间 group 之外，并把 weight producer 提升到覆盖多个 H/W tile 的合法层级 | weight LOAD calls/bytes/reload 下降，weight SRAM 不超限 |
| minimum access | 只能确认组合思想 | 当前 knobs 不显式表示 residence mode，也不以分类请求形态作为目标 | 独立模板增加整数 `mode`；先硬约束，后对实际 Lowered TIR 的请求向量做 Pareto/shortlist | stable candidate ID、TIR hash、静态/runtime 核对和板端 latency |

这里“提升到某层级”仍是 P3 待验证设计。TVM `compute_at` 的合法位置必须由实际 schedule/lowering 决定；本文件不预先指定一个未经编译验证的精确 axis。

## 3. 本地重实现伪代码

以下伪代码由本项目定义，不是 Cheng 等论文算法：

```text
function schedule_residency(cfg, output, mode):
    create original VTA input/weight caches and accumulator scope
    apply original tile entities

    if mode == ORIGINAL:
        preserve original reorder and cache placement exactly

    if mode == INPUT_STATIONARY:
        order/group spatial tiles so one input tile covers multiple CO tiles
        place input cache at the nearest legal loop outside that CO reuse region
        keep weight placement local unless required for correctness

    if mode == WEIGHT_STATIONARY:
        order/group CO before spatial tiles
        place weight cache at the nearest legal loop outside the H/W reuse region
        keep input placement local unless required for correctness

    if mode == PAPER_INSPIRED_HYBRID:
        create bounded CO/H/W groups
        keep weights across a spatial group and inputs across a CO subgroup

    preserve inp/wgt/acc scopes, dma pragmas and GEMM tensorization
    lower; reject illegal SRAM/tensorize shapes
    measure classified logical DMA signature
```

## 4. AutoTVM knob 合同

原模板已有：

```text
tile_b, tile_h, tile_w, tile_ci, tile_co,
oc_nthread, h_nthread
```

P3 实验模板可新增：

```text
mode ∈ {0,1,2,3}
```

编码固定为：`0=original`、`1=input_stationary`、`2=weight_stationary`、`3=paper_inspired_hybrid`。首轮 `oc_nthread=h_nthread=1`。只有前三种模式通过正确性和 TIR 机制验证后，才允许把本项目自定义的 `reuse_h/reuse_w/reuse_co` 加入后续空间。

原 `conv2d_packed.vta` 不追加 mode；新模板显式导入 TopHub tile entities，不能复用旧 `config.index` 作为身份。

## 5. 不允许的等价性推断

- 不能因为 output reorder 改成 `H→W→CO` 就称其等于论文 input-prioritized；
- 不能因为 `compute_at` 外提就称已实现论文的 overwrite avoidance；
- 不能把 `cache_read` 的逻辑 LOAD 数称为 AXI burst 数；
- 不能把本地 Pareto/request-shape tuner 称为论文 minimum-access tuner；
- 不能在未 lower/build/正确性验证前断言某个 cache lifetime 合法。

