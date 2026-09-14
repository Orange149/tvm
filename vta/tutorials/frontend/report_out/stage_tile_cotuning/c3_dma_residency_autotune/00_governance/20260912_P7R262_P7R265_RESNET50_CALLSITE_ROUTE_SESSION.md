# P7R262--P7R265：ResNet50 同 workload 的图节点级驻留隔离

日期：2026-09-12  
开发板 boot：`aa7a3e5c-021d-4d6b-88ae-7f696faa567c`  
结论：已从“一个 workload 的四个实例统一替换”推进到“只替换一个精确 Graph JSON 节点”；
单节点的逻辑 DMA 变化被精确隔离，但本轮没有可主张的整图加速。

## 1. 为什么继续做这一项

P7R232 的 ResNet50 完整图中，R50B workload 对应同一生成函数
`tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_3`，在 Graph JSON 的节点
57、72、85、98 共出现四次。原有 `ExplicitResidencyDispatch` 以
`workload + ConfigEntity` 为键，因此一条 route 会统一改变四个实例，不能表达“第一个实例用
barrier，其余三个保留 original”。

这会限制部署搜索：即使某种驻留方式只适合某个图位置，workload 级广播也只能全开或全关。

## 2. P7R262：不能把 TOPI 查询序号当 call-site

先后做了三次编译期 query-occurrence 探索。前两次 route 没有进入实际 schedule；第三次把标签送到
TOPI 查询边界后，同一目标 workload 在一次 Relay build 中被查询 7 次，而 Graph JSON 中真实执行
实例只有 4 个。原因是 TOPI 查询还包含 compiler probe/cache 等过程，其时序不是 Relay 图节点身份。

因此该原型已全部回退，不能把“第几次 TOPI 查询”包装成“第几个网络层”。三个 P7R262 目录只保留
为空的失败构建目录，不计候选失败、板端失败或性能实验。这个负结果确定了实现边界：真正的
call-site 控制必须绑定稳定图节点/函数身份，不能依赖编译器内部查询顺序。

## 3. P7R264：精确图节点组合

P7R264 run02 从 P7R231 已资格化的 original/barrier 两个完整图模块出发：

- 两者 Graph JSON 完全相同，120 个参数的语义哈希相同；
- 目标函数只出现在节点 `[57, 72, 85, 98]`；
- 只把节点 57 的 `func_name` 改为带 `_c3_callsite0` 的别名；
- 一个 AArch64 wrapper 通过自己的 DSO 路径精确 `dlopen` barrier 图模块并 `dlsym` 原函数；
- incumbent 图模块导入 wrapper，其他三个节点仍解析到 incumbent 原函数；
- 8 个构建产物、构建器源码和 wrapper 源码均有 SHA256，构建 manifest 为
  `49b0a979...8af851`，上板前状态为 `cross_built_unmeasured`。

这是一种部署期 Graph JSON/二进制组合机制，不是 TVM Relay/AutoTVM 原生的 call-site 搜索接口。

## 4. P7R265：真实 FPGA 隔离验证

run02 在同一 clean-start 中同时构造：

1. `incumbent_all4`：四个节点全部 original；
2. `callsite0_barrier`：仅节点 57 使用 weight-resident-barrier。

冻结合同先规定三个正确性种子、7 轮平衡交错计时，以及单次推理必须观察到：weight LOAD
`-393,216 B`、weight LOAD calls `-432`、synchronize `+16`。执行结果如下：

| 指标 | 结果 |
|---|---:|
| correctness calls | 6/6 输出逐元素相同且非零 |
| timing calls | 14/14 输出逐元素相同且非零 |
| weight LOAD bytes/推理 | -393,216 B |
| weight LOAD calls/推理 | -432 |
| synchronize/driver runs | +16/+16 |
| incumbent median | 406.375 ms |
| callsite0 median | 406.017 ms |
| 名义变化 | +0.088% |
| 配对获胜 | 4/7 |

DMA 差值恰好等于 P7R228 单算子的一份，也是 P7R232 四实例广播差值的四分之一。这说明节点 57
被独立替换，而节点 72/85/98 没有被误改。两次 P7R265 复跑都得到相同的精确 DMA 差值；run02 作为
源码哈希完整的正式记录。

0.088% 和 4/7 不支持性能加速结论。合理解释是单节点节省只占约 406 ms 完整图的一小部分，新增
16 次同步基本抵消收益；本实验贡献是把“驻留×tile 选择的作用域”从 workload 广播收窄到精确图
节点，并验证共享内存访问影响不会泄漏到另外三个同 workload 实例。

## 5. 可写与不可写

可以写：本文已实现一个 fail-closed 的图节点级组合原型；同一 workload 的四个 ResNet50 实例中，
只替换一个实例时，真实 FPGA 的权重 LOAD 下降严格等于一份孤立算子差值，输出保持一致。

不能写：已经实现 Relay/AutoTVM 原生 call-site tuner；0.088% 是显著 FPS 提升；Graph JSON 节点号可
跨重新编译直接复用；逻辑 LOAD calls 是物理 AXI burst；单 boot 等于跨硬件泛化。

正式部署清单必须同时绑定 Graph JSON 哈希、节点号、原/别名函数、两份图模块、参数语义、执行器
源码和硬件身份。重新融合、改 shape、改 Relay pass 或重编译后都必须重新发现节点并资格化。

## 6. 证据入口

- `07_grouped_holdout/20260912_p7r264_resnet50_callsite0_composed_graph_build_run02`
- `07_grouped_holdout/20260912_p7r265_resnet50_callsite0_composed_graph_board_run02`
- 构建 manifest SHA256：`49b0a9792ee943dd66078f4a835a7d3a2f71ecf7e0e2f68898ff2265ea8af851`
- 板端 contract SHA256：`10beedc64faecba10aa381be0007390f823882f4be2abe0a8c9d16073ae578c3`
- 板端 summary SHA256：`a4228891cd3fa90fd46477e4a929ece5ceeb47b95e71e9fea6dbb772bd56a5b9`
- 7 项单测：原有 workload dispatch 4 项 + 新增 call-site graph patch 3 项。

P7R265 run01 的功能结果同样通过，但缺少 executor 源码哈希；只作为复跑一致性补充，不作为最终
封存入口。
