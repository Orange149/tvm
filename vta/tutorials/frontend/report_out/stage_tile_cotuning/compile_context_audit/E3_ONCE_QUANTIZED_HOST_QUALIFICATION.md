# E3：一次量化后切分的主机资格化

日期：2026-09-08。**三组局部 tail 的共同参考已建立，主机六臂全部逐元素一致；完整 E3 尚未完成。** 本轮没有新 VTA 上板、u-dma-buf 绑定、AutoTVM 或性能测量。保留原 int8 回绕语义，不是修复量化政策或证明模型精度合格。

后续已完成[三组局部板端 mono/shared/copy 资格化](E3_VTA_CPU_TAIL_QUALIFICATION.md)，复用旧 VTA producer、检查 CPU view 物理范围且输出/逻辑 DMA 全通过；仍不是完整四臂性能、整网精度或 K2 并发安全实验。以下保留为主机实验当时的范围和结果。

## 1. 共同参考与实验范围

直接读取上一轮归档的三份完整量化 Relay JSON：`05..07 / 10..12 / 15..17`，不重新标定或量化。将 tail 的两个 int8 输入表达式提取为 producer 的 tuple 输出，再用新参数替换为 consumer 输入；检查依赖闭合。把 consumer 参数重新代回原表达式后，三组都与原图结构精确一致。

这是**局部 joined stage 的一次量化**，不是整个 ResNet18 一次量化。共同参考仍包含 `int32 add/ReLU → int8 cast` 的既有回绕行为；所有版本在反例处都保留同样的负值，不能写成“43 个错误已修复”。

每组运行以下六个版本，输入仍是原 E2 的 uniform、normal、zeros，共九组固定输入：

| 版本 | 执行方式 | 边界表示 |
|---|---|---|
| A | 单 LLVM GraphExecutor，原图 | 无跨 Executor 边界 |
| B | 单 LLVM GraphExecutor，在两个 cut 输入处再加 stop_fusion | 无跨 Executor 边界 |
| C-int8 | 两个 LLVM GraphExecutor，consumer 通过 zero-copy API 使用 producer 输出 | 两个 int8 tensor |
| D-int8 | 两个 LLVM GraphExecutor，set_input 物化拷贝 | 两个 int8 tensor |
| C-float32 | 两个 LLVM GraphExecutor，共享 float32 交接 | 显式 ×1/16 编码、×16 后 cast int8 解码 |
| D-float32 | 同上，但物化拷贝交接 | 同一精确编码/解码 |

float32 编码对全部 256 个 int8 值可精确往返（单元测试穷举），不会修改 scale 或饱和规则；它也不是任意 float32 输入都无损的通用量化器。它只说明现有 float32 contract 可以承载这个特定量化格点，代价是较大 payload 及显式 adapter。

## 2. 实际结果

五个变体分别与 A 比较，共 **2,634,240 个元素比较，0 差异**；A 也逐项检查与上一轮 frozen joined 输出一致。原来 layer3 的 1 处、layer4 的 42 处回绕反例包含在内。

共享版本未对 consumer 执行边界 set_input copy；生产者输出 NDArray 保持存活和地址稳定，并在每个样本前将 consumer 原 storage pool 清零，仍获得正确结果。拷贝版本检查了目标独立地址及内容一致。注意本仓库 `SetInputZeroCopy` 改的是 op 参数 DLTensor，而 `get_input` 返回原 storage pool；**不能把 get_input 的地址当成共享 op 指针实测**。对应 introspection 字段留空，依据 API 源码、输入更新和输出回归资格化。所有这些均为主机内存，不是板端物理地址或缓存一致性证据。

## 3. 融合并非被笼统证明“不受切图影响”

三个切点的两个输入上**本来就有 stop_fusion**，所以 A/B 的 FuseOps primitive 签名多重集合完全相同。按原 int8 接口切开后，producer 与 consumer 的 primitive 签名多重集合之和也与 A 完全相同。这是这三个既有屏障位置上的复用证据，而非所有切点不影响融合。

另做了事前记录的主机反事实：只去掉这两个原有屏障，所有数值算子和常量保持不变。三组的 primitive 数均由 **6 → 5**，另比较 **526,848 个元素，0 差异**。Graph JSON 显示 residual cast/add/ReLU 合入一个 conv-containing primitive，另一分支的 cast 也改变了所属 primitive。故这个修改确实改变了融合上下文；它不是正式 VTA 调度，也未经过 graph_pack、tensorize、SRAM/DMA 或板端合法性验证。

| 编译版本（各层相同） | primitive 调用数 |
|---|---:|
| 原图 A / 再加屏障 B | 6 / 6 |
| 按原 int8 边切分 | producer 4 + consumer 2 |
| 精确 float32 传输 | producer 6 + consumer 2 |
| 去掉两个原屏障的主机反事实 | 5 |

这是 Relay FuseOps 签名和 Graph JSON 证据，不是完整 VTA TIR、tile 排名或性能对照。主机构建日志中的 untuned 提示也不能用于推断 VTA incumbent 失效；本轮不采用任何主机耗时结论。

## 4. 与共享内存切图的具体关系

只看这一个双张量边界，不含 CPU prefix/suffix 的其他边：

| 层 | int8 payload/B | float32 payload/B |
|---|---:|---:|
| layer2 | 200,704 | 802,816 |
| layer3 | 100,352 | 401,408 |
| layer4 | 50,176 | 200,704 |

在共同数值参考下，**选择什么 dtype/layout 的切分接口、是否插入转换，以及是否在既有融合屏障处切分**，会影响共享边界容量与编译 primitive 集合。这些是 AutoTVM 的单 workload tile 结果之外的外层变量。

但本表不能被写为现有流水已节省 75% DDR：int8 边界尚未接入正式 u-dma-buf pipeline，float32 版本还有 adapter，逻辑 payload 不等于物理 DDR traffic。旧 E6-S 仍使用正式 float32 contract，不用这里的 int8 数字回填旧账本。

额外结构审计发现：从共同量化图抽出的 float32 producer，与上一轮**独立量化 main+projection producer 的 Relay body（含常量）结构完全一致**，三组均通过。因此这三对的已有数值差异定位在 tail 的表达方式，不是 producer qparams 变化。下一轮可优先复用旧 producer 做板端对照；这不等于已完成板端新 consumer 资格化。

## 5. 后续与退出条件

1. 优先在冻结 VTA producer 后接“保留共同参考语义”的 CPU tail，做板端 shared/copy 数值资格化；保留旧 float32 tail 作为不同数值政策的对照，不能混算公平性能。
2. 补正式流水真实激活、block-aligned 负对照和旧 Experiment C 原始输入归因；当前九组局部输入不能替代这些。
3. 若要修复回绕，单独冻结精度政策及参考再比较，不将加 clip、浮点 tail 或宽整数直接当成等价修复。
4. 去屏障只作为主机融合敏感性证据；本月不因此自动扩大 VTA tile 搜索。只有正式 VTA 编译上下文差异且原计划 gate 满足时才进入 E4。

## 产物与复现

- [六臂正式结果](e3_once_quantized_run2/summary.json)、[事前协议](e3_once_quantized_run2/preregistered.json)：每层六个编译模块，共 18 个 LLVM build，两个 consumer 实例分别用于 shared/copy。
- [去屏障反事实](e3_barrier_release_run1/summary.json)、[事前协议](e3_barrier_release_run1/preregistered.json)：另 3 个 LLVM build。
- [合并审计](e3_host_qualification_summary.json)：核验来源哈希、primitive 多重集合、producer 结构与边界字节。
- `e3_once_quantized_run1` 是未完成的脚本资格化尝试：InferType 重建参数对象后，旧参数对象身份检查误报，未产出数值结论；保留原目录，不混入正式结果。修正并通过测试后在 run2 完整重跑。

```sh
TVM_NUM_THREADS=1 TVM_THREAD_POOL_SPIN_COUNT=0 /home/orange/miniconda3/envs/vta-resnet/bin/python vta/tutorials/frontend/qualify_once_quantized_tail_split.py --output /tmp/e3_once_quantized_reproduction
TVM_THREAD_POOL_SPIN_COUNT=0 /home/orange/miniconda3/envs/vta-resnet/bin/python -m unittest discover -s vta/tutorials/frontend -p test_once_quantized_tail_split.py -v
```

反事实及合并脚本默认对账正式归档 `e3_once_quantized_run2`，不是自动读取上述 `/tmp` 重跑目录。完整 E3 仍保留未完成状态。
