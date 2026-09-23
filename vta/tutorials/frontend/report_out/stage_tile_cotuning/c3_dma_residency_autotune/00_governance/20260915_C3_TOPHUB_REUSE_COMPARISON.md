# C3 新复用方案与 TopHub：同一真实卷积的板端对照

日期：2026-09-15。状态：`COMPLETE; PRIMARY_EVIDENCE_AUDIT_PASS; NO_TOPHUB_SPEEDUP_IN_THIS_CASE`。

## 1. 直接回答

本轮在一个有对应 TopHub 配置的真实 ResNet18 卷积上重新实测，**新方案没有超过 TopHub**。
这不是拿另一个形状、另一块板的旧时间比较，也不是将 synthetic 最优点外推到整网。

对象是 `64→64、56×56、3×3、stride=1、padding=1`，packed input/weight 均为 int8，
与 VTA 卷积模板相同的 shift/clip/cast 输出。当前开发板 boot 为
`61017e99-8c27-4857-bf50-faadefd96b7c`。

| 本轮方案 | 算子中位延迟 | 相对紧邻 TopHub 的延迟变化 | 快于紧邻 TopHub 的轮数 |
|---|---:|---:|---:|
| TopHub 原配置，当前板重测 | 5.9733 ms | 参考 | — |
| 同 TopHub tile，仅输入复用 | 5.9784 ms | +0.16%，基本持平 | 3/10 |
| 本轮最佳输入+权重组合，ALT_251 | 6.3015 ms | +5.43%，更慢 | 0/10 |
| 本轮最佳权重驻留，ALT_251 | 6.7900 ms | +13.68%，更慢 | 0/10 |

TopHub 行是全部 320 次基线计时的中位数。各新方案的百分比使用它自己的 10 组
`TopHub—新方案—TopHub` 紧邻对照，两侧基线取平均，再汇总中位数；不强行用同一个全局基线
计算每一行。输入复用的 0.16% 差异不作为显著退化或加速主张。

“本轮最佳”只表示预先选定的四个替代 tile 和 TopHub tile 内的实测最佳，不代表扩展空间全局最优。
本轮没有做整图计时，不能据此报告 ResNet18 FPS 或整网相对 TopHub 的快慢。

## 2. TopHub 身份与可比性

本地 `/home/orange/.tvm/tophub/vta_v0.10.log` 有两条精确 workload 匹配记录。按缓存中的历史性能
选择第 22 行的较优 incumbent，只用旧时间选择参考配置，**不把旧时间当本轮实测数据**。

该记录来自 `pynq_1x16_i8w8a32_15_15_18_17`，不是原生 AXU5EVB 调优记录。本轮将其完整实体
重新定位到当前 AXU5EVB、通过本地和板端正确性再测量，结论是“当前板重测的该形状 TopHub 缓存
配置”，不是“击败/未击败官方认证的 AXU5EVB 全空间最优”。

原记录配置 index=828，tile_h=[7,8]、tile_w=[1,56]、tile_ci=[4,1]、tile_co=[2,2]，
oc_nthread=2、h_nthread=1。历史 index **不直接用于当前空间索引**：

- genuine baseline 用原 `conv2d_packed.vta` 模板及完整 `ConfigEntity` 实例化；
- 当前 residency 空间按实际内层 tile 和 categorical knobs 匹配；split 外层的 -1 表示推导值；
- genuine source 与匹配的 mode0 产生相同 lowered-TIR SHA256；mode0 重测为 5.9726 ms，
  与 genuine baseline 基本一致。这是 adapter 的同程序对照，不计作独立机制改进。

全部方案共用同一个已微验证的 fresh-completion 实验 libvta.so，SHA256 为
`e6acc970751e55b1350cdd1ed7e0625540a9c773745080a2674886961db47e6e`。
原驱动、原 DSO、此前冻结源码与证据目录均未覆盖。output-valid 映射仍是已实测的行为假设，
不是生成寄存器文件认证；本轮不混入原 DSO 的绝对时间。

## 3. 为什么同 tile 改善仍未超过 TopHub

ALT_251 是单线程、tile_h=7、tile_w=56、tile_ci=1、tile_co=2。该 tile 的四种方案实测为：

| 同一个 ALT_251 tile | 中位延迟 | input LOAD bytes | weight LOAD bytes | LOAD calls | 提交/同步数 |
|---|---:|---:|---:|---:|---:|
| original | 7.2068 ms | 501,760 | 294,912 | 128 | 1 |
| input | 6.6476 ms | 250,880 | 294,912 | 64 | 1 |
| weight | 6.7900 ms | 501,760 | 36,864 | 66 | 3 |
| input+weight | 6.3015 ms | 250,880 | 36,864 | 33 | 2 |

组合相对这个 tile 的 original 延迟下降约 **12.56%**，但这个 original 本来就比 TopHub 慢，
因此组合仍未超过 TopHub 的 5.97 ms。这里是同启动交错计时的中位数描述，不冒充两启动确认。

与 TopHub 的 745,472 B LOAD 相比，组合降到 287,744 B，下降 **61.40%**；输出 STORE 都为
200,704 B，完整逻辑 LOAD+STORE 下降 **48.38%**，但延迟仍增加 5.43%。这直接反驳
“只要共享内存搬运最少就一定最快”。这些是 VTA 逻辑请求，不是物理 AXI burst 或实测总线带宽。

更具体的共享内存语义是：TopHub 的 input/weight 逻辑 reload ratio 约为 2.43/7；组合为 1.25/1。
延长输入生命周期和保持权重有效，确实减少了重复读取；输入的 1.25 仍包含空间 tile 的 halo 重读。
另一方面，TopHub 使用两个 oc virtual threads，而本地 barrier 驻留要求单线程，并且组合增加一次
同步/提交。tile 的 h 从 8 变成 7，计算分段和命令流也改变。它们是已观察的差异，尚不能把净慢
5.43% 全部归因于某一个原因。

算子计时内的 device_run_wait 中位数也从 TopHub 的约 5.544 ms 变成组合的约 6.054 ms，
所以差异不只是网络传输或结果校验开销。不能把该等待指标当 pure-instruction 硬件计数器。

此外，**同 TopHub tile 的 input 模式没有减少 LOAD bytes/calls**：两者均为
input 487,424 B、weight 258,048 B、112 次 LOAD。本轮这个配置下仅开启模式开关没有实现有效
增量访存下降，不能将它写成“输入流量减少但没提速”。该线程/tile 组合的加载作用域须单独分析。

## 4. 执行完整性、成本与失败

候选在目标性能暴露前限定为：genuine TopHub source 1 项、同 TopHub tile 四模式 4 项、四个
固定替代 tile 各四模式 16 项，共 **21 个身份**。全部尝试真实 lowering，17 项通过三 seed FSim
和 ARM 交叉编译，4 项 lowering 拒绝。其中 TopHub tile 的 W/IW 因 oc_nthread>1 不受支持；
两个其他 W 因 compact DMA 约束失败。失败没有替换成另一个有效配置。

有效板端会话重新完整执行：健康门 3 次、17 身份×3 seed 正确性 51 次、显式 warmup 17 次、
十轮紧邻对照 480 次计时，共 **551/551 调用正确**。17 身份对应 16 个不同 TIR，source/mode0
重复是预先安排的等价性检查。逐调用审计确认逻辑 DMA bytes/calls、同步对应提交数和静态程序
一致；无隐藏 warmup，无 driver timeout，boot 前后相同。RPC 自身父子进程已清理。

本地成功准备连续墙钟 **53.551 s**。成功板端 W0→T1 为 **104.467 s**，T0→T1 为
**79.493 s**；T0 之前的部署/重载/RPC 就绪约 24.975 s。这里是完整池对照实验成本，不是某个
搜索算法达到质量目标所需成本，也不能拿来替代 cold-start autotune 实验。

前一个 board run01 完成 71 次正确调用后，首次计时发生 RPC kShutdown，0 次有效计时；整次
性能会话无效。新入口未保留 timer-helper 模块句柄，修正为整个 session 保持句柄后 run02 完成。
这是实验 harness 生命周期修复，不是 VTA 算子优化。run01 已记录的 W0 成本下界 30.903 s，
完整失败墙钟 unknown，不填零；所有旧输出保留，不拼接。更早 prepare run01 因缓存注释解析
失败，亦保留；修复后才产生本地冻结 contract，没有读取新性能选择候选。

## 5. 主证据与入口

- [本地冻结与逐候选资格](../07_grouped_holdout/20260915_c3_tophub_h1_prepare_run02/contract.json)
- [成功板端结果与逐调用原始证据](../07_grouped_holdout/20260915_c3_tophub_h1_board_run02/summary.json)
- [独立主证据审计](../07_grouped_holdout/20260915_c3_tophub_h1_audit_run01/summary.json)
- [失败会话](../07_grouped_holdout/20260915_c3_tophub_h1_board_run01/invalid_session.json)

新入口为 `run_vta_c3_tophub_reuse_comparison.py --phase prepare` 和
`run_vta_c3_tophub_reuse_retained_timer.py --phase board`，独立审计入口为
`audit_vta_c3_tophub_reuse_comparison.py`。CPU-only adapter 测试 3/3 通过。

当前结论：**四方案同 tile 的访存与时间改进成立；超过 TopHub 不成立于本轮有限候选和这个卷积。**
不外推到其他层、不宣称新模式永远无法超过 TopHub、不宣称 TopHub 为全空间理论最优，也不以
负结果触发标签暴露后的无限补点。整网更快仍须真正部署后单独测量。
