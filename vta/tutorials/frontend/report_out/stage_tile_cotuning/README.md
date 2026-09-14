# Stage–tile 有界迭代：一次反馈与上板验证

日期：2026-09-06。分支：`exp/vta-stage-tile-cotuning`。
原分支快照：`55ec5feac39db09e9fee3f0ae6300afa04d23bd7`。

当前状态：**冻结 Top-20 内的一次 stage–tile 技术闭环已经跑通，但尚未得到相对原 Top-20 高性能调度的有效改进。**
`iteration2_validated` 覆盖 Top-20 的 4 种 topology、5 个唯一 VTA stage 和 10 类实际 workload；共做 32 次候选测量，为 2 类 1×1 stride-2 workload 各补充不超过 2 个合法 repair 候选。原 fallback 仅用于发现 workload；最终对照日志对每个 workload 选择已通过独立量化 LLVM 参考的固定 tile，避免将数值错误的 fallback 当成性能基线。

原生流水线的 2×2 对照来自同一次板卡启动、每组 20 帧、无 warmup，因而目前定位是 **pilot validation**，不是跨 boot 置信区间。它足以证明闭环可执行，但还不足以声称已找到全局硬件最优或得到可泛化的 tile 规则。

2026-09-07 完成了以 TopHub 为不可删除 incumbent 的第一轮真实有界 AutoTVM：Top-20 合并为 10 类 workload，每类测 incumbent 和最多 7 个单轴邻居，共 80 次；32 次通过，41 次编译失败，7 次数值失败。安全接受规则要求 incumbent 与候选在同一 direct runner 均正确，且候选至少快 2%；本轮 **0 个候选被接受**，因此 4 种 topology 的重排结果不变。这只说明 TopHub 在该单轴邻域中是局部最优，不是“没有运行 AutoTVM”，也不证明全局最优。

候选级 profile 给出了可执行的剪枝信号：可比样本中，改变 `tile_w` 后 LOAD 请求中位增至 4.5 倍、LOAD payload 增至 4.173 倍、运行时间增至 3.374 倍；`tile_h` 与 `tile_co` 的 LOAD 请求中位均增至 2 倍；10 个 `tile_ci` 邻居全部编译失败；virtual-thread 邻居不减少 DMA 且更慢。完整表见 `iteration6_safe_overlay_final/TILE_DMA_ANALYSIS.md`。两类 projection 的 TopHub 配置在 isolated runner 中失败而完整 Relay stage 正确，所以稀疏实验日志现在显式叠加在 TopHub 上，缺失项不再被 repair 配置替换。

测量还复现了一个协议风险：连续运行非法候选后，不重载 PL 时同一 TopHub stage 出现数值错误；重新写入 SHA-256 `7bf1ac...28d6` 的 bitstream 后，五个 stage 全部恢复逐元素正确，耗时分别为 68.201、67.528、68.342、48.649、19.599 ms。正式协议因此规定 AutoTVM 搜索与完整 stage 验证之间必须重载固定 bitstream并重启 RPC。

随后启动 stage 内存实验 A/B。实验 A 对 4 种 topology 的所有 VTA stage 导出 workload occurrence、TopHub config 与 tile 因子；10 类 workload 跨 stage 全部保持同一配置，证明本 ResNet18 候选集可按唯一 workload 调优一次，而无需展开 stage×tile 笛卡尔积。实验 B 将单 workload profile 按真实卷积 occurrence 聚合；两个在裸模板 checker 中失败的 projection 改用数值正确的完整 Relay unit 上板补测，耗时分别为 3.350 ms 和 1.847 ms。5 个完整 stage 的 input payload、weight payload 和 STORE payload 均与聚合预测完全一致，LOAD 总字节误差仅为 -0.12% 至 -0.16%；LOAD 请求数仍有 -1.92% 至 -6.34% 的残差，来自 conv-only 聚合没有包含的 ACC load、ALU 和图级指令。受控的 03..15→03..16 扩展中，新增 projection 的 input、weight 和 STORE 增量与 isolated unit 完全相同，仅多出 2 个 ACC LOAD、2048 B ACC payload 和 6 条 driver 指令。证据见 `stage_memory_experiments/EXPERIMENT_AB.md`。

编译期 DMA 提取器随后直接遍历已选 TopHub config 的 lowered TIR，静态展开循环中的 `VTALoadBuffer2D`/`VTAStoreBuffer2D`。8 个 direct-template 正确 workload 的 LOAD/STORE calls、总 payload、input payload、weight payload 与 STORE payload均和板端 profile 精确一致；两个 projection 的卷积 input/weight/STORE 也精确一致，Relay unit 额外部分是 ACC/图级请求。10 类 workload 的 `R_input` 为 1.72--4.29，`R_weight` 为 1.00--7.00，`R_store` 全为 1.00。这证明 tile 确定后，逻辑 DMA 碎片和重复载入可以在编译期取得，不必为每个切图上板 profile；详见 `stage_memory_experiments/STATIC_DMA_EXTRACTION.md`。

端点 delta 表把上述 workload DMA 与图级边界 schema 合并到了用户关心的相邻切点。固定 VTA 起点 03 时，`03..15 -> 03..16` 加入 layer4 projection，静态卷积 LOAD 增加 64 次/217600 B，板端 stage 增量为 66 次/219648 B（残差仅 2 次 ACC LOAD、2048 B），STORE 增加 25088 B，同时 VTA→CPU 边界减少 100352 B；这是必须保留到资源模型比较的“内部 DMA 对边界”权衡。`03..16 -> 03..17` 不增加卷积 workload，静态和板端 LOAD/STORE 增量均为 0，边界再减少 100352 B；在内存特征上较长端点不劣，但最终仍需检查 CPU/VTA service 与流水平衡。该表可在完整 topology 上板前生成，详见 `stage_memory_experiments/CUT_ENDPOINT_DELTA.md`。

实验 C 又固定 layer4 的 5 个单元和相同 TopHub config，只在 block0/block1 之间插入 Executor 边界。30 次交错上板测量中，单 Executor、独立 ext_dev buffer 物化、共享 ext_dev buffer 的中位时延分别为 20.327、22.929、22.065 ms；物化减共享的配对中位为 1.290 ms，27/30 为正。物化组增加 100352 B、1 次 framework copy，共享组为 0；但三组的 522 次 LOAD、8736256 B LOAD、10 次 STORE、125440 B STORE 和 939 条 driver 指令完全一致。结论是该合法块边界不使 VTA DMA 变碎，stage 成本来自可消除的框架物化和不可由零拷贝消除的 Executor 交接；详见 `stage_memory_experiments/EXPERIMENT_C.md`。

实验 D 首轮把四种 topology 的完整流水和内存特征统一到同一表。A/B/C/D 的周期分别为 91.977/96.990/90.962/95.233 ms，VTA DMA 加框架物化的逻辑 demand 为 14.158/14.114/14.259/17.750 MB/帧。A/C 的 VTA DMA 完全相同而 C 多 100352 B 框架物化、周期却低 1.015 ms；D 比 B 多一个 island，LOAD payload 高 29.30%、权重 payload 高 49.61%、mutex wait 多 20.336 ms，但周期低 1.757 ms。这并非说明内存无代价，而是说明 CPU stage 重排可能隐藏或抵消内存代价，不能用单一总字节分数排序。首轮只有 4 个混杂样本，因此没有拟合 M2 参数；详见 `stage_memory_experiments/EXPERIMENT_D_INITIAL.md`。

开发板再次重启后完成了第二独立 boot 重放。固定 bitstream、runtime、192 MiB u-dma-buf 和所有 graphlib 哈希均与第一轮一致；四种 topology 的串行/流水输出哈希以及 DMA calls/payload 跨所有运行完全稳定。第二 boot 内两次 22 帧短运行暴露出 B/C 的明显波动，因此追加反向顺序 62 帧测量，A/B/C/D 为 10.955/10.020/10.818/10.840 FPS。第一 boot 的 `C>A>D>B` 细顺序变为 `A>D>C>B`，周期排序 Spearman 为 0.400；B 的 62 帧 interval P95 仍为 136.753 ms。故内存机制可重复，但不足以用两 boot 拟合 DDR 时间系数或宣称亚 1 FPS 排序优势；详见 `stage_memory_experiments/EXPERIMENT_D_CROSS_BOOT.md`。

第三独立 boot 进一步采用 4×4 Latin-square 顺序，每个 topology 在四个执行位置各运行一次 62 帧，共 992 帧。A/B/C/D 的四次中位 FPS 为 `10.896/10.091/10.756/10.489`，范围分别为 `10.567--10.928`、`9.973--10.404`、`10.383--10.851`、`10.463--10.588`。D 在 4/4 轮快于 B，中位差 0.444 FPS；A 在 3/4 轮快于 C，但中位差仅 0.125 FPS，四轮细顺序并不相同。三 boot 的输出哈希与每帧 LOAD/STORE calls/payload 仍完全一致。正式结论因此冻结为：tile 导出的 DMA 描述适合编译期端点比较、资源下界和支配剪枝，但这四个混杂 topology 不足以拟合独立 DDR 时间系数，也不报告亚 1 FPS 的拓扑优势；详见 `stage_memory_experiments/EXPERIMENT_D_THIRD_BOOT_LATIN.md`。

最终 M0/M1/M2 消融已在完整 972528 个执行配置、4623 种 topology 上重跑，并对 85 个 CPU segment 应用通过 grouped holdout 的 P5B 原子成本，对 87 个 VTA segment 应用 incumbent-tile 静态 DMA。四个冻结实测代表上，M0/M1/M2 的 Spearman 为 `-0.400/0.400/0.400`，regret@1 为 `7.39%/0/0`；M0 预测 `B>C>A>D`，M1 加入边界所有权、host work 与单 VTA mutex 后变为 `A>B>C>D`，命中第三 boot 中位实际 Top-1 A。M2 没有进一步变化：M1/M2 Top-20 20/20 相同，在全部候选中 DDR 下界成为瓶颈的数量为 0，最接近者也只有 M1 bound 的 17.27%。因此最终排序模型按计划回退到 M1，精确 DMA 只保留为切点增量、tile 退化和支配剪枝特征。按 stage placement 重复相同 8-trial 搜索原需 312 次，按 10 个唯一 workload 复用只做 80 次，减少 74.36%；详见 `stage_memory_experiments/MEMORY_MODEL_ABLATION.md`。

2026-09-07 已进一步定位并修复新构建丢失强基线的原因：`DispatchAudit` 作为非 fallback dispatch context 包住 `relay.build` 后，Relay 不再自动加载 TopHub。现在构建入口显式安装 TopHub，再在其内层审计；stage manifest 和 cache key 同时记录 `/home/orange/.tvm/tophub/vta_v0.10.log`、版本、大小及 SHA-256 `8eea6e...a6dc`。源码重编 topology B 得到的 VTA `graphlib.so` SHA-256 为 `6db838...c7ea`，与旧 incumbent 逐字节相同；9/9 workload 命中、0 fallback。开发板重启后重新部署 192 MiB u-dma-buf 和固定 HPC bitstream，第二次 22 帧流水丢弃前 2 帧为 **10.430 FPS**，距旧复测 10.724 FPS 为 -2.74%，输出均为 `a8c613584e081e5f`。冻结 Top-20 的全部 10 类 workload 也已审计为 10/10 TopHub incumbent，清单与本次证据见 `tophub_recovery/`。

2026-09-07 又重放了原 Top-20 的 topology B 高性能二进制。重新加载冻结 HPC bitstream 后，旧二进制用当前 runner 得到 10.724 FPS，且串行/流水各 22 帧输出逐字节稳定；历史 rank05 为 10.656 FPS。当前 bounded-tuned B 只有 3.482 FPS。DMA 对照表明，后者每帧 LOAD 请求为旧调度的 14.91 倍、LOAD payload 为 3.78 倍、设备等待为 3.39 倍。因此下文 2×2 的“固定 tile→有界调优”只是在一组新编译且整体较差的 schedule 内部改善，**不能解释成超过旧 Top-20，也不能用 3--4 FPS 替换原 Top-20 的 10--11 FPS**。完整证据见 `legacy_baseline_recovery/`。

同日还完成了原 Top-20 四种 topology 的标准化重放与聚合 DMA profile，FPS 为 10.310--10.994，均与历史量级一致。A（VTA 03..17）与 C（03..16）的 VTA LOAD/STORE 计数及 payload 完全相同，说明部分切点只改变框架边界、不改变 VTA 内部搬运；双-island D 相对单-island B 增加 25% 的 host→VTA 字节，并出现约 20.337 ms 的第二 VTA stage mutex wait。D 的 LOAD 请求数只比 B 多 4.05%，但 LOAD payload 多 29.30%、权重 payload 多 49.61%，说明请求数与字节类别必须联合建模。证据见 `legacy_topology_profiles/`。

## 已完成的一次反馈

五个唯一 VTA stage 在正确的固定-tile baseline 与有界调优日志下均与同一量化 LLVM 参考逐元素一致。其中位于最终两种切图的 stage 结果为：

| VTA stage | 固定 tile | 有界调优 | 变化 |
|---|---:|---:|---:|
| units 03..15（topology B） | 552.975 ms | 267.063 ms | -51.70% |
| units 03..12（topology D 前段） | 338.188 ms | 182.738 ms | -45.97% |
| units 15..19（topology D 后段） | 275.996 ms | 56.643 ms | -79.48% |

将新的 VTA stage 时间回填到冻结候选池，CPU、边界与 VTA host core-work 成本保持不变：

| 排名 | Top-1 切图 | 静态 score |
|---|---|---:|
| 反馈前 | B：CPU 00..02 / VTA 03..15 / CPU 16..20 | 556.412 ms |
| 反馈后 | D：CPU 00..02 / VTA 03..12 / CPU 13..14 / VTA 15..19 / CPU 20 | 243.450 ms |

这是“先固定 stage 候选、再用少量 tile 测量反馈一次”，不是 stage 与 AutoTVM 全空间的笛卡尔积联合搜索。原 Top-20 之外的切图可能在调优后进入前列，这是冻结 shortlist 的明确局限。

## 原生流水线 2×2 验证

用固定-tile baseline 和有界 AutoTVM best 分别编译 topology B/D。吞吐使用 `(N-1)/(last_completion-first_completion)`；四组的串行/流水 top-1 均一致，baseline 与 tuned 在同一 topology 上的最终输出也逐字节一致。

| 切图 | schedule | 串行中位时延 | 流水完成间隔 FPS |
|---|---|---:|---:|
| B | 固定 tile | 684.027 ms | 1.746 |
| D | 固定 tile | 751.094 ms | 1.601 |
| B | 有界调优 | 395.584 ms | 3.482 |
| D | 有界调优 | 369.450 ms | 4.112 |

在这四个新编译包内部，固定 tile 下 D 相对 B 的 FPS 低 8.32%；调优后 D 相对 B 高 18.09%，B/D 顺序发生反转。这可以作为“tile 可能改变不同 segment 相对代价”的先导现象，但由于没有把旧高性能 incumbent 纳入同一候选集，不能作为最终优化收益或新 Top-20 排名。

对 20 帧聚合的 VTA runtime profile 显示：

| 指标 | B：调优相对固定 tile | D：调优相对固定 tile |
|---|---:|---:|
| LOAD 请求数 | -61.41% | -72.29% |
| LOAD 请求 payload | -48.20% | -60.34% |
| 小 LOAD 请求数 | -62.76% | -73.36% |
| 带 stride 的 LOAD 请求数 | -62.76% | -80.86% |
| 权重 LOAD payload | -59.31% | -72.65% |
| 输入 LOAD payload | -22.98% | -26.18% |
| STORE 请求数 | -58.59% | -63.94% |
| STORE payload | 0% | 0% |

因此本轮找到的不是“总输出变少”，而是较少、较粗的搬运请求，并减少了随空间块重复载入的权重。这些数字是 runtime 记录的 DMA API 请求与 payload，不是 DDR 控制器的实际 AXI burst/byte；本次也没有硬件 compute-stall cycle 计数器。`driver_poll_wait_us` 只是 host 等待设备完成的时间，不能直接说成 compute 在等 LOAD。

## 完整 stage 的参考诊断结果

`equivalence_diagnostic/comparison.json` 使用同一量化 Relay 图在主机 LLVM 执行，绕开 VTA graph packing、schedule 和板端 runtime。测试的是 VTA 单元 03..17、确定性随机 stage 输入，不是全网络实际激活。

| 同参考比较 | 原 fallback | AutoTVM 调优版 |
|---|---:|---:|
| 输出元素数 | 25,088 | 25,088 |
| 不一致元素数 | 8,097 | 0 |
| 最大绝对误差 | 7.8125 | 0 |
| RMSE | 0.8257867104 | 0 |
| 同一 RPC session 连续两次 | 两次均相同偏差 | 两次均逐元素一致 |

这定位到“当前 fallback 不是可信对照”，并非证明所有 stage 都已正确，也没有确定是哪个 schedule/编译变换/runtime 环节引入错误。三个独立 pilot run 的 fallback 输出摘要不同，亦需进一步排查跨 session 稳定性。
该诊断不能直接否定旧部署包和旧实验。后续已在同板重放历史 topology B 包：重新加载冻结 HPC bitstream 后，串行与流水输出均恢复为与历史包相同的唯一哈希 `a8c613584e081e5f`，流水吞吐为 10.724 FPS。重放前若不重新配置 FPGA，旧二进制会出现不稳定输出，说明 bitstream 状态检查必须进入正式协议。

因此 `iteration2_validated` 建立的“正确固定 tile baseline”只适合作为该轮新编译候选的内部控制组。下一轮必须把已验证的历史高性能二进制作为 incumbent；新候选只有在正确性、stage 时间和实际流水 FPS 均不退化时才允许替换它。

## 目标与范围

先验证 `冻结 Top-20 → 合并相同 VTA stage/workload → 少量 AutoTVM 配置 → 实际 stage 应用日志 → 重测 → 原候选池重排`。
这不是对全部 972528 个配置重新求解。一次反馈本身用隔离 stage 时间重排；随后另用 2×2 原生流水线 FPS 实验验证排序反转。
CPU、边界和 VTA host core-work 仍使用历史成本；本次板端绝对耗时与历史值明显不同，不能据此声称排名预测已改善。

当前 Top-20 对应 4 种切图拓扑、5 个独立 VTA stage、10 类实际编译查询到的 `conv2d_packed.vta` workload。
线程不同但算子相同的候选复用调优结果；不是为每个候选单独调优。

## 已补齐的执行链

- `vta_autotvm_measure.py`：顺序交叉编译、直连现有 C++ RPC、独立 NumPy 卷积参考、每个配置逐元素校验、有效计时和 profiler 记录。无需 tracker，无硬件修改。
- `tune_resnet18_vta.py`：保留失败记录；不再因第 0 个配置编译失败而删除整个 workload；支持代表算子和真实 workload 文件。
- `vta_tuning_history.py`：检查日志成功记录、记录 SHA-256、审计实际编译时的命中/回退。严格覆盖的范围是 packed convolution，而不是所有 VTA/CPU 算子。
- 两个 stage 构建入口支持 `--tune-log` 和 `--require-tuned-vta`。native 构建缓存包含日志内容摘要，复用旧 package 时检查一致性。
- `run_stage_tile_iteration.py`：保留当前 fallback 配置，尝试 virtual-thread 和 width 邻居，每个 workload 最多 3 个初始配置；无正确配置的 1×1 stride-2 投影额外最多 2 个 full-width/shallow-height 探针。仍须通过真实编译和数值门槛。这是诊断驱动的候选启发式，不是已经证明的普适剪枝规则。
- `diagnose_stage_tile_equivalence.py`：独立执行量化 LLVM 与两种 VTA stage，保存输出和逐元素比较。参考共享量化器，不能代替独立框架的全网络参考。

本轮最终回归为 19 项 stage–tile 专项测试全部通过，覆盖记录有效性、TopHub 稀疏覆盖、日志内容改变后的 native cache key、配置预算、安全接受、ResNet18 workload occurrence、错误记录格式、LLVM 对照和重排记账；相关 Python 文件语法检查、新增 JSON 解析及 `git diff --check` 也已通过。

## 复现

板卡必须空闲，已启动对应 runtime 的 RPC。环境及二进制摘要见 `board_session.json`。

```sh
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux
TEST_DATA_ROOT_PATH=/tmp/tvm_test_data MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1 \
  /home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/run_stage_tile_iteration.py \
  --host 192.168.1.247 \
  --output-dir /tmp/stage_tile_iteration_new \
  --configs-per-task 3 --repeat 3
```

输出目录必须尚不存在，避免覆盖历史证据。默认进行严格 packed-conv 日志覆盖检查；不足 10 类时不会静默回退并宣布成功。
成功必须以输出目录的 `iteration_summary.json` 中 `status=completed` 为准；仅有 best log 不等于 stage 闭环完成。

## 首轮诊断证据

`smoke_c2`：首次直连 AutoTVM 测量 2 个配置均通过。双虚拟线程配置与当前 fallback 相同，不能把它相对单线程的收益算作超越旧基线。

`iteration1_retry2`：28 次测量，21 次通过，覆盖 8/10 类 workload。严格 stage 构建拒绝 8/10 覆盖，因此该目录不是完成的迭代。
失败集中在 1×1 stride-2 投影的部分配置，包括两个投影 workload 的所有初始邻居；失败配置没有进入 best log。
CPU LLVM 模板与独立参考在 1×1 stride-2、3×3 stride-1/2 上交叉验证通过；这不能单独定位 VTA 侧错误的根因。

`diagnose_p1_wide`、`diagnose_p2_wide`：每类额外 2 个不同长宽比配置均通过硬件数值校验。
这里只证明这些替代配置可用，不声称已经修复或解释原失败配置，也不以它们替代全网络参考检查。
这些 repair 候选源于本轮诊断，是探索性、事后形成的启发式；泛化能力需要新的 workload 或切图 holdout 验证。

### 一条已观察到的局部搬运规律

同一 C2 卷积、同一输入与双虚拟线程，只将 tile width 从 1 改为 2（`iteration1_retry2`）：

| 指标 | width=1，配置 583 | width=2，配置 591 |
|---|---:|---:|
| kernel 中位耗时 | 39.723307 ms | 20.774667 ms |
| load_buffer_2d 请求数 | 1792 | 896 |
| 请求的 load payload | 4,444,160 B | 2,609,152 B |
| 其中权重 payload | 2,064,384 B | 1,032,192 B |
| store_buffer_2d 请求数 | 224 | 112 |
| 请求的 store payload | 200,704 B | 200,704 B |

在这个具体 schedule 下，增加空间 tile 宽度减少空间分块次数，同一权重随空间块重复加载的次数随之减少；输出总字节不变而 store 请求变粗。
同轮配置 7 → 583 只改变虚拟线程，搬运次数/字节完全相同但耗时不同，也说明仅用总字节不能解释全部时间。

这些是软件 runtime 记录的请求和 payload，不是 AXI burst 数、DDR 控制器实测带宽或 compute 等待 load 的 stall cycle。
`small_calls` 阈值在本仓库是 payload <4096 B；它不是硬件坏 DMA 的判定阈值。
host `driver_poll_wait_us` 也不能等同 compute stall。

## 论文仍需完成的验证

1. 对已经完成的四 topology profile 补独立 boot 重复，并在需要时将双-island D 的聚合计数进一步归因到两个 VTA stage；当前机制规律不依赖微小 FPS 差异，但正式置信区间仍缺失。
2. 当前独立语义参考和实际流水线已通过；再补至少两个独立 boot 的配对重复，报告时延和稳态 FPS 的置信区间。
3. 将新 VTA stage 成本反馈到完整切图搜索，或明确将方法界定为有界 shortlist refinement；冻结 Top-20 可能漏掉调优后进入前列的拓扑。
4. 比较“历史高性能 incumbent”“只调 tile、不重新选图”“有界反馈”及其调优预算；任何候选若造成 LOAD/STORE 碎片、重复 payload 或实际 FPS 退化，不得因优于较差的新建 baseline 而被接受。
5. 将共享 DDR 边界物化与 DDR↔片上 SRAM 的 tile 重复搬运分开计量，再与已有双 slot 零拷贝做消融。片上优化不能自动等同跨 Executor 零拷贝收益。

AutoTVM 本身不是新增创新点；可检验的贡献应是预算受限的候选复用、内存代价反馈和选图决策改善。毕业是否足够仍需导师按学院标准确认，不以一次跑通或单个加速数值保证。
