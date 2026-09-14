# G0 第一阶段：重启恢复与自然流水 readiness 审计

日期：2026-09-08。状态：本阶段已完成；完整 G0 尚未完成，未进入 G1/G2/G3。

## 本轮回答的问题

固定已有 topology、stage 编译产物、tile、CPU 线程与共享 K2 机制，检查真实多帧流水中是否存在“一个 stage 正在运行，另一 CPU/VTA stage 刚完成输入/输出绑定”的可达方向。此次不重新 AutoTVM，不改 VTA 硬件，不实施延迟启动策略。

结论：存在自然可达的跨设备方向，而且不同冻结配置的方向分布差异明显，可以继续做受控 allow/wait discovery。但 encounter 不是有害争用证据，更不是可恢复性能收益；第三创新点仍未通过 G0。

## 恢复与可复现性

- 开发板：`root@192.168.1.247`；boot ID：`a5220e22-f562-40be-bbe9-058b6fbeed50`。
- SD 分区 `/media/sd-mmcblk1p2`，恢复 `/mnt/sd` 链接；加载既有 u-dma-buf 模块，`udmabuf0=201326592`（192 MiB），物理基址 `0x68500000`。
- 重新加载冻结 `vta_hpc.bit`，FPGA manager 为 `operating`。bitstream、libvta、libtvm_runtime 哈希见 [恢复日志](board_restore_a5220e22.txt)。未更换硬件设计或 runtime。
- CPU governor 为 `userspace`、频率 `1066666 kHz`。使用冻结脚本的各 stage 线程/CPU affinity；VTA host 保持原来的 default affinity，未声称已经达到 G0 confirmation 的固定 host 绑核协议。
- 本轮通过 native runner 独占顺序测量，未启动 RPC 服务；结束后没有残留测量进程，boot ID、FPGA 状态和缓冲大小复核一致。
- `/sys/class/thermal/thermal_zone0/temp` 不存在；收尾发现温度接口位于 IIO。PS 的 raw/offset/scale 为 `40412/-36058/7.771514892`，PL 为 `40145/-36058/7.771514892`，按 `(raw+offset)*scale/1000` 分别约 `33.84/31.76 °C`。这只是结束后的快照，不能补称全程温度窗口已受控；下一阶段应记录各 block 前后的 IIO 温度。
- 板端时钟与主机日历不一致，实验以 boot ID 和 `summary.json` 内的 host UTC 为准。

使用冻结包 `legacy_profile_rank01/05/07/13`，分别记作 A/B/C/D。每个 stage 的 graph、library、params 均对 manifest 做 SHA-256 校验，两项 runtime 也按冻结哈希验证。原 runner 未被覆盖；新 runner 独立部署在 `/media/sd-mmcblk1p2/vta_stage_pipeline_runner_g0_a5220e22`。

| 产物 | SHA-256 |
|---|---|
| 原 runner | `027ef4607a9bcdbac10d2e53e2b3b55f740624364e23b4f4795c28779188d311` |
| 本轮 trace runner | `67ab7d1f8e004113e9bd932cec8f32a1ddc5242163d048a66a3d7a29eabad641` |
| trace runner 源文件 | `46418bf28e01d314199ec6ff3f60ca022641669a4adbf7c14daa0d7ecf3b5412` |
| 本轮执行的编排脚本 | `76b9fda9e3c37f2d0473bb3a12c18f586d810a409798b8728dddbbdc4c48ea1f` |

交叉编译使用现有 SDK：

```sh
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux
aarch64-xilinx-linux-g++ -std=c++17 -O2 --sysroot="$SDKTARGETSYSROOT" \
  -I include -I 3rdparty/dlpack/include -I 3rdparty/dmlc-core/include \
  -I 3rdparty/vta-hw/include -I vta/include \
  vta/apps/native_deploy/vta_stage_pipeline_runner.cc \
  -o /tmp/vta_stage_pipeline_runner_g0_a5220e22 \
  -L build_axu_aarch64 \
  -Wl,-rpath-link,"$SDKTARGETSYSROOT/lib" \
  -Wl,-rpath-link,"$SDKTARGETSYSROOT/usr/lib" \
  -ltvm_runtime -lvta -ldl -pthread
```

## 新增观测与校验

native runner 新增 `queue_pop_ms`、`slots_acquired_ms`，输出已有内部计时对应的 `binding_done_ms`、`run_start_ms`、`run_end_ms`。时间戳放在 frame 的内存结构中，不新增逐事件文件写入；原有逐完成帧 JSON/flush 行为仍保留，因此不能声称完全无测量扰动。没有新增控制器、修改 queue/slot 协议或延长 mutex 范围。

共享 K2 下，binding-done 位于 slot 获取与零拷贝 view binding 之后、VTA mutex 获取与 graph invocation 之前，是候选门控边界。旧 copy 路径的 VTA mutex 包围 set/run/get，不能把该路径的 binding-done 当成同样的可门控边界；正式 readiness 分析只使用新 runner 的 shared 日志。

审计了 shared 的全部 1208 帧：时间戳单调、run 时长一致、两个 VTA island 的 mutex 持有区间不重叠、每个 slot 的 generation 递增、物理范围稳定且互不重叠、复用不早于上次 consumer run 结束，以及框架边界物化字段为 0，均通过。该审计不能替代独立的 owner 状态全轨迹、唯一帧输入、异常退出和 generation 故障注入测试；本轮重复使用冻结输入。

主机侧新增 6 个回归测试，覆盖区间并集去重、shell 换行解析、方向识别、generation/时序错误拒绝、旧日志缺失 readiness 的正确解释和输出不一致拒绝；全部通过。C++ 交叉编译、Python 语法检查和 `git diff --check` 通过。

## 实际运行结果

先用旧 runner 对每种 topology 做串行 8 帧、copy/shared 各 102 帧，共 848 帧；再用新 runner 做串行 8 帧、copy/shared 各 302 帧，共 2448 帧。合计 **3296 帧**，逐帧输出 FNV 和 frame 顺序均通过冻结结果校验。

以下 FPS 是新 runner 的实际多帧流水测量，不是静态排名；每种模式丢弃前 2 帧，使用余下 300 帧的首末 completion 计算 II，`FPS=1000/II`。P95 是 `final completion − queue0 Push 前的 submit`，包含提交阻塞，不是 completion interval。

| 冻结配置 | copy FPS | shared K2 FPS | shared II / ms | shared submit-to-complete P95 / ms | CPU–VTA graph-run 重叠并集 / 墙钟 |
|---|---:|---:|---:|---:|---:|
| A（rank01） | 10.613 | 10.944 | 91.375 | 603.963 | 76.9% |
| B（rank05） | 10.151 | 10.160 | 98.425 | 565.474 | 71.2% |
| C（rank07） | 10.436 | 10.857 | 92.109 | 605.778 | 76.5% |
| D（rank13） | 10.617 | 10.599 | 94.350 | 722.649 | 76.3% |

旧 runner 102 帧 copy FPS 为 A/B/C/D=`10.983/10.044/10.547/10.505`，shared 为 `10.808/10.029/10.820/10.625`。新旧结果没有显示数量级退化，但运行顺序及帧数不同，**不能据此资格化 instrumentation overhead，也不能据本轮 copy/shared 差值宣称加速**。正式开销需要同帧数交错 ABBA；确认收益还需要独立 boot。

这里的 overlap 是 host GraphExecutor run 区间，包含它内部的 host/runtime 工作，不是 FPGA compute 时间、DMA 活跃时间或 DDR stall。

## 可达方向：本轮最有价值的新结果

箭头 `active → ready` 表示前者正在 graph-run，后者刚完成 binding。分母是落在稳态 completion 观测窗口内的对应 stage ready 事件数，不统一假定为 300。

| 配置 | CPU stage0 active → VTA stage1 ready | VTA stage1 active → CPU stage0 ready | VTA stage1 active → CPU stage2 ready | CPU stage2 active → VTA stage1 ready |
|---|---:|---:|---:|---:|
| A | 110/297（37.04%） | 185/296（62.50%） | 111/299（37.12%） | 187/297（62.96%） |
| B | 2/298（0.67%） | 295/297（99.33%） | 4/299（1.34%） | 273/298（91.61%） |
| C | 102/298（34.23%） | 195/297（65.66%） | 100/299（33.44%） | 197/298（66.11%） |

A/C 的两侧都有较多 encounter；B 明显偏向 VTA 已 active 后 CPU stage0 到达，以及 CPU stage2 已 active 后 VTA 到达。这说明仅按无方向的“CPU 与 VTA 是否重叠”建表会丢失自然调度状态。不同配置同时含 stage 集合、边界与 CPU 线程等差异，当前不能把该分布差异单独归因于切点或 DMA。

D 的双 island 还暴露了重要排除项：`CPU stage0 active → VTA stage3 ready` 共 297 次，但只有 62 次在 ready 时没有 VTA mutex owner，其余 235 次此时本就不能立即启动 VTA。不能把 297 次全部算成门控能够改变的立即启动机会。D 的 VTA stage1/stage3 mutex-wait P95 分别约 `8.356/47.125 ms`；这是单物理 VTA 串行等待，不是 DDR 等待。`VTA stage3 active → CPU stage4 ready` 和反方向在本次窗口均未观察到，只能称本次未见，不能证明所有执行均不可达。

shared 的逐帧 producer+consumer slot-wait 总和 P95 为 A/B/C/D=`33.201/0.006/33.583/61.079 ms`。它们是共享缓冲 backpressure 的观测，不是可直接从 II 扣除的代价，也不能与 mutex wait 简单相加当成加速上限。

## 结论边界与下一步

1. 恢复成功、冻结产物可重放；新增记录能够区分 queue-pop、slot-ready、binding-ready、graph-run 与 VTA owner。
2. **自然可达性初筛通过，允许继续 G0 的受控 pair discovery；不是完整 G0 go。** 尚未测 `allow-now` 相对预定 `wait` 的 paired makespan，`protect/allow` 标签、可避免 penalty 与 opportunity ceiling 均为未知。
3. 下一步先依据 isolated/static 内存特征冻结 pair 与自然 active-age 桶，再实现 binding 后、mutex 前的逐样本 ready/start/done hook；两个方向分别比较相同 ready-state 的 allow/wait，加入 compute-only 负对照并记录 IIO 温度及固定 host affinity。不能用原先进程入口的 barrier 代替该 hook。
4. 完整 G0 仍需噪声门槛、自然 `protect/allow` 异质性、memory-path 归因和不重复计算的 opportunity ceiling；通过之前不实现 G2 控制器、不声称第三创新点成立。C1 的 E0/E1 编译融合审计也没有因本轮 trace 而自动完成。

## 产物与失败记录

- [旧 runner 基线汇总](boot_a5220e22_baseline_run2/summary.json)
- [新 runner 测量汇总](boot_a5220e22_trace_run1/summary.json)
- [readiness、物理范围与 generation 审计](boot_a5220e22_trace_run1/readiness_audit.json)
- 每个 rank 子目录保留 manifest、artifact SHA-256、实际执行命令、stdout 和逐帧 JSONL；汇总包含 runner/结果/执行脚本哈希，审计包含输入汇总与分析脚本哈希。
- 初次 `boot_a5220e22_baseline` 因网络沙箱拒绝 SSH，未做板端测量；`baseline_run1` 因 shell continuation 解析成独立换行参数，runner 在参数解析阶段退出，也未做推理。修复后重跑为 `baseline_run2`；失败目录保留，不计入成功样本。
