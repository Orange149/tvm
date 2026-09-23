# R18-H1 双 Tile × 八版本：线程、同步与驻留分离对照

用户批准的执行范围：本地准备→冻结→第一启动→暂停等待用户重启→第二启动→独立分析。
本实验不是整网实验，不改硬件/驱动/生产调度，不为 W/IW 解禁双线程，不增加调优候选。

## 固定设计

卷积为 `64→64, 56×56, 3×3, stride=1, padding=1`；i8/w8/a32 与 shift/clip/cast 语义固定。
T8/T7 分别使用内层高度 8/7；两者的 tile_b/w/ci/co 都是 1/56/1/2，h_nthread=1。

| 版本 | 模式 | oc_nthread | 额外驻留 barrier | 权重驻留 |
|---|---|---:|---|---|
| A | genuine/mode0 original | 2 | 无 | 无 |
| B | mode0 original | 1 | 无 | 无 |
| C | mode1 input_stationary | 1 | 无 | 无 |
| D | mode2 weight_loop_control | 1 | 无 | 无，不能冒称权重复用 |
| E | mode4 reload | 1 | 有 | 恢复重复加载 |
| F | mode4 resident | 1 | 有 | 有 |
| G | mode5 reload | 1 | 有 | 恢复重复加载 |
| H | mode5 resident | 1 | 有 | 有 |

T8-A 使用封存的完整 TopHub 实体，adapter 单独核对等价，不额外计作第17个测量身份。
配置按实体生成，不以旧 ConfigSpace index 查表。所有失败保留，不回退、不补点。

比较方向明确为基线→测试：A→B、B→C、B→D、D→E、C→G、E→F、G→H、A→F、A→H。
同版本 T8→T7 单列为 Tile 条件性，不用于其他因素的单变量归因。

## 资格与证据

最终 TIR 主机循环静态展开，检查实际 DMA、padding、DDR 边界，UOP kernel 大小与 UOP 循环因子所覆盖的
INP/WGT/ACC 地址；命令诊断另外检查 LOAD UOP 写入及 GEMM/ALU 引用范围。未能审计的形式直接拒绝。
StorageRewrite arena 按存储域报告；运行期有效占用峰值保持 unknown，不能借用累计装载量。

D/E、C/G 同时核对去除同步/依赖后的 TIR 结构与有序数据/计算事件，保留 SRAM 偏移、范围、归约组织。
E/F、G/H 核对 UOP 循环与后处理上下文、非权重访问和显式同步安排；权重请求必须减少。
地址、arena 和 UOP 差异仍是驻留机制的一部分，不称纯搬运时间。

每候选保留三 seed FSim、最终 TIR、交叉编译身份、静态地址证据和预热后命令打印。
FSim 的 driver_run_calls 是零：本地实际提交通过 VTA_QUEUE 和指令块核对，不能当成零次执行。
板端另查真实 driver_run_calls。原始 opcode 和互斥语义分类分别计数，NOP 不重复相加。

## 调用与环境协议

三个 seed 固定为 0/20250901/20260910；输出两两差异非零。调用前填 -113，结果必须逐元素一致。
同形状候选共享一组 output→data→weight 长期缓冲，更新内容不换句柄。诊断/计时是两个独立 RPC 会话，
同一 helper、算子库和 runtime；每候选每个 seed 显式预热一次。首次调用与预热后的记录按 phase 分账。

计时 helper 在 steady_clock 边界内只调用一次 main；loopback RPC 累加回归验证无隐藏调用。
helper 与算子模块句柄整个会话保留。诊断 helper 仅开启现有 DUMP_INSN/UOP，结束后恢复 debug=0，
不重编译另一套计算主体，不使用 FORCE_SERIAL。正式会话没有指令打印。

CPU3 亲和性必须通过进程/线程检查；记录 governor/频率/负载，不自动修改调频政策。
保持同一 fresh-completion runtime、bitstream 编程记录与健康检查。DDR instruction/UOP buffer 和提交阈值
统一为 33554432 B，replay disabled，非逐候选调节。该值是命令缓冲，不是片上 SRAM 容量。
仅停止自己创建且 cwd/exe 均匹配的 RPC，遇其他 RPC/板卡占用立即停止。

## 计时与分析

八行 Williams 顺序通过标签递增生成：

```text
A B H C G D F E
B C A D H E G F
C D B E A F H G
D E C F B G A H
E F D G C H B A
F G E H D A C B
G H F A E B D C
H A G B F C E D
```

每轮 Tile 先后交替，seed 顺序对所有候选相同并逐轮轮换。平衡仅针对行内候选组，不包括组内及跨边界。
缺候选时删减固定行、重报平衡，不称完整 Williams；保留所有实际调用。
全通过时每启动384个单次计时、48次正式预热；诊断/健康另计。

先取每候选每轮三个 seed 的中位数，再算 `100*(1-test/baseline)`，最后取八轮改善率的中位数。
两启动各≥2%且各赢≥6/8轮，才为稳定改善；反向对称为稳定退化。其余不是“没有影响”，也不触发重跑。
仅正确性、超时、身份、额外调用或启动变化等有效性问题允许新会话重试，旧失败记录保留，时间不拼接。
同步控制未匹配时不做同步归因；不将各百分比相加解释历史5.43%，不推断硬件等待周期。

## 操作入口

在 `vta-resnet` 环境和仓库根目录执行，所有输出目录必须不存在：

```text
python vta/tutorials/frontend/run_vta_c3_thread_sync_ablation.py --phase prepare --output-dir NEW_PREPARE
python vta/tutorials/frontend/run_vta_c3_thread_sync_ablation.py --phase freeze --contract NEW_PREPARE/contract.json --output-dir NEW_FREEZE
python vta/tutorials/frontend/run_vta_c3_thread_sync_ablation.py --phase board --contract NEW_FREEZE/contract.json --execute-board --output-dir NEW_BOOT1
# 用户实际重启之后；boot ID 不变时拒绝
python vta/tutorials/frontend/run_vta_c3_thread_sync_ablation.py --phase board --contract NEW_FREEZE/contract.json --execute-board --previous-boot-dir NEW_BOOT1 --output-dir NEW_BOOT2
python vta/tutorials/frontend/run_vta_c3_thread_sync_ablation.py --phase analyze --contract NEW_FREEZE/contract.json --boot-dirs NEW_BOOT1 NEW_BOOT2 --output-dir NEW_ANALYSIS
```

只有一个有效启动也可运行 analyze，但输出为等待第二启动，不给稳定性结论。只读本地分析允许省略 boot-dirs。

## 本次执行记录

- 当前状态：**TWO_BOOT_COMPLETE_AND_ANALYZED**。计划内板端执行与双启动分析完成，不再补测。
- 最终使用[成功准备 run07](../07_grouped_holdout/20260922_c3_thread_sync_prepare_run07/summary.json)和
  [冻结 run04](../07_grouped_holdout/20260922_c3_thread_sync_freeze_run04/contract.json)：16/16完整资格、
  8/8控制匹配、TopHub adapter等价，固定384个正式样本。最终回归26项通过，Python语法和空白检查通过。
- [有效第一启动 run06](../07_grouped_holdout/20260922_c3_thread_sync_boot1_run06/summary.json)：boot ID
  `912d78c0-273d-47e6-b874-57d49a9663e8`，governor为userspace；16/16板端资格通过，
  48次正式预热、384/384正式计时正确，隐藏调用为零，总完整算子调用534次，CPU3亲和性持续通过。
  诊断使用9090、计时使用9091，避免前一监听socket释放状态影响独立会话。
- 用户实际重启后，第二启动boot ID为`aac37acf-5ab2-441d-b9bd-7c335d302307`，与第一启动不同；
  [bitstream/u-dma-buf恢复](../07_grouped_holdout/20260922_c3_thread_sync_boot2_restore_run01/summary.json)及
  [taskset恢复](../07_grouped_holdout/20260922_c3_thread_sync_boot2_taskset_restore_run01/summary.json)均使用相同哈希与协议。
  [有效第二启动 run01](../07_grouped_holdout/20260922_c3_thread_sync_boot2_run01/summary.json)同样16/16资格通过、
  48次正式预热、384/384正式计时正确、隐藏调用为零，总完整算子调用534次。
- [双启动最终分析](../07_grouped_holdout/20260922_c3_thread_sync_dual_boot_analysis_run01/README.md)使用且仅使用
  boot1 run06与boot2 run01；旧时间与所有失败会话均未复用。两个启动各有313/384个正式profile与诊断逻辑
  签名完全相同，71个存在运行期指令数差异，涉及全部16身份；数据请求、提交数和输出全部通过。
  差异主要表现为流水线NOP数量变化，因此对应调用不能借用诊断dump作精确指令解释；未据此重采样或删点。
- 第一启动命令此前被安全审核拒绝；用户随后明确确认目的地 `192.168.1.247` 及部署范围，已重新执行。
  [boot1 run01失败记录](../07_grouped_holdout/20260922_c3_thread_sync_boot1_run01/invalid_session.json)
  已封存：初始只读预检因缺少 `udmabuf0` 停止，墙钟2.687594 s。
  尚未部署实验库、编程FPGA或调用算子；没有 `calls.jsonl` 或编程记录，没有计时样本。
- 用户已授权部署的内容：16个算子 `.so`、`c3_thread_sync_helper.so`、已绑定哈希的
  `tvm_rpc/libtvm_runtime.so/libvta.so`，目的地为板端独立 `/var/volatile/` 实验目录；
  bitstream 使用板端已有 `/lib/firmware/vta_hpc.bit`，核验哈希后加载，不改驱动或停止外部RPC。
- 后续只读SSH核查发现，本次启动既没有 `/dev/udmabuf0`，也没有 `/lib/firmware/vta_hpc.bit`；
  `/proc/modules` 仅列出dmaproxy。FPGA状态为operating，但不能据此确认当前硬件身份。
  boot ID为 `912d78c0-273d-47e6-b874-57d49a9663e8`，未观察到外部RPC。
  内核为 `Linux version 5.4.0 (oe-user@oe-host) (gcc version 9.2.0 (GCC)) #1 SMP Wed Mar 4 14:03:26 UTC 2026`，
  与历史成功TopHub run02的preflight完整字符串一致。根文件系统为RAM rootfs，`/var/volatile`为tmpfs。
- 本地bitstream SHA256为 `7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6`，
  与历史成功记录一致；已有 `u-dma-buf.ko` SHA256为
  `4c0aa5713225cd9b29548d9c1cd6e6127aeaafcc1f7d8650d68e3a0a527042c0`，
  与既有恢复脚本要求一致。此核查不是本次模块已加载或硬件资格通过的声明。
- 用户随后授权按历史文档恢复易失依赖。[恢复记录](../07_grouped_holdout/20260922_c3_thread_sync_boot_restore_run01/summary.json)
  核验并恢复192 MiB u-dma-buf，物理起址`0x68500000`；bitstream和模块哈希与历史约束一致，不写SD。
  板端镜像没有taskset，与既有文档记录一致；从同一SDK部署AArch64 taskset至tmpfs，并以RAM-rootfs临时链接验证
  `Cpus_allowed_list: 3`。[首个验证命令转义失败](../07_grouped_holdout/20260922_c3_thread_sync_taskset_restore_run01/invalid_session.json)
  保留，[新会话验证通过](../07_grouped_holdout/20260922_c3_thread_sync_taskset_restore_run02/summary.json)。
- 双启动已完成。最终只读收尾确认实验RPC清理；易失依赖仍只存在于RAM/tmpfs，不写SD。

当前本地证据示例（全部是生成程序/FSim命令数据，不是板端时间）：

| T8版本 | 输入+权重请求 B | 命令提交数 | 累计LOAD UOP B |
|---|---:|---:|---:|
| A 双线程原方案 | 745472 | 1 | 472 |
| B 单线程原方案 | 745472 | 1 | 236 |
| D W循环控制 | 745472 | 1 | 236 |
| E W reload＋barrier | 745472 | 3 | 472 |
| F W驻留＋barrier | 524288 | 3 | 1624 |
| C 输入复用 | 501760 | 1 | 428 |
| G IW reload＋barrier | 501760 | 2 | 428 |
| H IW驻留＋barrier | 280576 | 2 | 1580 |

例如D/E的数据请求及空间匹配，新增同步确实伴随提交和UOP装载变化；E/F权重请求下降又伴随UOP装载增加。
这些结果支持所设计的程序级对照；双启动时间分类如下，仍不推断硬件等待周期。

预定主比较（正值为右侧版本更快，括号内为boot1/boot2逐轮改善率中位数）：

| Tile | A→B线程 | D→E同步 | C→G同步 | E→F驻留 | G→H驻留 | A→F净效果 | A→H净效果 |
|---|---|---|---|---|---|---|---|
| T8 | 稳定退化（-18.042/-17.700%） | 未达稳定门槛（-1.766/-2.133%） | 未达稳定门槛（-1.239/-1.211%） | 稳定改善（+6.001/+6.043%） | 稳定改善（+5.231/+5.655%） | 稳定退化（-12.790/-12.730%） | 稳定退化（-5.141/-4.831%） |
| T7 | 稳定退化（-19.033/-19.444%） | 稳定退化（-2.121/-2.184%） | 未达稳定门槛（-1.069/-1.165%） | 稳定改善（+6.348/+7.387%） | 稳定改善（+6.089/+5.919%） | 稳定退化（-13.434/-12.767%） | 稳定退化（-4.445/-5.052%） |

结论边界：关闭虚拟线程的总程序代价在两个Tile上稳定且大于当前驻留收益；驻留相对匹配reload控制本身稳定有效，
但尚不足以超过双线程原方案。同步代价仅在T7 D→E达到稳定门槛，不能概括为所有同步安排都有≥2%代价。
输入复用B→C在T8/T7也分别稳定改善6.796–7.184%和7.544–7.662%；B→D遍历变化未达到稳定方向门槛。
除E版本T8→T7稳定退化约2.2%外，其余同版本Tile条件性比较未达到双启动稳定门槛。

历史准备失败保留：

- 初步16个候选的 lowering 与最终TIR地址检查全部通过，不代表已通过完整资格。
- 首次 prepare `20260922_c3_thread_sync_prepare_run01` 保留：采集器误用 FSim 的板端 driver 计数检查提交，
  导致资格拒绝；这是 harness 问题，没有访问板卡、没有性能结果。修正后从新目录重新准备，不覆盖原记录。
- prepare run02：16个候选通过FSim与地址检查，但旧资格模块导入重新设置了LD_LIBRARY_PATH，SDK初始化拒绝。
  修复仅针对每次SDK子进程启动前的环境清理；run03重新完整资格，未拼接旧结果。
- boot1 run02因板端缺少taskset在启动RPC前停止；boot1 run03暴露首调用NOP与预热签名误比较；prepare run04
  因沙箱禁止loopback socket退出。修复后run04完整保留，不覆盖。
- boot1 run04完成16/16诊断，但timing RPC因9090释放竞态无法绑定；boot1 run05完成两会话及48次预热，
  随后暴露正式profile差异被错误当作会话失效。两者均无有效正式结果，不与run06拼接；修正分别落实为双端口和
  “保留签名差异但继续核对请求、提交与正确性”。
