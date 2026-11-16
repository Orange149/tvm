# CPU/VTA 推理时间残差验证手册

本实验验证静态服务时间与真实流水时间之间的差值来自哪里。主实验复用当前单图、单输入的
CPU00..02 → VTA03..17 → CPU18..20；留出图为 CPU00..02 → VTA03..15 → CPU16..20。
图、权重、输入和 LLVM 参考保持封存，CPU/VTA 之间使用普通复制。

默认源是 `20260916T124859Z_single_image_audit_bfec496d`，只接受其合同 SHA 正确、
`SINGLE_IMAGE_SMOKE_PASS` 和所需文件哈希全部通过的状态。两张图的完整 live frontier 都保留，
尤其 VTA03..15 的两个输出分别传入 CPU16..20。准备 run 的当前 runtime 库身份为
`libtvm_runtime.so=9b08d159...`、`libvta.so=87243808...`；不能替换成旧 Top20 package 的库。
源 runner（92be60...）与新计时 runner 分别记录身份。源二进制仅作审计锚点；主要计时基线是同一个
新 runner 关闭新增计时开关。旧 C1 proxy 仅供带身份的诊断，不用新标签重拟合。

## 1. 冻结与离线对账

在仓库根目录执行。先选一个尚不存在的 run 名称；已存在的目录会拒绝，不覆盖旧结果。

```sh
residual_python=/home/orange/miniconda3/envs/vta-resnet/bin/python
residual_run_dir=vta/tutorials/frontend/report_out/stage_tile_cotuning/thesis_supplement_experiments/residual_validation/20260916_cpu_vta_residual_run01

$residual_python vta/tutorials/frontend/run_cpu_vta_residual_validation.py freeze --run-dir "$residual_run_dir"
$residual_python vta/tutorials/frontend/run_cpu_vta_residual_validation.py offline --run-dir "$residual_run_dir"
```

`freeze` 只核查并复制所需程序、输入、参考和部署文件；`offline` 只读分析源中已测得的串行与流水
数据。它们不会连接开发板。若另选源 run，在 `freeze` 加 `--source-dir`，随后阶段都使用已冻结合同。

关注 `contract.json`、`contract.sha256`、`analysis/board_artifact_bindings.json` 和
`analysis/OFFLINE_ZH.md`。程序路径位于 `raw/programs/`，两种拓扑的完整输入位于
`raw/component_inputs/`，全网输入位于 `raw/inputs/`，参考位于 `raw/reference/`。

## 2. 当前 boot 的六配置探索

```sh
$residual_python vta/tutorials/frontend/run_cpu_vta_residual_validation.py pilot --run-dir "$residual_run_dir" --execute-board
```

此阶段交叉编译独立 runner，链接封存的当前 runtime，核查实际 boot、频率、governor 和文件身份，
部署到新的板端目录。每个新 CommandQueue 前执行带 boot 绑定收据的上下文 reset。
期望的频率是 1066666 kHz、governor 是 userspace；实际值在预检前为 unknown，报告以实际记录为准。
第一轮包含大量正确性检查和独立流水窗口，按开发板速度可能超过 30 分钟。

| 配置 | CPU前/VTA/CPU后线程 | CPU前 mask | VTA host mask | CPU后 mask | spin count |
|---|---|---|---|---|---:|
| P0 | 4/1/4 | 0,1,2,3 | 0,1,2,3 | 0,1,2,3 | 0 |
| P1 | 4/1/1 | 0,1,2,3 | 0,1,2,3 | 0 | 0 |
| P2 | 4/1/2 | 0,1,2,3 | 0,1,2,3 | 0,1 | 0 |
| P3 | 3/1/1 | 0,1,2 | 0,1,2,3 | 0 | 0 |
| P4 | 3/1/1 | 0,1,2 | 0,1,2,3 | 3 | 0 |
| P5 | 4/1/4 | 0,1,2,3 | 0,1,2,3 | 0,1,2,3 | 300000 |

所有配置 queue depth 为 2。三个 block 的顺序固定为
`P0,P1,P2,P3,P4,P5`、`P5,P4,P3,P2,P1,P0`、`P2,P3,P4,P5,P0,P1`。

先用同一个新 runner 的 P0 比较计时开关 off/on、on/off。两个配对窗口的 II 扰动都不超过 2%
时，计时开关可进入主要结果；超出时使用关闭计时开关的主时延，并保留开关对照。
源旧 runner 的短等价比较可以补充执行，它不是必须验收门。新增服务对照须匹配同一 boot 和新
executable；需要补测的三个孤立配置是 CPU00..02 的 3 线程/mask0,1,2、CPU18..20 的 2 线程/
mask0,1，以及 CPU18..20 的 1 线程/mask3。

每配置先做串行和流水各 10 次完整输出正确性检查：整数精确，浮点使用合同中固定的容差，所有输出
必须有限且 frontier 完整。正确性失败后不继续性能测量。每个性能流水窗口共 307 帧：去掉前 5 帧
warmup 和随后 2 帧 settling，保留 300 次 completion、299 个间隔。II 由这 299 个间隔的平均值计算。
每个 block 的串行对照为 20 次，去掉前 5 次，保留 15 次。stage 服务使用 block 内均值，探索阶段
比较三个 block 均值的中位数，不把填充、排空或 stage 增量逐项加到 II。

只有预声明的 P0→P1、P0→P2、P0→P5、P3→P4 可以被选择。候选 CPU 服务变化绝对值须至少 5%，
并且同一侧 CPU 在至少一个 arm 的三个 block 都满足 busy fraction ≥90%，stage completion II
与全局 II 相差不超过 5%。选择最大绝对分数变化；平局按配置 id 字典序、stage index 排序，不优先
某个 CPU stage，不按 FPS 选择。选中结果封存在 `analysis/selected_contrast.json`。

没有合格 CPU 对照时才运行独立的 P0 queue depth 1/2/4、4/2/1 分支。任一匹配窗口中 VTA 的流水
平均 run 比串行高 5% 时，才采集两种拓扑的 VTA profile 分支。分支结果不回填 P0–P5 分数，profiler
分解不额外叠加到已包含 submit/sync/DMA/wait 的 run time。

## 3. 三个独立 boot 的所选对照

`pilot` 若选中对照，会在当前 boot 上另采集两种拓扑各两个平衡 block，顺序为 control→contrast
和 contrast→control。这些新复测窗口与先前探索窗口分别封存，不复用探索窗口作为复测样本。

用户随后对开发板进行两次真实重启，每次恢复可用环境后各执行一次：

```sh
$residual_python vta/tutorials/frontend/run_cpu_vta_residual_validation.py replicate --run-dir "$residual_run_dir" --execute-board
```

第一次对应第二个 boot，第二次对应第三个 boot。控制器核对新的 kernel boot id，重新启动 RPC
不构成独立 boot。程序不自行重启开发板。若没有 `selected_contrast.json`，不要执行复测命令；
报告本次没有选到可验证对照。

最终统计单位是三个 boot 内的配对均值，不把帧或 block 当作独立启动。机制复现门是三个 boot
效应同方向、变化绝对值的中位数至少 5%，并且配对 boot 均值的 Student t 95% CI（df=2）排除零。
主图与留出图分别报告；留出失败不能由换配置或重拟合旧 proxy 补救。

## 4. 输出、停止和解释范围

```sh
$residual_python vta/tutorials/frontend/run_cpu_vta_residual_validation.py analyze --run-dir "$residual_run_dir"
```

分析写入新的 `analysis/report_*` 目录，包含 `analysis.json`、`CONCLUSION_ZH.md` 和可导出的 SVG
stage 时间线。`timeline.jsonl` 记录动作、耗时和身份；`raw/sessions/` 保留逐帧完成、实际 stage
服务及可用的等待计时。

超时、数值失败、输入/程序漂移、输出不完整都会保存原始证据并停止，失败 run 不再采样。
不要删除失败标记、覆盖会话或原地改合同；修复后使用全新的 run 目录冻结并重跑相应协议。

统一加 34 ms 只是旧数据的校准诊断。若 CPU 并发服务明显增长、II 跟随 CPU 服务并且受线程或
独占 mask 改变，支持 CPU 争用机制；VTA 若只在特定图或并发中变慢，则需要分别检验段迁移误差
和共享资源。时延与软件 PMU/profile 本身不能把干扰全部归因到 DDR，也不能测得硬件 compute-only
时间或 DMA 与计算的物理重叠。单输入、当前 runtime、三个 boot 的结果不代表 ImageNet 准确率、
任意网络的固定开销或可迁移的新静态服务模型。
