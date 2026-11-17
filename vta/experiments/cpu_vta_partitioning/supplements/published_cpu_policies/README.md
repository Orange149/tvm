# 已发表 CPU 资源方法的复现与 TVM 适配

## ResNet50 迁移与同域消融（2026-11-17）

新增 [ResNet50-v2 迁移验证](resnet50_validation/README.md) 和
[五组同域消融](resnet50_ablation/README.md)，复用冻结的运行库、执行器及真实组件成本。
迁移验证为两种方法各 Top-20、40 项共同实测；后续消融为五份 Top-20 与三个基线桥接，
去重 101 项、202 个会话、4,949 次完整推理。两批测量分别报告，不跨批相减解释评分贡献。

对应入口为 `run_resnet50_validation.py`、`run_resnet50_top20_board.py` 和
`run_resnet50_ablation.py`。首次排名的已封存控制器限制、恢复步骤和外部数据依赖见各包说明。
`monitor_resnet50_status.py` 是独立只读监控工具；保留它不代表当前有后台任务运行。
Git 保存代码与协议；编译模块、成本、冻结名单和原始测量仍在本地忽略目录 `artifacts/`、
`runs/` 中，需要另行备份，不能仅凭源码提交恢复已经完成的实验数据。

## ResNet18 与 YOLO 共同实测（2026-11-12）

以 [最新共同实测](../../../../../paper/my_paper/analysis/20261112_common_tail_comparison/FINDINGS_zh.txt) 和 [研究汇总](../../../../../paper/my_paper/analysis/20261112_goal_synthesis/CONCLUSIONS_zh.txt) 为当前比较口径。两个网络共 88 个配置、176 个会话、4,312 次完整推理；输入 0，同一启动，8 帧预热、16 个完成时间戳、24 帧尾部。

| 方法 | ResNet18 静态第一名 FPS | ResNet18 前20最佳 FPS | YOLO 静态第一名 FPS | YOLO 前20最佳 FPS |
|---|---:|---:|---:|---:|
| Tarnawski 平台适配 | 7.653 | 7.880 | 4.610 | 5.068 |
| 已有最强完整模型 | 12.674 | 13.102 | 5.201 | 5.373 |
| 计算区间线程份额修正 | 12.674 | 13.102 | 5.201 | 5.373 |

最强完整模型开启线程份额、线程组服务、主机 CPU 需求和主机集中约束。历史 ResNet18 全域静态第一名的约 13.001／13.043 FPS 保留在旧批次中；最新同一配置测得 12.674 FPS。不同批次的数值不能直接相减归因。

- 共同实测控制器：[run_scope_tail_comparison.py](run_scope_tail_comparison.py)；冻结名单及部署包在 `artifacts/run_scope_tail_comparison_20261112_01/`，原始会话在 `runs/run_scope_tail_comparison_20261112_01/`。
- 已有实验查漏补缺：[离线审计](../../../../../paper/my_paper/analysis/20261112_experiment_gap_audit/REPORT_zh.txt)。该报告的下一步是待决定的实验设计。
- 隔离／流水误差：[当前时间线](../../../../../paper/my_paper/analysis/20261112_current_pipeline_timeline/)、[主机与锁回放](../../../../../paper/my_paper/analysis/20261112_current_host_lock_replay/)。诊断性回放与完整 DP 排名分别保留。
- 清理与恢复：[2026-11-12 清理记录](../../docs/reproduction_and_cleanup_20261112.txt)。历史控制器仍被冻结来源和实验依赖引用，保留原路径；删除的是经逐文件核验的重复导出和无用中间文件。

## 历史方法与批次记录

以下按原批次保留方法和观察结果。各节中的“最新”“当前”、窗口和 FPS 均指该节对应批次，不替代上面的共同实测结果。

本目录先核对原文、原公式和原代码，再把有明确出处的机制接到项目已有的局部 profile＋静态 DP。CPU 资源策略是采用的既有方法，不作为项目新创新点。原方法参考命令均为本地操作；新的精简试跑仅在显式传入 `board ... --execute-board` 时连接开发板。

## 逐线程核时与主机放置：最新ResNet18闭环

复用775项真实组件成本中的隔离线程份额，修正逐核负载，再联合已有主机／线程组规划约束。
两种加权模型各通过完整966240306组合与独立枚举；六项匹配诊断后，验证联合版的冻结名单。
没有新增profile、改运行库或重启开发板。最终58项、116会话、1682次完整推理已完成并校验归档。

|名单|一次预算FPS|五次预算FPS|二十次预算FPS|
|---|---:|---:|---:|
|原均分核时静态模型|11.310|11.819|11.869|
|逐线程份额＋资源包络，严格排序|12.957|12.987|13.373|
|新评分按资源行为筛选|12.957|13.218|13.218|

严格排序二十次预算提高12.67%，静态第一名提高14.56%；行为筛选五次预算更好，二十次预算比新严格排序低约1.16%。
严格新名单20项全部同分，最佳的第16位是身份并列次序；[同分排序指标](runs/resnet18_thread_work_combined_top20_01/analysis/ranking_with_ties.csv)另行报告。
不同helper负载不可互换：固定计算配置，主机核心1／2／3分别约10.60／12.81／10.61 FPS，慢化在相应计算阶段间转移。
结论只支持当前ResNet18单启动观察；并发内部等待分解及其他网络通用性仍未完成。

[完整证据与结论](../../../../../paper/my_paper/analysis/20261108_thread_work/README.md) ·
[方法、公式与开发消融](resource_envelope/THREAD_WORK_zh.md) ·
[最终实测](runs/resnet18_thread_work_combined_top20_01/analysis/REPORT_zh.md)

## 资源包络及资源行为筛选：当前结果

新增独立入口`run_envelope_top20.py`、[公式与依据](resource_envelope/DESIGN_zh.md)及[操作说明](resource_envelope/OPERATIONS_zh.md)。
保留完整966240306个资源分配组合，额外考虑主机放置不确定性与CPU线程组累计服务，独立枚举核验通过；没有新增profile或修改运行库。

三份20项名单并集57项，共114会话、1653次完整推理，已验收归档：

|名单|一次预算FPS|五次预算FPS|二十次预算FPS|
|---|---:|---:|---:|
|原静态模型严格前20项|11.450|12.025|12.025|
|新资源包络严格前20项|10.456|10.873|11.240|
|新包络按资源行为筛选20项|10.456|12.279|12.818|

**新包络严格排序退化，不作为已验证有效的全局评分采用。** 行为筛选二十次预算提高约6.59%，但第一名退化；它只支持预算压缩的局部价值。
六项匹配诊断中，固定历史计算分配、主机核心2且同组串行仍达到13.133／13.161 FPS，未被保送到任何新Top-20。
当前继续核查已有隔离逐线程份额对线程预算和主机位置联合选择的影响，尚未完成总目标。

[本轮完整证据及结论](../../../../../paper/my_paper/analysis/20261108_resource_envelope/README.md) · [同批实测](runs/resnet18_resource_envelope_top20_01/analysis/REPORT_zh.md)

## AutoTVM覆盖筛选：新Top-20共同实测

新增独立入口`run_diverse_top20.py`和[操作说明](selection_top20/README.md)，批次为`resnet18_autotvm_diverse_top20_01`。保留原完整DP与775项组件成本、87项DDR记录，使用未经修改的AutoTVM `submodular_pick`函数；原静态前五名保留，其余名额覆盖九种上下文、三类CPU执行方式与切图／资源特征。分类内使用1.15倍预测值限制，不采用全局阈值，不引入历史特殊配置，不读取旧FPS决定名单。

旧两份20项名单仍形成28项并集；新三份名单并集为40项，全部冻结后在同一启动中共同测量。每项1帧资格检查＋28帧性能窗口，共80个会话、1160次完整DNN推理，无新增组件或CPU共同帧流标定。运行库和正式流水执行器均不重编；等待机制只做实际二进制与源码的只读核验。新协议保存在独立包内部，旧批次身份与结果保持可校验。

本批已`COMPLETE`：50项本地检查通过，自包含部署包迁移及旧批次校验通过；40个方案、80个会话、1160次完整推理全部完成，1194份原始文件逐项重算哈希与摘要通过，部署文件复核和同步完成。

| 名单 | 一次预算FPS | 五次预算FPS | 二十次预算FPS |
|---|---:|---:|---:|
| 原纯静态20项 | 11.626 | 11.719 | 12.177 |
| 原差异化20项 | 11.626 | 11.719 | 12.177 |
| AutoTVM覆盖筛选20项 | 11.626 | 11.719 | 12.431 |

二十次预算本轮提高约2.08%；一次、五次预算共用前五名，相同结果来自设计。新名单第17项为本池最佳，由冻结的贪心覆盖步骤选入：CPU0→VTA1→CPU2→VTA3–4→CPU5–9→VTA10–19→CPU20，四个CPU阶段分别2／2／2／4线程，核心集合`{0,2}/{2,3}/{1,3}/{0,1,2,3}`；VTA主机固定核心2、控制线程共享四核、门控关闭。其预测／实测完成间隔为79.452／80.446 ms，未在两份旧20项名单内。

测量及回收约470.6秒；比旧28项并集多测12项、348次完整推理，约增加142.7秒。本次没有新增组件标定或编译，实际部署负载235.2 MiB。源码与二进制核验发现了等待路径，但未证明它是吞吐差距原因。结果属于单图、单启动短窗口观察；这次同时改变分类阈值、覆盖种子与贪心筛选，不能把收益单独归因于其中某一步，不自动追加确认或扩大标定。

[完整结果与原始证据](../../../../../paper/my_paper/analysis/20261108_autotvm_diverse_top20/REPORT_zh.md) · [方法与公式](../../../../../paper/my_paper/analysis/20261108_autotvm_diverse_top20/METHOD_zh.md) · [40项实测](../../../../../paper/my_paper/analysis/20261108_autotvm_diverse_top20/configurations.csv) · [三份名单](../../../../../paper/my_paper/analysis/20261108_autotvm_diverse_top20/three_top20_lists.json) · [验收](../../../../../paper/my_paper/analysis/20261108_autotvm_diverse_top20/verification.json)

## 完整DP候选CPU重排与Top-20验证

`run_frame_stream_top20.py`与独立`frame_stream_top20`包建立批次`resnet18_dp_frame_stream_top20_01`。完整域仍为21单元、172个阶段、最多3个VTA岛、9种上下文、全部合法核组及开放并发／同组串行。原DP公式不变，原始成本和旧实验继续保留。

准备阶段核验775项隔离成本及87项VTA流量记录，重新运行完整DP和独立枚举。纯静态20项与既有差异化20项必须与同成本旧结果一致；两份名单的并集最多30项，不附加历史优秀配置。当前重跑覆盖4,623种切图和966,240,306个增强组合，冻结并集28项。

只对这个有界候选池标定CPU共同帧流：每阶段一次资格调用，再共同处理14帧；2帧预热、8个完成时间戳、4帧尾部。支持2—4个CPU阶段、请求的控制线程位置，以及覆盖set/run/get的同组串行。每帧全部CPU工作完成时间为各阶段结束时间的最大值，整体间隔取`ceil((C9−C2)/7)`整数纳秒。新预测只取`max(原容量预测, CPU整体间隔)`，不叠加阶段慢化时间或核时。

CPU工作、frontier输入、线程映射、控制位置、门控和执行器均等价时才复用同一个CPU记录；CPU微基准不运行VTA，主机位置的差别仍保留在原容量模型中。新Top-20及哈希在任何完整流水计时前冻结。随后测量整个候选池，比较原纯静态、原差异化和重排名单在1／5／20次完整流水预算下的最佳FPS；额外CPU标定成本另列，不称总调优成本相同。

```bash
python run_frame_stream_top20.py prepare
python run_frame_stream_top20.py validate --bundle artifacts/bundles/<bundle-id>
python run_frame_stream_top20.py board run --bundle artifacts/bundles/<bundle-id> --execute-board
python run_frame_stream_top20.py board resume --run-dir runs/resnet18_dp_frame_stream_top20_01 --execute-board
python run_frame_stream_top20.py rank --stage static --run-dir runs/resnet18_dp_frame_stream_top20_01
python run_frame_stream_top20.py rank --stage frame-stream --run-dir runs/resnet18_dp_frame_stream_top20_01
python run_frame_stream_top20.py status --run-dir runs/resnet18_dp_frame_stream_top20_01
python run_frame_stream_top20.py analyze --run-dir runs/resnet18_dp_frame_stream_top20_01
```

完整流水使用原冻结执行器，先单帧资格检查，再运行8帧预热＋16个完成时间戳＋4帧尾部，16个时间戳对应15个间隔。输入0、普通复制、队列深度2；一次启动，不测锚点、不按漂移拒收、不自动延长或追加测量。错误输出、位置、哈希、启动身份及缺帧仍拒收，断线后重放原始凭据，不重复执行已验收任务。

该方法是完整DP后的候选标定重排，不把组合成本当成DP局部转移成本，也不保证候选池外的全域最优。结果属于单启动短窗口观察；是否值得采用，要同时看选点和额外准备成本，不自动开启全域CPU共同帧流profile。

### 本轮已完成结果

批次状态`COMPLETE`：28项CPU共同帧流、28项单帧流水资格检查、28项流水性能，共84个会话；实际1,365次CPU阶段调用（91次资格＋1,274次帧任务）和812次完整DNN推理。1,152份原始文件逐项重算哈希通过，运行后部署文件校验及同步完成。120项本地测试通过，其中28项为新增测试；QEMU实际执行、错误输出取消、自包含迁移及旧批次校验通过。

| 冻结名单 | 一次预算FPS | 五次预算最佳FPS | 二十次预算最佳FPS |
|---|---:|---:|---:|
| 原纯静态20项 | 12.046 | 12.046 | 12.061 |
| 原差异化20项 | 12.046 | 12.046 | 12.061 |
| CPU共同帧流重排20项 | 11.719 | 12.061 | 12.061 |

新模型池内平均绝对预测误差从17.502降至12.780 ms，但静态第一名吞吐下降约2.72%；五次预算仅提高约0.12%，二十次预算没有额外收益。按事前条件，五次预算数值改善且二十次预算不退化，因而分析程序标记为正向观察；这一弱条件命中不能证明新增标定具有明显实际价值，也不能解释为统计显著或等效。

原第一名与新第一名的切图相同，均为CPU0–2→VTA3–4→CPU5–7→VTA8–19→CPU20，前两段均四线程共享四核。原方案CPU20为单线程核心0，新第一名CPU20为三线程核心`{1,2,3}`；本次新方案反而较慢，不据此认定具体慢化机制。池内最佳切图为CPU0–2→VTA3–4→CPU5–7→VTA8–18→CPU19–20，CPU线程数4＋4＋1，主机和控制共享四核、门控关闭。它在原纯静态名单第16位、差异化名单第11位、新名单第3位；三份20项名单都包含它。

额外CPU标定与回收约157.4秒，全部流水资格／性能与回收约331.3秒，合计约489.2秒，不含部署预检。静态准备、组件凭据核验、DP／独立枚举及重放合计约818.8秒；新执行器封装准备约18.8秒。板端有效负载184.0 MiB，仅编译一个CPU微基准执行器，阶段模块、运行库和正式流水执行器未重编。

本轮未证明值得把共同帧流作为默认Top-20筛选步骤。原DP、差异化选点及历史结果保留，不追加测量或扩展全域标定。此次只评价固定28项池，不能判断池外历史优秀配置的本轮表现。九种上下文参与了静态搜索，但只有入池的上下文获得实测；其余上下文不据此判断实际性能较差。

[自动结果](runs/resnet18_dp_frame_stream_top20_01/analysis/REPORT_zh.md) · [28项配置与实测](../../../../../paper/my_paper/analysis/20261108_dp_frame_stream_top20/configurations.csv) · [预算对照](../../../../../paper/my_paper/analysis/20261108_dp_frame_stream_top20/selection_budgets.csv) · [上下文覆盖](../../../../../paper/my_paper/analysis/20261108_dp_frame_stream_top20/context_coverage.csv) · [CPU逐阶段服务／等待](../../../../../paper/my_paper/analysis/20261108_dp_frame_stream_top20/cpu_stage_service_wait.csv) · [冻结顺序与验收](../../../../../paper/my_paper/analysis/20261108_dp_frame_stream_top20/verification.json) · [原始凭据索引](../../../../../paper/my_paper/analysis/20261108_dp_frame_stream_top20/evidence_index.json)

![同预算选点与预测误差](../../../../../paper/my_paper/analysis/20261108_dp_frame_stream_top20/selection_and_error.png)

## CPU共同帧流：8会话、300次阶段调用

`run_frame_stream.py`与独立`frame_stream`包验证“每帧每个CPU子图各运行一次”的资源节拍，批次为`resnet18_cpu_frame_stream_01`。它不修改旧`cpu_policies`源码身份，不执行VTA推理，也不模拟原图阶段依赖或VTA等待。

沿用两个冻结切图及四种配置：4＋4完全重叠、2＋2完全重叠、2＋2分开、3＋1分开。第一种切图的CPU20固定单线程核心0也参加每帧工作；CPU17–20的两个frontier输入完整保留。计算worker逐线程固定，caller取核组最小编号；生产与主线程共享四核，CPU门控关闭。

共同生产端按原图CPU阶段顺序投递帧号0–13，每阶段独立深度2队列。队列满则生产者等待，队列空则阶段等待；没有逐帧屏障、没有额外背景循环。每阶段在自己的线程池中先做一次完整输出资格检查，再处理2帧预热、8帧计入、4帧尾部，结束后核对完整输出。两个切图分别有3／2个CPU阶段，合计`4×(3+2)×15=300`次阶段调用，不是300次完整DNN推理。

组完成时间是同帧所有CPU阶段结束时间的最大值，CPU完成间隔为`(C9−C2)/7`。新分数只取`max(原容量分数, CPU组完成间隔)`，整数纳秒仅最后一次上取整；不修改原核时、VTA或DDR公式。原15项隔离成本和8项完整流水结果按原始凭据核验引用，评分模块不读取FPS参考。先冻结新排名，再打开独立分析参考文件，按预算一次／两次报告池内选点损失，误差与相关性作为辅助指标。

```bash
python run_frame_stream.py prepare
python run_frame_stream.py validate --bundle artifacts/bundles/<bundle-id>
python run_frame_stream.py board run --bundle artifacts/bundles/<bundle-id> --execute-board
python run_frame_stream.py board resume --run-dir runs/resnet18_cpu_frame_stream_01 --execute-board
python run_frame_stream.py status --run-dir runs/resnet18_cpu_frame_stream_01
python run_frame_stream.py analyze --run-dir runs/resnet18_cpu_frame_stream_01
```

一次启动、输入0；不测锚点，不按漂移拒收、加测或重启。断线后按任务哈希、boot ID及持久化原始凭据续跑，已验收任务不重测。该组成本依赖全部CPU子图和放置组合，只用于已知配置开发回放；第一种切图还补入了CPU20，不把变化全部归于限量，也不直接推广到完整DP或其他DNN。

### 2026-11-08已完成结果

批次`resnet18_cpu_frame_stream_01`状态为`COMPLETE`：8个会话、20次资格调用及280次帧任务，共300次CPU阶段调用。96份已验收原始文件逐项重算哈希通过；每阶段恰好完成帧0–13，没有补跑、额外背景调用或新的完整DNN推理。8个会话从首项启动到末项验收约59.25秒，包含执行和数据回收，不包含本地准备及板端预检。

| 切图 | 原模型／共同帧流模型的平均绝对误差 | 共同帧流选出的CPU分配 | 该配置上一批完整流水FPS |
|---|---:|---|---:|
| CPU0–2→VTA3–4→CPU5–7→VTA8–19→CPU20 | 11.523／4.587 ms | 前段3线程`{0,1,2}`、中段1线程`{3}`；CPU20固定1线程核心0 | 11.798 |
| CPU0–2→VTA3–16→CPU17–20 | 9.140／5.699 ms | 前段2线程`{0,1}`、后段2线程`{2,3}` | 10.008 |

新约束在两个四项池中都把原实测最佳配置排在第一名，预算一次和两次的池内完成间隔损失均为0；原容量模型的一次预算损失分别为4.75%和3.73%，两次预算也均为0。三种方法的前两名都覆盖池内最佳，因此新约束此次改善的是一次预算选点，不能声称两次预算仍有收益。

与上一轮持续饱和CPU共跑相比，共同帧流的选点相同，没有额外选点吞吐收益；平均误差从20.096／8.923 ms降到4.587／5.699 ms。新评分的排序与这两个池的旧实测排序均一致。结果满足事前规定的“值得开展未见配置验证”条件，仍是已知配置开发回放。

三阶段切图的前两名新预测为94.915／95.061 ms，仅差0.146 ms，单次短窗口不能证明排序稳定。第一种切图比旧CPU饱和测试多执行了CPU20，也不能把全部改善归于限量。CPU微基准和旧完整流水来自不同boot；新微基准未包含VTA主机并发、真实图依赖和VTA访存。此处FPS均为上一批结果引用，没有新测或独立确认吞吐提升，不把组合成本直接写进单阶段DP表。

新自包含包为`452424b19638b5b8bcdb8456`，只编译新增执行器；阶段模块和运行库未重编。24项新测试及相关回归共142项通过，QEMU完成CPU20及双输入阶段执行、异常取消验证，迁移后的部署包和旧批次校验通过。额外禁止读取旧FPS文件后重放评分，冻结的预测哈希保持不变。

[完整结果](runs/resnet18_cpu_frame_stream_01/analysis/REPORT_zh.md) · [8项预测与既有实测](../../../../../paper/my_paper/analysis/20261108_cpu_frame_stream/predictions.csv) · [逐阶段服务／等待表](../../../../../paper/my_paper/analysis/20261108_cpu_frame_stream/stage_service_wait.csv) · [预算比较](../../../../../paper/my_paper/analysis/20261108_cpu_frame_stream/selection_budgets.csv) · [原始凭据索引](../../../../../paper/my_paper/analysis/20261108_cpu_frame_stream/evidence_index.json) · [验收与计数](../../../../../paper/my_paper/analysis/20261108_cpu_frame_stream/verification.json) · [准备及测量成本](../../../../../paper/my_paper/analysis/20261108_cpu_frame_stream/preparation_and_measurement_cost.json)

![三个预测版本与上一批流水完成间隔](../../../../../paper/my_paper/analysis/20261108_cpu_frame_stream/predictions_vs_reference.png)

## 受控并发profile与跨切图验证

`run_contention.py`提供独立批次`resnet18_controlled_contention_02`。固定两个切图，各测四线程完全重叠、两线程完全重叠、两线程分开、三线程加单线程分开；全部为逐线程固定，主机／控制共享四核，CPU门控关闭。验证切图CPU17–20的两个frontier输入均保留。

首批`resnet18_controlled_contention_01`在单VTA持续回放时因协调锁阻塞超时，保留26项已验收会话和失败凭据；修复为独立设备轮转锁后，第二批重新采集全部数据，不混用两个执行器的成本。

本轮48个会话：15项精确物理位置的隔离成本、20项受控并发、4项额外线程探针、8项完整流水和1项原执行器桥接。每个组件会话在同一阶段线程池中先做输出与实际线程映射资格检查，再预热2次并计入8次。较快CPU组件持续施加干扰，额外调用单独记录；VTA各岛由各自主机线程按顺序使用单个加速器。全流程只用输入0，可在同一次启动完成；不按时间波动自动拒收、加测或重启。

两种切图的原容量模型、增加CPU共跑服务约束、增加CPU和VTA共跑服务约束，均在任何本轮完整流水计时前冻结。新约束仅与原分数取最大值；并发进程CPU时钟不分摊给各阶段，不用流水结果拟合系数。正式组件与流水关闭线程探针，4个独立诊断会话保留读取包络与逐线程核时／等待，未知计数不补零。

```bash
python run_contention.py prepare
python run_contention.py validate --bundle artifacts/bundles/<bundle-id>
python run_contention.py board run --bundle artifacts/bundles/<bundle-id> --execute-board
python run_contention.py board resume --run-dir runs/resnet18_controlled_contention_02 --execute-board
python run_contention.py status --run-dir runs/resnet18_controlled_contention_02
python run_contention.py analyze --run-dir runs/resnet18_controlled_contention_02
```

SSH断开后使用相同启动续跑；已有原始数据通过哈希、任务身份、boot ID和原始解析重放后复用，不重新执行已验收窗口。错误输出、绑核或执行失败保留凭据并停止，不自动重测。代码更改后需使用部署包内`sources/contention_controller/run_contention.py`恢复原批次。分析提供逐阶段隔离／共跑／流水耗时、三个模型误差和选点、探针扰动，以及全域并发签名数量估算；本轮不修改原完整DP，也不自动补齐全域并发profile。

第二批首个流水会话已成功执行，但主机校验曾将首帧错误套用28帧声明。`validation_repair/`保留修复前状态、适配器源码及哈希、原始文件重放结果；仅将首帧校验声明改为1帧，执行器、任务、成本和事前预测均不变，已有会话不重新执行。当前统一入口会自动使用该批次归档的适配器查询、续跑和分析；也可直接运行`python runs/resnet18_controlled_contention_02/validation_repair/compat.py status --run-dir runs/resnet18_controlled_contention_02`。

### 本轮已完成的结果

第二批状态`COMPLETE`：48个会话、8个正式流水配置，615份已验收原始文件哈希通过；实际完成252次完整推理（含桥接）及1,159次独立组件调用（包含资格、预热和额外背景调用），两种调用不混为同一个计数。相关测试118项通过。成本及预测在首个完整流水会话之前冻结，完整DP保持不变。

| 切图 | 原容量模型选点FPS | 增加CPU共跑约束选点FPS | 吞吐变化 | 再加入VTA持续干扰的选点FPS |
|---|---:|---:|---:|---:|
| CPU0–2→VTA3–4→CPU5–7→VTA8–19→CPU20 | 11.263 | 11.798 | +4.75% | 11.798 |
| CPU0–2→VTA3–16→CPU17–20 | 9.648 | 10.008 | +3.73% | 8.267 |

保持两个阶段均为两线程，仅把第二阶段从`{0,1}`移到`{2,3}`，两种切图的FPS分别从7.460升至9.927、从6.345升至10.008。核心重叠确实会造成性能损失，但原容量模型已经较准确地预测了这一匹配干预的完成间隔差距：预测34.480／57.694ms，实测33.316／57.679ms。不能把全部误差都归为“原模型没有考虑竞争”。

新CPU共跑约束虽改善了两个静态选点，第一种切图的平均绝对预测误差却由11.523增至20.096ms；另一种由9.140降至8.923ms。加入持续VTA干扰后，第二种切图的选点比原模型慢14.32%。两项修正均未满足“两种切图误差均下降、选点不退化、匹配放置方向正确”的全部继续条件，因此不接入完整DP。

持续共跑属于压力测试，不能当作完整流水的等价成本。例如第一种切图四线程共跑时，前段完成8次，中段实际完成20次；真实流水每个阶段对每帧只执行一次。这种背景工作强度差异是过度惩罚的解释线索，不作为已分离的唯一原因。另一方面，三线程／单线程在第一种切图的隔离阶段约为71.95／70.76ms，而第二种切图变为71.95／120.13ms：分配必须随子图负载改变，不能固定采用3＋1。

修订后的自包含准备包为`a32daf05cfb604dcf12cff24`，仅修正控制器校验和交付入口，执行器、运行库、模块、位流及输入／参考与实测包`b5eabe34bdc6828937976136`逐文件一致，没有追加板测。原始状态、校验修订和失败批次均保留。

[完整中文结果](runs/resnet18_controlled_contention_02/analysis/REPORT_zh.md) · [逐阶段对照](runs/resnet18_controlled_contention_02/analysis/stage_comparison.csv) · [轻量数据归档](../../../../../paper/my_paper/analysis/20261108_controlled_contention/analysis.json) · [冻结与哈希验收](../../../../../paper/my_paper/analysis/20261108_controlled_contention/verification.json)

![事前静态第一名的实测吞吐](../../../../../paper/my_paper/analysis/20261108_controlled_contention/selected_throughput.png)

## 最新验证：逐核成本与调用线程位置

`run_core_model.py`及[操作说明](docs/CORE_MODEL_VALIDATION_zh.md)提供固定切图下的四组对照。`resnet18_core_model_validation_01`已完成43项任务、439次推理：11项共同隔离成本、225／1024项资源域独立核验、去重后的10项流水测量，113项本地测试及运行后验收通过。

逐线程核时将本池平均预测误差从15.67%降至13.60%，但未改善排序和静态第一名吞吐；调用位置四格也未观察到收益。本池最佳为不重叠3＋1配置，12.633 FPS。完整结果和证据见[2026-11-07小范围验证归档](../../../../../paper/my_paper/analysis/20261107_core_model_validation/README.md)。原完整DP、成本及候选名单保持冻结，不将局部精度改善当作已验证的全域优化。

## 第二轮诊断：CPU计算区间的逐线程核时与等待

`run_run_scope.py`提供独立低扰动诊断入口，[操作说明](docs/RUN_SCOPE_DIAGNOSIS_zh.md)列出固定任务及计数口径。`resnet18_prediction_gap_02`已完成22项任务、310次推理，71项本地测试、迁移及运行后校验通过。只编译新的可选探针执行器，原运行库、模块、公式和排名保持冻结。

静态第一名中，前段四线程run墙钟增加约62%，CPU运行核时增加约9%；中段墙钟增加约129%，CPU核时增加约7%。两个调用线程均在核心0，实际run范围读到了明显调度等待和线程份额不均。详细证据、探针扰动与解释限制见[低扰动板端诊断归档](../../../../../paper/my_paper/analysis/20261107_run_scope_diagnosis/README.md)。不把流水观测值回填为事前成本。

## 首轮诊断：隔离成本与流水阶段耗时的差距

`run_diagnosis.py`提供独立的小规模诊断入口；[操作说明](docs/prediction_gap_diagnosis.md)记录任务、探针及恢复语义。批次`resnet18_prediction_gap_01`已完成20项任务、299次推理，32项诊断测试和35项原运行／恢复回归测试通过。复用八个冻结模块和原执行器，未修改原成本、公式、DP或Top-20。

四项同启动隔离标定与旧成本的差异均不足0.8%；静态第一名的两个四线程CPU阶段进入流水后，服务时间分别增加约69%和105%，增量主要在run调用内部。详细探针采集到了调度等待，但自身使完成间隔增加约7%–13%，不能据此直接分摊普通会话的损失。完整结果、原始证据索引及未知项见[2026-11-07板端诊断记录](../../../../../paper/my_paper/analysis/20261107_prediction_gap_board/README.md)。

## 冻结基准：九种资源上下文与差异化20项

当前批次为 `resnet18_open_cpu_resource_02`，统一入口是 `run_open_resources.py`，协议位于 `configs/open_resources/protocol.json`。完整操作及成本近似见 [操作手册](docs/OPEN_RESOURCES_RUNBOOK_zh.md)。下面的 `run.py` 命令及两环境协议保留用于原试跑。

本轮已完成：核验并引用775项组件成本，采集87项VTA流量记录，实测去重后的40项流水配置，状态为 `COMPLETE`。190项本地测试、全域DP/独立枚举及运行后文件校验均通过。方法、实际公式、来源身份、完整结果和待查假设集中记录在 [2026-11-07方法与实验归档](../../../../../paper/my_paper/analysis/20261107_cpu_resource_model/README.md)，附40项结果及340项CPU线程曲线。

本轮纯静态20项最佳12.157 FPS，差异化20项最佳12.438 FPS；单列历史参照13.235 FPS仍未进入两份名单。归档保留这一漏选及预测误差，后续研究从这份冻结基准分析理论与实测差距，不覆盖本批排名或原始数据。

新批次从历史证据提炼子图线程收益、复用核组累计负载平衡、主机避开计算热点等规律，并搜索九种明确环境：四核共享；主机固定任一单核但仍与计算共享；真正为主机及生产／收集保留任一单核。支持任意CPU核心集合、开放并发，以及兼容核组上的可选同组串行。所有VTA岛累计到一个物理加速器。

第一批已验收261项新增主机位置成本。第二批核验并引用原514项及这261项，复用全部775项成本，只补87项极短流量任务；不重新编译172个模块。修复首批发现的浮点并列误差后，评分采用守恒的整数纳秒会计，原始profile不改；流量参数同时通过真实执行器解析与板端两帧诊断。首批失败记录保留，已验收成本不重测。输出纯静态20和“10个静态优选＋5个不同切图＋5个不同资源策略”，加对照和历史参照去重后最多43个流水配置。历史参照单列，不占候选名额。输入0、短窗口、一轮，不按波动自动拒收或加测。CPU同核数置换、逐核核时分配和历史DDR带宽参数的近似均明确记录。

```bash
python run_open_resources.py prepare --calibration-source runs/resnet18_open_cpu_resource_01
python run_open_resources.py validate --bundle artifacts/bundles/<bundle-id>
python run_open_resources.py board run --bundle artifacts/bundles/<bundle-id> --execute-board
python run_open_resources.py board resume --run-dir runs/resnet18_open_cpu_resource_02 --execute-board
python run_open_resources.py status --run-dir runs/resnet18_open_cpu_resource_02
python run_open_resources.py analyze --run-dir runs/resnet18_open_cpu_resource_02
```

准备命令要求完整域DP与独立枚举已验收；上板排名只读取隔离组件成本和静态模型。各环境未进入实测池时，报告其静态覆盖，不判断其实际性能。原 `run.py`、旧批次、部署包和原始凭据不修改。

## 单图、短窗口的完整 DP 试跑

`configs/pilot.json` 冻结本轮协议。两种资源方案均由三个方法搜索：四核计算、主机及控制共享四核；或三核计算、核心3承担主机及控制。核心编号没有速度优先含义。沿用共同量化图、21计算单元、最多三个VTA岛、172个冻结模块和固定绑定执行器，准备过程不编译新切图或修改旧批次。

- 全过程仅输入0。组件预热2帧、计入8帧；流水预热8帧、保留16个完成时间戳、尾部4帧。16个时间戳对应15个完成间隔，整体进程CPU核时也取同一首尾窗口。
- 统一新测514项隔离成本：CPU85×4＝340，VTA87×两种主机位置＝174。旧成本不混写为本轮测量。CPU其他同核数mask为同构核置换近似，并非每个集合都已实测。
- 每方法、每资源方案的完整DP Top20均由独立枚举核对，再合并为该方法总Top20。增强20＋单线程映射5＋原前缀模型5去重后最多30个流水配置，各测一轮；排名在任何流水计时前冻结。
- 不测锚点，不按5%波动拒收，不自动延长窗口、增加轮次或重启。资格输出、文件哈希、实际绑核、帧完整性和boot身份仍需有效。连接中断可在同一boot续跑；跨boot不能混入这个单启动批次。
- 控制线程位置真实执行，但静态模型没有其独立核时输入。CPU隔离表共用不代表两种并发流水竞争完全相同；原前缀公式及其门控关闭语义保留。结果只描述本轮单图、单启动短窗口，报告波动和排名误差。

```bash
# 本地准备、测试、封装；没有SSH
python run.py prepare
python run.py validate --bundle artifacts/bundles/<bundle-id>

# 预检和测量可用同一次启动；不自动重启
python run.py board run --bundle artifacts/bundles/<bundle-id> --execute-board
python run.py board resume --run-dir runs/resnet18_published_policy_dp_pilot_01 --execute-board
python run.py status --run-dir runs/resnet18_published_policy_dp_pilot_01
python run.py analyze --run-dir runs/resnet18_published_policy_dp_pilot_01
```

新协议使用独立调度与凭据校验路径，不恢复已因稳定性停止的旧主批次。正式组件与流水均固定为同一个执行器；VTA保持真实的主机affinity语义，执行器选择不由CPU绑核模式推断。包内保留源代码快照，修改代码后使用原包的 `sources/published_cpu_policies/run.py` 继续旧批次。

## 已核实的原方法

| 方法 | 原问题和输入 | 核分配／绑定的实际含义 | 本项目的使用边界 |
|---|---|---|---|
| Pipe-it | 每层、每种核类型／核数的时间矩阵；算法1–3平衡层分配并合并阶段 | 作者运行时使用单核固定 workers、互不重叠计算组、组互斥；图像 caller 跨组迁移 | 采用执行资源契约，通过 TVM 接口实现；项目外层 DP 和融合阶段成本继续保留 |
| Packrat | 同一模型的 `L(线程数, 子批量)`；DP 联合选择副本、线程和批量 | 实例分配固定且不重叠的核组；论文服务栈声明 worker pinning，公开 microbenchmark 可核验组级 affinity | 原问题优化器单列复现。不同 CPU 子图不能伪装成同模型副本；batch=1 也不能拆成多个正批量副本 |
| Flexi-BOPI | 阶段资源类型／核数、资源顺序和切点；完整流水实测驱动 HGP/BO | 同簇具体核心编号被视为等价；没有披露作者逐 TID 绑核调用 | 保留可变阶段核数和配置表示；不引入 BO，不声称完整复现其优化器 |

主要来源：

- [Pipe-it 作者页面与 Code 链接](https://staff.fnwi.uva.nl/a.pathania/publication/tcad-19/)，[固定提交运行时](https://github.com/Elliezza/CLCode/tree/ffa5d82c188398805dcbdfc50ffb7b23a41c57cf/multi_split)。原文算法1–2在打印第9页，算法3在第10页，式9／11／12给出阶段成本、吞吐目标和合并条件。Table VI 也使用实际测量层时间运行其 DSE。
- [Packrat 原论文](https://proceedings.mlr.press/v267/bhardwaj25a.html)，[固定提交优化器](https://github.com/msr-fiddle/packrat/blob/18dc6c77eff5d5b402b2b895d376f0977d5a0dd0/utils/optimizer.py#L135)。§3.2–3.4 给出 profile、DP、固定资源组。
- [Flexi-BOPI DOI](https://doi.org/10.1016/j.ins.2024.120984)，本地 `paper/VTA/Flexi-BOPI.pdf` 第7–10页。作者代码截至本次查询未找到，查询范围与限制保存在审计文件，不能据此断言不存在。
- [官方 ACL v21.02 绑定实现](https://github.com/ARM-software/ComputeLibrary/blob/7dcb9fadb98cad05fca72de3273311d570d98b4e/src/runtime/CPP/CPPScheduler.cpp#L249)。它是 Flexi-BOPI 所用框架的能力证据，不是作者实际调用方式的证据。

详细页码、代码位置、提交、检索范围及适配差异见 `docs/*_audit.json`。

## 原公式与已实现模块

`cpu_policies/pipeit.py` 独立实现论文算法1、2及式9的层成本加和。保留原算法严格不等式与后缀移动，未用全局 minimax 枚举替换启发式。算法3的打印伪代码与文字没有完整规定列表修改后的遍历行为；实现强制指定解释变体，当前仅作为参考，不能用来声称复现作者全部 DSE 结果。CPU-only、单核类型的算例不是 CPU–FPGA 端到端验证。

`cpu_policies/packrat.py` 独立实现原数学问题：

\[
\min \max_j L(t_j,b_j),\quad \sum_jt_j\le T,\quad\sum_jb_j=B.
\]

\[
F(t,b)=\min_{t',b'}\max\{F(t-t',b-b'),L(t',b')\}.
\]

保留缺失 profile 的不可行性和非单调曲线。作者原函数在外部固定提交上实际运行，与公式实现分开记录。合成输入复现了零批量边界及并列回溯差异；这不意味着作者论文实测结果错误。作者仓库未找到显式 LICENSE，原 Python 源码未复制进项目，独立实现只采用数学方法。

`cpu_policies/runtime_contract.py` 将选好的不重叠物理核组映射为 TVM 配置。它不新造线程数或主机核选择规则。TVM caller 是 worker0，配置 N 线程应为 caller＋N−1 helpers；Pipe-it 作者的并行 kernel 调用者则等待全局 worker 池，不能机械复制线程计数。主机与 producer／collector 的位置独立描述，不进行全进程 affinity 覆盖。

多个非相邻 CPU 子图复用同一核组，属于本项目适配：必须显式启用、按组串行，并由原静态模型累计该物理资源负载。多个 VTA 岛也仍需按单个物理加速器累计。原论文的独立阶段 `max(stage_time)` 不能直接代替这些累计约束。

## 方法核对阶段的接入记录

1. CPU 固定核组、计算 worker 固定、按核组串行的执行约束有原作者代码出处，通过现有 TVM 执行器适配。
2. 线程数和组分配由项目已有的静态 DP 在实测成本上选择；不把项目 DP 冠名为 Pipe-it 原搜索算法，不把依赖子图当成 Packrat 副本。
3. 核对阶段先保持共享四核；本轮精简试跑已扩展为共享四核与三核计算／一核主机控制两域搜索，不引入“主机总固定核心2”的启发式；VTA主机工作继续由项目已有模型记账。
4. 直接评估采用成熟约束后的效果，取消“先在所有切图证明统一超过5%才允许接入”的前置门槛。本轮只做短窗口试跑，不设置性能波动门槛或独立确认；正确性、亲和性及数据身份仍保留。
5. 外层融合阶段使用其真实整段 profile。这是项目的 interval-profile 适配，不将整段时间冒充 Pipe-it 原式9的逐层加和输入。

`adapt-existing` 已对旧候选实际完成数据／参数接通，生成单核 worker 映射与 `--cpu-worker-pinning`、`resource_group` 命令预览；没有执行该命令。

`requirements` 已核对采用原共享主机设置、固定绑定 CPU、代表性同构核成本假设时的输入缺口：CPU 340项＋VTA 87项＝427项；旧缓存中匹配103项（CPU 16、VTA 87），缺CPU 324项。其他物理核组的成本将是明确的核置换假设，不能称为每个mask都已实测。现有固定绑定镜像检查为12项，最大差约2.02%；正式导入仍需重放原始凭据。该清单不启动测量、不生成新排名。

`inspect-existing` 已重放16项 CPU profile 的原始证据；这些仅覆盖四个已融合 CPU 阶段的一到四线程曲线。不能将其当作完整21单元的逐层矩阵，也不能从中声称完成原论文的端到端效果复现。

## 本地命令

```bash
python run.py verify --output artifacts/reference_verification.json
python run.py pipeit --input examples/pipeit_workflow.json
python run.py packrat --input examples/packrat.json
python run.py runtime --input examples/runtime.json

# 固定提交作者文件已在外部缓存时，原样运行两段作者方法体作对照
python run.py verify --author-optimizer /tmp/packrat_source_audit/upstream/utils/optimizer.py \
  --output artifacts/reference_verification.json

python run.py inspect-existing --run-dir ../cpu_strategy/runs/resnet18_resource_policy_dp_01 \
  --output artifacts/existing_profiles.json
python run.py requirements --run-dir ../cpu_strategy/runs/resnet18_resource_policy_dp_01 \
  --output artifacts/profile_requirements.json
python run.py adapt-existing --run-dir ../cpu_strategy/runs/resnet18_resource_policy_dp_01 \
  --candidate-id cfg_225e6997520baf04e59fc957 --output artifacts/tvm_contract_preview.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -c pyproject.toml -q
```

所有 examples 都明确标为合成数据。`predicted_fps` 对毫秒输入使用 `1000/interval`。核验结果、源码哈希和 profile 输入清单写入忽略目录 `artifacts/`；原实验源码快照、成本、排名、选点及板端数据均未改写。`third_party/pipeit_reference` 是带原许可的只读源码证据，不是另一个长期维护或编译使用的原生执行器。

上述历史清单记录方法核对阶段的缺口；精简试跑按新协议统一采集514项。当前进度和实际结果以新批次 `state.json` 及 `analysis/` 为准；准备包不代表板测完成。旧约13FPS诊断结果没有被重新标为论文方法成果。

## Tarnawski基线与分阶段消融

独立入口为 `run_staged_ablation.py`，操作与边界见 [staged_ablation/README.md](staged_ablation/README.md)。
本轮先整理历史正反例和基线契约，再冻结排名并共同测量；Pipe-it和GCAPS作为执行组织与资源拆分依据，
不作为独立性能柱。2023年TVM流水论文留待后续，本轮不实现；也不开展YOLO或贝叶斯优化。

基线为明确的Tarnawski设备负载目标平台适配，不继承本文的主机CPU容量修正。
主消融交叉“CPU粗粒度／细化”和“主机需求排除／计入”；三个补充版本仅测前五项。
原前缀方法和失败探索保留在证据链中，只有能揭示规律时引用，不冒充外部论文基线。

[证据演进表](artifacts/staged_ablation_01/evidence/REPORT_zh.md)及原始来源索引已生成。
批次为 `runs/resnet18_staged_ablation_01`；是否完成以其 `state.json` 为准。
