# C3 补充实验计划：输入复用、权重复用及组合的收益与边界

2026-09-23 新增跨启动正结果：[YOLOv3-tiny Y00 输入复用与权重驻留双启动确认](20260923_C3_Y00_INPUT_WEIGHT_DUAL_BOOT.md)。
固定真实 Y00 卷积层的 h16/w52 Tile，input-only→input＋首次使用全权重驻留在两个独立 boot 分别改善
2.281%和2.125%，均6/6轮获胜；original→组合分别改善10.321%和10.175%。增量配对保持输入请求、
计算、静态arena、累计UOP装载和同步匹配，权重请求从239,616 B/52次降至4,608 B/1次。
这证明传输压力层上组合机制具有跨启动稳定净收益，但整层权重仅4,608 B，不能替代P1/P2部分驻留证据。

2026-09-23 新增部分驻留本地边界：[R18-H1 T8 P1/P2 部分权重驻留本地资格](20260923_C3_PARTIAL_RESIDENCY_LOCAL_GATE.md)。
P1-S/P2-S 均通过地址、依赖、三seed FSim、单提交和零额外drain检查，但累计UOP装载与P4-S同为
1,624 B；其权重请求反而为202,752 B/147,456 B，高于P4-S的36,864 B。因此预注册UOP门槛未通过，
不冻结上板候选。该负结果说明驻留粒度只有在lowering实际复用μop模板时才可能形成性能优势。

2026-09-23 新增开发里程碑：[R18-H1 T8 首次使用分阶段全权重驻留](20260923_C3_VTHREAD_STAGED_RESIDENCY_PILOT.md)。
固定 genuine O2 Tile 与双输出虚拟线程，对比 O2、前置全装载 P4-E 和首次使用分阶段装载 P4-S。
P4-S 仍只请求36,864 B权重、单次提交且累计UOP装载保持1,624 B/16次；它把首个有效MAC前必须完成的
权重请求从36,864 B降到9,216 B，代价为权重LOAD从2条增至8条、总指令从223增至228条。
一次启动的54/54正式计时全部正确：P4-S相对P4-E配对中位改善0.562%（4/6轮胜），相对O2改善
1.211%（6/6轮胜），均未达到预注册2%开发门槛。因此分阶段首次装载方向成立但收益较小；它没有降低
现有驻留方案的UOP代价。下一步进入P1/P2部分驻留的本地编译可行性与地址安全检查，不先扩展T7或Tile搜索。

2026-09-22 新增开发里程碑：[R18-H1 双虚拟线程权重驻留](20260922_C3_VTHREAD_WEIGHT_RESIDENCY_PILOT.md)。
以 genuine T8 O2 为骨架新增双线程 reload/驻留程序，不改 runtime、驱动或硬件。三候选均通过最终地址、
依赖、三 seed FSim、命令签名、ARM 交叉编译和板端资格。驻留相对双线程 reload 控制改善17.159%，6/6轮胜；
相对 genuine O2 改善1.191%，6/6轮胜，但未达到预注册2%开发门槛，故仅称方向一致的轻微改善，不称稳定加速。
随后完成T7同协议验证：VS相对VR改善20.976%、相对O2改善1.126%，两项均6/6轮胜；净改善同样未达2%。
双Tile统一命令表显示T7/T8的VS都只需2次权重LOAD和1次提交，而VR分别需16/14次LOAD及17/15次提交。
该结果证明权重驻留可在两个Tile上保留双输出虚拟线程并接近/略优于各自O2；仍只限一个层、同一boot。
下一步是独立启动确认，通过后再进入部分驻留/Tile联合选择；旧八版本实验不复跑。

2026-09-22 新增用户批准实验：[R18-H1 双 Tile × 八版本线程/同步/驻留对照](20260922_C3_THREAD_SYNC_ABLATION.md)。
独立入口已实现，最终16/16完整本地及板端资格、8/8控制匹配、TopHub等价及26项回归通过。
两个不同boot均完成16/16板端资格、48次正式预热和384/384正确计时，隐藏调用为零；双启动分析已完成。
结果显示：关闭虚拟线程稳定退化约17.7%–19.4%；驻留相对匹配reload控制稳定改善约5.2%–7.4%，
但驻留版本相对双线程原方案仍稳定退化约4.4%–13.4%。同步代价仅一个T7对照达到稳定门槛。
正式结果、失败会话分账和证据边界以该文档的执行记录为准；不扩展为硬件等待周期或整网结论。
旧实验与以下历史收尾状态保持不变，不以本轮准备工作冒充新增板端性能结论。

2026-09-22 统计勘误：[ResNet18 SRAM 统计纠错与可引用口径](20260922_RESNET18_SRAM_ACCOUNTING_CORRECTION.md)。
撤回旧 stage-search 的“整网 SRAM 利用率 38.28%”：容量单位错误，且分子不是编译分块分配。
重读封存 TIR 后，R18-H1 TopHub 的 INP/WGT/ACC 静态 arena 占比分别为 56.64%/3.52%/87.50%；
仅限该算子配置，不是整网峰值或动态平均占用。原始实验记录不变，缺少的整网指标保持 unknown。

制定日期：2026-09-15。状态：`A_B_C1_C2_D_COMPLETE; STOPPED_AT_PLANNED_CLOSEOUT`。

新增用户请求已完成：[同一真实卷积的新复用方案与 TopHub 实测](20260915_C3_TOPHUB_REUSE_COMPARISON.md)。
R18-H1 有精确 workload 的 TopHub 缓存实体，genuine source 重新定位当前板并与全部方案同库计时。
本轮551/551调用、480/480计时正确，逐调用DMA/提交审计通过。TopHub 5.9733 ms，输入同tile
5.9784 ms基本持平，有限池最佳IW/W为6.3015/6.7900 ms，分别慢5.43%/13.68%。组合虽减少
61.40% LOAD、同替代tile原方案快12.56%，仍未超过TopHub；这是一个真实层负结果，未测整图。
缓存为PYNQ实体retarget/requalified，并非原生AXU官方最优认证。失败会话保留、不拼接性能。

最新执行：[节点D双启动机制与权衡定稿](20260915_C3_REUSE_NODE_D_CLOSEOUT.md)。逐轮核验1318次完整调用、
832次正确计时；I/W/IW按双启动2%及各6/8胜轮门槛分别11/10/11组稳定快于O。组合10组两启动最快，
4个独立reload配对均稳定受益。四张主图、三张主表、资源/命令/Pareto及全部失败成本分账已生成。
重要负结果：M3跨tile最小bytes首点约慢3.2倍，因此不新增通用bytes排序、在线搜索或整网正贡献。
原计划明确允许缺失指标为unknown；历史早期timer/contract缺口不覆盖补造，以收尾账本披露。
本轮完成后停止，统一覆盖证明及生成寄存器认证仍非本轮完成主张。以下暂停状态均为历史记录。

最新执行：[C2第二启动收尾](20260915_C3_REUSE_FIXED_DRIVER_C2_CLOSEOUT.md)。实际新boot已确认，原样复测
48/48三seed正确、416/416计时正确、48个独立profile与静态DMA一致。两启动都为10组组合最快、
1组输入单项最快；C2的W0/T0墙钟51.180/35.623 s。重启临时依赖恢复4.879 s另列，早期preflight
失败完整墙钟unknown，不填零。原库、候选和冻结源码不变，按计划暂停于C2；D离线配对与图表尚未完成。
下方C1暂停和更早状态保留历史，以本段为当前执行状态。

最新执行：[实验版完成握手与C1收尾](20260915_C3_REUSE_FIXED_DRIVER_C1_CLOSEOUT.md)。用户已同意修改driver，
本轮以独立生成/构建/部署的实验库验证新完成事件，不覆盖原driver或B冻结源码。42次微验证、90提交
通过；关闭诊断日志后完整C1为48/48三seed正确，416/416计时正确，48个profile与静态DMA一致。
11组四种不同程序均实际双复用，10组组合最快、1组输入复用最快；W0/T0墙钟61.635/36.001 s。
在C1完成处暂停，C2须用户实际重启。0x1c output-valid是假设的板端行为验证，不是生成映射认证。
以下旧失败/暂停记录保留历史，不再代表当前审批或C1状态；旧性能会话仍无效，不拼接到新结果。

最新执行：[C1完成边界与暂停报告](20260915_C3_REUSE_C1_COMPLETION_BOUNDARY.md)。板端健康门3/3，
48身份资格46通过、2拒绝；后续M2另一IW计时输出错误，418调用后整次性能会话作废。
独立42调用诊断发现7次立即错误均在不重新执行、延后读取后变正确；这是runtime返回早于输出稳定
的直接证据，不是新的加速结果。下一节点为独立完成握手验证；因原计划明确不修改driver，
实验版driver修复当时须先由用户同意。原B池及既有C3证据只读不变；该失败尝试不是正式C1结果。

后续只读审计已完成：`20260915_c3_reuse_completion_audit_run03`，460/460调用的dispatch身份、
逻辑DMA和submission与冻结程序核对，4项测试通过，并交付完成可消费边界图与逐提交CSV。
三组M2 IW的FSim首段包含全部LOAD/STORE，第二段为64 B且无数据DMA的控制尾段；
纠正“先独立预加载权重再计算”的推断。仍缺与当前bitstream绑定的生成寄存器定义和逐段MMIO证据，
不猜测valid位或直接写寄存器作为修复。诊断缺失的完整逐调用墙钟保持unknown，不能填零。

节点A已完成：[正式汇总与核验](20260915_C3_REUSE_NODE_A_CLOSEOUT.md)。54条历史配对、12个输入目录/
319个文件前后哈希一致、23项测试通过；R18四方案口径为19组含回退可解析、1组四模式均实测、
0组四种不同二进制。节点B现已完成：[本地机制验证、成本和冻结结果](20260915_C3_REUSE_NODE_B_CLOSEOUT.md)。
M1/M2/M3各4/3/4组，共11组本地有效双复用四元组（44主身份）和4个reload控制，48/48交叉编译
通过，四种不同TIR/二进制组数11。最终合同为`20260915_c3_reuse_node_b_freeze_run02`，56项测试
通过、3项旧sim-target测试跳过；新入口四模式FSim测试均执行。此段保留B收尾历史，C1最新状态见上。
下文保留原执行规范及适用边界，以最新结果和修订为准。

### 2026-09-15 节点C1计时协议修订（新板端标签采集前）

新入口为 `run_vta_c3_reuse_board.py --phase prepare|board`，不改动已经冻结的B节点源码或48个身份。
正式板端准备目录为 `20260915_c3_reuse_c1_prepare_run02`；run01计时辅助库编译失败，保留失败记录，
没有连接开发板或取得目标标签。新增实验辅助库仅注册 `c3.reuse.timed_once`，不修改VTA runtime。

- 旧`time_evaluator(number=1,repeat=1)`隐含执行两次，无法逐调用检查warmup输出。本轮以板端
  `steady_clock`包围一次同步`main`调用，调用后逐元素核验；模块函数查找、RPC、输入拷贝及输出
  核验排除于算子延迟，全部包含在连续会话W0/T0墙钟中。辅助库经本机RPC原地递增测试验证无隐藏warmup。
- 三seed正确性后，每个通过身份另做一次显式warmup并检查输出，成本单列。所有8轮均保留。
  独立profile另执行一次完整算子，计时调用自身的逻辑profile也保留；统一按完整算子调用归一化。
- 在SSH/bitstream哈希/FPGA/u-dma-buf/tmpfs及独占RPC检查通过后，以同一冻结bitstream重新配置FPGA，
  再在新tmpfs目录启动已核验的runtime。只终止本次记录且cwd一致的自有RPC，不触碰其他会话。
- 4个reload控制各按所属family组成W-resident/W-reload/IW-resident/IW-reload独立四路Latin块，
  每块8轮。既有主块的W/IW时间不直接与另一时间段的reload时间混作同轮因果对照。
- 原冻结family顺序、四模式Latin顺序、三seed和7200秒上限不变。C1完成后仍暂停，C2必须由用户
  实际重启；不把同boot重复运行写成跨启动复现。

新入口6项测试通过；B节点冻结产物保持只读，准备期无新目标FPGA正确性或性能标签。

### 2026-09-15 节点B执行修订（板端标签仍不可见）

1. 真实完整 ConfigSpace：M1=1296、M2=1024、M3=320，边界=1280。固定线程及复用作用域筛选后
   分别有210/180/48/240个family；按原盐和max-min规则冻结64/64/48/64个检查顺序，边界在获得
   两个original-lowering合法family后停止。
2. 旧`/tmp/vta-hw-fsim`不存在。独立worker显式加载已保留的`build/libvta_fsim.so`，核验其构建宏
   与板端ABI一致；保持`TARGET=axu5evb`生成相同lowered TIR，仅执行模块的host改为`llvm`。
   因此这是板端参数下的本地FSim，不是新硬件实测；bitstream哈希只是后续必需指纹，尚未观测。
3. 新增独立的分支感知LOAD/STORE计数器；保存packed源区域、SRAM目标、循环及条件路径，
   以runtime和FSim指令计数交叉验证。不替换旧证据的计数器或覆盖历史结果。
4. 在StorageRewrite后、设备地址lowering前增加只读捕获，保存arena跨度；跨buffer跨度之和
   仍是上界，不冒充统一同时存活峰值。捕获前后最终TIR结构一致由回归测试验证。
5. 本地run01识别出冷/热命令差异：首次初始化与后续调用相差一条指令，而三seed输出和DMA
   可以一致。run01全部保留，仅作协议诊断，不用其“签名不一致”推断候选数值无效或组合不可行。
   run02每worker增加一次seed0预热并检查输出，之后仍要求三个正式seed命令一致；预热、失败
   和所有调用计入成本。family中任一模式FSim失败，则其余模式不继续执行，避免无谓验证；
   四模式的所有lowering尝试及未完成FSim的family身份仍完整保留。
6. 补充reload控制只通过实验模板进入：保留对应W/IW循环与barrier，恢复reduction-scope权重LOAD。
   只有compute作用域、输入/输出DMA及同步相同，且权重搬运确实增加的控制才接受；其SRAM地址
   和allocation仍可能改变，故只能称“匹配循环与同步的reload对照”，不能称完全隔离的物理因果对照。

准备合同：`07_grouped_holdout/20260915_c3_reuse_node_b_prepare_run02/contract.json`；
执行目录：`07_grouped_holdout/20260915_c3_reuse_node_b_local_run02/`。本修订没有读取新FPGA性能。

7. M2在64-family上限内只得到3组，仍满足积极确认门槛；保留4/3/4共11组，不扩域补齐48主身份。
8. SDK初次因继承`LD_LIBRARY_PATH`拒绝启动编译器。恢复到新目录
   `20260915_c3_reuse_node_b_cross_recovery_run01`，只对子进程隔离该变量，保持资格/TIR/候选不变；
   48个身份均成功交叉编译，原48次setup失败完整保留。资格与恢复分别计时，不拼接成连续session。
9. 最终使用`freeze_vta_c3_reuse_confirmation.py --phase freeze`封存完整五文件协议和全部二进制，
   family顺序seed=20260910，八轮四模式Latin顺序及extra-profile单完整调用已冻结。
   最终入口/合同：`07_grouped_holdout/20260915_c3_reuse_node_b_freeze_run02/contract.json`。
   prepare提案JSONL缺口和初稿freeze的results缺口不修改旧目录，以最终规范包及上游合同补全定位。

## 1. 本轮要回答什么

本轮先补机制证据，不同时开发新 tuner、完整覆盖验证器和新的内存分配器。要回答：

1. 输入优先究竟减少哪些输入加载，又增加哪些权重加载？
2. 权重驻留在哪些 tile 上足以抵消 barrier/submission 代价？
3. 两种机制能否在同一 tile 上同时减少输入、权重加载？组合何时优于单项，何时不适用或更慢？
4. 扩展驻留空间是否提供原始模板之外的正确、较快配置？这些规律给后续搜索提供什么信息？

主创新定位不变：固定 FPGA 容量下的低成本配置搜索。机制分析是搜索依据；只有新策略经过独立确认，才增加新的搜索贡献。不能因为图更详细，就声称排序更有效。

本文数值语义保持不变；改变的是加载、驻留和执行结构。Cheng 三类机制是功能级独立重实现，不声称作者源码级复现或首次提出。USMP 的大小、生命周期冲突和偏移规划属于互补能力，不列为本轮原创。

## 2. 已有证据与真正缺口

### 2.1 当前证据基线

下列数字来自已有文件，不是本计划的新实验结果。

| 证据 | 已完成 | 应保留的边界 |
|---|---|---|
| R18 原始 ConfigSpace 合法性 | 4384 次 real lowering，569 合法，总无效率 87.02% | 这是实际编译结果，不能把所有淘汰都归因于便宜的容量公式 |
| R18 FPGA 完整池 | 214 身份，207 正确，1035 个五轮计时样本 | 孤立算子资格不等于整图资格 |
| R18 original/input 同 tile 配对 | H1 15 对、10 对更快；H2 11 对、4 对更快；H3 14 对、9 对更快 | 原始胜负含测量噪声；不直接称显著加速 |
| Y00 输入复用 | 五个配对均加速，改善 7.97%--28.80% | 一个几何的正结果 |
| Y02 权重复用 | 权重 LOAD -92.31%，总 LOAD -54.48%；两次启动延迟改善 25.28%/23.55% | 一个 family 的跨启动正结果，不等于跨 workload |
| P7R513 四方案分析 | 19 个 family 可按实测或资源回退解释 | 其中只有 H3 的一个 family 四模式均实际测量；H2 的18组包含 original 回退 |
| R18 组合方案 | 有孤立算子正确且较快的候选，也有整图错误后回退 | 唯一四模式均实测的 family 没有减少搬运字节，不能充分验证双重复用协同 |

关键输入目录相对 C3 根目录：

- `07_grouped_holdout/20260914_p7r513_resnet18_cheng_four_scheme_analysis_run01/`：四方案记录；
- `07_grouped_holdout/20260914_p7r510_resnet18_merged_board_pool_run01/`：完整板池；
- Y00/Y03、Y02、R50 的既有证据按现有 artifact ledger 定位，不猜测文件名或拼接缺失记录。

### 2.2 必须分开的四种数量

报告中分别计数：`four_resolved_with_fallback`、`four_measured`、`four_distinct_programs`、`effective_dual_reuse`。

- 回退到 original 不算优化模式的独立实测。
- 四模式都测量，不保证四者生成不同程序。
- mode flag 为输入/权重驻留，不保证实际减少加载。
- 双重复用必须通过同 tile DMA 和加载作用域确认，不能从模式名称推断。

## 3. 节点 A：整理旧证据并纠正口径，完成后暂停

### A1. 独立只读聚合

新增聚合器，读取既有合同、资格、计时和 profile，先校验哈希。旧 runner、结果和 oracle 全部保留不覆盖。不要重新执行4384点扫描或214点板池。

按 `workload + complete ConfigEntity + implementation/hardware context` 建立 same-tile family，分别统计 original/input、original/weight、original/combined 配对。跨启动同一 family 单独列，不按多个 workload 计数。

每对保存：三类张量及 ACC/UOP 的 bytes/calls、同步和 submission、instruction/UOP peak、算子延迟、已记录的 driver 时间、正确性和缺失原因。缺失值为 `unknown`，不能填零。

对每个配对报告延迟变化、搬运变化、请求变化；中位数分别计算，不能把多个中位数相互代入。对缺少逐张量 calls 或实际分配跨度的旧证据列缺口，不根据总量虚构分解。

### A2. 优先解释已有反例

固定至少四类案例：H1 输入减少但权重增加、H2 输入模式几乎无作用、Y02 权重 LOAD 大幅减少且净加速、H3 请求减少但 submission 增加且算子变慢。

从现有 schedule/lowered TIR 解释 LOAD 所在循环和复用作用域。仅对小案例做独立卷积索引枚举校验，不对207点逐乘加建立大图。本轮不要求完整地址—消费者—异步覆盖证明。

输出 `reuse_pairs.csv`、`family_coverage.csv`、`counterexamples.md`、`summary.json` 和哈希。新报告明确说明 P7R513 的19组包含回退；主创新文档追加最新口径，保留历史记录。

**A 验收与暂停：**重现40个 R18 input 配对和19/1的四方案口径；解释正反案例，并报告缺失项。先向用户汇报，不自动开始新上板。

## 4. 节点 B：小型四方案池的本地资格与冻结，完成后暂停

### B1. 固定硬件与三类机制测试几何

当前仓库配置是 batch=1、block_in/out=16、int8 输入/权重，input/weight/ACC/UOP SRAM 为32/256/128/32 KiB。执行时以 `vta.get_env()`、构建配置及 bitstream 指纹共同核对；不匹配则停止，不能静默换规格。

| ID | CI→CO；输入空间；kernel/stride/padding | 整层 int8 权重 | 作用 |
|---|---|---:|---|
| M1 | 64→64；28×28；3×3/1/1 | 36,864 B | 大空间、输入与权重重复搬运 |
| M2 | 128→128；14×14；3×3/1/1 | 147,456 B | 较高权重驻留占用与 reduction 分块 |
| M3 | 128→256；14×14；1×1/2/0 | 32,768 B | 1×1、下采样及小请求边界 |

这些是机制测试几何，不自动称未见网络或真实模型层。完整 workload 仍由任务模板生成，禁止把 ConfigSpace 一律写成1280点。

### B2. 四种机制与实际效果分类

固定模式编号：O=`original/0`、I=`input_stationary/1`、W=`weight_resident_barrier/4`、IW=`input_weight_resident_barrier/5`。不用 feasibility-only mode2 或旧 nominal hybrid mode3替代。

同一 family 的完整 ConfigEntity 相同，首轮固定 `oc_nthread=h_nthread=1`，隔离虚拟线程影响。保留至少两个 output-channel outer tile 和两个 spatial tile；否则应标为退化对照，不能解释为积极复用案例。

M1/M2 优先保留两档 `tile_co`、两档空间 tile，以及可用的不同 `tile_ci`。M3 的不同空间 tile 以7×7输出的真实可用 split 为准。候选按离散 knob max-min 多样性和固定盐 `c3-reuse-mechanism-v1` 生成确定顺序，不用 latency 排序。

每个几何最多检查64个 family；每个展开四模式。先逐 family original/modes real lowering，再三 seed FSim（0、20250901、20260910）；只对全通过 family 计算结构效果并选点。记录所有尝试、失败和开销，不能藏掉筛选成本。

每个几何目标选4个 family：其中至少2个满足 O 的 input/weight reload ratio 均大于1，且 IW 相对 O 的 input 与 weight bytes 都严格下降；另2个按 knob 多样性选取，保留退化或权衡案例。最多12个四元组、48个主身份。

reload ratio 的分母为实际访问的唯一源区域，而不是无条件使用整张量大小；padding零填充不属于源DDR字节。结构效果选择只使用 lowering/FSim信息，不读取新 FPGA 性能。这是针对机制的资格筛选，不把它当作自然搜索空间的无偏样本。

若不足目标，不扩展几何、不按板端速度补点。输出不足原因并暂停：可能是模板 compact/padding 约束、组合整层驻留限制，也可能是本地实现不能产生预期复用。此时先决定是否修机制，不能继续以模式名称宣称已复现双重复用。

### B3. 超容量与同步对照

增加一个固定边界：128→256、14×14、3×3/1/1，整层权重294,912 B，大于256 KiB。取至多2个 original-lowering 合法 family，记录四模式的适用状态。本轮组合应拒绝；这只证明当前整层驻留实现受限，不证明 tile 级权重复用原则上不可能。

审核现有 local qualification 对 W/IW 共用 `full_weight_fits_sram` 的 gate。对 W 报告它是当前框架的额外保守限制，不能与“weight tile自身放不下”混淆。本轮不放宽生产门槛。

对首个已选 M1/M2 family 各准备 W-reload、IW-reload 两个补充控制：保持对应模式的循环顺序和显式 barrier，恢复 reduction-scope 权重加载。最多4个额外身份，仅放入实验模板，不修改原模式。lowering 后确认 barrier次数和其他改变；不一致则撤销控制并报告，不能称纯同步因果对照。

### B4. 本地交付与暂停

保留指令请求源/目标区域、加载作用域和分块计数；以区间/矩形和multiplicity表示，不逐乘加生成巨图。源区域差集只代表潜在复用，未由程序保留的数据仍按实际LOAD计数。

核对 FSim、lowered-TIR 和 runtime 逻辑 bytes/calls；遇到条件LOAD先求值相应分支，不能机械乘循环次数。对有效 family 保存 TIR、实际分配跨度、三 seed输出及命令签名，并交叉编译。

新入口采用 `--phase audit|prepare|local|freeze|board|analyze`、`--contract`、`--output-dir`；board阶段必须提供显式 `--execute-board`。可拆成小脚本，但使用同一协议。复用既有 identity、参考数据、编译、缓冲复用及板端基础设施；现有 R18 runner/分析器有硬编码的三几何、五轮计时和合同校验，不能未经适配直接套用。

冻结 `contract.json`、候选及 family顺序、seed、源码、TIR、二进制、模板、配置和硬件哈希。最多48主身份＋4控制；边界拒绝项不上板。完整产物目录必须拒绝覆盖。

**B 验收与暂停：**至少两个几何各有两组本地有效双重复用四元组，才进入积极组合确认；不足则报告机制负结果。发布冻结池与代码测试，不立即上板。

## 5. 节点 C：真实 FPGA 四元组实测，分两次启动确认

### C1. 第一启动与健康门

先核对 SSH身份、boot_id、bitstream、runtime/u-dma-buf及文件系统状态。部署到既有 tmpfs 路径，不重新写SD卡镜像。
原“不修改driver”约束已由用户明确批准的独立实验版完成握手替代：仅使用资格化的新DSO，不覆盖原driver、
原DSO或B冻结源码；四模式统一协议。重启后临时bitstream/module须恢复并分账，不能将缺失前置条件算候选无效。
完成已资格化 original 健康探针三seed；异常则停止。

同一几何按已资格化 output-data-weight顺序复用buffer。保持布局；不得为不同模式随意改变分配顺序。不同形状的分配与生命周期分别记账，确认不进入不释放分配耗尽路径。

每个身份三seed逐元素正确性，首错即停；失败身份无性能标签。其余模式可以保留单项结果，但 family 不进入完整四元组协同统计。不能静默回退 original 或用其他 tile 补齐。

每个完整四元组做8轮平衡交错计时：以四模式4×4 Latin轮转顺序重复两遍，family顺序由冻结seed确定。额外控制在独立平衡块测量。首轮用于8轮统计，不根据结果丢弃慢轮；必要warmup另记成本。

每个计时调用都检查输出；失败使该身份的性能资格失效。总主计时最多384个/启动，另有控制计时。FSim/provenance trace在正式timing关闭。profile获取是否导致额外运行由合同列明，所有实际调用计账。

独立profile汇总 `driver_run_total_us`，除以实际完整算子调用数，**不能除以 submission 数**；默认称 driver累计执行时间。本地对应Cheng pure-instruction指标的近似值须说明包含的驱动/同步范围，不当成纯计算时间。

完成第一启动后汇报四元组分类与错误，不自动要求用户重启。

### C2. 第二启动与中断处理

用户重启并恢复健康门后，原样复测同一冻结池；每启动分别保存结果，不把两个启动拼成一次连续session，也不计为两个workload。

单次session预设7200秒生命周期；发生断电/RPC失效保存 `invalid_session.json`，该session计时不用于正式配对。完整重跑使用新目录。保留失败成本，既报告有效会话成本，也报告含失败/重试的总投入；持续异常停止排查，不能无限重试。

## 6. 节点 D：分析组合、时间—空间权衡与搜索空间价值

### D1. 三类核心结论

每个完整family分别回答：I是否减input、W是否减weight、IW是否同时减两者，以及代价来自哪里。保存逐张量数据，不只比较总bytes。

定义：

```math
\Delta_{i,m}=B_i(m)/B_i(O)-1,\qquad
\Delta_{w,m}=B_w(m)/B_w(O)-1.
```

`effective_dual_reuse`要求IW的两个变化均小于0。`combination_fastest`要求IW实际中位延迟低于O/I/W，但稳定性能结论还要求第二启动方向一致。

为说明加法交互，报告 `I_T=T(IW)-T(I)-T(W)+T(O)`；负值表示相对O的加法模型下有利交互，**不等于IW必然最快**。两种模式可能同时改变分配和执行结构，不将该式解释为完全隔离的物理因果机制。

“稳定加速”默认要求两个启动都至少改善2%，且各启动至少6/8平衡轮获胜；未达者记为微小差异、混合方向或噪声敏感，不作显著性保证。所有原始配对仍展示，不能只选胜者。

### D2. 空间与时间分账

报告有效驻留数据需求、lowered SRAM分配跨度、对应容量比例，以及页对齐 instruction/UOP backing。未知生命周期时不求虚构的统一同时存活峰值。

实际硬件SRAM容量未变化；高占用比例不称增加了芯片SRAM。命令backing缩小也不称减少整个192MiB u-dma-buf预留。USMP负责固定程序的偏移规划，本轮不做USMP真实VTA地址重写，也不把冲突分析重新列为原创。

### D3. 原始空间与扩展空间

在完成测量的同一tile集合上比较：

```math
T_O^*=\min_{x\in X} T(x,O),\qquad
T_E^*=\min_{x\in X,m\in\{O,I,W,IW\},\ correct}T(x,m).
```

O是扩展池的组成部分，所以同一有效测量池中扩展最小值不会更差，但不保证严格更快。此为小型机制池最优，不是全ConfigSpace全局最优。

若有 exact workload/hardware匹配的TopHub，单独做correctness资格后作为事后参考；不根据它筛选新family，不强制追求超过它。无匹配或错误则报告不可用。配对收益与相对最优original/TopHub收益分开。

### D4. 下一步搜索是否值得修改

只读比较已有 bytes/calls 排序与这些机制解释，列出H1失败、H2成功等案例的可能原因。同步控制能支持哪种解释就写哪种，未隔离的差异写关联，不硬归因。

本轮不启动新的27次在线调优，也不再复跑HW-Aware/ML²Tuner大实验。若语义只能解释而未改善决策，作为机制支撑定稿。若发现可计算、成本可控且能改变错误选择的信息，再另冻独立搜索确认协议；目标性能标签已暴露的池只能用于开发，不能重新包装为前瞻。

本轮不把synthetic卷积强行插入ResNet18。完整图沿用已完成证据；只有新候选exact匹配真实模型任务且通过图上下文资格，才另行制定传递实验。

## 7. 必须交付的图表、测试和账本

四张主图：

1. 输入/权重加载作用域与消费者复用示意；
2. 四方案逐张量bytes/calls和8轮算子延迟，按same-tile分组；
3. SRAM分配占用、同步/submission与性能权衡，包含不适用和反例；
4. 同tile集合中best-original与best-extended对照，TopHub仅在资格通过时另列。

三张主表：旧证据覆盖及回退口径、四元组正确性/实际复用分类、两个启动的收益与完整成本。

每次保存不可覆盖的 `contract.json`、`timeline.jsonl`、`results.jsonl`、`summary.json`、`artifact_hashes.json`；记录W0进程启动、T0板端就绪、T1本节点完成。lowering/FSim/cross/FPGA/profile/timing/warmup和失败全部分账。没有本轮搜索器时不把实验总墙钟称time-to-quality。

测试覆盖：same-tile身份；回退不算独立模式；同TIR/二进制的去重与退化标记；原始统计复现；模式0/1/4/5；padding和条件LOAD计数；缺失指标拒绝；未来FPGA性能标签隔离；失败不获latency；八轮顺序平衡；调用归一化；布局合同；中断不拼接；原有结果只读。

## 8. 顺序、成本上限与最终口径

| 节点 | 操作 | 上限/停止条件 | 完成后 |
|---|---|---|---|
| A | 已有证据聚合与反例解释 | 不上板、不重扫大池 | 暂停汇报 |
| B | 新三几何本地筛选、边界和冻结 | 每几何最多64 family；48主＋4控制身份 | 暂停，确认能真实双重复用 |
| C1 | 第一启动correctness和8轮计时 | 冻结池；任何身份错误不补点 | 暂停，等待下一启动 |
| C2 | 第二启动原样确认 | 不换源码、tile或规则 | 暂停汇报 |
| D | 汇总机制、空间价值、反例及后续选择 | 不自动扩展tuner/整图实验 | 更新论文证据与停止 |

积极组合证据的最低目标：至少两个几何、每几何两组实际四模式正确且IW同时减少input/weight bytes；性能必须忠实报告，不能要求组合一定胜出。达不到时，区分“实现不能形成复用”“资源不适用”“访存减少但净时间无收益”，这些都是需要保留的不同结果。

最终可以写：本文在固定存储容量下实测tile与输入/权重驻留的搬运、分配和同步权衡，并据此建立搜索依据。只有独立后续搜索确认支持时，才能再写这些新增语义降低搜索成本。

参考来源：Cheng 等 [FGCS DOI](https://doi.org/10.1016/j.future.2025.108165)；[HW-Aware Initialization](https://arxiv.org/abs/2205.15568)；[ML²Tuner 2024原文](https://mlforsystems.org/assets/papers/neurips2024/paper6.pdf)；[USMP官方设计](https://discuss.tvm.apache.org/t/rfc-unified-static-memory-planning/10099)。本计划不声称这些既有方法为原创。

## 9. 2026-09-23 Y00 权重传输压力开发实验

为检验 R18-H1 的小幅收益是否与所选层偏计算密集有关，本次没有按结果反复搜索算子，而是从已有
YOLOv3-tiny 层和冻结候选中选择 Y00：`16→32、208×208、3×3、stride=1、padding=1`，固定
Y00F05 的 `tile_h=4、tile_w=26、tile_ci=1、tile_co=1`，将 `oc_nthread` 固定为2。该层原程序
重复权重请求明显，整层权重仅4,608 B，适合构造同 Tile 的 O2/首次使用全驻留配对。

驻留变换增加了严格的单归约组路径：Y00 的输入通道归约只有一组，lowering 后 `k_o_outer` 被消去，
因此不扩张权重地址，只在 `(i2_outer,i3_outer)=(0,0)` 时加载两个虚拟线程各2,304 B的权重。
原 H1 多归约组路径保持不变。完整回归为11/11通过，包含 H1 原合同以及 Y00 三个 seed 的 FSim
逐元素精确校验。

同 Tile 控制匹配结果如下。逻辑 bytes/calls 是编译后指令要求的数据请求，不等于外部总线物理流量。

| 指标 | Y00F05-O2 | Y00F05-PS |
|---|---:|---:|
| WGT LOAD bytes | 1,916,928 | 4,608 |
| WGT LOAD calls | 832 | 2 |
| INP LOAD bytes/calls | 2,202,240 / 832 | 2,202,240 / 832 |
| OUT STORE bytes/calls | 1,384,448 / 832 | 1,384,448 / 832 |
| ACC/INP/WGT arena | 13,312 / 5,376 / 4,608 B | 相同 |
| UOP累计装载 | 128 B | 128 B |
| submission | 1 | 1 |
| 指令数 | 6,674 | 5,844 |
| 全部样本中位耗时 | 27.084601 ms | 24.3631385 ms |

冻结的六轮两顺序平衡实验共36个正式单调用计时样本；两个候选各三个 seed 的板端正确性、诊断签名
和正式计时输出全部通过，无隐式调用。六轮配对改善为9.779%--10.151%，中位改善10.035%，
驻留版6/6轮获胜。这是 boot ID `ffdf1803-e65f-475e-963b-2d3f7273a816` 上的一次启动开发信号，
尚不是跨启动稳定性结论，也不推广为整个 YOLOv3-tiny 的收益。

这个结果支持“层/Tile 的传输压力决定驻留收益上限”，但仍不能凭程序级数据声称精确减少了多少
Compute 等待周期。按256 MAC/周期、100 MHz 的分析峰值，Y00 的199,360,512个有效 MAC 对应
约7.788 ms 理想下界，而 O2 实测约27.085 ms；此前 H1 的理想下界约4.516 ms、实测约5.881 ms。
两者差异说明 H1 已更接近计算下界，权重请求即使减少也更可能被计算覆盖；Y00 则有更大的非计算
关键路径空间。Y00 中 WGT请求减少99.760%、总LOAD减少约46.424%，而时间减少10.035%，也再次
说明传输量减少不会一比一转化为延迟减少。

本次关键产物：

- `07_grouped_holdout/20260923_c3_y00_memory_pressure_prepare_run01/`：本地资格、TIR、地址、FSim、命令与交叉编译；
- `07_grouped_holdout/20260923_c3_y00_memory_pressure_freeze_run01/`：候选、二进制哈希与36次调用顺序；
- `07_grouped_holdout/20260923_c3_y00_memory_pressure_board_run01/`：板端逐调用原始记录；
- `07_grouped_holdout/20260923_c3_y00_memory_pressure_analyze_run01/`：配对结果与空间—传输—指令—时间表。

下一步不应再把 H1 当作唯一性能代表。应把 H1 保留为低收益/近计算下界对照，把 Y00 作为高权重
重载正例；在不读取新性能标签的前提下，用 `可消除WGT bytes / MAC`、原始总LOAD构成和分析计算
下界筛选少量真实网络层，再做独立启动确认。只有跨层、跨启动仍能区分高收益与低收益层，才能把
“联合选择 Tile 与驻留粒度”提升为第三创新点的选择方法，而不是单个层的特例优化。

## 10. 2026-09-23 Y00F06 输入复用＋首次使用权重驻留实验

Y00F05 的同 Tile PS 虽证明权重驻留有效，但其24.36 ms仍不能代表充分调优后的 Y00 程序。本实验转向
此前冻结且板端正确的 Y00F06：`tile_h=16、tile_w=52、tile_ci=1、tile_co=1、oc_nthread=1`。
先保留 mode1 输入优先循环，再新增实验 mode9，仅通过 post-StorageRewrite 变换把每个空间 Tile 的
整层权重LOAD改为首次空间 Tile执行一次。该模式要求整层权重可放入WGT SRAM且 `oc_nthread=1`，
不增加逐组drain。直接改成双虚拟线程会把INP请求从794,368 B恢复到1,588,736 B，因此被本地资格
检查排除，没有作为“组合”上板。

本地14项回归全部通过，包含H1、Y00F05和Y00F06三seed FSim。I与IW的计算哈希、INP/OUT请求、
ACC/INP/WGT arena、同步次数和最大kernel μop一致；WGT请求从239,616 B/52次降至4,608 B/1次，
减少98.077%。三版本的静态与板端汇总如下：

| 指标 | O | I | IW |
|---|---:|---:|---:|
| INP LOAD bytes/calls | 1,588,736 / 104 | 794,368 / 52 | 794,368 / 52 |
| WGT LOAD bytes/calls | 239,616 / 104 | 239,616 / 52 | 4,608 / 1 |
| OUT STORE bytes/calls | 1,384,448 / 104 | 相同 | 相同 |
| ACC/INP/WGT arena | 53,248 / 15,552 / 2,304 B | 106,496 / 15,552 / 4,608 B | 与I相同 |
| UOP累计装载 | 208 B | 460 B | 460 B |
| submission | 1 | 1 | 1 |
| 指令数 | 843 | 479 | 428 |
| 全部样本中位耗时 | 19.407719 ms | 17.809603 ms | 17.406429 ms |

六轮三版本全排列、每版本每轮三个seed，共54个正式单调用样本。全部输出正确，诊断签名匹配，
无隐式额外调用。配对结果：

- O→I：中位改善8.189%，6/6轮获胜；
- I→IW：增量中位改善2.281%，6/6轮获胜；
- O→IW：组合中位改善10.321%，6/6轮获胜。

这比Y00F05更直接地支持“联合选择”的必要性：IW不是在一个较差O2 Tile上单独减少权重，而是在
已经减少输入请求的程序上继续消除98.077%的权重请求，同时没有改变输入请求、μop累计装载、arena
或submission。增量收益只有2.281%，仍说明权重请求并非全部处于未隐藏关键路径；但它已超过预注册
的2%开发阈值，且六轮方向一致。

第一次板端run01在任何算子调用前因通用runner硬编码`_O2`健康基线而fail-closed；该目录完整保留，
没有性能样本。修复为读取冻结协议首版本后，重新执行全部prepare/cross/freeze形成run02，未修改旧
冻结合同，也未拼接会话。有效结果仍来自 boot ID
`ffdf1803-e65f-475e-963b-2d3f7273a816` 的一次启动，因此尚不能声称跨启动稳定。

有效产物：

- `07_grouped_holdout/20260923_c3_y00_input_weight_prepare_run02/`；
- `07_grouped_holdout/20260923_c3_y00_input_weight_freeze_run02/`；
- `07_grouped_holdout/20260923_c3_y00_input_weight_board_run02/`；
- `07_grouped_holdout/20260923_c3_y00_input_weight_analyze_run02/`。

下一步优先级调整为：先在新boot原样复测冻结run02，确认I→IW的约2.3%小增量不是启动相关波动；
随后把同一“保留现有高质量Tile/循环，仅增加受约束驻留”的构造应用到未参与本轮开发的真实网络层，
按静态可消除WGT bytes/MAC分层验证。只有新层也能由静态规则区分收益，才进入完整YOLO图的conv2
替换及端到端验证；暂不优先开发P1/P2部分驻留。

## 11. 2026-09-24 mode 10：组合安全的局部完成语义

后续 YOLO-608 与历史 ResNet18 整图实验暴露了一个比“传输是否减少”更基础的问题：mode 5 候选
在孤立算子中通过三 seed FSim/FPGA 正确性，却在多个驻留路由组合进同一图后产生随执行顺序变化的
错误输出。全图 `VTA_DEBUG_FORCE_SERIAL` 能恢复 YOLO 双路由正确性，因此问题被收敛为正常依赖编码
下的程序组合安全，而不是参数、Graph JSON、路由命中或算法语义错误；尚不能进一步声称已经找到了
某一条精确硬件 token/race 根因。

新增 mode 10 `input_weight_resident_composable` 保持 mode 5 的 Tile、循环、驻留和数据请求安排，只在
驻留命令段边界生成局部 `VTASetDebugMode(32) → VTASynchronize → VTASetDebugMode(0)`。全局
debug flag 保持0。最终地址审计器只接受静态的 `[32, 0]` 转换，并将其作为有序控制事件记录；未知
debug值或未闭合序列仍然失败。

本地与二进制资格结果：

- YOLO-608 两个真实 `1×1` 路由均通过 StorageRewrite 容量、最终 DDR/SRAM/UOP/GEMM/ALU 地址、
  依赖闭合、三 seed FSim 精确相等和跨 seed 命令结构一致性；反汇编仅目标函数6和10各含两次
  `VTASetDebugMode`。
- ResNet18 历史失败 H3 路由也通过相同门槛；反汇编仅 `clip_cast_7_compute_` 含两次控制调用。
- mode 5 保持不变，旧失败证据未被覆盖；mode 10 使用新的 schedule version 和 candidate ID。

板端无计时正确性结果均来自 boot `c9d73f4c-c0df-474a-9a74-a28a2f8955e2`，未重载 bitstream：

| 网络与路由 | 历史 mode 5 整图结果 | mode 10 整图结果 |
|---|---:|---:|
| YOLO-608，两条驻留路由 | 组合图错误 | 3/3 seed × 8输出全相等 |
| ResNet18，H3驻留路由 | 0/6正确，每次1000 mismatch | 3/3 seed × 1输出全相等 |

冻结后又执行了单启动、位置平衡的开发计时；每种方案9个单次样本，全部输出相等、0隐式调用：

| 网络 | 对照 | 中位配对改善 | 获胜数 | 解释 |
|---|---|---:|---:|---|
| YOLO-608 | scoped composable vs stock | +13.484% | 9/9 | 强开发正例 |
| YOLO-608 | scoped composable vs global force-serial | +12.298% | 9/9 | 局部完成保留了全局串行丢失的并行性 |
| YOLO-608 | global force-serial vs stock | +1.323% | 9/9 | 全局串行仅保留很小净收益 |
| ResNet18 | composable H3 vs 同H1/H2安全H3-original | +0.056% | 6/9 | 未形成稳定净收益 |
| ResNet18 | composable H3 vs stock | -0.250% | 1/9 | 正确但不应部署为加速方案 |

这些是单 boot 开发证据，不是双启动正式确认。用户报告重启后观测到的 boot ID 仍未变化，因此不能
把本轮称为独立第二启动。结果也不能推出“所有驻留都应加入局部串行”；它支持的是把完成策略作为
候选身份和选择维度。

第三创新点据此加强为：

> **面向固定 VTA 的组合安全、编译后代价感知驻留—Tile 联合选择。编译器为候选显式生成驻留与完成
> 语义，利用容量、最终地址、数据请求、μop、同步和组合资格约束候选，并在有限板端预算下选择是否
> 部署；选择单位从彼此独立的单层程序提升为带兼容约束的图级候选集合。**

这里的关键新增不是“加一个 barrier”，而是三点：候选正确性合同覆盖程序组合；局部完成避免全图
串行；选择器允许在 YOLO 强正例中采用 mode 10、在 ResNet18 低收益案例中回退。下一正式阶段需要
在真正变化的 boot ID 上原样复测，不再修改 mode 10；之后 selector v3 才能把
`completion_policy/composability_certificate` 纳入候选身份，并使用新的未见任务验证，不能把本轮已
暴露性能的 YOLO/ResNet18 再称为 holdout。

详细证据与下一步见
`00_governance/20260924_C3_COMPOSABLE_RESIDENCY_MODE10.md`。

## 12. 2026-09-24 selector v3 候选集合门禁与 YOLO-608 六轮确认

在 mode 10 的开发计时之后，已把“组合安全”从人工判断实现为三层机器可读证书：单候选本地证书、
静态候选集合证书和精确图上下文证书。当前冻结产物包含3个候选证书、2个静态集合证书和2个图上下文
证书，覆盖 YOLOv3-tiny 与 ResNet18。候选/静态证书不声称整图正确；只有绑定精确 graph、params、
候选集合和多 seed 板端零 mismatch 的图证书允许进入计时。

selector v3 在原 v2 静态代价排序之前增加硬语义门禁，并把部署单位修正为
`exact_graph_candidate_set`。这是必要的，因为 YOLO-608 的13%级收益属于两个 route 同时替换后的
完整图程序，不能归给其中任意一个单层 candidate ID。旧 mode 5 barrier 候选默认拒绝并要求重新生成
mode 10；未知完成策略 fail-closed；incumbent 始终保留为回退。

本轮为 YOLO-608 新冻结了6轮确认合同，stock 与 scoped-composable 在首位各出现3次；每个程序每轮
使用3个固定 seed，先求轮内中位数，再做轮间配对。结果如下：

| 项目 | 结果 |
|---|---:|
| candidate set ID | `d29aa9dac2c7af8ad39e544ba7308053d174972a61f8e317d8fa0d53f966c47c` |
| 中位配对改善 | 13.413% |
| 获胜轮数 | 6/6 |
| 正式计时调用 | 36 |
| 完整图调用（含显式预热） | 48 |
| 输出正确性 | 全部相等 |
| 隐式额外调用 | 0 |

产物哈希完整通过。但板端报告的 boot ID 仍为
`c9d73f4c-c0df-474a-9a74-a28a2f8955e2`，与此前开发实验相同。因此该结果被登记为第一个正式
boot 结果，而不是第二个独立启动。机器门禁当前输出：

- YOLO-608：`PROMISING_REQUIRES_DISTINCT_BOOT_CONFIRMATION`；
- ResNet18 H3：`FALL_BACK_TO_INCUMBENT`，不再为低收益分支消耗正式确认预算；
- 总体：`deployment_authorized=false`。

下一步只需在 boot ID 真正变化后，对完全相同的 YOLO candidate set、二进制、runtime 和6轮合同执行
一次 boot-2。只有两个 boot 各自满足中位改善至少2%、至少5/6轮获胜，selector v3 才输出
`DEPLOY`。在此之前不能把单启动13.413%写成跨启动正式收益。

新增关键产物：

- `07_grouped_holdout/20260924_c3_composability_certificates_run01/`；
- `07_grouped_holdout/20260924_c3_yolo608_composable_confirmation_boot1_run01/`；
- `07_grouped_holdout/20260924_c3_selector_v3_development_evidence_run01/`。

## 13. 2026-09-24 从 mode10 特例到通用图候选集合

selector v3 已进一步去除对数字 mode10 和固定 YOLO route 的方法级依赖：新增版本化完成策略注册表，
将局部 completion、普通依赖执行、旧未闭合 barrier 和全局串行诊断明确分型；新增精确图候选集合
身份，绑定 workload、图节点、生成函数、完成策略和候选证书。有界集合枚举只在声明的 per-task 池与
`max_replacements` 内称完整，组合积超过上限时直接拒绝，不静默截断。

真实 YOLO-608 两 route 已按 workload 的输入/权重形状唯一映射到节点38/函数6和节点61/函数10，
冻结为通用 candidate set `0eaab412...33377`。旧正确性与 boot-1 时间通过 graph、params、候选成员
及函数作用域的精确相等迁移到新身份，迁移记录明确 `latency_values_modified=false`。集合级短名单只
含 incumbent 与该候选集合，二者均具备图计时资格；部署门仍只因缺不同 boot ID 而回退。

第三创新点由此可以收敛为“组合安全的驻留—Tile 联合选择”，但当前仍不能声称跨任务通用性能：
方法接口与真实两-route实例已经成立，外部有效性还需要新的 latency-unseen 图任务、三 route 以上
组合，以及相同预算下对 bytes/calls/random 的选择比较。详细边界见
`00_governance/20260924_C3_SELECTOR_V3_GENERALITY_AND_NOVELTY.md`。

新增产物：

- `07_grouped_holdout/20260924_c3_yolo608_graph_candidate_set_run01/`；
- `07_grouped_holdout/20260924_c3_selector_v3_graph_set_plan_run02/`。该修订产物把无图级预测分数时
  的角色明确记为`selector_v3_static_fallback`，并冻结`performance_model_claim=false`；因此当前结果
  只证明集合级资格、计时身份与回退门禁闭合，不证明已建立completion-aware性能预测模型。

## 14. 2026-09-24 completion-aware selector v3 合同与数据准备度

已新增独立的`c3_selector_v3_model.py`，模型单位严格限定为“精确图候选集合相对其incumbent”，而非
孤立算子。特征包括请求/调用/UOP/指令/提交/同步的相对量、arena差值，以及replacement数、局部完成
数、阻塞完成数、debug状态转换数、函数作用域数和完成策略计数。时延字段禁止进入特征。

模型训练采用按执行任务留一：缺失值填补和median/IQR缩放均只在训练折拟合；正则选择先最小化每个
留出任务等权的simple regret，再比较macro MAE，避免候选较多的任务主导模型。只有同时满足下列条件
才允许生成`completion_aware_ridge`：

- 至少8个精确图执行任务、40个候选集合—incumbent比较，目标覆盖率至少80%；
- 至少3个scoped-completion任务和3个纯dependency-token任务；
- 全部训练行硬件指纹一致、图上下文正确、FPGA正确，并具有匹配incumbent聚合指标；
- 推理分数必须由冻结合同现场产生，调用方预填的预测值会被忽略。

现有证据准备度审计的结果是：v2的121行/11任务均为孤立算子标签，不能改名为图级标签；旧barrier
不能替代scoped-completion语义。YOLO只有一个两route集合、一个boot标签，且旧集合产物缺少匹配的
incumbent聚合指标；ResNet18只有开发负例。因此当前 admitted training rows 为0，正式合同按设计
冻结为`static_fallback`，不是伪造一个模型。真实YOLO计划run03已经读取该合同，输出
`model_contract_not_completion_aware_ridge`和`performance_model_claim=false`。

通用集合构造器现已支持同时聚合candidate与incumbent指标；指标不参与program/set ID，因此不会因
补充性能特征改变程序身份。后续每个新图池在板测前就应生成这两套匹配指标，并把选择数据和确认数据
分离。只有数据门槛达到后，同一graph-set plan接口才会切换到模型排序。

新增产物：

- `07_grouped_holdout/20260924_c3_selector_v3_model_readiness_run02/`；该版本额外绑定模型源码哈希；
- `07_grouped_holdout/20260924_c3_selector_v3_graph_set_plan_run04/`。

## 15. 2026-09-24 同预算完整池评价协议

已在读取任何新图池时延前冻结selector v3的同预算评价器。B1/B2/B3均表示额外尝试1/2/3个新
candidate set，incumbent共同成本另列；FPGA-invalid尝试消耗预算且不补位。每个方法只能决定尝试
顺序，已测候选中仍按真实时延选最快者。

对照固定为incumbent-only、bytes-only、calls-only、command-only、selector v3和1000个固定随机
顺序（seed 20260924）。报告simple regret、相对incumbent改善、oracle+1%/+2%命中和invalid尝试数。
oracle仅指“该参考任务事先声明且完整测量的池内FPGA-valid程序”，不能推广为所有VTA程序最优。
selector排序只有在每个候选分数都来自冻结的completion-aware合同才启用；调用方注入的分数不会被
接受。选择数据仍不得作为最终稳定改善确认数据。

冻结产物`07_grouped_holdout/20260924_c3_selector_v3_evaluation_protocol_run02/`不读取性能标签、不接触
开发板，仅预注册方法、预算、指标和边界，因此不能单独证明selector优于任何基线。

## 16. 2026-09-24 mode10 前驱敏感性纠正与 mode11 入口排空

第11--13节中把 mode10 称为“组合安全”的表述现被后续跨程序诊断收窄：mode10 在其已经执行过的
精确 YOLO-608/ResNet18 图上下文中仍保留正确性与开发计时证据，但不能再作为任意 VTA 程序序列中的
生产级完成策略。YOLO-576 诊断证明 mode10 独立执行、仅预加载其他 DSO、跟随 barrier 程序时可以
正确；跟随普通 input-stationary 或 original 程序时会错误。重建 host CommandQueue 无效，而在
debug flag 仍为0时执行一次空 `VTASynchronize` 可令3/3前驱—后继对恢复正确。因此证据支持的是
FPGA 侧依赖/FIFO 状态需要在进入 force-serial 段前排空，不支持把差异归因于 DSO 符号冲突，也不支持
声称已定位到某个具体周期或硬件 token。

新增 mode11 `input_weight_resident_transition_safe` 保持 mode10 的 Tile、循环、驻留和数据请求语义，
把完成合同改为：

```text
入口 VTASynchronize（debug=0）
→ VTASetDebugMode(32)
→ 驻留命令段
→ VTASynchronize
→ VTASetDebugMode(0)
→ 函数正常结束同步
```

mode10 仍保留为历史前驱敏感反例，但其策略被标记为 `production_eligible=false`；旧 mode5 和 mode10
都必须重新生成成 mode11 才能进入 selector。mode11 的入口排空成为候选身份与代价特征，不能把它
隐去后沿用 mode10 的旧 latency 标签。

当前完成的 mode11 资格证据为：

- 聚焦回归 46/46 通过；入口和段尾同步使用不同 TIR 节点，避免静态审计漏计；
- 3个真实 YOLO-576 `1×1, CO=256` 执行任务、9个 Tile 家族、45个五模式候选全部通过实际
  lowering、StorageRewrite 后地址/容量/依赖审计和三 seed FSim，45个程序均不等价；
- 9个 mode11 候选的 debug 转换均为 `[32,0]`，第一个同步事件位置均为0，静态同步多重度均为
  `[1,1,1]`；45/45 AArch64 交叉编译且 re-lowered TIR 哈希一致；
- 同一 boot `c9d73f4c-c0df-474a-9a74-a28a2f8955e2` 上，对3个任务分别冻结11个 fresh-RPC/
  bitstream-reload case，覆盖独立三 seed、同 Tile 的 Input/Original 前驱、跨 Tile 前驱和三程序混合
  前缀；合计33/33 case、129次算子调用、0 mismatch，未请求 latency。
- 随后按完整45候选冻结顺序在一个连续 RPC 会话中执行三 seed FPGA correctness：45/45候选、
  135/135调用全部正确，只使用1个wave，没有失败后重载、候选替换或结果拼接，仍未请求 latency。

这组结果只证明精确测试任务和转换中的前驱安全性改善。它仍是单 boot；没有覆盖所有 VTA 程序、完整
真实完整图或 mode11 性能。完整45候选同会话 FPGA correctness 已通过，因此下一门禁允许冻结只含
Original、Input 和 mode11 的开发计时池；该池必须标记任务 latency 已暴露、mode11程序未计时，不能
冒充 strict-unseen holdout。旧 mode10 的13%级开发结果不能自动迁移成 mode11收益。

详细记录见 `00_governance/20260924_C3_TRANSITION_SAFE_MODE11.md`。关键产物：

- `07_grouped_holdout/20260924_c3_yolo576_transition_safe_route_pools_run01/`；
- `07_grouped_holdout/20260924_c3_yolo576_transition_safe_local_qualification_run01/`；
- `07_grouped_holdout/20260924_c3_yolo576_transition_safe_board_pool_freeze_run01/`；
- `07_grouped_holdout/20260924_c3_yolo576_transition_safe_cross_compile_run01/`；
- `07_grouped_holdout/20260924_c3_yolo576_transition_safe_matrix_y256_run01/`、`..._y512_run01/`、
  `..._y1024_run01/`；
- `07_grouped_holdout/20260924_c3_yolo576_transition_safe_full_pool_correctness_run01/`。

## 17. 2026-09-24 mode11 开发计时与 selector 负结果

在任何 mode11 时延暴露前，已冻结只含 Original、Input 和 mode11 的27程序开发计时池：3个真实
YOLO-576 `1×1, CO=256`任务，每任务3个 Tile 家族、每 Tile 3种模式。任务本身已有历史时延，故
`task_latency_unseen=false`；只有 mode11 的精确程序在冻结时尚未计时。本轮不构成 strict-unseen
holdout，也不用于跨网络或跨启动结论。

同一 boot `c9d73f4c-c0df-474a-9a74-a28a2f8955e2` 的完整开发会话中，27/27程序通过三 seed FPGA
correctness；共完成711次算子调用，其中81次资格正确性、36次选择计时、108次确认计时、486次完整池
参考计时。全部输出正确，隐式额外调用为0，会话未拼接。

机制级结果支持 mode11 在这组访存请求很重的真实 `1×1` 层上有净收益：3/3池内 oracle 都是
mode11；9个同 Tile 的 Input→mode11 配对全部获胜，每个配对均为6/6轮获胜，配对改善中位数为
52.951%，范围为5.588%--73.790%。同 Tile 权重逻辑请求从 Input 的393,216--21,233,664 B降为
mode11的65,536--262,144 B；同时实际提交数由1增至3，部分候选 UOP 成本也上升。因此证据仅支持
“在这些层中传输节省超过了入口排空和闭合提交成本”，不等同于测得了DRAM等待周期或带宽瓶颈。
这也解释了为什么不能从H1的约1%收益推广到所有层：驻留收益具有明确的工作负载与 Tile 条件性。

本轮也得到必须保留的 selector 负结果。冻结的多角色短名单在B3只命中2/3个池内 oracle；简单
bytes-only在B2已命中3/3，calls-only也优于以提交/UOP为字典序首项的静态兜底。失败任务中，多个
mode11程序都具有很低的权重请求，但不同 Tile 的时延差异很大；原短名单只保留一个最低字节 Tile，
遗漏了同为mode11的oracle Tile。故当前数据证明了驻留机制，不证明 selector v3 优于简单基线。

下一版 selector 必须作为新的开发版本冻结：先在低传输候选窗口内保留 Tile 多样性，再生成
B1/B2/B3前缀；不得用本轮暴露任务作为未见验证，也不得把静态支配当作速度保证。冻结后再选择新的
latency-unseen任务验证。当前没有完整图、跨boot或性能模型成功主张。

新增关键产物：

- `07_grouped_holdout/20260924_c3_yolo576_transition_safe_timing_freeze_run01/`；
- `07_grouped_holdout/20260924_c3_yolo576_transition_safe_timing_boot1_run01/`；
- `07_grouped_holdout/20260924_c3_yolo576_transition_safe_timing_analysis_run02/`。

## 18. 2026-09-24 selector v4、YOLO-544 严格未见 boot-1 与生产资格纠正

在 YOLO-544 的任何板端时延暴露前，已冻结 selector v4：它不是性能模型，而是在编译后逻辑请求较低
的固定窗口中，用归一化 Tile/线程坐标做确定性 farthest-first 排序；B1/B2/B3 是同一个顺序的前缀，
窗口外候选不被安全剪枝。608和576已经暴露，因此新分辨率审计按预注册顺序选择544；该图含11个唯一
VTA执行任务，其中3个真实 `1×1, CO=256`任务生成了每任务15个五模式程序。静态/FSim资格为43/45；
CI1024任务的冻结incumbent因最终TIR审计展开超界而整体不进入参考评测，没有替换incumbent或补点。

余下两个完整任务共30个程序全部交叉编译，并在时延之前完成30/30候选、90/90调用的三seed FPGA
正确性。boot-1完整会话又完成30/30正确性、24次选择、72次确认和540次完整池参考计时，共726次调用，
全部输出正确、0隐式调用、会话未拼接。按原始15程序孤立算子池，v4在B1命中2/2池内oracle；bytes-only
和calls-only均在B2命中，command-only到B3仍未命中。六个同Tile Input→mode11配对全部6/6轮获胜，
中位改善36.981%，范围13.427%--69.433%。这只是一种YOLO分辨率、两个任务和一个boot的证据，不能
据此宣称selector已具备跨网络通用优势。

本轮同时暴露并保留一个协议缺陷：两个15程序池的孤立算子oracle都是旧
`input_weight_resident_barrier` mode5，但现有完成策略注册表早已把mode4/mode5标记为
`production_eligible=false`并要求重新生成为mode11。原冻结器在v4排序前漏用了这道硬门，因此原
B1结果只能称“孤立算子参考池选择”，不能称可部署选择成功，也不能直接进入完整图。

修复采用独立新模块，不修改已冻结v4源码与历史产物：`c3_selector_v4_deployment.py`先调用既有
`candidate_disposition`，再把生产合格的新候选交给label-free v4排序；legacy候选仍可留在完整机制
参考池，但不得占用部署预算。聚焦回归68/68通过。看过boot-1时延后的审计拒绝了6个mode4与6个mode5
程序；每任务留下incumbent加8个合格新程序。在这两个九程序生产子池中，mode11均为最快程序，修复后
顺序均在B1命中。该结果只是`strict_unseen_claim=false`的事后协议审计，不能反向重标为未见验证；
下一新holdout必须在时延前执行生产门。

因此当前最准确的创新点表述进一步收敛为：

> **组合安全、编译后代价感知的驻留—Tile联合选择：先以容量、最终地址、FPGA正确性和完成策略硬门
> 区分参考候选与可部署候选，再在生产合格池中依据请求/命令代价与Tile多样性安排有限测量，最终以
> 精确图上下文正确性和独立启动确认决定部署。**

boot-1中相对冻结pool incumbent的65%/92%确认值不代表生产TopHub或整网收益。板卡当前报告的boot ID
仍为`c9d73f4c-c0df-474a-9a74-a28a2f8955e2`，与boot-1相同；在boot ID真正变化前不执行或登记boot-2。
当前也没有完整图正确性或端到端性能结论。

新增关键产物：

- `07_grouped_holdout/20260924_c3_selector_v4_diverse_development_freeze_run01/`；
- `07_grouped_holdout/20260924_c3_yolo544_selector_v4_board_pool_freeze_run01/`；
- `07_grouped_holdout/20260924_c3_yolo544_selector_v4_fpga_correctness_run01/`；
- `07_grouped_holdout/20260924_c3_yolo544_selector_v4_timing_boot1_run01/`；
- `07_grouped_holdout/20260924_c3_yolo544_selector_v4_boot1_analysis_run01/`；
- `07_grouped_holdout/20260924_c3_yolo544_selector_v4_deployment_gate_posthoc_audit_run01/`。
- `07_grouped_holdout/20260924_c3_selector_v4_production_gate_freeze_run01/`。

## 19. 2026-09-24 修复后首个 strict-unseen YOLO-512 池

YOLO-544时延暴露且生产门合同冻结后，分辨率注册表按原顺序硬排除608/576/544，并在不读取任何新
latency时选择512。YOLO-512静态图合同为114,504,192 B，小于192 MiB；含11个唯一VTA执行任务，11个
任务均有至少8个静态候选身份。真实lowering门生成3个K1/CO256任务、45个五模式候选：
`CI256,H32`、`CI512,H16`、`CI1024,H16`，每任务3个Tile家族。

本地最终地址/容量/依赖与三seed FSim资格为43/45；43个通过程序全部FSim正确且互不等价。
`CI256,H32`为13/15，不补点、不作为完整池；后两个任务均15/15，按稳定task ID形成两个完整参考池。
冻结器在时延前把每任务15程序的孤立参考池与每任务9程序的生产池分开；生产B1/B2/B3只在通过
completion-policy/composability硬门的Original、Input和mode11中形成。30/30程序随后交叉编译通过。

correctness-only板端执行保留了三个基础设施记录：run01在第14个候选后因`/var/volatile`满而无法部署
新wave；run02在候选调用前因bitstream文件缺失失败；run03在候选调用前因重启后`taskset`悬空链接
失败。旧实验的板端临时副本从volatile清理后，使用SD上的同哈希bitstream和历史同哈希taskset恢复；
恢复器新增只删除悬空软链接的处理，不绕过CPU3或哈希门。以上失败会话均未拼入最终结果。

fresh run04结果为28/30 FPGA-correct、2个FPGA-rejected、86次算子调用、3个wave、未采集latency。
两个拒绝均是各任务第三Tile家族的Original，分别在seed0产生36,703和35,080个mismatch；二者不是
incumbent，也不在冻结B3中，因此每任务生产池剩8个FPGA-valid程序，而incumbent与B1/B2/B3均保持
有效。拒绝候选不被替换，仍计入完整池资格成本；oracle只允许在事先声明池的FPGA-valid子池内定义。

计时合同现已冻结为`FROZEN_AFTER_FPGA_CORRECTNESS_BEFORE_FIRST_LATENCY`，明确
`latency_values_collected=false`、invalid消耗预算且不补位。当前boot ID仍为
`c9d73f4c-c0df-474a-9a74-a28a2f8955e2`，因此尚未执行YOLO-512选择/完整池时延。这个池是修复后的
首个真正生产门前置holdout，但在实际时延出现前不能主张selector命中oracle或优于基线。

新增关键产物：

- `07_grouped_holdout/20260924_c3_yolo_unseen_resolution_prepare_run05/`；
- `07_grouped_holdout/20260924_c3_yolo512_transition_safe_route_pools_run01/`；
- `07_grouped_holdout/20260924_c3_yolo512_transition_safe_local_qualification_run01/`；
- `07_grouped_holdout/20260924_c3_yolo512_selector_v4_production_board_pool_freeze_run02/`；
- `07_grouped_holdout/20260924_c3_yolo512_selector_v4_production_cross_compile_run01/`；
- `07_grouped_holdout/20260924_c3_yolo512_selector_v4_production_fpga_correctness_run04/`；
- `07_grouped_holdout/20260924_c3_yolo512_selector_v4_production_timing_freeze_run01/`。
