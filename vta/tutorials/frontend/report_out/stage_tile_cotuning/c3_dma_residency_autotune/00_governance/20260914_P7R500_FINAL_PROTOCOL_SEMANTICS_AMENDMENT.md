# P7R500--P7R501：ResNet18 文献对齐最终协议的信息与成本语义修订

状态：`FROZEN_BEFORE_ANY_MERGED_POOL_FPGA_OR_RESNET18_FULLGRAPH_LABEL`

P7R500 不改变 P7R499 已冻结的 214 个候选身份、候选顺序、ARM 二进制、20 个搜索种子、预算或
整图执行协议。它只修正完成性审计发现的两项分析语义，避免上板以后再依据结果修改规则。

## 1. ML²Tuner 的删失标签

此前实现会把“lowering 和交叉编译成功、但未被 Model A 提升到 FSim/FPGA”的候选标为最终
`valid`。这是错误的乐观标签，因为设备异常和数值错误仍未知。修订后：

- lowering/交叉编译失败是可确认的负标签；
- 编译成功但未提升的候选保持 `unknown/censored`，不进入 Model V 训练；
- 只有经过 FSim 和 FPGA correctness 的候选才获得最终正/负有效性标签；
- 板后按 workload 留一报告 Model V 的 precision、recall、F1、accuracy、nDCG、top-20 valid
  yield 和全池基率；P/A 仍报告 RMSE 与 nDCG，A+DMA 仍是独立消融。

## 2. HW-Aware 初始化成本

此前性能回放只给 E0 中真正进入性能测量的候选计时，没有把生成 E0 的完整 presampling lowering
计入第一次命中之前；同时 validity 排序会被随后的性能 XGB 覆盖。修订后：

- 每个 seed 先按完整 original ConfigSpace 重建真实 presampling 次序；
- H1/H2 各预付 1000 次 lowering，H3 预付完整 480 次；
- 使用 P7R484--P7R486 保存的逐候选实测 wall，而不是估算平均值；
- 已经 presample 的 original 后续被性能测量时不重复收取 lowering 成本；
- balanced valid-E0 之后，使用“共同 visible-feature performance XGB 排名 + 目标域
  lowering-validity XGB 排名”的 rank-sum 选择，确保 validity bias 真正参与后续搜索。

真实重建得到 60 条 seed×workload presampling 轨迹。以 seed 57001 为例，H1/H2/H3 的累计
lowering 墙钟分别为 24.824011、22.458119 和 8.620643 秒。这些成本从此位于首次性能 dispatch
之前，因而 HW-Aware 即使最终更早命中优解，也不能靠隐藏初始化工作获得不公平优势。

## 3. 验证与边界

- 新增删失标签、编译失败负标签、未提升候选隔离、Model V 指标和 HW-Aware 不重复计费测试；
- 相关测试 26/26 通过；完整 60 条 presampling 轨迹由真实 P7R470/P7R484--P7R486/P7R491
  身份重建并通过一致性检查；
- P7R500 本身未连接开发板、未读取目标 FPGA correctness 或 latency，不能产生任何性能主张。

冻结产物：

`07_grouped_holdout/20260914_p7r500_resnet18_final_execution_amendment_run01/`

P7R501 在同样无标签条件下再补齐汇总输出字段：六类策略均报告 gross candidates、compiler
attempts、FPGA dispatch、kernel invocation、逻辑 DMA bytes/calls，以及 lowering、FSim、
cross-compile、correctness、timing 五阶段墙钟；oracle+2% 和 +5% 均报告首次命中的 trial 与真实
累计 wall。它仍不改变任何候选或板端执行顺序，并 supersede P7R500 作为最终分析语义合同：

`07_grouped_holdout/20260914_p7r501_resnet18_final_execution_cost_amendment_run01/`

P7R502 又统一了三对三 clean-start 的起点：官方空历史 AutoTVM-XGB 与本文执行器都必须在 T0 前
核对默认 runtime 文件哈希和 RPC 工作目录；本文执行器还会验证 ResNet18 source contract 的完整
artifact ledger。该修订避免一边在未知 runtime 上起跑，也不引入任何目标标签：

`07_grouped_holdout/20260914_p7r502_resnet18_clean_start_preflight_amendment_run01/`

P7R503 修正完整 214 点板池的 RPC 生命周期。TVM `session_timeout` 是整个 session 的最长持续时间，
旧默认 120 秒无法覆盖至少 642 次 correctness 和最多 1070 次 timing invocation，且中断后又禁止
拼接。完整池专用上限因此改为 7200 秒；这不是把单次算子超时放宽，而是防止服务器按总时长主动
杀死仍在正常执行的不可拼接 session：

`07_grouped_holdout/20260914_p7r503_resnet18_long_rpc_amendment_run01/`

P7R504 在同样无标签条件下补齐每次板池实验的统一审计文件。正常和失败路径都会生成
`contract.json`、`timeline.jsonl`、`results.jsonl`、`summary.json` 和 `artifact_hashes.json`；失败
路径还保存 `invalid_session.json`、继续抛出异常并明确禁止拼接。文献对齐本地测试在 FSim 环境下
50/50 通过。该修订不改变候选、板端顺序、二进制、seed、预算或测量次数：

`07_grouped_holdout/20260914_p7r504_resnet18_common_audit_artifacts_amendment_run01/`

P7R505 把相同要求扩展到策略所选完整图和两条 clean-start 路径。三类 runner 的失败 session 都会
保留五类统一审计文件和 `invalid_session.json`；本文在线搜索还会在当前候选发生 RPC/设备异常时
先记录该 gross dispatch、已完成调用和实际墙钟再使整次 session 失败。官方 XGB CLI 对预先存在的
输出目录只抛出 `FileExistsError`，不再向用户旧证据写入失败文件。完整本地套件 54/54 通过：

`07_grouped_holdout/20260914_p7r505_resnet18_all_board_artifacts_amendment_run01/`

P7R506 修正 Cheng 四方案中的 pure-instruction 单位。`timed_reused_call` 的 runtime profile 汇总
time evaluator 执行的两次完整 `main`，所以每算子 pure-instruction 必须除
`time_evaluator_total_kernel_invocations`；不能除 `driver_run_calls`，后者会把 barrier 模式一次算子
内部的多个 submission 错当成多个算子。新增 34 次 driver call/2 次完整算子的测试，完整套件
55/55 通过：

`07_grouped_holdout/20260914_p7r506_resnet18_pure_instruction_normalization_amendment_run01/`

P7R507 将该指标改为 fail closed：若任一正确候选的 timing profile 缺少
`driver_run_total_us` 或完整算子调用数非正，板后适配直接失败，不能把缺测值填成 0 ms 后参与
Cheng 最快方案选择。完整套件 56/56 通过：

`07_grouped_holdout/20260914_p7r507_resnet18_pure_instruction_failclosed_amendment_run01/`
