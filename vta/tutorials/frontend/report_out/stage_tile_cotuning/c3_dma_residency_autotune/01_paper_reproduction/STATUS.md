# C3-P1 状态

任务编号：C3-P1-A1  
阶段：2026 论文规格与复现合同  
状态：`full_text_audited`  
Gate：`CHENG2026_FULL_TEXT_METHOD_AND_SEARCH_BOUNDARY_AUDIT_COMPLETE`

## 结论

- 论文方法级复现：正文规格已具备；作者源码/commit 不可得，因此不能称 code-exact。
- 当前本地 `weight_resident_barrier`：只能称相同高层目标的功能级替代，不是论文 weight-reuse 实现。
- 论文相对本文搜索创新的边界：已由全文确认。

## 原因

用户提供的 9 页正式全文已经核对 Algorithm 1/2、Fig. 1--3、Table 1/2、四种方案、编译器/runtime
修改位置、SRAM 适用条件及三网络实验。论文没有报告 AutoTVM ConfigEntity 搜索算法、trial budget、
time-to-target、regret 或等预算 tuner 对照；其“tuning”是四种数据访问/调度方案的构造与逐层选择，
不是搜索效率算法。这使本文可辩护的增量收敛为驻留模式与 tile 的联合搜索次序及搜索成本，而非
输入优先、权重驻留或 SRAM 回退本身。

## 产物

- `PAPER_METHOD_SPEC.md`
- `PSEUDOCODE_TO_VTA_MAP.md`
- `REPRODUCTION_GAPS.md`
- `SOURCE_LEDGER.md`
- `STATUS.md`
- `HANDOFF.md`
