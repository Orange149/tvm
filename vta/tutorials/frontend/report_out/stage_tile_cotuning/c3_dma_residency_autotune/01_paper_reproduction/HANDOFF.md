# C3-P1 Handoff

任务编号：C3-P1-A1  
阶段：P1 论文规格与复现合同  
状态：`full_text_audited`

> 2026-09-12 更新：下述最初检索失败是历史记录。用户随后提供正式全文，正文边界以
> `PAPER_METHOD_SPEC.md`、`REPRODUCTION_GAPS.md` 和 `SOURCE_LEDGER.md` 的最新版本为准。

## 已完成

- 核对 ScienceDirect 正式版、SSRN 2025 预印本元数据、DOI、ResearchGate/OUCI 记录；
- 将 input-prioritized、on-chip weight reuse、minimum-data-access 逐项拆成已证实、合理推断和未知；
- 明确记录当前无法提供论文正文页码、图号、表号或伪代码，未臆造；
- 映射到当前 `vta_conv2d.py` 的 output reorder、cache_read、`compute_at(k_o)`、AutoTVM knobs、DMA pragma 和 tensorize 行号；
- 建立本平台独立重实现合同和禁止等价推断清单；
- 决定模式名降级为 `paper_inspired_hybrid`。

## 文件与代码变更

只新增 `01_paper_reproduction/` 下六份 Markdown。没有修改 Python/C++、schedule、`00_governance` 或其他实验目录。

## 检索与失败

- Web 检索成功获得正式摘要/highlights/引言片段和预印本元数据；
- ScienceDirect 正文和 SSRN PDF 均返回 403；ResearchGate 明示无全文；
- 命令行下载因当前网络不可达未取得文件；未绕过访问控制；
- 未发现作者公开代码或补充材料。

## 更新后的 Gate

- 方法规格：通过，正文已足以实现功能级复现；
- code-exact reproduction：仍不通过，因为作者源码和 TVM commit 未公开；
- 本地 barrier 与论文 weight reuse 等价：不通过，二者实现语义不同；
- 搜索问题边界：通过；全文没有定义 AutoTVM 等预算搜索方法或搜索成本实验。

后续可以按正文复现具体循环和权重复用语义，但不得臆造论文 tuner；论文没有给出该 tuner。

## 未解决风险

- 若论文正文中的 input-priority 或 weight reuse 不等于本地 stationarity 设计，B1 只能是受启发基线；
- 论文平台未知，不能把其约 10% 与本地 ResNet18/AXU5EVB 直接比较；
- 本地 schedule 变换能否通过 StorageRewrite、DMA 注入和数值验证仍待 P3。

## 下一任务建议

主代理先验收本阶段。只有 P0 板端身份/manifest 解锁且 P2 强基线恢复通过后，才允许 P3 修改 schedule；当前板端阻塞与 P1 文献结论相互独立。
