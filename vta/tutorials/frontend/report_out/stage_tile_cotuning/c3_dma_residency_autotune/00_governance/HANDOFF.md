# C3-P0 Handoff

> 历史说明：本文件记录最初 P0 交接，不是当前执行状态。P7R348 的权威状态见
> `progress.json`、`C3_CCF_C_SHARED_MEMORY_INNOVATION.md` 和
> `20260913_P7R337_P7R348_R50E_OPERATOR_PROXY_HOLDOUT_SESSION.md`；当前板端已恢复并完成 R50E
> cheap-proxy 前瞻留出，下面的“板端仍被禁止”只描述当时的 P0 状态。

任务编号：C3-P0-A1  
阶段：P0 治理冻结与证据盘点  
状态：`conditional_completed`

## 已完成

- 冻结主仓/VTA 子模块 commit、dirty diff hash、porcelain hash 和脏项数量；
- 冻结 VTA config、TopHub、位流、u-dma-buf ko、模型参数、输入、核心 schedule/runtime/driver/runner/调优工具和历史强基线 manifest；
- 冻结 10 个 workload 的 development/grouped-holdout 分组和 3 个额外几何 holdout；
- 冻结 candidate ID、B0--B8、P6/P7 总预算、correctness seeds、2%/5%门槛、统计单位和故障恢复流程；
- 明确 B2026 学术边界和 claim 降级规则；
- 未修改 schedule，未连接开发板，未运行性能实验。

## 验收结论

G0 本地治理部分通过：可以进入 P1，只读整理论文复现合同。

G0 板端部分未通过：P2/P3 和任何性能实验仍被禁止。原因是本次重启后的 SSH RSA 指纹与用户 known-hosts 及两份历史冻结指纹均不同，尚未由开发板本地终端确认。

## 下一步

1. P1 子代理只读论文与源码，写 `01_paper_reproduction/`；不得改 schedule。
2. 用户从板端串口/本地终端确认 `ssh-keygen -lf /etc/ssh/ssh_host_rsa_key.pub` 的输出。
3. 主代理使用实验专用 known-hosts 连接，补齐本次 boot 的 board manifest。
4. 通过 TopHub canary 和 topology-B 5% 恢复门槛后，才接受 G2 并允许 P3。

## 未解决风险

- B2026 正文现已取得并核对；方法级细节缺口解除，但作者源码/commit 仍不可得，本地 barrier 也不等于论文实现；
- 当前 boot 的 bitstream、ko、runtime、driver、runner、频率、governor、RPC 和 boot ID 都未认证；
- 工作树已有用户改动，后续任何源码 patch 必须先逐文件检查重叠；
- 历史 topology-B 参考来自既有证据，本次 reboot 尚未复测。
