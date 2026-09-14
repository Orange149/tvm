# 无开发板阶段执行补充

日期：2026-09-10  
触发：开发板 SD 卡 EXT4 元数据损坏且分区 100% 占满；用户要求先完成所有不依赖开发板的工作。

## 用户目标调整

当前实现目标优先改为“复现最接近论文的数据访问与调度方法”，不要求为了形式新颖而强行设计不同算法。学术证据标签仍按可获得材料决定：在未取得正文/代码前，本地实现记为 `paper_inspired_hybrid`；若用户后续提供合法全文，则新建补充记录并逐项核对，允许升级为 exact reproduction。

## 临时阶段权限

在 P2 板端 Gate 尚未通过时，允许并行进行以下本地工作：

- `P2-local`：TopHub 覆盖、原模板、compile-only、FSim/LLVM oracle 和静态 DMA 基线；
- `P3-local`：独立驻留模板的最小实现、lower/FSim/TIR/DMA 机制测试；
- `P4-local`：stable candidate ID、离线特征 schema、失败分类和现有数据适配。

这些工作不得：

- 声称恢复了 10.724 FPS 或任何板端性能；
- 把 FSim 时间作为开发板时间；
- 将逻辑 DMA 请求当作 AXI 物理事务；
- 修改 runtime、driver、RTL、bitstream 或 u-dma-buf；
- 覆盖原模板、TopHub 或用户现有脏改动。

## 本地 Gate 标签

- `G1_PAPER_INSPIRED_CONTRACT`：P1 文献合同完成；exact reproduction 暂不成立。
- `G2_LOCAL_BASELINE`：仅代表本地 TopHub/编译/仿真/正确性链恢复，不解锁性能主张。
- `G3_LOCAL_MECHANISM`：仅代表新 schedule 能正确 lower/FSim，且 TIR/DMA 机制有预注册变化。
- `G4_LOCAL_TOOLING`：仅代表候选身份和离线特征工具通过单元测试。

真正上板前仍必须：修复并留出 SD 空间、确认文件系统错误不再增长、冻结新 boot/bitstream/ko/runtime/runner/频率/RPC 指纹、运行 TopHub canary，并恢复 topology-B 到历史值 5% 范围内。

