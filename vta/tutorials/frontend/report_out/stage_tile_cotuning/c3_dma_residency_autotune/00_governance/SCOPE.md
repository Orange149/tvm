# C3 实验范围冻结

冻结日期：2026-09-10  
状态：`CONDITIONAL_G0`（允许 P1 只读研究；禁止 P2/P3 上板，直至 SSH 主机指纹和本次 boot 实物哈希确认）

## 研究问题

在固定 AXU5EVB/VTA 位流、量化方案、阵列、片上 SRAM 和显式 DMA 共享内存路径下，研究不同卷积几何应选择何种数据驻留方式，以及是否能用 Lowered TIR 中的 DMA 请求形态和候选自身命令资源证书，减少板端无效试验并保护强基线。

方法暂定名：**面向 VTA 显式 DMA 共享内存的驻留策略与请求形态协同调优方法**。

## C3 内部边界

C3 包含：

- B2026 的可审计复现或受其启发的重实现；
- `original/input_stationary/weight_stationary/hybrid` 按 workload 选择；
- 从最终 Lowered TIR 提取分类 LOAD/STORE、请求数、粒度、stride、padding 和 reload；
- 片上 SRAM/UOP 合法性检查，以及胜者的 instruction/UOP/FINISH/replay 资源证书；
- TopHub 与 B2026 双 incumbent 保护；
- grouped replay、前瞻候选和最终 stage/流水验证。

C3 不包含：

- VTA RTL、ISA、DMA engine、硬件预取器或 FPGA 重新综合；
- u-dma-buf ko 修改；
- 把逻辑 DMA descriptor 当作物理 AXI transaction、DDR stall 或带宽实测；
- 把 B2026 原样复现、一般 weight stationary 或普通 SRAM 合法性过滤称为本文首创；
- 重复申报 C1 的 CPU--VTA 划分或 C2 的双 slot/queue 定容。

## 固定平台

- 开发板：AXU5EVB，目标地址 `root@192.168.1.247`。
- VTA：`BATCH=1`、`BLOCK_IN=BLOCK_OUT=16`，input/weight/acc/UOP SRAM 分别为 32/256/128/32 KiB，目标频率 100 MHz。
- 共享内存：u-dma-buf 目标容量 192 MiB；它是研究载体，不单独构成创新。
- 原模板 `conv2d_packed.vta` 和 TopHub 记录必须保持可用。新驻留模式使用独立实验模板；不得向原模板直接追加 mode 后继续解释旧 `config.index`。
- 新候选搜索先使用默认大 command backing。旧 T2656/11008 B 只对旧 schedule 有效，不是新候选硬上限。

## 当前工作树保护

冻结时主仓有 8 项 modified/submodule 状态和 74 项顶层 untracked 状态；VTA 子模块内 `src/axu5evb/axu5evb_driver.cc` 已修改。所有这些都视为用户已有工作，不清理、不 reset、不覆盖。精确状态由 `FROZEN_MANIFEST.json` 中的 commit、diff hash 和 porcelain hash 标识。

## 当前板端阻塞

本次重启后 SSH 提供 RSA 指纹：

```text
SHA256:stfSRlTnQCxxVE7h5vuZQyTN1Bp3MD67S6ULLL6n9Zg
```

它与两份历史冻结指纹均不同；历史指纹也彼此不同，提示板端 host key 可能随重启/根文件系统变化而重新生成，但这不足以自动认证当前主机。用户在板端本地/串口确认之前，不使用 `StrictHostKeyChecking=no`，不更新用户 `~/.ssh/known_hosts`，不上板运行。

## 阶段权限

- 当前允许：P1 文献与源码只读分析、治理文档和独立实验脚本准备。
- 当前禁止：P2 强基线板端恢复、P3 及以后上板、修改原 schedule、启动候选搜索。
- 解锁条件：用户确认当前 RSA 指纹；创建实验专用 known-hosts；读取并冻结 boot ID、FPGA state/bitstream、u-dma-buf ko/容量、runtime、driver、runner、CPU governor/frequency、RPC 参数；随后通过 TopHub canary。

