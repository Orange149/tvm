# C3-S S0：输入池别名审计与真实分配基线

日期：2026-09-08。结论：**静态安全类别与实际分配基线可对账，可进入受限 S1；尚未实现借池或证明优化收益。**

## 冻结与方法

A/B/C/D 固定为 rank01/05/07/13，不根据节省大小选拓扑。
`s0_static_run1/{A,B,C,D}.json` 记录所有 graph、graphlib、params、原运行库哈希、
各 stage 的全部 entry/alias/readers/writers，以及逐边分类。
数据包来自既有 Top-20，逐文件匹配旧 G0 冻结 SHA-256；未重编译计算图。

审计工具 `audit_shared_buffer_storage.py` 读取真实参数名字，并检查完整 ELF 符号表排除
linked-parameter lookup。首版仅 CPU→VTA、独占 storage-id 的普通输入池。
未知、反向边、内部别名、参数及跨边冲突均不能借用；多张量必须整边通过。
`eligible` 仅表示静态条件成立，输出始终 `static_only=true, runtime_qualified=false`。

源码生命周期依据：GraphExecutor 的 `SetInputZeroCopy` 只改该 input entry 对应的
op 参数；`SetupOpExecs` 收集全部直接使用，原 `data_entry_`/`storage_pool_` 仍保留。
因此独占输入池在绑定 slot1 后，没有其他内部 entry 继续访问原池 slot0。
这不适用于多个 entry 共用同一 storage-id 的情形。S1 仍须检查真实地址范围、
接口和 owner，保留 slot 状态机，并用跨帧压力验证，不能把源码审查当实测安全证明。

## 上板基线（不是性能确认）

Boot：`a5220e22-f562-40be-bbe9-058b6fbeed50`；CPU 1066666 kHz；
FPGA operating；u-dma-buf 201326592 B，物理基址 0x68500000。
bitstream SHA-256 与既有基线一致，无重部署/重启。

每拓扑按旧 runner+旧 driver、新 runner+旧 driver、新 runner+诊断 driver 三模式，
各启独立进程，2 帧 warmup +22 正式帧。12 次运行、264 个正式输出的 FNV1a64 均与
冻结旧路径相符。这是固定输入的兼容性烟测，不是多输入逐元素对照、1000 帧压力或无损 FPS 证明。
没有借池，三模式都是 external-K2；没有覆盖旧 runner/runtime/package。
构建与命令、stdout/stderr、原始 JSONL、预注册和结果见 `build_s0_run1`、`s0_board_run2`。
`s0_board_run1` 在 SSH 沙箱阶段失败，无板端样本；保留空目录，不混入结果。

诊断只增加计数和只读快照，观察本身不初始化/分配池；首次快照 initialized=0、used=0。
原图池与新增 slot 的预测/实测在四拓扑均逐字节一致，本次各池分配均已对齐，padding=0。

| 拓扑 | 图/参数后实际 B | 新增 external-K2 B | warmup/稳态总高水位 B | 可借池数 | 预测可省 B | 预测 slot 降幅 | 预测总高水位降幅 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 9,038,848 | 1,806,336 | 77,954,048 | 1 | 802,816 | 44.44% | 1.03% |
| B | 8,880,640 | 2,207,744 | 78,197,248 | 1 | 802,816 | 36.36% | 1.03% |
| C | 9,013,760 | 2,007,040 | 78,129,664 | 1 | 802,816 | 40.00% | 1.03% |
| D | 12,880,896 | 2,609,152 | 82,598,912 | 2 | 1,003,520 | 38.46% | 1.21% |

右三列都是 **预测**，尚无 pool-anchor-K2 实测。四拓扑合计 5 个候选张量；
A/B/C 的 stage1 输入均为 802816 B，D 的 stage1/stage3 分别 802816/200704 B。
其余 5 条 VTA→CPU 边按 v1 规则保留 external，不表示这些边永远无法复用。

## 关键修正：不能遗漏运行时队列

四拓扑 warmup 均额外分配 67108864 B，分配计数增加 2；warmup 后到22帧结束高水位增长为0。
与源码 `CommandQueue` 的 uop/insn 两队列、每队列 `VTA_MAX_XFER=(1<<25)` 的
`BaseQueue::InitSpace` 预分配一致。按源码及计数将其归为运行时队列，而不是 graph pool 或 slot。
这说明此前只相加 graph pool+slot 会明显低估总分配；本次不修改队列大小，该方向不在计划范围。
22 帧稳定仅是短程证据，1000 帧压力仍未完成。

## S0 门槛与下一步

明确独占输入类别、非零节省、实测分配对账三项已满足；源码未发现该受限类别的未解决别名风险。
允许进入 S1，但必须落实 owner 持有、物理验证与整边回退，不能仅根据本 JSON 直接启用借用。
第三项仍未成立：`--shared-buffer-plan`、借池构造、物理拒绝案例、逐帧不同输入、压力、三 boot ABBA
均待实现/验证。预注册的 ≥20% 是新增 slot 分母，总高水位预计仅约1%，两者必须同时展示。
