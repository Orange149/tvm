# 第三创新点：命令共享内存资源证书结果

状态：`FULL_POOL_LOCAL_PASS; EXACT_W05_RUNTIME_MANIFEST_ATTESTED`  
日期：2026-09-11

## 1. 解决的问题

原 VTA runtime 使用 `VTA_MAX_XFER=2^25`，为 instruction 和 UOP 两个队列各申请 32 MiB
FPGA 可访问连续内存，共 64 MiB。它保证简单，但容量与实际模型、调度及命令流无关。在嵌入式
系统中，这会挤占 u-dma-buf 连续物理区，减少可同时容纳的 tensor slot、模型或 Executor 数量，
也增加连续分配失败与碎片化风险。它主要是空间问题，不应强行解释成 FPS 优化。

## 2. 与具体开发板参数无关的规则

对硬件指纹和源码均固定的允许部署身份集合 `S`，令 `B_q(c,b)` 为候选 `c` 的第 `b` 次提交在
队列 `q` 上序列化后的字节数。安全容量为：

```text
C_q(S) = align_A(max_{c in S, b in submits(c)} B_q(c,b))
```

其中 `q∈{instruction,UOP}`，`A` 是目标 allocator 的对齐粒度。本实现保守采用 4 KiB；runtime
最低要求为 256 B。`B_instruction` 包含依赖收尾和 FINISH，两个队列必须在提交设备前同时检查。

证书失效键不是“config 编号”，而是：

```text
(hardware fingerprint, workload shape, residence mode,
 complete ConfigEntity, lowered-TIR hash, runtime-source hash)
```

若有 `J` 个同时存活的 CommandQueue，则空间上界为：

```text
M_command = sum_j(C_instruction(S_j) + C_UOP(S_j))
```

只有生命周期不重叠的队列才能复用同一 backing。因此它可以与论文第二点的 slot/lifetime 规划
组合，但“每个队列多大”与“几个队列同时活着”是两个不同问题。

## 3. 十个 workload、197 个身份的集合验证

利用此前冻结且正确的 197 个身份重新执行缩容压力验证，覆盖 W00--W09 和五类模式：original、
input-stationary、weight-stationary、paper-inspired hybrid、weight-stationary barrier。

| 指标 | 结果 |
|---|---:|
| instruction 最大峰值 | 23,008 B |
| UOP 最大峰值 | 10,820 B |
| 集合级 instruction 容量 | 24 KiB |
| 集合级 UOP 容量 | 12 KiB |
| 总容量 | 36 KiB |
| 相对 64 MiB requested backing 降幅 | 99.9451% |
| 缩容正确性/峰值/容量复验 | 197/197 |

逐身份单独定容时，总容量分布为：最小 8 KiB、中位 8 KiB、P90 12 KiB、最大 28 KiB。这说明
单一模型身份通常无需为整个候选池最坏情况付费；但是否按身份、Executor 或全局 allowlist 定容，
必须由实际部署生命周期决定。

留一 workload 检验中，使用另外九个 workload 推导的固定容量可覆盖 9/10；W08 需要额外三个
instruction 页。这个负例非常重要：36 KiB 不是新的万能常数，未见形状必须重新编译并生成证书。

证据：`20260911_p7r87_full_pool_reduced_command_capacity_run01`、
`20260911_p7r88_full_pool_command_capacity_analysis_run02`。

## 4. W05 真实 u-dma-buf/FPGA 验证

最终派发集 `{TopHub575, bounded-hybrid574}` 的离线峰值为：

| 身份 | instruction | UOP |
|---|---:|---:|
| TopHub575 | 3744 B | 1480 B |
| hybrid574 | 3808 B | 1424 B |

按页对齐得到 4096 B + 4096 B。最终板端合同使用隔离 tmpfs runtime，不覆盖默认 runtime，
从同一 RPC 会话调用新增只读接口 `vta.runtime.queue_capacity_status`，取得：

```text
instruction capacity = 4096 B, peak = 3808 B
UOP capacity         = 4096 B, peak = 1480 B
submissions          = 6
```

TopHub575 与 hybrid574 各三个 seed，共 6/6 FPGA 逐元素正确；boot、FPGA、192 MiB u-dma-buf
前后不变，4057 秒错误水位后无新增存储错误。实验只写 `/var/volatile`，结束后默认 RPC 已恢复，
容量环境变量已清除。由此可主张：对这两个精确 W05 身份，真实 VTA/u-dma-buf 路径在 8 KiB
requested command backing 下正确工作，相对原 64 MiB 减少 99.9878%。

证据：`20260911_p7r89_w05_board_command_capacity_contract_run07`、
`20260911_p7r90_queue_capacity_status_build_run01`、
`20260911_p7r90_w05_board_command_capacity_run03`。

### 4.1 从调优候选到部署证书的闭环

旧派发只携带 `config_index`；一旦 ConfigSpace 顺序改变，同一数字可能指向另一组 tile。新增 v2
派发携带完整 `ConfigEntity`、`candidate_id` 和 TIR hash，AutoTune 在测量前按实体语义匹配，索引
只保留为调试字段。随后按两个不同集合分别生成资源证书：

| 集合 | 身份 | 推导容量 | 资格状态 |
|---|---:|---:|---|
| 调优 allowlist + fallback | hybrid494、hybrid574、TopHub575 | 8192+4096 B = 12 KiB | `local_provisional` |
| 最终部署子集 | hybrid574、TopHub575 | 4096+4096 B = 8 KiB | `qualified_reduced_capacity` |

12 KiB 与 8 KiB 的差异是方法的正证据：`C_q` 随精确 allowlist 的最大峰值变化，不能写成板级
常数。最终 8 KiB 证书由已有同会话 FPGA 容量状态、两个身份各三个唯一 seed、二进制 hash、
runtime/source hash 和 boot 证据升级；调优集合中的 hybrid494 尚未做缩容板端验证，因此其 12 KiB
证书严格停留在本地 provisional 状态。

当前 replay 路径未产生可观测的命令资源记录，证书显式设置 `replay_policy=disabled`。这意味着
部署若需要 replay，必须先重新采集 capture/replay 峰值并生成新证书；当前结果不再笼统声称
“FINISH 和 replay 都已验证”。

证据：`20260911_p7r91_w05_deployment_command_signatures_run01`、
`20260911_p7r92_w05_semantic_dispatch_run01`、
`20260911_p7r93_w05_autotune_command_resource_certificate_run02`、
`20260911_p7r94_w05_final_command_resource_certificate_run02`、
`20260911_p7r95_w05_qualified_command_resource_certificate_run03`、
`20260911_p7r96_w05_autotune_resource_chain_validation_run02`。

### 4.2 前瞻 manifest--runtime attestation

为消除“证书绑定源码、实际执行却加载另一版二进制”的缺口，新的本地结构签名从每个隔离
FSim worker 的 `/proc/self/maps` 解析真实映射的 `libvta_fsim.so`，5/5 身份均绑定同一二进制
SHA-256。随后由当前源码交叉编译新的 ARM runtime，在未执行 FPGA 前生成内容寻址的 pending
manifest；它固定两个精确身份、4 KiB+4 KiB 容量、单队列生命周期和 `replay=disabled`。

板端仅将该 runtime 放入新的 `/var/volatile` 隔离目录。新实验各运行 TopHub575、hybrid574
三个 seed，6/6 逐元素正确；同会话只读状态为：

```text
manifest_id = 9792b9222410584f0ac259fe16488facf34020dfe4e2f7974755315f56a51e8e
instruction capacity/peak = 4096/3808 B
UOP capacity/peak         = 4096/1480 B
submissions               = 6
replay_policy             = disabled
```

qualifier 直接核对上述 manifest/replay 字段、源码与 ARM/FSim 二进制哈希、两个身份各三个 seed
和容量峰值后签发 ready manifest。pending 与 ready 的 manifest ID 完全相同，说明结果附着于
预先冻结的合同，而不是事后重新定义合同。默认 RPC 恢复到 tmpfs，冻结错误水位之后无新增存储
错误，全程未 reboot、poweroff 或写 SD。

另一个不接触开发板的负向控制在同一 ready manifest 环境下主动调用 capture/replay API；实际
加载且哈希匹配的 FSim runtime 对两次调用都返回“被 deployment manifest 禁用”。因此
`replay=disabled` 是运行时强制策略，而非仅供文档展示的元数据。

证据：`20260911_p7r98_w05_manifest_command_signatures_run01` 至
`20260911_p7r108_w05_manifest_replay_policy_fsim_run01`；正式 ready manifest 为
`20260911_p7r106_w05_ready_deployment_manifest_run02`。

## 5. 失败记录与边界

- `p7r90 run01` 在执行 FPGA 前因后台 shell PID 与真实 RPC PID 不同而停止；默认 RPC 随后恢复。
- `p7r90 run02` 已取得 6/6 正确，但 fork 子进程的 stderr 不进入父 RPC 日志，故不能用日志证明
  实际容量；它只作为“缩容环境下正确”的中间证据。
- `p7r90 run03` 通过同会话只读接口直接返回容量、峰值和提交数，是正式板端容量证书。
- 197/197 是本地 FSim 集合安全证据；只有 W05 两个身份完成真实 FPGA 缩容验证。
- 当前版本的 W05 结果已完成 prospective manifest--runtime attestation；旧 p7r95 证书仅对当时
  runtime 历史版本有效，不替代 p7r105 run02。
- 未验证动态形状、replay、完整 stage 的所有提交、allocator 内部碎片或 FPS；这些不能从当前结果
  外推。
- Banerjee 等 2021 年工作已覆盖 VTA tiling search、DRAM bytes 削减、冗余 LOAD 消除和分层
  正确性，因此本文不能把这些单独写成首创；AEx 等也已研究片上 instruction-memory sizing。
  本文的窄增量是固定 VTA/ISA 下 host 共享 instruction/UOP backing 的 exact-allowlist 证书、
  三态准入、incumbent 回退和 runtime fail-closed 执行契约。

## 6. 论文中的统一表述

这不是“把 64 MiB 改成 8 KiB”的板级常数优化，而是：固定 FPGA 后，编译/离线 dry-run 为
硬件允许的调度身份生成 DMA、片上容量、正确性和命令空间证书；AutoTune 只测证书允许的候选，
部署时再按 allowlist 峰值和生命周期分配共享内存，并由 runtime 在提交前 fail-closed 检查。

因此第三点同时包含：硬件信息驱动的搜索空间压缩、驻留/虚线程联合调度、真实硬件正确性
allowlist、强 incumbent 回退，以及面向 u-dma-buf 的命令资源定容。具体容量随硬件、形状和
调度变化，方法本身不依赖 config574 或 AXU5EVB 的固定编号。
