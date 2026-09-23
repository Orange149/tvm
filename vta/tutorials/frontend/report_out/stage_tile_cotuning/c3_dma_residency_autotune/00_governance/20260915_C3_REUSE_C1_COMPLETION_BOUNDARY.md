# C3 C1执行节点：runtime返回—共享输出可消费之间的完成边界

状态：`C1_ATTEMPT_INVALID_TIMING; COMPLETION_BOUNDARY_DIAGNOSED; C1_RESTART_PENDING; C2_NOT_STARTED`。
第一启动为 `17cec1fe-4bb5-4f9e-90fb-eaf0e5f1b2b8`。不是完整C1性能结果，也不是第二启动。

## 1. 本次实际执行

新的 `run_vta_c3_reuse_board.py` 不修改Node-B冻结源码、44主身份或4个reload控制。
`c3_reuse_timed_once.cc` 仅作为实验RPC辅助库，板端steady_clock包围一次同步main调用，无隐式warmup，
调用后核验输出。本机RPC原地递增测试证明每个观测严格执行一次；该入口及资格检查共6项测试通过。
其后旧机制/审计加新增资格回归共54项测试通过，包含实际四模式FSim执行。

正式准备 `07_grouped_holdout/20260915_c3_reuse_c1_prepare_run02` 完成健康探针lowering、三seed FSim、
ARM编译、辅助库编译与源码/runtime绑定，耗时5.284 s，不含新目标板端标签。
run01因辅助库编译未携带SDK sysroot失败；保持只读，不覆盖，也不计为候选机制失败。

板端run01在SSH前置阶段受执行环境网络限制失败，未部署或调用FPGA。申请网络执行后使用新目录run02。
SSH使用既有已核验主机密钥，bitstream哈希匹配，FPGA operating，u-dma-buf为201326592 B。
运行限定新tmpfs目录，重配同一bitstream，不修改SD镜像、驱动或正式runtime。

正式尝试 `07_grouped_holdout/20260915_c3_reuse_c1_board_run02`：

| 阶段 | 完整算子调用 | 结果 |
|---|---:|---|
| original健康门 | 3 | 3/3正确 |
| 48身份三seed首错即停 | 141 | 46身份全通过；2个M2 IW拒绝 |
| 显式warmup | 46 | 46/46正确 |
| 八轮计时进行部分 | 228 | 最后一次M2 IW输出错误，立即停止 |
| 合计 | 418 | 全部有dispatch begin和独立结果记录，无隐藏warmup |

两次correctness拒绝分别为M2D0140 IW(seed0，896 mismatches)与M2D0078 IW(seed20250901，4120 mismatches)。
M2D0071 IW通过三个seed及warmup，但主计时第一轮seed0出现10752 mismatches。
后者的0.569466 ms是**错误调用观测，不能作为latency标签**。

本次运行W0→失败47.072 s、T0→失败22.505 s；已完成逻辑LOAD178140160 B、STORE12368384 B、
FPGA driver invocation1006次。418个完整算子与1006个submission不可混为一个计数。
全部driver_timeout_calls为0，说明“驱动未报timeout”不等于输出可消费。
整个会话已标记 `all_timing_invalid=true`；七组已经完成的八轮也不进入正式性能统计。
没有拼接、补齐family、静默original回退或筛选最优点。失败成本保留，不能藏在有效成本之外。

## 2. 独立有限诊断：没有第二次执行，输出后来变正确

`diagnose_vta_c3_reuse_completion.py` 在新的tmpfs/RPC会话中复用同一冻结ARM二进制/runtime，
只检查上面三个M2 family。选择基于错误身份，不基于latency排名。
每seed顺序固定为O→IW→W→IW；调用完成后立即核验，再不重新执行算子、仅延后10 ms读取同一output。
这是诊断读取，不是修复、warmup、正式计时或可消费同步方案。

产物 `07_grouped_holdout/20260915_c3_reuse_completion_diagnosis_run01`：

- 全部42次完整算子完成：前/后健康门各3次，三个family×三个seed×四次诊断共36次。
- 立即读取7次错误；7/7在不重新执行情况下延后读取变正确，7次输出哈希均改变。
- 这些错误全部是IW；对应driver_run_total_us约136.2--139.7 μs。
- 健康门6/6、original和W立即读取均正确；没有新timeout或boot变化。
- 会话耗时28.411 s。诊断会话与失败C1会话分别记录，不能相加包装为连续T0→T1。

直接证据支持：**至少这些调用中，runtime返回先于共享输出稳定/可消费。**
这不是仅由SRAM容量或DMA bytes即可认证的性质。三个IW的实际weight arena都是147456 B(256 KiB的
56.25%)；ACC arena分别7168/50176/14336 B，容量公式本身不能解释这些完成错误。

HLS源码在compute读取指令并取得依赖后才`done=0`，FINISH再`done=1`；AXU驱动提交后只等待
`COMPUTE_DONE_RD_OFFSET`值为1。这使“上次FINISH的done在本次compute尚未更新时仍可见”成为优先
核验假设，但不能仅凭mode名称推断IW的提交结构。
但当前没有本bitstream生成的AXI-lite done/valid寄存器波形或逐submission MMIO记录，
所以不能宣称具体stale-done寄存器根因已证实，也不能直接把源码推断当成硬件证明。

### 独立账本/队列审计修正

正式只读审计为 `07_grouped_holdout/20260915_c3_reuse_completion_audit_run03`。
418次失败会话与42次诊断调用全部按dispatch顺序、candidate/seed/phase、单完整调用、无隐藏warmup、
最终TIR语义DMA与FSim submission数量核对，共460/460；4项审计测试通过。
审计run01因诊断记录未保存完整host-call墙钟字段失败，保留初稿；run02补为unknown后对账完成，
run03另增加可消费边界图。诊断只有完整会话墙钟与RPC执行墙钟，缺少逐调用完整墙钟时不能填零。

冻结FSim队列给出三组M2 IW的两段结构：

| family | 首段 LOAD / STORE B | 尾段 insn B / LOAD / STORE B |
|---|---:|---:|
| M2D0140 | 219136 / 25088 | 64 / 0 / 0 |
| M2D0078 | 176128 / 25088 | 64 / 0 / 0 |
| M2D0071 | 194048 / 25088 | 64 / 0 / 0 |

以上数值以run03审计CSV为准。首段已包含整个算子的LOAD/STORE，不能写“独立权重预加载提交后再
提交计算”。尾段仅可按无数据DMA称控制尾段，不能从总指令长度推断其全部opcode。
板端profile只有完整算子累计driver统计，未记录逐段start/done/valid，仍不能定位误判完成发生在哪一段。
源码提供WR=0x10/RD=0x18，但未找到与当前bitstream绑定的生成寄存器定义或波形，不能猜测RD+4的
valid位或直接写WR清零作为修复。

## 3. 对共享内存语义的价值

已经验证LOAD作用域改变与实际搬运；现在又发现**可消费时刻**是不可省略的语义条件：

```text
驻留减少LOAD
  -> 改变submission/FINISH结构
  -> runtime必须认证“本次”完成
  -> CPU/VTA下一消费者才可安全读取或覆盖共享区域
```

这与USMP固定程序的生命周期/偏移规划不同，也不是推翻Cheng的驻留规律。
它限定的是本地barrier独立重实现与当前runtime组合的执行安全条件。
FSim同步执行和随机seed正确性均不能代替真实FPGA完成握手。
同时，小driver时间或偶尔正确的快速调用也不能证明程序更快。

## 4. 下一节点及暂停边界

先停止正式八轮计时，不能简单增加10 ms sleep后称已修复并继续报告加速。
下一步应冻结独立完成握手验证协议：核对当前bitstream寄存器定义；逐submission记录start/done/valid；
验证FINISH对应本次提交且LOAD/STORE均完成。若需要修改driver，只建立隔离实验runtime，与原runtime
分别绑定、分别资格；不得覆盖原库、修改原48身份或复用无效会话的计时。

原计划明确本轮不修改driver；因此实验版driver修复需要先取得用户同意，再作为新增C0节点执行。
确认后重新封存C1计时协议并完整重跑；C2仍须用户实际重启。当前没有新的正式四方案latency、
跨启动稳定性或整图FPS结论。所有原有C3搜索/整图证据保持有效，不被本轮失败覆盖。

已确认本次自有RPC父/timeout子进程均结束；tmpfs证据保留，无SD数据删除。
