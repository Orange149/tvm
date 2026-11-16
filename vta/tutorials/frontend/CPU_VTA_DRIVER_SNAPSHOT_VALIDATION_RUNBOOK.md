# 低扰动 driver / VTA host 线程累计计数验证

独立协议以完成三 boot 的 `20260917T121038Z_all_stage_v1` 为资产来源。
从该协议冻结 runner 源码派生新的实验 runner，只增加默认关闭的
`--vta-driver-snapshots`；不改 canonical runner、旧控制器、运行库、驱动、
bitstream、Tile、频率或驻留策略。

板端只读核查已经确认 `CONFIG_FTRACE=n`、`CONFIG_SCHEDSTATS=n`，同时
`CONFIG_SCHED_INFO=y`、`CONFIG_TASK_DELAY_ACCT=y`，并存在线程 schedstat。
不挂载新 tracefs、不改内核或全局跟踪设置。schedstat 是否实际工作，由性能
数据中的 runqueue 和 dispatch 累计计数增长验证；没有增长就报告 unknown。

每个 VTA run 的前后探针读取：

- 现有 `VTADriverProfilerStatus` 累计计数器，计算该 run 的 calls、insns、
  timeout、poll_iters、total、MMIO、post-start sleep、poll wait 差值。
- `CLOCK_THREAD_CPUTIME_ID`，仅描述单 VTA host worker 的线程 CPU 时间。
- `/proc/self/task/TID/schedstat`，取已就绪后在运行队列等待的累计 ns 和
  dispatch 次数差。CONFIG_SCHEDSTATS=n 时 runtime 字段可能为零，不使用它。
- 同一 steady clock 的绝对快照区间和 benchmark origin，以及前后探针耗时。

前置探针成本放在 set_ms，后置成本放在 get_ms，保持 inclusive service
严格等于 set+run+get，run 区间不包含探针本身。线程 CPU 快照窗口包含探针，
schedstat 采样发生在该窗口内部；报告窗口和探针预算，不能把它当作毫无
边界误差的逐次调度轨迹。流水的 stage_process_cpu_ms 始终为 null。

固定图 `CPU00:02 → VTA03:17 → CPU18:20`、线程4/1/4、四核mask、spin=0、
queue depth=2 和普通复制。B为CPU互斥，C为全CPU/VTA阶段互斥。
串行及B/C快照开启各10帧完整输出校验；前后串行服务各20帧；性能顺序固定：

1. B off、B on、C off、C on。
2. C on、C off、B on、B off。

每性能会话307帧，丢前7帧，300个完成时间戳、299个间隔。
逐提交日志、dump、checkpoint、稳定性sleep全部关闭，结果JSON延后写出。
同时检查每配对的II和VTA run扰动，阈值各2%；主性能始终采用off。
若任何配对超限，on数据仅用于诊断，不用它闭合off性能幅度。
匹配串行服务的各阶段前后漂移阈值5%。正确性、完成超时、资产/身份/合同
漂移、缺帧、计数减少、错误线程或计时顺序失败均停止，保留部分证据。
不删除慢帧，不在同目录静默重跑。

两套计数存在重叠，分析必须分别看：

```
run ≈ driver_total + host_outside_driver
driver_total = MMIO + post_start_sleep + poll_wait + driver_other

host snapshot wall ≈ thread_CPU + runqueue_wait + other_non_CPU
```

第二式是带微小采样边界误差的host窗口分解；它与第一套driver计时重叠。
runqueue_wait是Linux可运行线程排队时间，不是硬件DDR等待。
other_non_CPU包含poll睡眠、timer延迟、硬件等待等；不能命名为纯硬件计算。
post_start_sleep同样与硬件运行重叠，不能直接视为可消除的固定额外成本。

执行（环境 `/home/orange/miniconda3/envs/vta-resnet/bin/python`）：

```
python run_cpu_vta_driver_snapshot_validation.py freeze --run-dir NEW_DIRECTORY
python validate_cpu_vta_driver_snapshot_host.py --run-dir NEW_DIRECTORY
python run_cpu_vta_driver_snapshot_validation.py pilot --run-dir NEW_DIRECTORY --execute-board
python run_cpu_vta_driver_snapshot_validation.py analyze --run-dir NEW_DIRECTORY
```

只用协议自有tmpfs/RPC，每新VTA命令上下文采用原验证的完成重置协议。
结束停自有RPC并恢复firmware链接。一个boot补充只能作探索性定位；需要
新增计数器机制的跨boot统计时，必须补实际kernel reboot，不借用旧boot
缺少计数的数据。FPGA实测频率、硬件compute-only、物理DDR、历史34ms仍
保留unknown；不修改正式静态公式或候选排名。

准备阶段的v1/v2目录保留，均未执行板端推理。v1被板前计时守恒核查取代；
v2有一项过时的源顺序断言失败。正式v3冻结修正后的探针记账与测试，默认
schema、CPU兼容、计数及19项Python检查全部通过后才允许板端执行。
