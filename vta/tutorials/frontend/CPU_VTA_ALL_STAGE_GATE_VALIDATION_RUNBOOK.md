# 同流水线程池的全阶段互斥诊断

冻结来源为已完成的CPU-only互斥三boot协议。新实验目录独立，4/1/4线程、四核mask、spin=0、queue depth=2、同图/输入/权重/运行库。

A正常重叠，B仅CPU阶段FIFO互斥，C所有CPU/VTA阶段FIFO互斥。新runner由旧协议冻结C++源码确定性派生；原canonical runner及旧控制器不修改。新增`--stage-gate-scope cpu|all`，默认cpu，已有门模式默认off。

门等待先于stage_start，覆盖set/run/get与输出快照；VTA mutex在admission取得后进入，admission在下游Push前释放。异常沿用RAII释放及取消唤醒。
JSON为兼容派生继续使用`stage*_cpu_gate_*`时间戳名称，明确的`stage_gate_scope`区分cpu/all；scope=all时VTA也有admission时间戳。这些字段不代表VTA硬件计算周期。

CLI使用`/home/orange/miniconda3/envs/vta-resnet/bin/python`：

```sh
python vta/tutorials/frontend/run_cpu_vta_all_stage_gate_validation.py freeze --run-dir NEW_UNIQUE_RUN
python vta/tutorials/frontend/validate_cpu_vta_all_stage_gate_host.py --run-dir NEW_UNIQUE_RUN
python vta/tutorials/frontend/run_cpu_vta_all_stage_gate_validation.py pilot --run-dir NEW_UNIQUE_RUN --execute-board
python vta/tutorials/frontend/run_cpu_vta_all_stage_gate_validation.py replicate --run-dir NEW_UNIQUE_RUN --execute-board
python vta/tutorials/frontend/run_cpu_vta_all_stage_gate_validation.py analyze --run-dir NEW_UNIQUE_RUN
```

首boot先做serial/A/B/C各10帧完整输出验证，serial前后各20帧，丢5帧。off/observe-all两次II配对均不超过2%时A采用observe，否则锁定off。三个性能block顺序ABC、CBA、BCA，每会话307帧，丢7帧，以300完成时间戳的299间隔计II。关闭checkpoint、dump、稳定性sleep与逐提交日志，延后只写JSON文本。

预声明主指标是B→C首CPU平均run减少；尾CPU、VTA、CPU run之和与II为次指标。中位主变化至少5%、B有CPU/VTA执行重叠、C全帧leases互斥、匹配serial首CPU漂移不超过5%，触发复现。额外采集BC/CB两个全新平衡block作为boot1；后续两个boot由用户实际重启，各做完整输出、前后serial与两个固定配对。三boot df2 Student-t，单boot只作探索性证据。

C改变资源约束，II很可能变长，不以吞吐改善为成功。C若接近serial服务，支持共活动解释；若仍偏离，剩余线程池、缓存/内存布局与执行路径差异仍未闭合。即使恢复，也不能分开调度/cache/DDR分量，不从墙钟推导hardware compute-only。实际FPGA频率unknown；正式静态公式与候选排名不改。

正确性、超时、身份或门区间合同失败即停，保留原始与部分结果、不静默重试。只清理本协议RPC和归档会话，保留原SD/库/其他进程。每次native上下文使用原reset/完成协议。
