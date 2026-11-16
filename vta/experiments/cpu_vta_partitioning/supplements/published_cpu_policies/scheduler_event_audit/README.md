# 调度事件接口：零DNN自测

```bash
python run_scheduler_event_audit.py prepare
python run_scheduler_event_audit.py board --execute-board
python run_scheduler_event_audit.py status
python run_scheduler_event_audit.py analyze
```

只运行一次独立原生程序，依次执行单线程忙运行、两个线程同核竞争、已知nanosleep、显式换核四种短场景，共五个有限测试线程。没有DNN、FPGA访问、周期采样线程、sysctl调整、ftrace开启或重启。

通过`PERF_COUNT_SW_DUMMY`、`context_switch`与`sample_id_all`记录任务级`PERF_RECORD_SWITCH`，时间戳使用`CLOCK_MONOTONIC`。只请求TID、TIME、CPU，未配置采样周期或频率。每个线程使用独立64KiB环形缓冲区，结束后读取；溢出、丢失、错误TID、错误时钟、缺失事件和CPU时间不守恒均拒绝。

`board`复用已验收记录；远端尝试标记防止断线后自动重做已启动测试。接口不可用或校验失败时保留原始数据，不以恒零表示没有等待，不自动改用另一接口或扩大缓冲。

此接口提供进入／离开CPU及抢占标记。它没有精确唤醒时刻；非抢占离开不总是睡眠，离核区间不全是CPU排队。控制实验的已知操作只用于验证接口，不能将其中最长间隔作为DNN固定慢化常数。

QEMU只核验程序加载，不替代板端能力自测。成功也不代表实际DNN的记录容量、完整性和扰动已经验证；新DNN诊断另行开展。
