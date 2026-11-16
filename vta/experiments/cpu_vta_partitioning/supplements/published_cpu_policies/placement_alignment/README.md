# 主机位置与worker逐内核对齐

默认离线核查已有三个配置，不更改模型、候选、程序或性能结果。

```bash
python run_placement_alignment.py analyze
python run_placement_alignment.py prepare-counter
```

计数器准备只在本地交叉编译并用QEMU核验加载，不证明板端接口可用。可选板端自测是独立授权动作，必须明确调用`board-self-test --execute-board`。自测只执行两个约150ms CPU忙段及一个100ms睡眠段，不执行DNN，不访问FPGA，不重启，不改系统参数，不使用采样或新增后台采样线程。

自测请求每TID、每CPU的计数型TASK_CLOCK，同时读取全CPU TASK_CLOCK及线程CPU时钟。必须通过迁移后的核心定位和总量守恒；不按enabled/running比率放大计数。权限错误、接口不可用、恒零或CPU筛选不生效均不能作为有效分解。

核对依据：[perf_event_open接口](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)、Linux v5.4 kernel/events/core.c中的task-clock事件实现。供应商内核仍需原生正负对照验证，QEMU不替代计数资格。
