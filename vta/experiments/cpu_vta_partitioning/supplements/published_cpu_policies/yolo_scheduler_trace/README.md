# YOLO单配置调度时间线

冻结切图CPU0–2→VTA3–6→CPU7–8→VTA9–12→CPU13。CPU线程数为4／3／1，核心集合为0–3／1–3／0；逐线程固定，CPU门控关闭，主机与控制线程共享四核。

```bash
python run_yolo_scheduler_trace.py prepare
python run_yolo_scheduler_trace.py validate
python run_yolo_scheduler_trace.py qemu
python run_yolo_scheduler_trace.py board --execute-board
python run_yolo_scheduler_trace.py status
python run_yolo_scheduler_trace.py analyze
```

`board`也用于断线续跑，只执行尚未验收的任务，boot变化则停止。先核验新旧两个执行器的全部8个输出，再做新执行器探针关／开、旧执行器桥接，各一轮；共5个会话、146次完整推理。正式窗口为8帧预热、16个完成时间戳、24帧尾部，以保证生产线程在计数停止时仍存活。无锚点、漂移拒收、自动延长或自动加测。

只在新准备目录的冻结源码副本增加可选SessionProfile实现。复用原运行库、模块、等待机制、输入与参考；复用已验证的原等待观察适配库，以记录阶段、并行启动、逻辑worker、TASKSTATS及调用线程尾部。

调度事件使用任务级PERF_COUNT_SW_DUMMY、PERF_RECORD_SWITCH和CLOCK_MONOTONIC，每个目标TID有独立128KiB数据环，12个目标TID包括8个CPU计算槽位、2个VTA主机、生产与收集线程。没有周期采样线程。每线程附带全CPU TASK_CLOCK以核验事件时间线与CPU量的包络。没有确定状态的首尾区间保留未知。

探针开／关共用同一新执行器；旧执行器仅作桥接。CPU量、抢占离核、非抢占离核、TASKSTATS排队和未知区间分别报告，不相加形成静态惩罚。记录丢失、溢出、错误身份、绑核或输出均拒绝；只选择第一个完整覆盖帧作为例图，不挑最慢帧。

本轮只解释一个已知配置，不修改评分、DP或Top-20，不将流水慢化／落核比例回填为通用成本。PERF_RECORD_SWITCH没有精确唤醒时刻；非抢占离开不总是睡眠，所有离核时间也不全是CPU排队。
