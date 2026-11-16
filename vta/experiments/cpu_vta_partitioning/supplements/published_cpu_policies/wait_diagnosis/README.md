# YOLO等待诊断与一次阻塞等待对照

固定切图CPU0–2→VTA3–6→CPU7–8→VTA9–12→CPU13。
仅比较四线程重叠共享和三线程计算分组，均关闭门控，主机与控制共享四核。
原模块、执行器及运行库不变；只编译LD_PRELOAD适配库。

适配库通过公开C ABI包装并行启动／回调、屏障及实际运行库的Yield路径。
观察模式调用原函数；阻塞模式只替换屏障和已识别的任务完成等待。
Yield使用线程局部计数，函数边界记录墙钟及线程CPU时间，结束后统一输出。
原等待、关闭探针的会话完全不加载适配库。

在上级目录运行：

```bash
python run_yolo_wait_diagnosis.py audit
python run_yolo_wait_diagnosis.py prepare
python run_yolo_wait_diagnosis.py validate
python run_yolo_wait_diagnosis.py board run --execute-board
python run_yolo_wait_diagnosis.py board resume --execute-board
python run_yolo_wait_diagnosis.py status
python run_yolo_wait_diagnosis.py analyze
```

本批实际板测和分析使用独立验收兼容入口：

```bash
python run_yolo_wait_diagnosis_compat.py board resume --execute-board
python run_yolo_wait_diagnosis_compat.py analyze
```

原执行器只在资格检查中填充主动worker_slots。兼容入口在正式测量中使用已验证的单核affinity重建槽位，仍逐项核对实际TID；不增加探针、不改变任务或程序哈希。
首次拒收的正式隔离会话直接回收原结果，未重测；修正源码和失败记录见新批次PASSIVE_WORKER_COMPATIBILITY.json、VALIDATOR_RECOVERY_AUDIT.json。

第一阶段分别测2种隔离配置和2种流水配置，探针关／开，每项先做单帧资格检查。
只有四线程本批慢化仍存在、目标等待Yield被捕获，且探针对目标阶段的影响不超过慢化差值20%，才测一次阻塞策略。
组件2＋8帧，流水8＋64＋32帧；不测锚点、不漂移拒收、不自动重测或重启。
上限24个原生会话、840次完整DNN推理、44次隔离阶段调用，初始化调用单列。

内核schedstats／ftrace关闭。非运行时间不等于调度等待；零schedstat不解释为无等待。
轨迹属于整段会话，不伪称正式64时间戳对应的逐帧profile。
不把多个线程的等待墙钟相加成完成间隔，不将本批结果回填静态成本或自动扩大实验。
