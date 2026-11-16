# CPU–VTA静态预测误差的小规模诊断

本批次只定位误差来源，不改静态公式、成本表、DP或Top-20。原始基准为
`resnet18_open_cpu_resource_02`及部署包`8a942b2047c031339168f9d2`。

执行器、运行库、输入0和八个阶段模块均按原哈希复用，不增加执行器编译。
独立入口为`run_diagnosis.py`，默认批次`resnet18_prediction_gap_01`。

## 测量安排

单次启动内共20个任务、299次推理：

|工作|数量|窗口|
|---|---:|---|
|隔离CPU：前段四线程／三线程，中段四线程／核心3单线程|4|2次预热＋8次计入|
|上述组件的完整输出和实际绑核资格检查|4|每项1次|
|三个完整流水的完整输出和实际绑核资格检查|3|每项1次|
|每个流水分别普通／详细／普通计时|9|8次预热＋16个完成时间戳＋4次尾部|

三个流水为原静态第一名、原历史参考主机固定核心2、相同历史计算配置但主机共享四核。
后两个只改变VTA主机位置；它们与静态第一名存在多项不同，不当作单因素配对。
不加锚点，不设漂移拒收，不自动延长或追加测量，不自动重启。

## 命令

```bash
python run_diagnosis.py prepare
python run_diagnosis.py validate --bundle artifacts/bundles/<bundle-id>
python run_diagnosis.py board run --bundle artifacts/bundles/<bundle-id> --execute-board
python run_diagnosis.py board resume --run-dir runs/resnet18_prediction_gap_01 --execute-board
python run_diagnosis.py status --run-dir runs/resnet18_prediction_gap_01
python run_diagnosis.py analyze --run-dir runs/resnet18_prediction_gap_01
```

准备及校验完全在本地运行，不加载TVM或调用SSH。板端命令显式要求`--execute-board`。
恢复依据原任务身份、启动ID及原始凭据，不重复计入已完成任务；换启动则拒绝混合。
输入错误、缺帧、重复帧、绑核错误及哈希错误仍属于无效执行。

## 如何读结果

先比较旧隔离、同启动隔离、流水内的set/run/get，区分旧profile变化和并发变化。
阶段均值取完成序号8–23的16帧；资源覆盖用全部28帧裁剪至C8–C23并取并集。
CPU组等待、VTA锁等待和队列等待分别列出，不相加并发阶段的进程CPU时钟。

详细会话增加队列事件及原生session profile。必须保留`profile.json`并绑定到原始凭据，
但不要求平台一定提供全部计数：无效／不支持的PMU、全零但不能证实启用的调度计数
标为未知，不作为零开销。内核设置保持原样。

探针采样由collector在收到完成8／23后触发，没有精确采样边界字段，因此不等于严格C8–C23
的逐阶段CPU核时；可能遗漏已退出的producer。PMU为捕获线程的汇总，不能直接归因到单阶段。
详细队列计时可能影响调度，必须与前后普通会话比较，不能自动认为无扰动。

`analysis/`提供流水FPS、阶段对照、队列／阶段时间轴、TID角色与计数可用性、探针扰动和报告。
本次仍无法区分的原因保留未知项，另行制定最小必要探针；不凭历史最佳FPS反向拟合成本。
