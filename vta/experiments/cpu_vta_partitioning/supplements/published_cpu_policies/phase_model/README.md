# 同钟组件与有限时序回放

这是双网络Goal下的开发诊断，不修改原评分、DP、运行库或正式执行器。旧源码和凭据保持原身份。

`run_phase_profile.py`只构建可选阶段观察执行器；`run_phase_model.py extract`投影隔离工作；`predict`冻结四个开发配置的时序预测；`analyze`在预测冻结后读取共同流水测量。原始数据和来源位于`artifacts/cross_network_model_01/phase_profile`及`runs/cross_network_phase_profile_01`。

```bash
python run_phase_profile.py status
python run_phase_model.py predict
python run_phase_profile.py resume --scope pipelines --execute-board
python run_phase_model.py analyze
```

只有显式`--execute-board`可以连接板子。已验收任务重放凭据，不重复执行；boot或程序身份变化不能混入。全过程不重启开发板。组件采集已完成，不能因为重新分析而重跑。

隔离输入包含每次并行启动的worker计算CPU需求、调用线程串行区段和完成依赖。回放采用有理数纳秒，最后向上取整。完成等待动态产生，不把隔离caller尾部作为固定计算量再次相加。CPU阶段线程池在队列空时不计为计算工作。

CPU等份时间共享和VTA主机的理想放置是待验证假设。主机run CPU先于/后于设备非运行区间分别冻结报告；不按FPS挑一种。生产、收集、调度粒度、唤醒延迟及缓存干扰仍未建模；模型没有伪称这些开销实测为零。因此结果不是严格性能边界，也不能直接作为全域DP局部成本。

本轮四项为已知开发配置，未改善YOLO的静态第一名，暂不提升为全域方法。完整证据和限制见论文目录的`20261110_cross_network_model/REPORT_zh.md`。
