# 双网络六项资源组合验证

计划位于`artifacts/cross_network_model_01/phase_validation/PLAN.json`，在采集新组件和新流水性能前冻结。每网络三个资源组合使用同一已知切图；排除本Goal开发池的配置身份后，按预定核心轮换规则选点。这里的未见范围是资源组合，尚未包含新切图泛化。

每网络只补前段/中段的两线程隔离记录，共4种新签名、12个原生会话、84次CPU阶段调用。原模块、运行库和可选阶段执行程序均复用；真实ARM QEMU核验四种组件输出及线程映射。

完成隔离投影后，静态排名在读取任何新流水性能前冻结。比较旧成本容量模型、匹配新局部成本后的相同容量模型、保留真实VTA设备调用顺序的CPU时序回放。并列保留主机工作聚合及驱动内部CPU先/后的对照。主预测始终为`interleaved_before`，不根据流水结果选择顺序。

```bash
python run_phase_validation.py prepare
python run_phase_validation.py qemu
python run_phase_validation.py components --execute-board
python run_phase_validation.py project
python run_phase_validation.py rank
python run_phase_validation.py pipelines --execute-board
python run_phase_validation.py analyze
```

`components`和`pipelines`均可重复调用来恢复：已验收任务只核验原始凭据。两者都要求显式`--execute-board`；boot必须与本研究之前的采集一致。资格/输出、线程、hash及帧完整性仍须通过。没有重启、锚点、漂移拒收、自动延长或自动性能重测。

六项流水各一次资格、一次8/16/4窗口，共12会话、174次完整DNN推理。CPU调用和完整DNN推理单独记账。只在每网络三个配置的池内比较预算1/2/3，完整DP和Top-20仍待后续；不能把这个小池结果当作全域性能。

回放假设CPU同核活跃线程公平分时，主机任务在允许集合内按事件时刻的负载选核；保留串行区段、并行完成汇合和深度2队列。VTA互斥覆盖整段set/run/get，匹配冻结执行器。阶段等待和完成尾部不会再按固定计算量重复相加。真实Linux时间片、唤醒、控制开销及缓存干扰仍未完整恢复。

通过条件：相对匹配成本的容量模型，两网络选点均不变差，至少一种改善；驱动内CPU先/后安排若改变选点，继续保留未解决的不确定性。没有达到条件时如实报告，不增加配置以追求有利结果。
