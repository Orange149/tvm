# 固定核组分支离线评估

运行环境：`/home/orange/miniconda3/envs/vta-resnet/bin/python`。在上一级目录执行：

```bash
python run_policy_branch_eval.py run
python run_policy_branch_eval.py validate
```

没有板端命令。入口阻止网络、SSH和shell调用，仅允许本地主机C++枚举器及编译器。

沿用原主模型的非均匀线程份额、线程组服务需求和VTA主机CPU容量；主机集中项关闭。新分支完整枚举至少两个互不重叠计算核组，同组可被非相邻阶段复用并串行，不同组并行。此组织方式借鉴Pipe-it；不是新增预测公式或完整复现Pipe-it。

输出原完整域Top-20、固定组Top-20和交替选择的10＋10名单。原域完整保留，不删除共享配置；固定组仅是第二条候选生成路径。候选选定和哈希冻结之前禁止读取流水性能。

分析分别核验ResNet18短窗口、YOLO短窗口和YOLO固定核组匹配实验的原始凭据，不混合历史批次。每个预算必须测齐其全部前缀才报告完整预算性能。另列已测池回放，其选点偏差及未知项明确保留。新完整域Top-20有未测项时，不能宣称已经验证提升。

默认产物位于`artifacts/policy_branch_eval_01`，可读报告位于`paper/my_paper/analysis/20261110_policy_branch_eval`。旧批次只读。
