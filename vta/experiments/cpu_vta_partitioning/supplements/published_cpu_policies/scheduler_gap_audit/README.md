# 子图线程曲线与剩余误差诊断

本包复用原有成本、完整模型静态第一名和YOLO匹配配置，不修改旧模型、搜索域或结果。CPU排队采用逐TID TASKSTATS计数，必须先在本机启动中通过忙任务竞争／睡眠自测。

```bash
python run_scheduler_gap_audit.py curves
python run_scheduler_gap_audit.py audit --execute-board
python run_scheduler_gap_audit.py self-test --execute-board
python run_scheduler_gap_audit.py prepare
python run_scheduler_gap_audit.py validate
python run_scheduler_gap_audit.py qemu
python run_scheduler_gap_audit.py board run --execute-board
python run_scheduler_gap_audit.py board resume --execute-board
python run_scheduler_gap_audit.py status
python run_scheduler_gap_audit.py analyze
```

自测不执行DNN；QEMU只核验加载、逻辑线程和CPU完整输出，不证明内核计数可用。新启动自建实验边界，不将旧启动的成本来源改写为本批实测。

六个对象（3个组件、3个流水）各一资格、无探针与有探针会话，共18会话、627次完整DNN、63次CPU阶段调用。初始化和空回调另行保存。没有重启、自动重测或漂移门槛。

探针保持原等待策略。启动前读取已知worker的逐TID计数以观察进入回调前的排队；回调、屏障、阶段及控制工作保留时间与计数读取包络。未知读取时点不能当成精确唤醒轨迹，多个线程及嵌套范围不能相加成流水完成间隔。

原始凭据验收并同步后才清理远端大输出；哈希、boot、实际线程及帧错误均拒绝。续跑只接续未验收任务。能力不支持时不重刷内核；诊断扰动过大时保留无探针性能并限制归因。
