# 剩余误差离线审计

仅复用三个固定配置的隔离输入与已有控制CPU轨迹。原评分、完整DP、Top-20、模块、运行库和旧批次均不改动。

```bash
python run_model_semantics_audit.py audit
python run_model_semantics_audit.py project
python run_model_semantics_audit.py replay
python run_model_semantics_audit.py analyze
python run_model_semantics_audit.py validate
# 顺序完成上述步骤；重入验证并复用已冻结回放
python run_model_semantics_audit.py run
```

没有板端命令；入口拒绝网络及SSH/SCP进程。198项情形由原允许主机核集合全枚举和两种控制口径确定，不按实测挑选。

ISOLATED_INPUTS只包含原隔离工作量、资源定义和原模型自检向量；DIAGNOSTIC_CONTROL_INPUTS显式标记为已有流水控制CPU观测，不能进入静态预测入口。投影只解码凭据的task/raw/files/boot元数据，不解码summary/FPS；分析在全部回放封存后才正常读取旧性能值。

新增控制执行单元保留有界队列与反压，完成时间仍在最后阶段完成时记录。输出统计、JSON和文件写入的源码范围分别核查；队列阻塞不当作CPU工作，未测析构/回收保留未知。

固定主机落核是可能轨迹假设，范围不是严格上下界。模型仍是连续理想公平共享，没有真实CFS时间片、唤醒、迁移和vruntime仿真；不填猜测调度参数。并列最后完成者保留为集合，避免编号顺序制造假拖尾。

回放封存后复用旧结果生成诊断报告。没有新的真实FPS，也不把诊断改善当成未见配置的预测有效性。
