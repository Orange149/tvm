# P7R449 状态

状态：`SEARCH_PREFIX_VALID; END_TO_END_RUN_INVALIDATED_AFTER_BUDGET_STOP`

空历史 XGB 在共同 T0 后运行至 602.940 s，共记录 159 个候选。逐候选 clean-start 隔离有效，搜索
前缀可用于调试；但 600 s callback 通过异常停止 AutoTVM 后，旧 tuner 没有清除全局 `in_tuning`
标志，导致后续 stock 完整图构建失败。该会话没有达到共同 T1，不得计入最终 A/B 成本或整图结果。

执行器已增加异常路径上的全局 tuning 状态和 XGB worker pool 清理。必须从空日志重新运行，不能
将本会话搜索墙钟与另一进程的整图结果拼接。
