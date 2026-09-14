# P7R447 run01 状态

状态：`INVALIDATED_DURING_MEASUREMENT_INFRASTRUCTURE_AUDIT`

本次从零 AutoTVM-XGB 运行发现，随机候选可能触发板端 VTA runtime 的 fatal check 并终止 RPC。
旧 direct runner 记录该候选失败后会继续使用已经失效的 RPC，使后续候选受到会话污染。因此人工
停止该运行；其日志不得计入 AutoTVM 成本、成功率或性能比较。

修正后的 P7R445 runner 在每个完成本地构建、准备上板的候选之前执行同一 bitstream reload 和
tmpfs default-RPC clean start。编译期失败候选不接触板端但仍计入 gross proposal 和墙钟。
