# C3 P2-local status

Status: `PARTIAL_COMPLETED_CONCURRENT_SOURCE_CHANGE`

本地证据本身给出强正结果：TopHub 10/10、0 fallback；10 个冻结配置均可为 AXU5EVB 交叉编译；10 workload × 3 冻结 seed 的 FSim/NumPy 逐元素比较全部正确；静态 DMA 输出与历史冻结文件逐字节相同；相关单元测试通过。

但本 run 不能标为完整通过。执行期间 `vta/python/vta/top/vta_conv2d.py` 被另一并发任务修改，SHA-256 从冻结的 `b7fc752b...` 变为 `6f9bb140...`。本代理没有修改源码，发现后依预注册停止。22:43 生成的 TIR 文本属于变更后版本，只作后续重构审计输入，不纳入“冻结 original 未变”的结论。

开发板未连接，因此 TopHub canary、当前 boot manifest、topology-B 及历史 10.724 FPS 均未复测；完整 G2 仍不可判定。
