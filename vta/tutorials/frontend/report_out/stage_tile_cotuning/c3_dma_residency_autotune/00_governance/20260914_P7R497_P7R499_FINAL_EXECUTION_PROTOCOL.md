# P7R497--P7R499：ResNet18 文献对齐最终执行协议

状态：`FROZEN_BEFORE_ANY_MERGED_POOL_FPGA_OR_RESNET18_FULLGRAPH_LABEL`

## 已补齐的实现

P7R497 先冻结 214 点板后分析的六策略、20 seeds、预算 10/20/50/75/96 和两种不可混淆的成本
口径。随后发现整图部署必须保留每条搜索轨迹的终点 candidate identity，因此在仍未读取任何 R18
性能标签时补充该字段，并由 P7R499 重新绑定实现哈希；P7R497 保留为被明确替代的旧协议。

P7R498 对 `relay.testing.resnet.get_workload(num_layers=18)` 做实际量化、graph-pack 和 AutoTVM
任务提取。共得到 13 个唯一卷积 workload，H1、H2、H3 各精确命中一次：

- H1：64→64，56×56，3×3，stride 1；
- H2：256→256，14×14，3×3，stride 1；
- H3：256→512，14×14，1×1，stride 2。

这一步关闭了“独立算子几何可能并不属于所测整图”的来源风险。后续完整图只允许使用该 Relay
Testing ResNet18，不允许拿 MXNet ResNet18 的结果替换。

P7R499 绑定 21 个实现源码哈希和 P7R494/P7R496/P7R498 输入证据，并冻结：

1. 214 点完整板池的三 seed fail-fast 正确性和全部正确点五轮平衡计时；
2. Random、pool-XGB、HW-Aware+XGB、ML²Tuner P/V/A、Cheng minimum-access、本文 DMA
   multifidelity 的 20-seed 等预算回放；
3. ML²Tuner `A+DMA` 只作为消融，不包装成第七种主方法；
4. Cheng original/input/weight/combined 的 same-tile 四方案分析，只有 SRAM 不适用才回退 exact
   original，lowering/FSim/FPGA 错误不允许静默回退；
5. 固定 seed 57001、budget 50 后，为 pool-XGB、Cheng、ML²Tuner、本文方法各选 H1/H2/H3
   三条精确路由；相同路由签名只构建一次；
6. stock 与四种策略图使用三个确定性随机输入检查全部输出，再做七轮平衡计时；
7. H1 上空历史官方 AutoTVM-XGB 和本文流程各做三次完整 clean-start，实际计量 W0→T1 与
   T0→T1，而不是事后把零散 phase 相加。

## 计时边界与失败口径

- W0：host 进程启动，尚未进行开发板 preflight、bitstream 和 RPC 准备；
- T0：冻结 bitstream 已重载且新的默认 RPC 就绪，调优工作从此开始；
- T1：选中配置进入完整 ResNet18，通过三个输入的全部输出等价并完成七轮计时。

编译失败、板端错误、RPC 重启和恢复均计入 gross 与墙钟。任何断电、RPC 或设备执行中断都会生成
`invalid_session.json`，整次运行不得与另一会话拼接。完整池的实际公共资格成本和策略反事实 lazy
成本分别报告，不能相加后冒充同一种实测在线轨迹。

## 当前仍未产生的结果

目前没有 214 点池的 FPGA correctness、operator latency、pool oracle、Model P/A 性能指标、Cheng
最快方案一致率、ResNet18 整图 latency/FPS 或三对三 clean-start 结果。阻塞原因不是脚本缺失，而是
开发板当前 Dropbear RSA 公钥与此前串口核验记录不同。客户端观察到的指纹为
`SHA256:u7LMrzTebC0p8iNcXkRoTweS6DEqa0EYmM0uhFpxLX0`；在串口重新读取并确认完整公钥前，严格 SSH
验证会继续拒绝连接，不能用 `StrictHostKeyChecking=no` 绕过。
