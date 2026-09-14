# P7R227--P7R233：Relay 显式驻留分派与 ResNet50 传递

## 目的

关闭 R50A/R50B 仍停留在独立算子模板的缺口：把已通过多级资格的 residency mode 和完整
ConfigEntity 真正送入普通 Relay/VTA 编译与 graph executor，并验证完整 ResNet50 图。

## 实现

- 新增 `vta.top.ExplicitResidencyDispatch`。route key 为 AutoTVM exact workload，route value 同时
  绑定 complete ConfigEntity、candidate ID、public mode 和 implementation mode。
- `schedule_conv2d_packed` 在 schedule 入口解析 route；不存在显式 route 时维持 original，配置身份
  不一致时报错，避免 silently 选择错误 residency。
- 由于 Relay schedule 可能在线程池中执行，解析会沿 AutoTVM DispatchContext 链寻找 route，并以
  exact query 返回的 ConfigEntity 对象身份完成 schedule 绑定。
- 四项单元测试覆盖默认 original、mode4 命中、错误 config 拒绝和非法 mode 拒绝。

## 运行序列

1. P7R227 run01：Relay lowering 成功，链接时裸交叉编译器缺少 sysroot。修复为解析环境 `CXX` 和
   `LDFLAGS`；该失败不计候选 invalid。
2. P7R227 run02：最小 Relay A/B 交叉构建成功，same graph/params、different TIR/binary，route
   schedule 1/1 命中。
3. P7R228：clean-start 板端最小 Relay 图 6/6 correctness、14/14 timing 正确；25.674497 ->
   25.083010 ms，驻留 +2.358%、7/7。
4. P7R229 run01：MXNet 随机 ResNet50 延迟初始化失败，未进入 VTA build；补一次 dummy forward。
5. P7R229 run02/P7R230：随机参数整图成功并 7/7 更快，但最终 logits 全零，仅保留为开发记录。
6. 下载官方 Gluon `resnet50_v2` 参数后，P7R231 重新交叉构建；两版 107 个 lowered 参数内容一致。
7. P7R232：clean-start 预训练整图三 seed 的 1000 个 logits 全部非零且 A/B 全等；14/14 timing
   输出全等，406.700897 -> 405.361804 ms，驻留 +0.330%、7/7。

## 关键解释

整网 profiler 的 weight LOAD 差值为最小 Relay 算子差值的四倍，因此 exact workload 在 ResNet50
中有四个执行实例。当前分派粒度是 workload，不是层 ID；这既解释了整网计数，也规定了后续
call-site manifest 的必要性。整网只改善 0.33%，因为目标四层的局部 DMA 收益被其余 VTA 层与 CPU
尾部摊薄。该结果用于证明落地，不改变以搜索成本为核心的论文主张。

板端 boot 为 `aa7a3e5c-021d-4d6b-88ae-7f696faa567c`；实验后 FPGA 为 `operating`、u-dma-buf 为
201326592 B、RPC 位于 tmpfs runtime、无新增 EXT4/mmc 错误。

