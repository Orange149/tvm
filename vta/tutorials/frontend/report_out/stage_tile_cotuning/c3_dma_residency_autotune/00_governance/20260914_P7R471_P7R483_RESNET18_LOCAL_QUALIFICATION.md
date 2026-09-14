# P7R471--P7R483：ResNet18 文献对齐池的本地资格与上板冻结

状态：`LOCAL_QUALIFICATION_AND_CROSS_COMPILE_COMPLETE; BOARD_BLOCKED_BY_CHANGED_SSH_HOST_KEY`

## 1. 执行范围

本轮严格消费 P7R470 在目标标签前冻结的三个 ResNet18 几何，并且没有连接 FPGA、读取 latency、
TopHub cost 或 pool oracle。每个候选依次执行真实 VTA lowering、ML²Tuner 风格隐藏编译特征、逻辑
DMA 特征、依赖检查、三 seed FSim 和结构命令签名。Cheng 权重驻留要求完整层权重能够放入 weight
SRAM；因此 H2 的两个权重模式都记为 `not_applicable` 并绑定 same-tile original fallback。

首批每个几何均不足 12 个本地合法身份，按冻结协议仅由合法点数量触发 max-min 下一批 24 tile。
H1/H3 分别在累计 72/48 tile 后停止；H2 因完整权重不适用且 original/input 合法率低，继续到
144 tile 才达到门槛。补点从未查看板端性能标签。

## 2. 资格结果

| 几何 | 实际完整 original 空间 | 累计 tile | 四模式 gross 身份 | not applicable | lowering 成功 | 三 seed FSim 通过 | 最终本地合法池 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R18-H1 | 2304 | 72 | 288 | 0 | 30 | 13 | 13 |
| R18-H2 | 1600 | 144 | 576 | 288 | 29 | 12 | 12 |
| R18-H3 | 480 | 48 | 192 | 0 | 38 | 25 | 25 |
| 合计 | 4384 | 264 | 1056 | 288 | 97 | 50 | 50 |

671 个身份在 real lowering 失败，47 个在 lowering 后被 FSim/命令签名门拒绝。50 个通过身份全部
具有三 seed 一致命令结构；模式分布为 original 17、input-stationary 12、weight barrier 9、组合
方案 12。该结果支持“固定 FPGA 的搜索空间需要真实编译/仿真资格”，但尚不支持任何性能优劣。

P7R482 将 50 个身份、TIR 哈希、ConfigEntity、模型/硬件指纹、隐藏特征、逻辑 DMA 和命令特征冻结
为上板池，几何计数为 H1=13、H2=12、H3=25。P7R483 随后完成 50/50 ARM 交叉编译并保留哈希绑定
二进制，总墙钟 59.908 s；无失败、无板端接触。

## 3. 当前阻塞

板端只读预检没有越过 SSH 身份验证。P7R 前序串口公钥对应的 known-host key 与当前 Dropbear 提供
的 key 不一致；OpenSSH 报告当前远端 RSA 指纹为
`SHA256:u7LMrzTebC0p8iNcXkRoTweS6DEqa0EYmM0uhFpxLX0`。出于 fail-closed 原则，没有使用
`StrictHostKeyChecking=no` 接受新密钥，也没有启动 P7R484 上板目录。

恢复条件是在开发板串口执行：

```sh
dropbearkey -y -f /etc/dropbear/dropbear_rsa_host_key
```

将完整 Public key portion 与上述客户端 SHA256 指纹交叉确认后，才可重建 known_hosts，执行
clean bitstream/fresh RPC、50 点三 seed fail-fast 正确性和所有正确点五轮平衡计时。任何停电、RPC
中断或执行异常都会使整次 session 作废，不能和下一次拼接墙钟或 timing pool。

## 4. 证据边界

- P7R471/P7R473/P7R475/P7R477/P7R479/P7R481 是本地资格证据，不是 FPGA 有效性或性能标签；
- `not_applicable` 不计作编译失败，但仍保留在 gross proposal 成本，并明确回退 same-tile original；
- 逻辑 VTA LOAD/STORE 仍不是物理 AXI burst；
- 50/50 交叉编译成功不意味着 50/50 FPGA 正确；
- 在完成整池板端计时以前禁止连接 oracle，也不能运行最终六策略质量回放或整图结论。
