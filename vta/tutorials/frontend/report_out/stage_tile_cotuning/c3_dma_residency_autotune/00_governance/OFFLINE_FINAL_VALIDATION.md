# C3 离线阶段最终验证记录

日期：2026-09-11  
状态：`OFFLINE_READY_FOR_BOARD_PREFLIGHT`  
范围：仅本机编译、静态分析、FSim 与 AXU5EVB 交叉编译；未使用 SSH、网络 RPC 或开发板。

## 1. 最终离线产物

- P4j run02：197 个本地合格候选的严格无 RPC、direct-local FSim command dry-run；197/197 seed-0 正确，32/32 mode-4 drain 与既有证据一致。
- P5c：197 候选、10 workload 的无标签 command-aware shortlist；预算 4/8/16 均保护 incumbent，排序通过 label-poison invariance 测试。
- P6b：W00/W02/W09 各 original、最佳 input 降幅、最佳 barrier-weight 降幅，共 9 项机制 canary；9/9 绑定已有 FSim 与 AXU 资格，未来板端顺序与判定门槛已冻结。

P4j run01 只使用本机 `rpc.LocalSession()`，没有联网或访问开发板，但违反 P4j 的严格 no-RPC 预注册协议。因此 run01 已标为 `superseded`，只保留审计轨迹；P5c 明确只消费 run02。

## 2. 综合回归

在同一工作树和 `/tmp/vta-hw-fsim` 配置下执行核心 C3 测试集合：

```text
138 passed in 9.82s
```

覆盖理论边界、稳定候选身份、驻留候选生成、CoProcSync 屏障语义、依赖审计、多 seed FSim 资格、AXU 交叉编译资格、代表/全池命令画像、无标签 shortlist 和两类未来派发合同。

增量构建检查：

```text
cmake --build build --target tvm vta_fsim -j2
ninja: no work to do.
```

JSON/JSONL 全量解析检查：

```text
files: 135
records: 1825
invalid: 0
```

`git diff --check` 退出码为 0。

## 3. 最终关键哈希

- P4j run02 `artifact_hashes.json`：`e4c7f82e8959d87c2fc2410123541a3ae4f82910d571e2a02511a8467f90e181`
- P4j run02 `results.jsonl`：`7a875ed1d4d9404187e7976e53851fdea35c8b94117711d79200e5382499e179`
- P4j run02 `summary.json`：`fbcb1ba119d8a024b94e8cef618d1a5d685d7fa75a70a0813fbf5b2ffdb7713b`
- P5c `artifact_hashes.json`：`21475166073a164f265776172a333e01384cb8222c0ace1d06fb25a563f37fd8`
- P5c `shortlist.json`：`7ff96142d28f0ddea2a153ada86d8f4baad421cf7c4357887d8d87beab542031`
- P6b `artifact_hashes.json`：`b2ce0719d77a7fd806bf1adf67ea5a7f849eef5946e6f7499324b629177013f2`

各 run 自身的 output/source ledger 均由对应阶段复验，无 mismatch。

## 4. 不能越过的结论边界

离线阶段已经证明：机制可 lower、可模拟正确执行、可为目标 ARM 平台生成模块，且能提取稳定的静态 DMA/命令结构并据此冻结派发顺序。

离线阶段没有证明：当前 boot 可加载、真实 runtime DMA 与静态描述一致、板端数值正确、单次推理或 FPS 提升、搜索预算减少、instruction/UOP backing 可安全缩到 dry-run 峰值，或 G2/G5/G6 已通过。FINISH 仍是 runtime 源码推导，replay 在当前日志中不可观测。

开发板恢复后必须从环境/SD/boot 指纹和 TopHub canary 开始，随后执行已冻结 P6b；不能根据新测结果改候选或判定阈值。
