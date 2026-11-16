# C3 selector v4：YOLO-544 strict-unseen boot-1与生产门纠正

## 结论

selector v4在YOLO-576暴露开发池上形成并冻结，在YOLO-544任何板端时延出现前固定候选、静态特征和
B1/B2/B3顺序。两个真实YOLO-544 `1×1, CO=256`任务的完整孤立算子池中，v4以B1命中2/2池内
oracle，bytes-only和calls-only均到B2命中；但两个oracle都是完成策略不具生产资格的legacy mode5。
因此原结果是有效的孤立算子池评价，不是可部署selector成功。

遗漏的生产硬门已作为独立模块补到排序器之前。事后审计显示，每任务的生产合格子池包含incumbent、
三个Original、三个Input和三个mode11，共9个程序；两个子池的oracle均为mode11，静态v4顺序均在B1
命中。因为该重放已经读取boot-1时延，它只能验证协议修复方向，不能获得新的strict-unseen标签。

## 未见链路

1. 608、576分辨率被暴露注册表排除；按预注册顺序选择544。
2. YOLO-544静态图合同为119,471,104 B，小于192 MiB；含11个唯一VTA执行任务。
3. 三个K1/CO256任务生成45个五模式程序；静态/FSim为43/45。
4. CI1024任务的冻结incumbent最终TIR审计展开超界，因此不替换incumbent、不补点、不进入参考池。
5. 两个完整任务的30个程序全部cross通过，随后30/30候选、90/90调用FPGA正确且未采集latency。
6. 此后才执行boot-1计时，完成726次算子调用，全部正确、0隐式调用、会话未拼接。

## boot-1结果

| 项目 | 结果 |
|---|---:|
| 完整参考任务 | 2 |
| 每任务孤立算子程序 | 15 |
| v4首次命中孤立池oracle | 2/2在B1 |
| bytes-only / calls-only | 2/2在B2 |
| command-only | B3仍为0/2 |
| 同Tile Input→mode11 | 6/6配对均6/6轮获胜 |
| 同Tile配对改善 | 中位36.981%，范围13.427%--69.433% |
| boot ID | `c9d73f4c-c0df-474a-9a74-a28a2f8955e2` |

两个原确认候选相对冻结pool incumbent的中位改善分别约65%和92%。incumbent仅是生成池时按固定规则
选出的Original程序，不代表充分调优TopHub或生产部署；这些百分比不得写成整网提升。

## 协议缺陷与修复

原板池同时包含Original、Input、mode4、mode5和mode11。mode4/mode5适合作为孤立算子机制参考，
但完成策略注册表已把它们标为非生产。原冻结脚本没有在v4排序前调用既有`candidate_disposition`，
导致B1选择了mode5。

新增`c3_selector_v4_deployment.py`执行以下顺序：

```text
完整本地正确参考池
  → completion-policy / composability硬门
  → 生产合格新候选
  → label-free v4 Tile多样性排序
  → B1/B2/B3前缀
```

legacy程序仍保留在另命名的完整参考池，用于回答“机制上哪个孤立算子最快”，但不进入部署预算。
mode11还必须通过本地组合证书；任何候选进入图前仍需精确图上下文正确性。

该顺序已由`20260924_c3_selector_v4_production_gate_freeze_run01`冻结，绑定v4排序器、生产门、
完成策略注册表和可组合性实现的源码哈希；冻结过程不读取候选或时延。相关聚焦回归68/68通过。

事后审计拒绝12个legacy程序（6个mode4、6个mode5），两个九程序生产子池的最快者都是mode11，
修复后B1均命中。审计文件明确记录`performance_labels_read=true`、`strict_unseen_claim=false`和
`deployment_authorized=false`。

## 当前证据边界

- 可以主张：在两个新任务上，mode11相对同Tile Input均有稳定boot-1净收益；完成策略必须是选择硬
  约束；低请求窗口内保留Tile多样性比旧单一角色短名单更合理。
- 不能主张：mode5可部署、修复后的生产子池重放仍是未见验证、selector已跨网络普遍优于bytes、
  已有跨boot稳定性、完整图正确性或端到端收益。
- 当前板卡boot ID没有变化，不能执行或登记独立boot-2。

## 下一门禁

1. 在真正变化的boot ID上，可对原始孤立参考池做boot-2复现；其用途仅是机制和池内排序稳定性。
2. 部署主线必须使用生产门后的mode11候选，并先生成精确图候选集合与多seed完整图正确性证书。
3. selector方法的下一次strict-unseen验证必须换新任务，并在其latency之前调用生产门冻结预算。
