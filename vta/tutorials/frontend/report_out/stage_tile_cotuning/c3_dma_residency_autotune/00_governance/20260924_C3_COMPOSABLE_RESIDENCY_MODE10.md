# C3 mode 10：组合安全的驻留候选与跨网络开发证据

状态：`MECHANISM_IMPLEMENTED; TWO_NETWORK_GRAPH_CORRECTNESS_PASSED; YOLO_ONE_BOOT_CONFIRMATION_PASSED; DISTINCT_BOOT_PENDING`

## 1. 结论

mode 10 把“驻留数据是否正确”扩展为“驻留程序与其他图节点组合执行时是否正确”。它保留 mode 5
的计算和数据驻留，仅在候选自己的驻留命令段使用局部 force-serial 完成语义，并在同步后恢复正常
模式。YOLO-608 两路由组合和 ResNet18 H3 均恢复整图精确正确性；全局串行不再是部署前提。

YOLO-608 的单启动开发计时显示局部机制相对 stock 改善13.484%，相对全局串行改善12.298%。
ResNet18 则仅相对安全 original 改善0.056%，相对 stock 退化0.250%。因此性能机会具有图和层的
条件性，选择器必须能够采用强正例并拒绝低收益方案。

## 2. 机制与证据边界

新增公开模式：`input_weight_resident_composable`，implementation mode为10。lowering生成：

```text
VTASetDebugMode(32)
驻留候选命令段
VTASynchronize
VTASetDebugMode(0)
```

全局 `vta_debug_flag=0`。最终TIR地址审计要求debug转换为空或精确`[32,0]`，并继续检查全部
DDR/SRAM/UOP/GEMM/ALU边界。反汇编范围审计要求控制调用只出现在被路由的目标函数中。

当前证据只能表述为：正常依赖编码下的组合执行失败，而局部串行完成恢复正确性。尚未定位到一条
精确token依赖或一个具体硬件race，因此不作更强微体系结构归因。

## 3. 结果

### YOLOv3-tiny 608

- 两个真实路由：`1024→256, 19×19, 1×1`与`512→256, 19×19, 1×1`；
- 两条mode 10均通过本地地址/容量/依赖和三seed FSim；
- scoped整图三seed、每seed八个输出全部与stock相等；
- 单启动9配对：scoped vs stock为13.484%，scoped vs global为12.298%，均9/9获胜。

### ResNet18

- 历史H3：`256→512, 14×14, 1×1, stride=2`；
- mode 5孤立算子正确但历史整图0/6正确；
- mode 10通过本地门槛，整图三seed的1000类输出全部与stock相等；
- 单启动9配对：相对同H1/H2的安全H3-original为0.056%且6/9获胜；相对stock为-0.250%。

## 4. 对第三创新点的影响

候选不再只由`Tile + O/I/W/IW`定义，还必须包含完成策略和组合资格。selector v3的硬过滤至少包括：

1. lowering、容量和最终地址合法；
2. 孤立FSim/FPGA正确；
3. 完成策略范围受控且退出后恢复正常模式；
4. 图上下文正确性通过，或属于已证明兼容的组合类。

性能排序再使用请求bytes/calls、UOP、指令、提交与完成策略代价。独立层候选分数不能直接相加为图级
可部署方案；选择对象应为满足兼容约束的候选集合。ResNet18结果证明“正确可组合”仍不等于“值得
部署”，因此必须保留incumbent回退。

## 5. 下一步冻结顺序

1. mode 10代码和本轮二进制保持冻结，不根据性能继续改完成边界。
2. 机器可读`composability_certificate`和候选集合兼容门禁已经实现；图级计时必须绑定精确
   candidate set，不再把集合收益归给任一单层candidate。
3. YOLO-608已完成boot-1六轮确认：中位改善13.413%、6/6轮获胜、输出全等；当前boot ID仍与
   开发轮相同，所以部署门禁继续拒绝。
4. 等真正变化的boot ID后，只对相同YOLO候选集合执行原样boot-2。ResNet18已因低于2%回退，
   不再消耗正式确认预算。
5. 双启动通过后，在新的未见任务/分辨率上冻结selector v3性能模型与验证集；当前YOLO/ResNet18
   都保持latency-exposed开发数据身份。

## 6. 主要产物

- `20260924_c3_yolo608_composable_local_run02/`
- `20260924_c3_yolo608_composable_build_run01/`
- `20260924_c3_yolo608_composable_board_run01/`
- `20260924_c3_yolo608_completion_scope_timing_run01/`
- `20260924_c3_resnet18_h3_composable_local_run01/`
- `20260924_c3_resnet18_h3_composable_build_run01/`
- `20260924_c3_resnet18_h3_composable_board_run01/`
- `20260924_c3_resnet18_h3_composable_timing_run01/`
- `20260924_c3_composability_certificates_run01/`
- `20260924_c3_yolo608_composable_confirmation_boot1_run01/`
- `20260924_c3_selector_v3_development_evidence_run01/`

所有板端产物记录同一boot `c9d73f4c-c0df-474a-9a74-a28a2f8955e2`，且明确
`bitstream_reloaded=false`、`board_reboot_requested=false`。
