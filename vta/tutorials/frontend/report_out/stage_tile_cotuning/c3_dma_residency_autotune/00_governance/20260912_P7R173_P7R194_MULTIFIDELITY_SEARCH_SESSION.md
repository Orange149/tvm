# P7R173--P7R194：在线多保真搜索与 Y05/Y06 严格验证

## 结论

本轮把第三创新点从“全池完成 lowering/FSim 后离线排序”推进为真正的 candidate×phase 按需付费
搜索。候选在时刻 0 只能看到 knob、SRAM 粗界和几何；完成 lowering 后才揭示精确 DMA，完成
FSim 后才揭示命令峰值，完成 FPGA gate 后才允许测 latency。主排序使用已冻结的共享内存服务代价：

```math
S=B_{dma}+65536N_{dma}+131072\max(N_{submit}-1,0).
```

Y05 证明固定 4→2 前沿可以在严格标签隔离池中第一次测量就命中 oracle，但也暴露合法空间极稀疏
时几乎扫完全池的问题。随后只用已暴露开发标签冻结“首 family 三模式存活数不超过 1 则切换
family-wave”的规则。Y06 是该规则冻结后的严格确认：24 个身份只需考虑 6 个、执行 1 次 FSim 和
1 次板端测量便命中完整池 oracle，统一阶段墙钟相对全池资格降低 26.95%。

## P7R173--P7R174：状态机与开发消融

- P7R173 汇总 Y00/Y03/Y04 完整池和 Y01 六点恢复池，共 78 个候选，补齐 lower、FSim、compile、
  FPGA correctness 和 measurement 的逐阶段成本与状态。
- P7R174 冻结 4→2 规则：硬件多样性选 4 个 lowering 前沿，再按已揭示的 service proxy 晋级 2 个。
- 已暴露开发池上，相对全池资格后排序，达到同一 oracle+2% 的总墙钟为 184.589→80.394 s
  （-56.45%），lowering 78→26、FSim 65→8；FPGA invocation 和逻辑 LOAD 相同。
- 跨 workload Model V 没有稳定优于无模型规则，因而只保留为 ML²Tuner 启发的负向消融。

## P7R175--P7R183：Y05 严格在线 holdout

- 几何：YOLOv3-tiny conv6，3×3，CI=64、CO=128、H=W=52。
- 在任何目标 lowering/FSim/FPGA/latency 之前，从 1728 个 ConfigEntity 中冻结 8 family × 3 mode。
- 在线前缀执行 23 次 lowering，其中 18 次失败；两个晋级 FSim 均通过。
- 第一块板端波次中两个候选均三 seed 正确、14/14 计时正确。第一次测量候选为 16.299433 ms。
- 完整池最终为 18 lower-invalid、2 FSim-invalid、4 FPGA-correct；独立完成后的 oracle 仍是同一候选，
  16.305223 ms。
- 一致回放中，固定前沿相对全池资格墙钟 49.468→39.339 s（-20.48%）。
- 限制：首波收集器实际同时认证和计时了两个晋级候选，因此候选顺序是前瞻的，但多出的收集工作
  不能包装成字面上的在线早停收益。

## P7R184--P7R185：生存率自适应规则冻结

Y05 表明固定宽度在稀疏合法空间会浪费 lowering。开发阶段比较后冻结如下规则，Y06 之后不得再改：

```text
先 lower 第一个硬件多样 same-tile family 的三个 mode
  ├─ 通过数 <= 1：sparse family-wave，逐 family 立即晋级最小 service 候选
  └─ 通过数 > 1：dense fixed 4→2 frontier
```

五个暴露 workload 上均能达到 oracle+2%。相对固定前沿，Y05 墙钟 -95.23%，Y01 -2.55%，Y00/Y03
不变；Y04 +4.21%。因此它是根据在线存活率选择资格粒度，不是每个 workload 都必然占优的万能规则。

## P7R186--P7R194：Y06 严格前瞻确认

- 几何：YOLOv3-tiny conv4，3×3，CI=32、CO=64、H=W=104。
- 完整 ConfigSpace 为 1536 个实体，201 个通过便宜容量粗筛；在任何目标结果前冻结 24 个身份和
  自适应源码 SHA-256 `57ca4e87f5d948647a02bff244c33277722e2bcdb8bef792ccef21f4255fefcc`。
- 首 family 0/3 lowering 通过，按冻结阈值进入 sparse path；下一个 family 为 2/3 通过，最低 service
  的 input-stationary 通过三 seed FSim，形成一个候选的首波。
- 首波候选 `661dd142...` 三 seed FPGA-correct、7/7 计时正确，中位 30.074251 ms。
- 完整 24 点结局：22 lower-invalid、0 FSim-invalid、0 FPGA-invalid、2 measured-ok；第二候选为
  49.925149 ms，完整池 oracle 仍是 `661dd142...`，独立中位 30.102661 ms。
- 严格在线 time-to-oracle：6 lower、1 FSim、1 compile、1 FPGA gate、1 measurement；gross 6/24。
- 统一诊断成本：adaptive 3.522092 s，exhaustive 4.821705 s（-26.95%），固定 4→2 为
  4.971984 s（adaptive -29.16%）。实际首波 canonical phase wall 为 3.688025 s，包含独立首波
  采集的真实编译、三 seed correctness 和代表计时成本。
- 一次推理的首选候选逻辑 LOAD 为 4,415,360 B/832 calls，STORE 为 692,224 B/832 calls；三 seed
  正确性加一份代表计时累计 5 次 kernel invocation、22,076,800 B LOAD。它们是逻辑 VTA 计数，
  不是物理 AXI burst。

## 失败与证据卫生

- P7R188 run01 因在线执行器在首 family 0/3 时错误停止，是实现失败；未连接开发板、未产生 latency，
  原样保留。run02 只修复“继续到下一 family”，未修改冻结阈值或候选。
- P7R193 run01 把输出目录参数误写为文件名，产生嵌套目录；它只是封装命令错误，不是实验失败。
  run02 为正式 staged pool。
- P7R190 的 22 条 `not_run_static_failed` FSim 占位行继承旧 Y01 helper 的描述性 `workload_id=Y01`；
  candidate ID、family、static failure 和最终状态均与 Y06 正确绑定，pool builder 也按 candidate ID
  连接，因此不影响结论。后续源码已在调用点把该字段规范化为实际 workload；冻结的 P7R190 原件
  不回写。
- Y05 的收集器偏差被显式记录；Y06 首波只有一个候选，不存在同类偏差。
- 本轮之后开发板 boot 为 `aa7a3e5c-021d-4d6b-88ae-7f696faa567c`；两个 SD 分区为 rw，dmesg 中
  ext4/mmc/I/O error 数为 0，新 RPC 正常，u-dma-buf 容量为 201,326,592 B。仅对当前 boot 有效。

## 主张边界

现在可以主张：在固定显式 DMA FPGA 上，以最终编译程序的共享内存服务代价排序，并依据在线
lowering 存活率选择资格粒度，能在一个严格未见的 1536-ConfigEntity 几何子池中减少达到完整正确池
oracle 的搜索墙钟和资格次数。这已经不是单纯的离线排序或工程部署。

仍不能主张：普适最优、物理 AXI 流量下降、整网 FPS 提升、Cheng 2026 机制原创、Model V 优于
ML²Tuner，或一次 Y06 验证足以保证跨网络泛化。下一阶段优先做更多网络/几何与 stage 传递，而不是
在 Y06 上继续调阈值。

## 关键证据哈希

- P7R173：`d689ae219545f3b49a95c7630890b7d520efcb713d3ea6e7c4ae52a9ad3079c7`
- P7R174：`a20679b2df21eaf6f7252b0066a699760ed2af39c6a5201151259b1ae73d4268`
- P7R183：`9fdd964bd17de87401c589d6c9ff446e639ba9ecdddbf585ae85903b4cf16009`
- P7R185：`8ea2ceae865622a00a748823bc38a5735abef043950e3526f1279039ed64e517`
- P7R187：`4893f3be29632c7b03841d6154721ae52745d0c67dad565a038b7a2d980a37bf`
- P7R188 run02：`b6c367f400dd7f93a5378a1ddfcdf1e08eb224771607f4b4fbcc6eb2bf68c94d`
- P7R189：`99b77161e84c55ab88f1601b1804476ebf8166bead8a642c531a2bc57d2f431f`
- P7R190：`80ec738de52324bde181662ead6df584e58cf0692497d75cede08a5c9f846468`
- P7R191：`e38f9ebf2e79388f2bd396e14f9f8a996ae856800ca53c832d313cb3202d6fe6`
- P7R194：`bf282b0724dfdcd710c3a0a32b4a78e240b0a2a232c36cc21b866882801df3c9`
