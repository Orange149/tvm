# P7R216--P7R226：ResNet50 扩张层严格留出与四池汇总

## 结论

在不修改 P7R185 冻结阈值、service proxy 或 adaptive policy 源码的前提下，R50B 为第四个严格
标签隔离 holdout。它验证了第三次 sparse path，并且第一次 FPGA 测量仍为完整池 oracle。加入
R50B 后，四个严格 holdout 的 20-seed 配对汇总中 adaptive 相对全池资格减少 70.83% gross、
26.83% 统一墙钟，相对固定 4→2 降低 15.96% 墙钟。

## 几何与冻结

- 来源：仓库 `python/tvm/relay/testing/resnet.py` 的 ResNet50 bottleneck later stride-1 unit conv3。
- 几何：CI=128、CO=512、H=W=28、K=1、stride=1、padding=0，是 bottleneck 的通道扩张卷积。
- 完整 ConfigSpace 3456 点，容量/复用粗筛 581 点；按新 seed 冻结 8 family，再展开为 24 个 v2
  original/input/weight-barrier 身份。
- P7R216/P7R217 在 target lowering、FSim、FPGA correctness 和 latency 前完成身份及策略冻结；
  `replay_vta_adaptive_multifidelity_search.py` 保持冻结哈希 `57ca4e87...`。

## 在线路径与完整池

- 首 family 0/3 lowering-pass，按阈值进入 sparse family-wave。
- 在线前缀共执行 6 次 lowering、1 次 FSim，冻结一个首波候选。
- 首波候选 3/3 FPGA-correct、7/7 计时正确，中位 25.274863 ms。
- 完整 24 身份只有 4 个 lowering/FSim-pass：original 3 个、weight barrier 1 个、input 0 个。
- 20 个 lowering 失败为 `dma_2d_pattern=10`、`allocation_capacity=6`、
  `dma_compact_buffer=4`。这再次证明容量公式不能代替真实 lowering。
- 4/4 候选三 seed FPGA-correct，28/28 计时正确；独立全池 oracle 与首波身份相同，为
  25.151352 ms。

一致回放中，adaptive 为 1.352282 s、gross 6；exhaustive 为 2.838543 s、gross 24；固定
4→2 为 2.668095 s、gross 24。adaptive 相对两者墙钟分别下降 52.36% 和 49.32%。三种策略达到
目标时均只测同一个 oracle，因此不声称 FPGA invocation 或目标 DMA 减少。

## 机制结果

R50B 唯一完整 same-tile 对是 F02 的 original/weight-resident-barrier。barrier 从 25.746468 ms
降到 25.151352 ms，改善 2.31%，七轮均更快；总逻辑 DMA 字节下降 15.95%、DMA calls 下降
42.86%、instruction peak 下降 94.43%，但 UOP peak 上升 193.10%。这是“访问减少—命令峰值变化—
同步后净加速”的新边界，收益远小于 Y02B00 和 R50A input pairs，说明 residence mode 必须与
几何/tile 联合选择。

## 四个严格 holdout 汇总

| 方法 | 四池全部命中率 | 中位 gross | 中位统一墙钟 |
|---|---:|---:|---:|
| 生存率自适应多保真 | 100% | 28 | 21.088 s |
| 固定硬件多样 4→2 | 100% | 82 | 25.094 s |
| 全池资格后 service 排序 | 100% | 96 | 28.822 s |

四个 target 都第一次测量命中完整池 oracle，是强正结果，但样本数仍小，不能宣称 universal
optimality。R50B 只有四个性能点；R50A/R50B 都是来自真实模型结构的算子级几何，不等于
ResNet50 stage 或整网 FPS。

## Ledger SHA-256

- P7R216 冻结合同：`4c99398f314dd381cccd747ff76de58be1883a4433b908eb396de314d0e74965`
- P7R217 adaptive registry：`db21df30db4351dcf9701b87ec203f8477fb0bf4383cbe5df706ce538e17d9ce`
- P7R218 live local prefix：`5e9b4ac65129f8ce8658d6c2034e53806f1551392195254edd1e11919241891a`
- P7R219 first board wave：`433397fbd8f1af2bed9f824916f86bf41f0580298d094f3a2529e534d8d761f4`
- P7R220 full local：`6cf4f5d47c0feb9c4d0dcd21127981b75faeb089821733eb66250f1c7480b06d`
- P7R221 full board：`835eeb15d9bd8f7a8c038a2393b9d3b12de6507b51bd7e3cea67ead27a872878`
- P7R222 canonical pool：`b8e21e644a5f72394b18212a37031b8532b44381b50e7ba1d064516d92c066ed`
- P7R223 multifidelity pool：`1218054dbc900764068e1838260aa65e3375d1adc0240b7c23b3c8d71be9d729`
- P7R224 holdout analysis：`908c03d8aa811d91b9c748689cb3f95b6845f7088c338d8acc42e9b45436f825`
- P7R225 same-tile analysis：`35e34ef71c585b513f4af1d178b66d6235fb1aeeeddc673a9c01a11e51125d28`
- P7R226 four-holdout aggregate：`428c154f884912e19bc2bdf1b2d9fd041e9a62a65abecc2f8d97f02df227131f`
