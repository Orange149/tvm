# Y08 实测容量预筛与 lowering 开销

已完成，四个独立 Python 进程按 baseline/prescreen/prescreen/baseline 顺序运行。
protocol.json 在计时前保存顺序、候选和规则源码哈希。只读取候选描述，没有
读取历史 static 分类用于在线筛选。规则来源于已暴露的开发池，故不是未见验证。

| 顺序 | 策略 | 主机进程墙钟 s | lowering 尝试 | 成功 |
|---|---|---:|---:|---:|
| 1 | baseline | 3.817223 | 24 | 6 |
| 2 | prescreen | 1.880519 | 13 | 6 |
| 3 | prescreen | 1.825490 | 13 | 6 |
| 4 | baseline | 2.087740 | 24 | 6 |

完整子进程墙钟包含解释器启动、导入、候选读取、预筛、instantiate、lowering、
成功TIR序列化/哈希及报告写入。没有FSim、DMA特征提取、交叉编译、FPGA或
搜索性能评价。因此不是端到端autotune墙钟。

四轮6个成功candidate_id及其TIR哈希集合完全一致。预筛节省11/24=45.83%的
lowering尝试，这一计数结果不依赖计时噪声。不能把它直接解释为45.83%的时间收益。

第一轮baseline明显慢于末轮，独立进程并不消除文件页缓存和宿主机冷启动。
两次prescreen都短于末轮baseline，但只有各两次重复，暂不主张稳定比例或
统计显著性。后续独立候选和更多平衡重复才能形成主结果。

保留每轮rows.jsonl的失败原因、逐点时间与TIR哈希，以及完整日志。原Y08
冻结搜索池与顺序不变；本次是独立的开发期lowering实验。
