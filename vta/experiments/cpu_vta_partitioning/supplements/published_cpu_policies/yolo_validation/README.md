# YOLOv3-tiny416 冻结方法验证

本批只比较Tarnawski单线程物理核适配、粗粒度CPU模型和本文主模型。公式直接复用`staged_ablation.model.score`及经过核验的搜索源；主机集中约束关闭。14个单元、159个模块、输入0、八输出和量化图保持不变。

入口在上一级`run_yolo_validation.py`。`audit`重放571条旧成本，`prepare`冻结新协议、控制器与成本来源索引；`validate`不连接开发板。`board run --execute-board`自动完成缺失组件、DMA流量、三组全域排名、部署、短筛选、选点冻结、较长窗口复核及分析。`board resume --execute-board`仅接续未验收任务，`status`查询，`analyze`重放并生成报告。可以用`--output`、`--run-dir`和`--local-config`指定独立位置。

成本来源可以来自不同启动，每条记录保留原boot、任务、凭据及原始文件哈希，不改写旧记录。本批流水工作则必须来自同一个boot；boot改变即停止，不混入新启动数据。

三个方法均严格排序Top-20。静态排名只读取本批冻结的隔离成本、DDR数据和资源定义。新旧流水FPS不参与排名；筛选结果只用于冻结预算1／5／20的复核选点。每项资格检查1帧；短测8＋16＋4帧；最多九个去重选点复核8＋64＋32帧，16／64个时间戳分别对应15／63个间隔。复核不再换点，不加锚点、漂移拒收或自动加测。

`artifacts/yolo_validation_01/`保存审计、跨启动成本来源索引、三组完整域搜索证明及自包含流水包；`runs/yolov3_tiny416_validation_01/`保存新任务和原始凭据，最终报告位于其`analysis/README.md`。旧迁移批次与ResNet结果只读。

结果限于单图、单启动观察，包含图内双头处理，不包含读图、预处理、解析框和NMS。标称VTA纯计算参照按YOLO各选点实际放置的卷积MAC数计算，不是严格整机吞吐上限。

本批已完成60项筛选、8项固定复核，共128个原生会话、2572次完整推理。暂停期间开发板发生重启，当前流水执行目录由`runs/yolov3_tiny416_validation_01/ACTIVE_EXECUTION.json`指向，原始报告在该目录的`analysis/README.md`。整理后的中文报告和三份Top-20 CSV在仓库`paper/my_paper/analysis/20261109_yolo_validation/`。

续跑使用独立的`run_yolo_validation_resume.py`保留准备阶段各条原boot凭据。执行器在门控关闭时省略JSON字段，`run_yolo_validation_gate_compat.py`采用既有ResNet验收器相同的缺省`off`规则，全部评分、程序、名单和窗口不变。修正及原失败记录位于当前执行目录的`GATE_VALIDATOR_COMPATIBILITY.json`和`FAILURE_01_GATE_FIELD.json`。复核已结束，无需再启动测量；`run_yolo_validation.py validate`可只读重验准备包及三组完整域证明，查询执行状态时将`--run-dir`指定到`ACTIVE_EXECUTION.json`中的目录。
