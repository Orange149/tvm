# YOLO固定核组与开放共享匹配验证

复用冻结的YOLO隔离成本和执行器，只比较静态选出的资源组织方式。
固定核组须有至少两个不同且互不重叠的核组；同组串行。
从原主模型第一名、固定组家族第一名、Tarnawski第一名构造最多三个切图／上下文区块。
每区块比较开放优选、固定组优选及固定组关闭串行；另加原Tarnawski第一名桥接。
去重后最多10个配置，每项一帧资格检查＋8/64/32正式窗口，无自动加测。

在上级目录执行：

```bash
python run_yolo_cpu_policy_check.py audit
python run_yolo_cpu_policy_check.py rank
python run_yolo_cpu_policy_check.py prepare
python run_yolo_cpu_policy_check.py validate
python run_yolo_cpu_policy_check.py board run --execute-board
python run_yolo_cpu_policy_check.py board resume --execute-board
python run_yolo_cpu_policy_check.py status
python run_yolo_cpu_policy_check.py analyze
```

排名不读取任何流水FPS。三类来源和全部任务在本批计时前冻结；旧结果、评分公式、运行库和执行器保持原身份。
发生boot变化时停止本批；已验收任务仅重放凭据，不再次测量。
分析输出逐区块匹配表、逐阶段隔离／流水服务、整体CPU核时、原始证据哈希及费用。
结果只表示单输入、单启动观察，不自动修改模型或开展下一轮Top-20。
