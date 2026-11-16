# YOLO单配置主机落核诊断

固定CPU0–2四线程／核心0–3，CPU7–8三线程／核心1–3，CPU13单线程／核心0；两个VTA岛、生产及收集线程共享四核，门控关闭。

只编译独立诊断执行器：原冻结源码的SessionProfile可选实现替换为逐TID、逐核心的计数型TASK_CLOCK，并记录两个既有末阶段完成时刻。阶段模块、运行库、等待策略和旧执行器保持原身份。

```bash
python run_host_placement_check.py prepare
python run_host_placement_check.py validate
python run_host_placement_check.py qemu
python run_host_placement_check.py board --execute-board
python run_host_placement_check.py status
python run_host_placement_check.py analyze
```

`board`也用于续跑：只接续尚未验收的会话，拒绝新的boot ID。范围固定为两项资格检查与三项8／64／32流水测量，合计5个原生会话、314次完整推理。新执行器计数开／关共用同一程序，旧冻结诊断执行器只作桥接。没有锚点、漂移拒收或自动加测。

计数只在收集线程取出完成编号8和71的窗口首尾读取，不新增采样线程，也不逐帧采样。每个活跃TID分别保留全CPU及核心0—3的原始计数，验证其总量守恒、角色身份和允许集合；不按enabled/running放大结果。输出保留计数读取窗口相对末阶段完成时刻的偏移，不能把它冒充完全同步的末阶段窗口。

分析只解释这一个已知配置的落核分布和探针影响；不将实测比例或慢化系数写回静态评分，不更新DP和Top-20。单启动结果不支持显著性或稳定收益结论。
