# 第一阶段：本地准备与验收

本阶段不连接板子。`prepare` / `validate` 不实例化 SSH，构建器拒绝 SSH/SCP/SFTP 命令。使用独立 SDK CMake 目录，不要求重编 host libtvm，不依赖旧清单中已丢失的同名运行库。

## 资产和身份

`local.json` 指向 host libtvm、PetaLinux SDK、共同量化 IR、调优历史、64+3 输入及 frontier 参考、位流、u-dma-buf 模块、历史合法段身份清单。已有示例针对本工作区环境，可复制后调整路径。输入清单的图 SHA256 必须与共同 IR 一致，逐文件校验输入和参考哈希。

编译缓存身份包含 host libtvm、编译 worker、TVM/VTA Python 源码、切图适配器、共同图、调优历史、VTA config 和 SDK 设置。命中缓存仍检查每个产物；身份变化重新编译。原始历史结果不修改。

SDK 初始化后的环境记录编译命令、编译和链接参数、工具实际路径与文件哈希、sysroot 路径及 SDK 环境子脚本。该记录进入阶段编译身份；原生 runtime/driver 的 CMake 目录按工具链身份与配置隔离。更换 SDK、工具或有效参数时重新编译；身份相同时可复用对应目录。编译阶段之间再次核验身份，准备期间发生变化则停止，重新运行 prepare。

运行依赖每次从当前 SDK sysroot 重新解析，包括加载器和间接依赖。先在临时目录核验完整 ELF 闭包及复制哈希，再发布到 runtime；缺失当前 SDK 文件不能由旧缓存中的同名文件补足。新闭包不保留已无引用的旧 SDK 库，新编译的 TVM/VTA 库单独核验，不被 SDK 同名库覆盖。中断后重跑会重新构造闭包，成功前不封包。

`plans/compiler_identity.json` 和 `plans/runtime_build.json` 记录共同工具链身份；`plans/sdk_runtime.json` 记录 SDK 库来源与哈希。来源绝对路径仅用于审计，部署与搬迁校验不访问源 SDK。包内校验检查三份工具链身份一致、SDK 库哈希和运行库归属完整。

准备命令在只读计算工作区路径后立即取得独占锁，随后才写入协议、编译身份或复制资产。锁持续覆盖编译、测试和封包；另一准备进程若未取得锁即退出，不得改写在用缓存。正常结束或抛出异常均释放锁，之后可按身份和哈希继续准备。

SDK 初始化前仅在子进程环境移除 `LD_LIBRARY_PATH`，避免 SDK 拒绝启动；使用系统 CMake/Ninja，避免 SDK 自带 CMake 3.15 不满足仓库最低版本。私有 `config.cmake` 明确关闭 LLVM/libbacktrace/gtest，启用 VTA FPGA 和 RPC，避免根目录配置覆盖参数。仅构建 `tvm_runtime`、`vta` 和现有原生执行器。

85 CPU 和 87 VTA 阶段全部本地编译。每段检查共同图重组精确等价、无额外量化、完整 frontier 类型。编译失败写入 `plans/compile_failures.json`，所有组统一排除；空域不可验收。线程数、affinity、轮询均为运行参数。

CPU 段使用标准 TVM PassContext，VTA 段才使用 VTA build_config。旧准备脚本对两类阶段都使用 VTA build_config，会通过 CPUAccessRewrite 向 CPU 算子插入 VTATLSCommandHandle/VTABufferCPUPtr，使 CPU 段也初始化 VTA 队列。本项目不沿用这一构建上下文；核验全部 CPU ELF 不含 VTA 外部符号，并在当前有 QEMU 的环境检查输入处理、多输入残差 frontier、末端输出三类 CPU 段的数值结果。这一修正对七组一致生效，O 的评分公式保持原样，旧成本不得复用。

## 包内容

包包含 `manifest.json`、`manifest.sha256`、`OFFLINE_READINESS.json`，以及 programs/runtime/runner/firmware/inputs/references/plans/sources。所有执行依赖使用相对包路径，保留源码快照、源码哈希和本地测试证据。SDK 的 aarch64 共享库和动态加载器一并打包，执行时显式使用包内 loader 和 library path。内核及设备接口仍由板子提供，板端资格检查验证。

包不包含编译中间目录。`artifacts/builds/<identity>/build_logs` 保留编译日志；测试日志和 XML 位于包内 `plans/`。`plans/compiler_identity.json`、`compile_coverage.json`、`frontier_validation.json`、`elf_dependencies.json` 提供审计依据。

`plans/board_payload.json` 是板端执行文件闭包，排除电脑端的 references/sources。`plans/board_execution.json` 冻结 DMA 分配、磁盘余量和诊断执行修订。完整包保留在电脑，第二阶段只按闭包生成压缩流直接解压到板上。即使缓存中留有失败阶段的旧文件，运行库检查、冻结包和板端清单也只纳入成功阶段。

`validate` 校验 manifest 身份、文件大小/哈希、aarch64 ELF、共享库闭包及所有标定输入路径。独立测试使用完整冻结域比较向量化搜索和独立标量枚举，也覆盖十次模拟启动、统计、身份错误和续跑。复制到新目录后再次 validate，确认不依赖历史目录执行。

部署与恢复测试实际运行本地 POSIX shell 和 tar，核验启动锁隔离、压缩流部署和补传、缺挂载/空间不足、无退出码时的日志回收、持久接受凭据后的清理及重试，以及诊断顺序平衡、漂移整块作废和编译失败后的统一收缩。测试用本地适配器替换 SSH 和硬件写操作，不连接板子。

断线恢复测试使用真实 `nohup` 后台任务：会话活跃时不执行日志 tar、全包哈希或清理，会话结束后按原身份接续且仅接受一次。QEMU 测试故意只提供三个诊断输入中的第一张，要求核验拒绝重复输入。统计测试在完整十次模拟启动结果上注入重复/错配 boot ID、缺凭据、缺原始数据、修改汇总等错误，验证分析及已完成轮次的续用均会拒绝异常。

研究判定测试覆盖反例：主机消融为零、专用核基线更优、不同预算各赢一部分比较，以及有性能收益但机制未确认。小于蓄水池容量的收缩域测试验证各候选均有抽样机会及固定种子复现。成本测试覆盖完整拒收、部分回收、重复快照、多启动和恢复后接受。真实 shell 测试分别使用本机 sh 与 Bash 的 sh 模式，向包装进程发送 HUP/INT/QUIT/TERM/KILL，检查子任务仍在运行时维护操作不能取得锁，普通退出则正常释放。

`OFFLINE_READINESS.json` 只有全部本地验收通过才写 `PREPARED_OFFLINE`。其中明确列出待板验证：完整数值正确性、实际 affinity/线程数、成本与预热窗口、启动独立性、驱动内核兼容及 PMU 可用性。不存在提前的真实排名或性能结论。

准备锁测试用独立进程竞争同一缓存且切换 SDK，核验锁拒绝后文件内容及修改时间均未变化；覆盖准备各步骤直到封包的锁持有，以及成功和异常后的释放。完整模拟实验上的分析回归覆盖源码变化、依赖版本变化、run 与包身份错配及缺失 state；拒绝时已有分析文件保持原样，确认未完成的路径同样受检。

工具链测试用真实 CMake/Ninja/g++ 构建最小工程，验证 SDK A→B→A 时执行器、runtime 和 driver 一致切换，且同身份缓存可复用；还覆盖主设置脚本未变但实际编译命令变化。依赖测试覆盖新 sysroot、同路径库更新、加载器更新、间接依赖缺失、旧库清除、身份错配和发布中断后的重跑。

## 复现

```bash
python run.py prepare --config configs/resnet18.json --local-config configs/local.json
python run.py validate --bundle artifacts/bundles/<bundle-id>
```

`prepare` 可在编译中断后重跑，已完成且身份/哈希一致的阶段直接复用。失败日志不会自动当作成功。源程序或协议变化后重新准备部署包，板测新建 run ID；不要把新代码接续进旧确认数据。

## 共同参考的目标架构

`inputs_manifest` 必须使用同一规范图的独立 AArch64 CPU 整图与完整边界探针参考（当前来源为 `20261017_arm_common_reference02`）。首层包含浮点卷积，x86 与 ARM 的累加差异可能越过量化舍入边界，因此 x86 参考不能作为 ARM 整数边界的逐元素精确判据。准备过程核验参考证明、原始输入的哈希与顺序及独立参考程序的哈希，并将来源文档封入包内。整数输出仍要求精确相等；不能用待验证划分自身的输出替代独立参考。历史参考仅提供数值判据，成本和性能仍在新实验中重新测量。
