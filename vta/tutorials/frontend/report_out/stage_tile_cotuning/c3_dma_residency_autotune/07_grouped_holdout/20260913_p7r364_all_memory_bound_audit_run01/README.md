# ACC/input 生命周期边界扩展：开发集回放

根据 vta_conv2d.py 的 store_pt 和 compute_at 推导单batch、无virtual-thread、
unit stride/dilation 的 packed int8/int32 公式。其他范围返回unknown，不剪枝。

- input-stationary ACC区域覆盖全部CO，不是tile_co。
- weight-resident-barrier的空间store区域包含width-group：外层宽度tile数为偶数
  时为2，否则为1。该倍数同时影响ACC和input，来自当前调度实现。
- ACC字节=驻留CO通道数×tile_h×区域宽度×4。
- input字节=tile_ci×block_in×(tile_h+KH−1)×(区域宽度+KW−1)。

使用P7R363相同12池288个不同身份，新增公式识别全部30个已知容量失败，保留
全部149个static-ok。相较weight-only，新增13个ACC和2个input容量失败识别。
逐点输出与输入哈希在子目录audit.json中。

这是从已看过失败记录的开发池推导后回放，不是独立验证，也没有真实编译
节省计时。未解释的109个非容量lowering失败仍需完整编译检查，不能说全部
非法点均已解决。公式不是完整allocation证明；源版本、布局改变后须重新验证。
当前只加入独立审计工具的--all-memories选项，没有改变Y08冻结搜索。
