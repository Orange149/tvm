# 网络恢复后的身份核验断点

最近一次严格SSH连接不再返回No route to host，而是主机密钥改变：

`SHA256:TBAs9E6m4vzRfxr8a3hrXT+/++wHoAAvgdpIeqYioTs`

旧记录文件为 `/tmp/vta_c3_known_hosts_20260913_reboot2`。未修改旧记录，未接受
新密钥，未登录或上传任何文件。已请求用户通过开发板串口核对RSA公钥指纹。
重启可能重新生成密钥，但网络中观察到的新指纹本身不能独立证明目标身份。

待用户确认后，为本次启动保存独立known_hosts记录，并继续P7R376恢复流程：
当前boot/storage诊断→RAM runtime/bitstream恢复→健康canary→原Y08冻结实验。
此前的网络不可达记录仍是历史证据，不再代表最新网络状态。
