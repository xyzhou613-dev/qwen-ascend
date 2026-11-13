# FNX 原容器开机恢复

本机专用的恢复工具：保留原 `bole-model-Flash` container ID、可写层和正式 p275 入口。用于两张固定 PCI BDF（`0000:83:00.0`、`0000:03:00.0`）在重启后 `/dev/davinciN` 枚举变化时恢复原本在线的服务。

`boot_restore.sh` 首先检查 Docker 的运行意图和 `HasBeenManuallyStopped`；尊重人工停机。若 9002 已健康，只确认正常 watchdog 存在，不重启 Engine。存在尚未健康的 Engine 或其他 NPU 任务时拒绝替换。

需要修改设备节点时，要求全部容器停止、dockerd 处于 inactive，随后仅修正原容器的两个设备节点。保留 device manager 等配置；原始 HostConfig 备份放在 0700 私有目录，备份文件 0600，不纳入 Git 或证据包。设备计划不含完整 ENV。设备/容器身份不符合预期时直接失败。

部署位置 `/root/ascend-dev/fnx-boot-restore`，systemd unit 为 `/etc/systemd/system/fnx-boot-restore.service`。启用后的 unit 在 Docker/network 之后执行，有 1000 秒超时。它不是实验自动续跑器，也不删除实验检查点。

校验：`python3 test_reconcile.py`；`bash -n boot_restore.sh`；`systemd-analyze verify fnx-boot-restore.service`。已有健康服务时的幂等复验应输出 `ALREADY_HEALTHY`，PID 不变。不要为验证它而重启生产主机。

本轮主机多次重启原因尚未确认；该工具修复恢复链路，不构成对重启原因的诊断。
