# Flash Next r140 增量升级

适用于已完成 r135 或更新版本部署、首次环境自检通过的 2×910B4 主机。本包累计包含 r137 残差融合、r138 专家收拢、r139 down 投影和 r140 decode 后处理融合，不包含权重。

验证矩阵：CANN 9.1.0、驱动 26.1.1、torch 2.10.0、torch_npu 2.10.0.post4。其他矩阵需从源码重建并做组件、真实层与完整请求验收。

上传压缩包及 SHA256 文件到同一目录，先校验并在临时目录检查启动脚本补丁：

```bash
sha256sum -c FlashNext-r140-increment.tgz.sha256
FNX_R140_STAGE=$(mktemp -d /root/fnx-r140-stage.XXXXXX)
tar -xzf FlashNext-r140-increment.tgz -C "$FNX_R140_STAGE"
python3 "$FNX_R140_STAGE/r140-enable.py" /root/ascend-dev/p275.sh --check
```

检查通过后停止模型、备份并安装。若有外部自动恢复任务，先暂停，避免安装期间启动模型。

```bash
/root/ascend-dev/modelctl.sh stop
FNX_R140_BACKUP="/root/ascend-dev/backup-before-r140-$(date +%Y%m%d-%H%M%S).tgz"
tar -czf "$FNX_R140_BACKUP" -C /root/ascend-dev fnx/fnx p275.sh
echo "回退备份：$FNX_R140_BACKUP"

tar -xzf FlashNext-r140-increment.tgz -C /root/ascend-dev
python3 /root/ascend-dev/r140-enable.py /root/ascend-dev/p275.sh
bash -n /root/ascend-dev/p275.sh
/root/ascend-dev/modelctl.sh start flash-next
```

必须先停止使用共享库的进程，再替换 `.so`。参数补丁兼容已验证的参数化及旧式启动脚本，保留端口、卡号、容器及其他配置；无法识别时不写文件。参数化脚本保留调用方环境覆盖，确保调用方没有覆盖为旧值。

选定参数：

```text
FNX_INJECT_NATIVE=1
FNX_EXPERT_GATHER=1
FNX_DECODE_DN_GMM=0
FNX_POST_NATIVE=1
```

`FNX_POST_NATIVE` 在源码中默认关闭。本次补丁为启动脚本设置默认 1；开启后融合已验证单 token 形状的激活、门控和后处理，保留原 BF16 舍入边界及 RMS 归约。其他输入使用原路径。

206 主机健康检查如下；其他主机沿用原端口。图捕获需要数分钟。

```bash
curl -fs http://127.0.0.1:9002/health
```

应包含 `service_revision=r140`、`post_native=true`、`expert_gather=true`、`decode_down_gmm=false`、`inject_native=true`、`ctx=262144`、`worker_alive=true`。升级后使用本机基线复测性能及精度。

仅回退本轮后处理：将 `FNX_POST_NATIVE=0` 写入启动环境后重启。完整回退：先停模型，解压记录的备份回 `/root/ascend-dev`，再按原入口启动。首次从 r135 升级时，完整回退更合适。

源码构建：

```bash
cd /root/qwen-ascend
bash kernels/inject/build.sh
bash kernels/expert_gather/build.sh
bash kernels/post_ops/build.sh
```

三个共享库须与配套 Python 源码一起安装。本包共八个运行文件：engine.py、server.py、native_inject.py、libfnx_inject.so、expert_gather.py、libfnx_expert_gather.so、post_ops.py、libfnx_post.so。

后续模型切换仍沿用原入口：

```bash
/root/ascend-dev/modelctl.sh start 27b
/root/ascend-dev/modelctl.sh start flash-next
```

详细结果见同目录 `FlashNext-r140-优化实测-20261111.md`。本轮未重跑历史 100 题。
