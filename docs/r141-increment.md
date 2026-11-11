# Flash Next r141 增量升级

适用于已完成 r135 或更新版本部署、首次环境自检通过的 2×910B4 主机。本包累计包含 r137 残差、r138 专家收拢、r139 down 投影、r140 后处理及 r141 MoE 输出融合，不含权重。

验证矩阵：CANN 9.1.0、驱动 26.1.1、torch 2.10.0、torch_npu 2.10.0.post4。其他矩阵先自检、重编译并做组件、真实层与完整请求验收。

上传压缩包及 SHA256 文件到同一目录，先校验和预检查：

```bash
sha256sum -c FlashNext-r141-increment.tgz.sha256
FNX_R141_STAGE=$(mktemp -d /root/fnx-r141-stage.XXXXXX)
tar -xzf FlashNext-r141-increment.tgz -C "$FNX_R141_STAGE"
python3 "$FNX_R141_STAGE/r141-enable.py" /root/ascend-dev/p275.sh --check
```

检查通过后暂停外部自动恢复任务，停止模型、备份并安装：

```bash
/root/ascend-dev/modelctl.sh stop
FNX_R141_BACKUP="/root/ascend-dev/backup-before-r141-$(date +%Y%m%d-%H%M%S).tgz"
tar -czf "$FNX_R141_BACKUP" -C /root/ascend-dev fnx/fnx p275.sh
echo "回退备份：$FNX_R141_BACKUP"
tar -xzf FlashNext-r141-increment.tgz -C /root/ascend-dev
python3 /root/ascend-dev/r141-enable.py /root/ascend-dev/p275.sh
/root/ascend-dev/modelctl.sh start flash-next
```

必须先停止使用共享库的进程，再替换 `.so`。补丁兼容已验证的参数化和旧式启动脚本，保留端口、卡号、容器及其他参数；无法识别时不写文件。参数化默认值仍允许调用方环境覆盖，需确认调用方没有覆盖为旧值。

选定参数：

```text
FNX_INJECT_NATIVE=1
FNX_EXPERT_GATHER=1
FNX_DECODE_DN_GMM=0
FNX_POST_NATIVE=1
FNX_MOE_REDUCE=1
FNX_MOE_EPILOGUE=1
```

源码默认关闭 `FNX_MOE_EPILOGUE`；本补丁启用它。新路径直接读取 down Cartesian 结果的对角块，融合路由归约和共享专家门控，保留每个 BF16 舍入边界及原 FP32 求和顺序。其他分支走原实现。DeltaNet 分块候选未启用。

206 主机健康检查如下；其他机器沿用原端口。图捕获需数分钟。

```bash
curl -fs http://127.0.0.1:9002/health
```

应包含 `service_revision=r141`、`moe_epilogue=true`、`post_native=true`、`expert_gather=true`、`inject_native=true`、`decode_down_gmm=false`、`ctx=262144`、`worker_alive=true`。随后用本机基线复测性能和精度。

仅回退本轮融合：`FNX_MOE_EPILOGUE=0` 重启。完整回退：先停模型，将记录的备份解压回 `/root/ascend-dev`，再按原入口启动。

本机编译器位于 Flash Next 容器中。源码构建写入临时目录，避免覆盖正在使用的共享库；其他环境替换实际容器名：

```bash
FNX_R141_CONTAINER=bole-model-Flash
FNX_R141_BUILD=$(docker exec "$FNX_R141_CONTAINER" mktemp -d /tmp/fnx-r141-build.XXXXXX)
docker cp /root/qwen-ascend/kernels "$FNX_R141_CONTAINER:$FNX_R141_BUILD/kernels"
docker exec -e ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest "$FNX_R141_CONTAINER" bash "$FNX_R141_BUILD/kernels/inject/build.sh" "$FNX_R141_BUILD/libfnx_inject.so"
docker exec -e ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest "$FNX_R141_CONTAINER" bash "$FNX_R141_BUILD/kernels/expert_gather/build.sh" "$FNX_R141_BUILD/libfnx_expert_gather.so"
docker exec -e ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest "$FNX_R141_CONTAINER" bash "$FNX_R141_BUILD/kernels/post_ops/build.sh" "$FNX_R141_BUILD/libfnx_post.so"
docker exec -e ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest "$FNX_R141_CONTAINER" bash "$FNX_R141_BUILD/kernels/moe_epilogue/build.sh" "$FNX_R141_BUILD/libfnx_moe_epilogue.so"
```

通过验证并停模型后，再将临时目录里的四个共享库与配套 Python 源码一起安装。增量包包含十个运行文件及参数补丁脚本。27B 配置及切换入口保留：

```bash
/root/ascend-dev/modelctl.sh start 27b
/root/ascend-dev/modelctl.sh start flash-next
```

详细结果见 `FlashNext-r141-优化实测-20261111.md`。本轮未重跑历史 100 题。
